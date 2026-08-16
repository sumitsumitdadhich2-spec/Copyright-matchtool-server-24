/**
 * The new candidate/verification system.
 * ===========================================================================
 * Input:  MatchedSegment[] from the untouched matching engine + the two
 *         original uploaded video files.
 * Output: the finalised segment list (accepted matches, plus rejected ones
 *         kept visible and flagged) with one durable record per range.
 *
 * The whole design is one loop and one decision:
 *
 *   for each short-clip range the engine matched:
 *     take its candidates (the engine's own pick + the engine's alternates,
 *     ordered by hash confidence)
 *     cut both clips, ask Gemini "same footage?" once per candidate
 *     when a candidate is accepted, only remaining candidates with a HIGHER
 *     hash confidence are still checked; among all accepted candidates the
 *     one with the highest hash confidence wins
 *     if none are accepted, the range is dropped (kept visible, flagged)
 *
 * Deliberately NOT here (all of it was in the old system and none of it
 * improved accuracy):
 *   - no duplicate of the matching engine; alternates come from the engine's
 *     own `getAlternateCandidatesForRange` export
 *   - no embedding gate, SSCD gate, dense re-scan, or degenerate guard
 *   - no cascade of fallback verifiers that each get a second opinion
 *   - no re-fingerprinting; the only source data is MatchedSegment + the files
 *
 * Gemini (server/gemini-vlm.ts) is used exactly as-is and is the sole judge.
 */

import {
  geminiConfigured,
  geminiVerifyVideoPair,
  getGeminiStatus,
  type GeminiVerdict,
} from '../gemini-vlm';
import {
  getAlternateCandidatesForRange,
  type MatchedSegment,
} from '../matching-engine';
import { cutClip, deleteClip } from './clip';
import { buildVerificationPrompt, VIDEO1_LABEL, VIDEO2_LABEL } from './prompt';
import {
  writeRecordAsync,
  type CandidateRecord,
  type VerificationRecord,
} from './store';
import { flagTimelineOutliers, timelineConsistencyScore } from './timeline';
import { readMatchVideoMetadata } from '../video-metadata';
// RETRY-only fresh full-movie search (see fresh-search.ts). Never used by the
// bulk pass; only recheckSegment (the manual Retry entry point) calls it.
import {
  freshSearchCandidates,
  signalTier,
  SIGNAL_THRESHOLD,
  type FreshCandidate,
} from './fresh-search';

// ---------------------------------------------------------------------------
// Tunables
// ---------------------------------------------------------------------------

/** Extra candidates fetched from the engine's pool, on top of its own pick.
 *  Default raised 5 → 8 (the max): ranking puts the likeliest alternates
 *  first and accepted candidates short-circuit the rest, so a deeper pool
 *  costs nothing on easy ranges and rescues hard ones where the right match
 *  ranked low. Override with VERIFY_MAX_ALTERNATES if needed. */
const MAX_ALTERNATES = clampInt(process.env.VERIFY_MAX_ALTERNATES, 8, 0, 8);
/** Task 5 kill switch — VERIFY_RANKING_ENABLED=0 restores pure
 *  hash-confidence ordering of the alternates. */
const RANKING_ENABLED = process.env.VERIFY_RANKING_ENABLED !== '0';
/** Task 5 ranking weights: rankScore = hashConfidence (0-100 scale) plus each
 *  0..1 signal times its weight. Neutral-safe: a candidate with no timeline
 *  information scores the same as one at 0.5 consistency. */
const RANK_SPAN_WEIGHT = clampNum(process.env.VERIFY_RANK_SPAN_WEIGHT, 10, 0, 100);
const RANK_FRAMES_WEIGHT = clampNum(process.env.VERIFY_RANK_FRAMES_WEIGHT, 10, 0, 100);
const RANK_TIMELINE_WEIGHT = clampNum(process.env.VERIFY_RANK_TIMELINE_WEIGHT, 15, 0, 100);
/** Hard hash-confidence floor for CANDIDATES (alternates only — the engine's
 *  own pick from the main matching system is never filtered here). Alternates
 *  below this % are not offered as candidates at all. The candidate pool this
 *  filter sweeps is the engine's full-movie, all-variant pre-dedup scan, so
 *  filtering the whole pool IS the exhaustive "recheck the entire reference
 *  movie in every variant" for >= this confidence. */
const MIN_CANDIDATE_HASH = clampNum(process.env.VERIFY_MIN_CANDIDATE_HASH, 80, 0, 100);
/** Fallback floor: when NO alternate reaches MIN_CANDIDATE_HASH anywhere in
 *  the pool, alternates strictly above this % are still offered so Gemini VLM
 *  can verify them — one of those may be the real match. Below this they are
 *  dropped unconditionally. */
const FALLBACK_CANDIDATE_HASH = clampNum(process.env.VERIFY_FALLBACK_CANDIDATE_HASH, 78, 0, 100);
/** Tiny-fluke guard: an alternate whose speed-corrected movie span covers less
 *  than this fraction of the short range's span is a fluke (e.g. a 0.24s
 *  movie sliver "matching" a 2.12s clip) and is dropped from the candidate
 *  list. Applies to alternates only, never the engine pick. */
const MIN_SPAN_RATIO = clampNum(process.env.VERIFY_MIN_SPAN_RATIO, 0.3, 0, 1);
/** RETRY-only hash floor: a Retry candidate below this % hash confidence is
 *  DROPPED outright; at/above it, the candidate goes to Gemini VLM which gives
 *  the final verdict. Applies to fresh-search candidates AND pool fallbacks. */
const RETRY_MIN_CANDIDATE_HASH = clampNum(process.env.RETRY_MIN_CANDIDATE_HASH, 75, 0, 100);
/** RETRY-only cap on the total candidates a single Retry verifies. */
const RETRY_MAX_TOTAL_CANDIDATES = clampInt(process.env.RETRY_MAX_CANDIDATES, 20, 1, 20);
/** RETRY-only duration rule for pool-fallback candidates: the candidate's
 *  speed-corrected movie span must cover at least this fraction of the target
 *  clip's duration (a 0.20s candidate for a 2s clip cannot exist). Fresh-
 *  search candidates are full-duration by construction. */
const RETRY_MIN_DURATION_RATIO = clampNum(process.env.RETRY_MIN_DURATION_RATIO, 0.6, 0, 1);
/** Ranges verified in parallel. Gemini's own RPM pacing lives in gemini-vlm.ts. */
const CONCURRENCY = clampInt(process.env.VERIFY_CONCURRENCY, 2, 1, 8);
/** A verdict below this confidence is not trusted either way. */
const MIN_CONFIDENCE = clampInt(process.env.VERIFY_MIN_CONFIDENCE, 55, 0, 100);
/** Frame-sampling rate handed to Gemini for both clips of a pair. */
const FPS = clampInt(process.env.VERIFY_FPS, 2, 1, 24);

function clampInt(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, Math.round(n)));
}

function clampNum(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number(raw);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

// ---------------------------------------------------------------------------
// Public contract
// ---------------------------------------------------------------------------

export interface VerifyRequest {
  /** Raw segments from matchVideosFromFiles(). Never mutated. */
  segments: MatchedSegment[];
  /** The engine's pre-dedup pool, used only to offer alternates. */
  candidatePool?: MatchedSegment[];
  /** Absolute path of the uploaded short clip (undefined when not retained). */
  shortVideoPath: string | undefined;
  /** Absolute path of the uploaded reference movie (undefined when not retained). */
  movieVideoPath: string | undefined;
  /** Where per-range records are written. */
  uploadDir: string;
  matchJobId: string;
  onProgress?: (done: number, total: number, message: string) => void;
  /** Live step-by-step log lines (used by the manual Retry flow so the UI can
   *  show exactly which candidate is being checked right now). Best-effort:
   *  the bulk pass never sets it. */
  onLog?: (message: string) => void;
}

export interface VerifySummary {
  /** False when verification did not run at all. `reason` says why. */
  ran: boolean;
  reason: string;
  rangesTotal: number;
  rangesVerified: number;
  accepted: number;
  rejected: number;
  unverifiable: number;
  /** Ranges whose match came from an alternate rather than the engine's pick. */
  switched: number;
  geminiCalls: number;
}

export interface VerifyResult {
  segments: MatchedSegment[];
  summary: VerifySummary;
}

/**
 * Verify every matched segment. Never throws and never rejects: any failure
 * degrades to "unverifiable" for that range and the engine's original segment
 * is kept, so a match job always completes.
 */
export async function verifyMatchedSegments(req: VerifyRequest): Promise<VerifyResult> {
  const total = req.segments.length;

  if (total === 0) {
    console.log('[Verify] No matched segments — nothing to verify.');
    return { segments: [], summary: emptySummary('no matched segments', 0) };
  }

  // --- Graceful degradation, loudly ------------------------------------------
  // The old system's worst bug was silently no-opping when its provider was
  // unconfigured. Every skip path below names itself explicitly.
  if (!req.shortVideoPath || !req.movieVideoPath) {
    const reason =
      'original uploaded video file(s) are no longer on disk — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  if (!geminiConfigured()) {
    const reason =
      'GEMINI_API_KEY is not set — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  const status = getGeminiStatus();
  if (status.dailyLimitReached) {
    const reason =
      'Gemini daily quota exhausted on every model — verification SKIPPED, all segments pass through unverified';
    console.warn(`[Verify] ${reason}.`);
    await writeSkippedRecords(req, reason);
    return {
      segments: flagTimelineOutliers(req.segments),
      summary: emptySummary(reason, total),
    };
  }

  console.log(
    `[Verify] Verifying ${total} matched range(s) with Gemini (model=${status.model}, ` +
    `concurrency=${CONCURRENCY}, alternates<=${MAX_ALTERNATES}, minConfidence=${MIN_CONFIDENCE}).`,
  );

  const summary: VerifySummary = {
    ran: true,
    reason: 'verified with Gemini',
    rangesTotal: total,
    rangesVerified: 0,
    accepted: 0,
    rejected: 0,
    unverifiable: 0,
    switched: 0,
    geminiCalls: 0,
  };

  // Stable range index by short-clip order, so record files line up with the
  // order the UI renders and survive a re-check later.
  const ordered = [...req.segments].sort((a, b) => a.shortStart - b.shortStart);
  const finalised: Array<MatchedSegment | null> = new Array(ordered.length).fill(null);

  // Task 1/5 debug metadata (fps, VFR) — read once, stamped on every record.
  const videoMeta = loadVideoMetaForRecords(req.uploadDir, req.matchJobId);

  let done = 0;
  let cursor = 0;

  const runWorker = async (): Promise<void> => {
    for (;;) {
      const index = cursor++;
      if (index >= ordered.length) return;

      // Task 5 timeline consistency ranks against the OTHER ranges of the clip.
      const neighbors = ordered.filter((_, j) => j !== index);
      const outcome = await verifyOneRange(req, ordered[index], index, neighbors, videoMeta);

      summary.geminiCalls += outcome.calls;
      summary.rangesVerified++;
      if (outcome.record.dropped) {
        summary.rejected++;
      } else if (outcome.record.candidates.some(c => c.verdict === 'accepted')) {
        summary.accepted++;
        if ((outcome.record.usedCandidateIndex ?? 0) > 0) summary.switched++;
      } else {
        summary.unverifiable++;
      }

      finalised[index] = outcome.segment;
      await writeRecordAsync(req.uploadDir, req.matchJobId, outcome.record);

      done++;
      req.onProgress?.(done, ordered.length, `Verified ${done}/${ordered.length} matched ranges`);
    }
  };

  await Promise.all(
    Array.from({ length: Math.min(CONCURRENCY, ordered.length) }, () => runWorker()),
  );

  const segments = flagTimelineOutliers(
    finalised.filter((s): s is MatchedSegment => s !== null),
  );

  console.log(
    `[Verify] Done: ${summary.accepted} accepted, ${summary.rejected} rejected, ` +
    `${summary.unverifiable} unverifiable, ${summary.switched} switched to an alternate ` +
    `(${summary.geminiCalls} Gemini call(s)).`,
  );

  return { segments, summary };
}

// ---------------------------------------------------------------------------
// One range
// ---------------------------------------------------------------------------

interface RangeOutcome {
  segment: MatchedSegment;
  record: VerificationRecord;
  calls: number;
}

async function verifyOneRange(
  req: VerifyRequest,
  primary: MatchedSegment,
  segmentIndex: number,
  neighbors: MatchedSegment[],
  videoMeta: VerificationRecord['videoMetadata'],
  /** When true (manual Retry), EVERY candidate is verified — no first-accept
   *  early stop. Among all Gemini-accepted candidates the one with the highest
   *  HASH confidence wins (same rule as the bulk pass). */
  checkAll = false,
  /** RETRY-only: a pre-built candidate list (fresh full-movie search) that
   *  replaces the pool-based collectCandidates. The bulk pass never sets it. */
  presetCandidates?: RankedCandidate[],
): Promise<RangeOutcome> {
  const candidates = presetCandidates ?? collectCandidates(primary, req.candidatePool, neighbors);
  req.onLog?.(
    presetCandidates
      ? `${candidates.length} candidate(s) queued for clip ${fmt(primary.shortStart)}s–${fmt(primary.shortEnd)}s ` +
        `(current pick + ${candidates.length - 1} from the fresh full-movie search).`
      : `Collected ${candidates.length} candidate(s) for clip ${fmt(primary.shortStart)}s–${fmt(primary.shortEnd)}s ` +
        `(engine pick + ${candidates.length - 1} alternate(s) from the candidate pool).`,
  );

  const record: VerificationRecord = {
    segmentIndex,
    shortStart: primary.shortStart,
    shortEnd: primary.shortEnd,
    recordedAt: Date.now(),
    candidates: candidates.map(c => ({
      segment: c.segment,
      checked: false,
      hashConfidence: c.segment.confidence,
      rankScore: c.rankScore,
      rankSignals: c.rankSignals,
      retrySignals: c.retrySignals,
    })),
    dropped: false,
    videoMetadata: videoMeta,
  };

  // The short-clip side is identical for every candidate of this range, so it
  // is cut once and reused across all Gemini calls.
  const shortClip = await cutClip(
    req.shortVideoPath!,
    primary.shortStart,
    primary.shortEnd,
    `short-${segmentIndex}`,
  );

  if (!shortClip) {
    const reason = 'could not cut the short-clip side with ffmpeg — range left unverified';
    console.warn(`[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}): ${reason}.`);
    req.onLog?.(`FAILED: ${reason}.`);
    record.skippedReason = reason;
    record.usedCandidateIndex = 0;
    return { segment: primary, record, calls: 0 };
  }

  let calls = 0;
  /** Bulk pass: hash confidence of the best candidate accepted so far. Once a
   *  candidate is accepted, only remaining candidates with a STRICTLY higher
   *  hash confidence are still worth a Gemini call — if one of those is also
   *  accepted, the higher-hash-confidence candidate wins the range. */
  let acceptedHash: number | null = null;
  try {
    for (let i = 0; i < candidates.length; i++) {
      const candidate = candidates[i].segment;
      const entry = record.candidates[i];

      if (!checkAll && acceptedHash !== null && candidate.confidence <= acceptedHash) {
        entry.reason =
          `skipped — an accepted candidate already has equal or higher hash confidence ` +
          `(${Math.round(acceptedHash)}% >= ${Math.round(candidate.confidence)}%)`;
        req.onLog?.(
          `Candidate ${i + 1}: SKIPPED — hash confidence ${Math.round(candidate.confidence)}% is not higher ` +
          `than the already-accepted candidate's ${Math.round(acceptedHash)}%.`,
        );
        continue;
      }

      req.onLog?.(
        `Checking candidate ${i + 1}/${candidates.length} @ movie ${fmt(candidate.movieStart)}s–${fmt(candidate.movieEnd)}s ` +
        `(hash confidence ${Math.round(candidate.confidence)}%)…`,
      );

      const movieClip = await cutClip(
        req.movieVideoPath!,
        candidate.movieStart,
        candidate.movieEnd,
        `movie-${segmentIndex}-${i}`,
      );
      if (!movieClip) {
        entry.verdict = 'unverifiable';
        entry.reason = 'reference clip could not be cut';
        req.onLog?.(`Candidate ${i + 1}: UNVERIFIABLE — reference clip could not be cut.`);
        continue;
      }

      let verdict: GeminiVerdict | null = null;
      try {
        verdict = await geminiVerifyVideoPair(
          movieClip,
          shortClip,
          VIDEO1_LABEL,
          VIDEO2_LABEL,
          buildVerificationPrompt({
            movieStart: candidate.movieStart,
            movieEnd: candidate.movieEnd,
            shortStart: candidate.shortStart,
            shortEnd: candidate.shortEnd,
            speedRatio: candidate.speedRatio,
          }),
          FPS,
        );
        calls++;
      } catch (e: any) {
        console.warn(
          `[Verify] Range ${segmentIndex} candidate ${i}: Gemini call threw (${e?.message || e}) — unverifiable.`,
        );
      } finally {
        deleteClip(movieClip);
      }

      entry.checked = verdict !== null;

      if (!verdict) {
        entry.verdict = 'unverifiable';
        entry.reason = 'Gemini returned no usable verdict';
        req.onLog?.(`Candidate ${i + 1}: UNVERIFIABLE — Gemini returned no usable verdict.`);
        continue;
      }

      entry.confidencePct = Math.round(verdict.confidence);
      entry.reason = (verdict.evidence || []).join('; ').slice(0, 400) || undefined;

      const trusted = verdict.confidence >= MIN_CONFIDENCE;
      if (verdict.same && trusted) {
        entry.verdict = 'accepted';
        console.log(
          `[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}) ` +
          `ACCEPTED candidate ${i} @ movie ${fmt(candidate.movieStart)}s ` +
          `(Gemini ${entry.confidencePct}%${i > 0 ? ', switched from the engine pick' : ''}).`,
        );
        req.onLog?.(`Candidate ${i + 1}: ACCEPTED by Gemini (${entry.confidencePct}% confident).`);
        if (!checkAll) {
          // Bulk pass: an accepted candidate settles the range UNLESS a later
          // candidate has a strictly higher hash confidence — those still get
          // their own Gemini check, and if also accepted, the higher hash
          // confidence wins. Everything else is skipped to save quota.
          acceptedHash = Math.max(acceptedHash ?? -Infinity, candidate.confidence);
          const higherLeft = candidates
            .slice(i + 1)
            .filter(c => c.segment.confidence > acceptedHash!).length;
          if (higherLeft === 0) break;
          req.onLog?.(
            `${higherLeft} remaining candidate(s) have a higher hash confidence than the accepted one — ` +
            `checking those too; the highest hash confidence accepted candidate will win.`,
          );
          continue;
        }
        // checkAll (manual Retry): keep verifying the remaining candidates so
        // the highest-hash-confidence accepted match wins, not just the first
        // acceptable one.
        continue;
      }

      if (!verdict.same && trusted) {
        entry.verdict = 'rejected';
        req.onLog?.(`Candidate ${i + 1}: REJECTED by Gemini (${entry.confidencePct}% confident it is different footage).`);
      } else {
        entry.verdict = 'unverifiable';
        entry.reason =
          `low certainty (${entry.confidencePct}% < ${MIN_CONFIDENCE}%)` +
          (entry.reason ? ` — ${entry.reason}` : '');
        req.onLog?.(`Candidate ${i + 1}: UNVERIFIABLE — low certainty (${entry.confidencePct}% < ${MIN_CONFIDENCE}%).`);
      }
    }
  } finally {
    deleteClip(shortClip);
  }

  // Winner selection among accepted candidates — both the bulk pass AND the
  // manual Retry (checkAll): when Gemini accepts more than one candidate, the
  // highest HASH confidence wins. Gemini confidence breaks ties, then the
  // earlier/higher-ranked candidate. This is the whole point of still checking
  // higher-hash candidates after a first accept.
  const acceptedEntries = record.candidates
    .map((c, i) => ({ c, i }))
    .filter(x => x.c.verdict === 'accepted');
  if (acceptedEntries.length > 0) {
    acceptedEntries.sort((a, b) =>
      (b.c.hashConfidence ?? 0) - (a.c.hashConfidence ?? 0) ||
      (b.c.confidencePct ?? 0) - (a.c.confidencePct ?? 0) ||
      a.i - b.i,
    );
    const winnerIdx = acceptedEntries[0].i;
    record.usedCandidateIndex = winnerIdx;
    const winner = candidates[winnerIdx].segment;
    console.log(
      `[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}) ` +
      `${checkAll ? 'full check' : 'bulk pass'}: ${acceptedEntries.length}/${record.candidates.length} accepted — ` +
      `winner is candidate ${winnerIdx} @ movie ${fmt(winner.movieStart)}s ` +
      `(hash ${Math.round(record.candidates[winnerIdx].hashConfidence ?? 0)}%, ` +
      `Gemini ${record.candidates[winnerIdx].confidencePct}%` +
      `${winnerIdx > 0 ? ', switched from the engine pick' : ''}).`,
    );
    req.onLog?.(
      `Check done: ${acceptedEntries.length}/${record.candidates.length} accepted — ` +
      `WINNER is candidate ${winnerIdx + 1} @ movie ${fmt(winner.movieStart)}s ` +
      `(hash confidence ${Math.round(record.candidates[winnerIdx].hashConfidence ?? 0)}%, ` +
      `Gemini ${record.candidates[winnerIdx].confidencePct}% confident).`,
    );
    return { segment: winner, record, calls };
  }

  // Nobody won. Keep the engine's own pick visible so the user can review it
  // and hit re-check, but mark it clearly as not a verified match.
  const anyRejected = record.candidates.some(c => c.verdict === 'rejected');
  record.dropped = anyRejected;
  record.usedCandidateIndex = 0;
  if (!anyRejected) {
    record.skippedReason = 'no candidate could be judged — Gemini gave no usable verdict';
  }

  console.log(
    `[Verify] Range ${segmentIndex} (${fmt(primary.shortStart)}-${fmt(primary.shortEnd)}): ` +
    `${anyRejected ? 'REJECTED' : 'UNVERIFIABLE'} after ${record.candidates.length} candidate(s) — ` +
    `keeping the engine pick, flagged for review.`,
  );
  req.onLog?.(
    `Full check done: no candidate accepted after ${record.candidates.length} candidate(s) — ` +
    `${anyRejected ? 'REJECTED (kept visible for review)' : 'UNVERIFIABLE (Gemini gave no usable verdict)'}.`,
  );

  return {
    segment: { ...primary, vlmRejectedKept: anyRejected },
    record,
    calls,
  };
}

/** A candidate plus its Task 5 ranking metadata (absent for the engine pick
 *  and when ranking is disabled). */
interface RankedCandidate {
  segment: MatchedSegment;
  rankScore?: number;
  rankSignals?: NonNullable<CandidateRecord['rankSignals']>;
  /** RETRY-only: the fresh search's 4-signal scores for this candidate. */
  retrySignals?: NonNullable<CandidateRecord['retrySignals']>;
}

/**
 * The engine's own pick ALWAYS first (its structure is untouched by Task 5),
 * then its alternates for the same short range — ordered by the Task 5 rank
 * score when ranking is enabled, by raw hash confidence otherwise. Alternates
 * pointing at essentially the same movie timestamp as an earlier candidate
 * are dropped — re-asking Gemini about the same footage is pure quota waste.
 */
function collectCandidates(
  primary: MatchedSegment,
  pool: MatchedSegment[] | undefined,
  neighbors: MatchedSegment[],
): RankedCandidate[] {
  const out: RankedCandidate[] = [{ segment: primary }];
  if (MAX_ALTERNATES === 0) return out;

  // The pool is the engine's full-movie, all-variant pre-dedup scan, so the
  // filtering below sweeps every candidate the exhaustive scan ever produced
  // for this range — the "recheck the whole reference movie in every variant"
  // happens by filtering the complete pool, not by re-running the scan.
  const rawAlternates = getAlternateCandidatesForRange(
    pool,
    primary.shortStart,
    primary.shortEnd,
    [],
    0.5,
  );

  // 1. Tiny-fluke guard: a candidate whose speed-corrected movie span covers
  //    only a sliver of the short range (e.g. 0.24s vs a 2.12s clip) is a
  //    hash fluke, not a real match — drop it outright.
  const shortSpan = Math.max(0, primary.shortEnd - primary.shortStart);
  const solid = rawAlternates.filter(
    a => speedCorrectedSpanRatio(a, shortSpan) >= MIN_SPAN_RATIO,
  );

  // 2. Hash-confidence floor: only alternates >= MIN_CANDIDATE_HASH% become
  //    candidates. If the entire pool has none, fall back to alternates
  //    strictly above FALLBACK_CANDIDATE_HASH% so Gemini VLM can still verify
  //    them — one of those may be the real match. Anything lower is dropped.
  const strong = solid.filter(a => a.confidence >= MIN_CANDIDATE_HASH);
  const alternates = strong.length > 0
    ? strong
    : solid.filter(a => a.confidence > FALLBACK_CANDIDATE_HASH);

  if (rawAlternates.length !== alternates.length) {
    console.log(
      `[Verify] Candidate filter for clip ${fmt(primary.shortStart)}s–${fmt(primary.shortEnd)}s: ` +
      `${rawAlternates.length} raw alternate(s) → ${solid.length} after tiny-span guard ` +
      `(min ratio ${MIN_SPAN_RATIO}) → ${alternates.length} after hash floor ` +
      `(${strong.length > 0
        ? `>= ${MIN_CANDIDATE_HASH}%`
        : `no alternate >= ${MIN_CANDIDATE_HASH}% in the whole pool — fallback > ${FALLBACK_CANDIDATE_HASH}% for Gemini verification`}).`,
    );
  }

  const ordered = RANKING_ENABLED
    ? rankAlternates(alternates, primary, neighbors)
    : alternates.map((segment): RankedCandidate => ({ segment }));

  for (const alt of ordered) {
    if (out.length >= MAX_ALTERNATES + 1) break;
    if (out.some(existing => Math.abs(existing.segment.movieStart - alt.segment.movieStart) < 1)) continue;
    out.push(alt);
  }
  return out;
}

/** How well a candidate's movie span covers the short range's span after
 *  correcting for the candidate's own playback speed. 1 = perfect coverage,
 *  0.11 = the 0.24s-sliver-vs-2.12s-clip fluke case. */
function speedCorrectedSpanRatio(segment: MatchedSegment, shortSpan: number): number {
  const speed = Number.isFinite(segment.speedRatio) && segment.speedRatio > 0
    ? segment.speedRatio
    : 1;
  const expectedSpan = shortSpan * speed;
  const actualSpan = Math.max(0, segment.movieEnd - segment.movieStart);
  if (expectedSpan <= 0 || actualSpan <= 0) return 0;
  return Math.min(expectedSpan, actualSpan) / Math.max(expectedSpan, actualSpan);
}

/**
 * Task 5: order alternates for verification by more than hash confidence.
 *
 *   rankScore = hashConfidence (0-100)
 *             + spanScore     × VERIFY_RANK_SPAN_WEIGHT      (default 10)
 *             + frameScore    × VERIFY_RANK_FRAMES_WEIGHT    (default 10)
 *             + timelineScore × VERIFY_RANK_TIMELINE_WEIGHT  (default 15)
 *
 *   spanScore     0..1 — how close the candidate's movie span is to the short
 *                 range's span after correcting for the candidate's own speed
 *                 (a 10s short range at 2x speed should cover ~20s of movie).
 *   frameScore    0..1 — multi-frame hash agreement: how many frames back this
 *                 candidate relative to the best-supported alternate, so a
 *                 high-average-but-3-frame fluke stops outranking a slightly
 *                 lower-average candidate backed by 40 frames.
 *   timelineScore 0..1 — agreement with the continuous movie section the
 *                 clip's OTHER segments point at (timeline.ts). Null (no
 *                 usable neighbours / non-linear edit) is NEUTRAL: scored as
 *                 0.5, identical for every candidate of the range.
 *
 * Ranking only changes the ORDER Gemini is asked in; it accepts nothing by
 * itself, so the worst possible outcome of a bad rank is spending quota in a
 * suboptimal order — never a wrong match.
 */
function rankAlternates(
  alternates: MatchedSegment[],
  primary: MatchedSegment,
  neighbors: MatchedSegment[],
): RankedCandidate[] {
  if (alternates.length === 0) return [];

  const shortSpan = Math.max(0, primary.shortEnd - primary.shortStart);
  const shortMid = (primary.shortStart + primary.shortEnd) / 2;
  const maxFrames = Math.max(1, ...alternates.map(a => a.frameCount || 0));

  const ranked = alternates.map((segment): RankedCandidate => {
    const speed = Number.isFinite(segment.speedRatio) && segment.speedRatio > 0
      ? segment.speedRatio
      : 1;
    const expectedSpan = shortSpan * speed;
    const actualSpan = Math.max(0, segment.movieEnd - segment.movieStart);
    const spanScore = expectedSpan > 0 && actualSpan > 0
      ? Math.min(expectedSpan, actualSpan) / Math.max(expectedSpan, actualSpan)
      : 0;
    const frameScore = (segment.frameCount || 0) / maxFrames;
    const timelineScore = timelineConsistencyScore(
      neighbors,
      shortMid,
      (segment.movieStart + segment.movieEnd) / 2,
    );

    const rankScore =
      segment.confidence +
      spanScore * RANK_SPAN_WEIGHT +
      frameScore * RANK_FRAMES_WEIGHT +
      (timelineScore ?? 0.5) * RANK_TIMELINE_WEIGHT;

    return {
      segment,
      rankScore,
      rankSignals: { hashConfidence: segment.confidence, spanScore, frameScore, timelineScore },
    };
  });

  // Primary order: raw HASH confidence — the highest-hash candidates are
  // always shown/asked first. Secondary order: the Task 5 rank score (span,
  // frame support, timeline consistency), which still breaks ties between
  // candidates the hash search believed in equally. Nothing was removed from
  // the rank signals; hash-first ordering was added on top.
  ranked.sort(
    (a, b) =>
      b.segment.confidence - a.segment.confidence ||
      (b.rankScore ?? 0) - (a.rankScore ?? 0),
  );
  return ranked;
}

/**
 * RETRY-only candidate assembly. Builds the final ordered candidate list that
 * recheckSegment hands to verifyOneRange:
 *
 *   1. The range's CURRENT pick is always candidate 0 (Gemini re-judges it —
 *      the previous verdict is never trusted).
 *   2. Fresh full-movie search results are the candidate source. Only when the
 *      fresh search is unavailable (fingerprints gone) does the saved engine
 *      pool serve as fallback — with the same Retry rules applied.
 *   3. HARD 75% hash floor (RETRY_MIN_CANDIDATE_HASH): any candidate below it
 *      is DROPPED outright; at/above it, Gemini VLM gets the final say.
 *   4. Duration rule for pool fallbacks: a candidate whose speed-corrected
 *      movie span covers < RETRY_MIN_DURATION_RATIO of the target clip's
 *      duration cannot exist (fresh-search candidates are full-duration by
 *      construction and were already filtered in fresh-search.ts).
 *   5. 4-signal tier ordering (structure/color/background/human-edge):
 *      all 4 > SIGNAL_THRESHOLD → verified FIRST; any signal below it →
 *      verified LAST; signals unavailable → middle. Hash confidence orders
 *      within a tier. Signals only filter/rank — Gemini decides.
 *   6. Near-duplicate movie locations are dropped, and the total list is
 *      capped at RETRY_MAX_TOTAL_CANDIDATES (20).
 */
function buildRetryCandidates(
  current: MatchedSegment,
  fresh: FreshCandidate[] | null,
  pool: MatchedSegment[] | undefined,
  onLog?: (message: string) => void,
): RankedCandidate[] {
  // The current pick is always re-judged by Gemini, never filtered here.
  const out: RankedCandidate[] = [{ segment: current }];
  const targetSpan = Math.max(0, current.shortEnd - current.shortStart);

  let source: RankedCandidate[];

  if (fresh) {
    // Fresh full-movie search results: apply the hard 75% hash floor, then
    // order by signal tier (all-4-above-threshold first, any-below last).
    const kept = fresh.filter(c => c.segment.confidence >= RETRY_MIN_CANDIDATE_HASH);
    if (kept.length !== fresh.length) {
      onLog?.(
        `${fresh.length - kept.length} fresh candidate(s) DROPPED below the ` +
        `${Math.round(RETRY_MIN_CANDIDATE_HASH)}% hash-confidence floor; ` +
        `${kept.length} go on to Gemini VLM for the final verdict.`,
      );
    }
    kept.sort(
      (a, b) =>
        signalTier(a.signals) - signalTier(b.signals) ||
        b.segment.confidence - a.segment.confidence,
    );
    const tier0 = kept.filter(c => signalTier(c.signals) === 0).length;
    const tier2 = kept.filter(c => signalTier(c.signals) === 2).length;
    if (kept.length > 0) {
      onLog?.(
        `Signal ranking (threshold ${Math.round(SIGNAL_THRESHOLD * 100)}% on all 4 of ` +
        `structure/color/background/human-edge): ${tier0} top-priority, ` +
        `${kept.length - tier0 - tier2} neutral, ${tier2} last-priority candidate(s).`,
      );
    }
    source = kept.map((c): RankedCandidate => ({
      segment: c.segment,
      retrySignals: c.signals ?? undefined,
    }));
  } else {
    // Fallback ONLY when fingerprints are gone: sweep the engine's saved
    // full-movie pool with the same Retry rules (duration + 75% hash floor).
    const rawAlternates = getAlternateCandidatesForRange(
      pool,
      current.shortStart,
      current.shortEnd,
      [],
      0.5,
    );
    const durationOk = rawAlternates.filter(
      a => speedCorrectedSpanRatio(a, targetSpan) >= RETRY_MIN_DURATION_RATIO,
    );
    const kept = durationOk.filter(a => a.confidence >= RETRY_MIN_CANDIDATE_HASH);
    onLog?.(
      `Fresh search unavailable — falling back to the saved candidate pool: ` +
      `${rawAlternates.length} alternate(s) → ${durationOk.length} after the duration rule ` +
      `(span >= ${Math.round(RETRY_MIN_DURATION_RATIO * 100)}% of the target clip) → ` +
      `${kept.length} at/above the ${Math.round(RETRY_MIN_CANDIDATE_HASH)}% hash floor.`,
    );
    kept.sort((a, b) => b.confidence - a.confidence);
    source = kept.map((segment): RankedCandidate => ({ segment }));
  }

  for (const cand of source) {
    if (out.length >= RETRY_MAX_TOTAL_CANDIDATES) break;
    if (out.some(existing => Math.abs(existing.segment.movieStart - cand.segment.movieStart) < 1)) {
      continue;
    }
    out.push(cand);
  }
  return out;
}

/** Job-level fps/VFR metadata for the records, read once per verify pass.
 *  Best-effort: null whenever the probe file is absent or unreadable. */
function loadVideoMetaForRecords(
  uploadDir: string,
  matchJobId: string,
): VerificationRecord['videoMetadata'] {
  try {
    const meta = readMatchVideoMetadata(uploadDir, matchJobId);
    if (!meta) return null;
    return {
      shortDeclaredFps: meta.short?.declaredFps ?? null,
      shortAverageFps: meta.short?.averageFps ?? null,
      shortIsVFR: meta.short?.isVFR ?? false,
      movieDeclaredFps: meta.movie?.declaredFps ?? null,
      movieAverageFps: meta.movie?.averageFps ?? null,
      movieIsVFR: meta.movie?.isVFR ?? false,
    };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Manual re-check of a single range (used by the preview UI's retry action)
// ---------------------------------------------------------------------------

export interface RecheckRequest extends Omit<VerifyRequest, 'segments' | 'onProgress'> {
  /** The range's current segment, as shown in the UI. */
  segment: MatchedSegment;
  /** Its stable range index — the record file to overwrite. */
  segmentIndex: number;
}

export interface RecheckResult {
  segment: MatchedSegment;
  record: VerificationRecord;
  /** True when a candidate was accepted this time. */
  accepted: boolean;
  /** Human-readable outcome for the UI toast. */
  message: string;
}

/**
 * Re-run verification for exactly one range and overwrite its record.
 * Same code path as the bulk pass, with one deliberate difference: Retry
 * verifies EVERY candidate (no first-accept early stop) and picks the
 * accepted candidate with the highest HASH confidence. It also never checks
 * the daily quota gate — a manual Retry always attempts the full check.
 */
export async function recheckSegment(req: RecheckRequest): Promise<RecheckResult> {
  if (!req.shortVideoPath || !req.movieVideoPath || !geminiConfigured()) {
    const message = !geminiConfigured()
      ? 'GEMINI_API_KEY is not set — cannot re-check this segment'
      : 'original uploaded video file(s) are no longer on disk — cannot re-check this segment';
    console.warn(`[Verify] Re-check of range ${req.segmentIndex} refused: ${message}.`);
    const record: VerificationRecord = {
      segmentIndex: req.segmentIndex,
      shortStart: req.segment.shortStart,
      shortEnd: req.segment.shortEnd,
      recordedAt: Date.now(),
      candidates: [{ segment: req.segment, checked: false, hashConfidence: req.segment.confidence }],
      usedCandidateIndex: 0,
      dropped: false,
      skippedReason: message,
    };
    await writeRecordAsync(req.uploadDir, req.matchJobId, record);
    return { segment: req.segment, record, accepted: false, message };
  }

  console.log(`[Verify] Manual re-check requested for range ${req.segmentIndex}.`);
  req.onLog?.(
    `Retry started — FRESH full-movie re-search of the target clip, then a full check: ` +
    `every surviving candidate gets its own Gemini verification and Gemini gives the final verdict.`,
  );

  // Retry rule 1: do NOT depend on the previous result — re-search the target
  // clip against the ENTIRE reference movie from the fingerprint files.
  // Falls back to the saved candidate pool only when fingerprints are gone.
  const fresh = await freshSearchCandidates({
    uploadDir: req.uploadDir,
    matchJobId: req.matchJobId,
    shortStart: req.segment.shortStart,
    shortEnd: req.segment.shortEnd,
    onLog: req.onLog,
  });

  const presetCandidates = buildRetryCandidates(
    req.segment,
    fresh?.candidates ?? null,
    req.candidatePool,
    req.onLog,
  );
  req.onLog?.(
    `${presetCandidates.length} candidate(s) will be verified (max ${RETRY_MAX_TOTAL_CANDIDATES}) — ` +
    `hash/signals only filtered and ordered them; Gemini VLM makes the final match/no-match decision on each.`,
  );

  const outcome = await verifyOneRange(
    {
      segments: [req.segment],
      candidatePool: req.candidatePool,
      shortVideoPath: req.shortVideoPath,
      movieVideoPath: req.movieVideoPath,
      uploadDir: req.uploadDir,
      matchJobId: req.matchJobId,
      onLog: req.onLog,
    },
    req.segment,
    req.segmentIndex,
    // A single-range re-check has no sibling ranges in scope; timeline
    // consistency degrades to neutral, which is exactly the safe behavior.
    [],
    loadVideoMetaForRecords(req.uploadDir, req.matchJobId),
    // Manual Retry always runs the FULL check: every candidate is verified
    // (up to one Gemini call each) and the accepted candidate with the
    // highest HASH confidence wins. A deliberate quota trade-off.
    true,
    // The fresh-search candidate list replaces the pool-based collection.
    presetCandidates,
  );

  await writeRecordAsync(req.uploadDir, req.matchJobId, outcome.record);

  const winner = outcome.record.candidates[outcome.record.usedCandidateIndex ?? 0];
  const accepted = winner?.verdict === 'accepted';
  const message = accepted
    ? `Confirmed as a match (Gemini ${winner.confidencePct ?? 0}% confident)`
    : outcome.record.dropped
      ? 'Gemini rejected every candidate for this segment'
      : outcome.record.skippedReason || 'Gemini could not judge this segment';

  return { segment: outcome.segment, record: outcome.record, accepted, message };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function emptySummary(reason: string, total: number): VerifySummary {
  return {
    ran: false,
    reason,
    rangesTotal: total,
    rangesVerified: 0,
    accepted: 0,
    rejected: 0,
    unverifiable: 0,
    switched: 0,
    geminiCalls: 0,
  };
}

/** One record per range explaining why nothing was checked, so the UI can say so. */
async function writeSkippedRecords(req: VerifyRequest, reason: string): Promise<void> {
  const ordered = [...req.segments].sort((a, b) => a.shortStart - b.shortStart);
  await Promise.all(
    ordered.map((segment, segmentIndex) =>
      writeRecordAsync(req.uploadDir, req.matchJobId, {
        segmentIndex,
        shortStart: segment.shortStart,
        shortEnd: segment.shortEnd,
        recordedAt: Date.now(),
        candidates: [{ segment, checked: false, hashConfidence: segment.confidence }],
        usedCandidateIndex: 0,
        dropped: false,
        skippedReason: reason,
      }),
    ),
  );
}

function fmt(seconds: number): string {
  return seconds.toFixed(2);
}
