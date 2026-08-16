/**
 * RETRY-ONLY fresh full-movie search.
 * ---------------------------------------------------------------------------
 * When the user hits Retry, the target short-clip range is re-searched against
 * the ENTIRE reference movie from the fingerprint files on disk — no
 * dependence on the previous match result or the saved candidate pool.
 *
 * How it works (duration-aware smart search):
 *   1. Load both fingerprint PreSets (short + movie) via the shared
 *      movie-asset cache, so repeated retries don't re-stream a 2-hour file.
 *   2. Probe stage: every movie frame is probed as a potential time-alignment
 *      with the target clip (cheap full-variant aHash on 3 spread frames).
 *      The best alignments survive to the run stage.
 *   3. Run stage: for each surviving alignment, every target-clip frame is
 *      compared (cross-variant aHash+dHash+pHash, mirror-aware) against the
 *      time-aligned movie frame. Maximal stretches of CONSECUTIVE frames with
 *      >= RUN_MIN_SIM% hash similarity and >= RUN_MIN_FRAMES frames become
 *      candidate seeds — even when the frames before/after the run don't match.
 *   4. Duration-aware extension: a run that only covers the MIDDLE of the
 *      target clip is extended in the movie on BOTH sides by exactly the
 *      short-clip time remaining before/after the matched frames, so every
 *      candidate spans the target clip's full duration. Tiny slivers (e.g. a
 *      0.20s "match" for a 2s clip) therefore cannot survive: they either
 *      extend to the full duration or are rejected at the movie's edges.
 *   5. Multi-signal scoring: per run, 4 signals are computed from the frame
 *      signatures — structure (hash), color (colorGrid), background
 *      (outer-ring color+detail), humanEdge (skinGrid+detailGrid).
 *      These are used for FILTERING/RANKING only; Gemini VLM stays the sole
 *      final judge of every candidate.
 *
 * This module lives in server/retry-verification/ ONLY. It never touches the
 * matching engine's own passes or server/verification/. It only READS the
 * engine's public exports and the fingerprint files server.ts already writes.
 */

import * as fs from 'fs';
import * as path from 'path';
import {
  streamPrecomputeFromFile,
  type MatchedSegment,
  type PreSet,
} from '../matching-engine';
import { getOrLoadMovieAsset } from '../movie-asset-cache';
import type { FrameSignature } from '../../src/shared/fingerprint';

// ---------------------------------------------------------------------------
// Tunables (retry-system only)
// ---------------------------------------------------------------------------

/** A run needs at least this many CONSECUTIVE matching frames to seed a candidate. */
const RUN_MIN_FRAMES = clampInt(process.env.RETRY_RUN_MIN_FRAMES, 5, 2, 50);
/** Per-frame hash similarity (%) required for a frame to belong to a run. */
const RUN_MIN_SIM = clampNum(process.env.RETRY_RUN_MIN_SIM, 80, 50, 100);
/** Maximum candidates the fresh search hands to verification. */
const MAX_FRESH_CANDIDATES = clampInt(process.env.RETRY_MAX_CANDIDATES, 20, 1, 20);
/** Probe stage: minimum cheap-probe similarity (%) for an alignment to survive. */
const PROBE_MIN_SIM = clampNum(process.env.RETRY_PROBE_MIN_SIM, 62, 0, 100);
/** Probe stage: how many best alignments move on to the expensive run stage. */
const PROBE_KEEP = clampInt(process.env.RETRY_PROBE_KEEP, 300, 20, 2000);
/** A candidate whose (edge-clamped) movie span covers less than this fraction
 *  of the target clip's duration is rejected — candidates must match the
 *  target clip's length. */
const MIN_DURATION_RATIO = clampNum(process.env.RETRY_MIN_DURATION_RATIO, 0.6, 0, 1);
/** Candidates whose movieStart is closer than this (seconds) are duplicates. */
const DEDUP_SECONDS = 1.0;

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

/** The 4 retry ranking signals, each 0..1. Null when frame signatures are
 *  missing from the fingerprint files (older fingerprints). */
export interface RetrySignals {
  /** Structural hash agreement over the run (aHash+dHash+pHash blend). */
  structure: number;
  /** colorGrid agreement over the run. */
  color: number;
  /** Outer-ring (non-center 4x4 cells) color+detail agreement — background. */
  background: number;
  /** skinScoreGrid + detailGrid agreement — human character / edge detail. */
  humanEdge: number;
}

export interface FreshCandidate {
  /** Same on-disk shape the whole app already uses — record format unchanged. */
  segment: MatchedSegment;
  /** Which target-clip frames actually matched (times, for logging). */
  matchedShortFrom: number;
  matchedShortTo: number;
  /** 4-signal scores for ranking, null if signatures unavailable. */
  signals: RetrySignals | null;
}

export interface FreshSearchResult {
  candidates: FreshCandidate[];
  /** Human-readable notes for the retry log. */
  notes: string[];
}

/**
 * Re-search the target range across the whole reference movie.
 * Returns null when the fingerprint files for this match job cannot be found
 * (caller falls back to the saved candidate-pool behavior). Never throws.
 */
export async function freshSearchCandidates(opts: {
  uploadDir: string;
  matchJobId: string;
  shortStart: number;
  shortEnd: number;
  onLog?: (message: string) => void;
}): Promise<FreshSearchResult | null> {
  const notes: string[] = [];
  const log = (m: string) => { notes.push(m); opts.onLog?.(m); };

  try {
    const paths = resolveFingerprintPaths(opts.uploadDir, opts.matchJobId);
    if (!paths) {
      log('Fresh search unavailable: fingerprint files for this match job were not found on disk.');
      return null;
    }

    log('Fresh full-movie search: loading fingerprints for the target clip and the entire reference movie…');
    const [shortLoad, movieLoad] = await Promise.all([
      getOrLoadMovieAsset(paths.shortFp, 'retry-preset', () => streamPrecomputeFromFile(paths.shortFp)),
      getOrLoadMovieAsset(paths.movieFp, 'retry-preset', () => streamPrecomputeFromFile(paths.movieFp)),
    ]);
    const sSet = shortLoad.value;
    const mSet = movieLoad.value;
    if (sSet.fps.length === 0 || mSet.fps.length === 0) {
      log('Fresh search unavailable: fingerprint files are empty.');
      return null;
    }

    // Target-clip frame window.
    const s0 = lowerBound(sSet.fps, opts.shortStart - 1e-6);
    let s1 = lowerBound(sSet.fps, opts.shortEnd + 1e-6) - 1;
    s1 = Math.min(Math.max(s1, s0), sSet.fps.length - 1);
    const targetFrames = s1 - s0 + 1;
    const targetDuration = Math.max(0.01, opts.shortEnd - opts.shortStart);
    if (targetFrames < 2) {
      log(`Fresh search: target range has only ${targetFrames} fingerprint frame(s) — too few to re-search.`);
      return { candidates: [], notes };
    }

    log(
      `Fresh search: scanning ALL ${mSet.fps.length} movie frames for the target clip ` +
      `(${targetFrames} frames, ${targetDuration.toFixed(2)}s) — old results are ignored.`,
    );

    // ── Stage 1: cheap probe of every possible time alignment ───────────────
    const probes = pickProbeIndexes(s0, s1);
    const kept = await probeAllAlignments(sSet, mSet, probes, s0);
    log(`Probe stage done: ${kept.length} promising alignment(s) kept out of ${mSet.fps.length} scanned.`);

    // ── Stage 2: consecutive-run detection on the surviving alignments ──────
    const movieDuration = mSet.fps[mSet.fps.length - 1].timestamp;
    const raw: FreshCandidate[] = [];
    let iter = 0;
    for (const alignment of kept) {
      const runs = detectRuns(sSet, mSet, s0, s1, alignment.delta);
      for (const run of runs) {
        const cand = buildCandidate(sSet, mSet, run, opts.shortStart, opts.shortEnd, movieDuration);
        if (cand) raw.push(cand);
      }
      if (++iter % 25 === 0) await yieldLoop();
    }

    // Dedup near-identical movie locations, best confidence wins.
    raw.sort((a, b) => b.segment.confidence - a.segment.confidence);
    const deduped: FreshCandidate[] = [];
    for (const c of raw) {
      if (deduped.some(d => Math.abs(d.segment.movieStart - c.segment.movieStart) < DEDUP_SECONDS)) continue;
      deduped.push(c);
    }

    // Signal-tier ranking (filtering/ranking only — Gemini VLM decides):
    //   tier 0: all 4 signals > threshold → verified FIRST
    //   tier 1: signals unavailable (old fingerprints, no signatures) → middle
    //   tier 2: any signal below threshold → verified LAST
    deduped.sort((a, b) => signalTier(a.signals) - signalTier(b.signals) ||
      b.segment.confidence - a.segment.confidence);
    const candidates = deduped.slice(0, MAX_FRESH_CANDIDATES);

    log(
      `Fresh search found ${raw.length} consecutive-frame run(s) ` +
      `(>= ${RUN_MIN_FRAMES} consecutive frames at >= ${RUN_MIN_SIM}% hash match), ` +
      `${deduped.length} unique movie location(s), keeping the top ${candidates.length} ` +
      `(max ${MAX_FRESH_CANDIDATES}).`,
    );
    for (const c of candidates.slice(0, 5)) {
      log(
        `  Candidate @ movie ${c.segment.movieStart.toFixed(2)}s–${c.segment.movieEnd.toFixed(2)}s: ` +
        `hash ${Math.round(c.segment.confidence)}%, matched clip frames ` +
        `${c.matchedShortFrom.toFixed(2)}s–${c.matchedShortTo.toFixed(2)}s` +
        (c.signals
          ? `, signals structure ${pct(c.signals.structure)} / color ${pct(c.signals.color)} / ` +
            `background ${pct(c.signals.background)} / human-edge ${pct(c.signals.humanEdge)}`
          : ', signals unavailable (no frame signatures in fingerprints)'),
      );
    }

    return { candidates, notes };
  } catch (e: any) {
    log(`Fresh search failed (non-fatal, falling back to saved candidates): ${e?.message || e}`);
    return null;
  }
}

/** Threshold used both here and by verify.ts for the "all 4 signals" rule. */
export const SIGNAL_THRESHOLD = clampNum(process.env.RETRY_SIGNAL_THRESHOLD, 0.60, 0, 1);

export function signalTier(signals: RetrySignals | null): number {
  if (!signals) return 1;
  const all = [signals.structure, signals.color, signals.background, signals.humanEdge];
  return all.every(v => v > SIGNAL_THRESHOLD) ? 0 : 2;
}

// ---------------------------------------------------------------------------
// Fingerprint file discovery — mirrors server.ts naming, read-only.
// ---------------------------------------------------------------------------

function resolveFingerprintPaths(
  uploadDir: string,
  matchJobId: string,
): { shortFp: string; movieFp: string } | null {
  try {
    const metaPath = path.join(uploadDir, `${matchJobId}_matchmeta.json`);
    if (!fs.existsSync(metaPath)) return null;
    const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
    const shortJobId = meta?.shortJobId;
    const movieJobId = meta?.movieJobId;
    if (typeof shortJobId !== 'string' || typeof movieJobId !== 'string') return null;
    const shortFp = path.join(uploadDir, `${shortJobId}_result.json`);
    const movieFp = path.join(uploadDir, `${movieJobId}_result.json`);
    if (!fs.existsSync(shortFp) || !fs.existsSync(movieFp)) return null;
    return { shortFp, movieFp };
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Hash similarity on PreSet flat arrays (mirror-aware, cross-variant)
// ---------------------------------------------------------------------------

function popcount32(x: number): number {
  x = x >>> 0;
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
}

function ham(a: Uint32Array, offA: number, b: Uint32Array, offB: number, words: number): number {
  let d = 0;
  for (let k = 0; k < words; k++) d += popcount32(a[offA + k] ^ b[offB + k]);
  return d;
}

/** Similarity (0..1) of one (short frame, variant) vs one (movie frame, variant). */
function pairSim(
  sSet: PreSet, si: number, sVi: number,
  mSet: PreSet, mi: number, mVi: number,
): number {
  const aWords = Math.min(sSet.aWords, mSet.aWords);
  const aBits = Math.min(sSet.aBits, mSet.aBits);
  const sAOff = (si * sSet.numVariants + sVi) * sSet.aWords;
  const mAOff = (mi * mSet.numVariants + mVi) * mSet.aWords;

  let aSim = 1 - ham(sSet.aFlat, sAOff, mSet.aFlat, mAOff, aWords) / Math.max(1, aBits);
  // Mirror-edit detection: short (normal) vs movie (flipped).
  if (mSet.faFlat) {
    const f = 1 - ham(sSet.aFlat, sAOff, mSet.faFlat, mAOff, aWords) / Math.max(1, aBits);
    if (f > aSim) aSim = f;
  }

  let dSim = -1;
  if (sSet.dFlat && mSet.dFlat && sSet.dBits > 0 && mSet.dBits > 0) {
    const dWords = Math.min(sSet.dWords, mSet.dWords);
    const dBits = Math.min(sSet.dBits, mSet.dBits);
    const sDOff = (si * sSet.numVariants + sVi) * sSet.dWords;
    const mDOff = (mi * mSet.numVariants + mVi) * mSet.dWords;
    dSim = 1 - ham(sSet.dFlat, sDOff, mSet.dFlat, mDOff, dWords) / Math.max(1, dBits);
    if (mSet.fdFlat) {
      const f = 1 - ham(sSet.dFlat, sDOff, mSet.fdFlat, mDOff, dWords) / Math.max(1, dBits);
      if (f > dSim) dSim = f;
    }
  }

  let pSim = -1;
  if (sSet.pFlat && mSet.pFlat && sSet.pBits > 0 && mSet.pBits > 0) {
    const pWords = Math.min(sSet.pWords, mSet.pWords);
    const pBits = Math.min(sSet.pBits, mSet.pBits);
    const sPOff = (si * sSet.numVariants + sVi) * sSet.pWords;
    const mPOff = (mi * mSet.numVariants + mVi) * mSet.pWords;
    pSim = 1 - ham(sSet.pFlat, sPOff, mSet.pFlat, mPOff, pWords) / Math.max(1, pBits);
  }

  // Same weighting scheme the engine uses: a+d+p (0.25/0.35/0.40) when all
  // three exist, a+d (0.55/0.45) otherwise, aHash alone as the floor.
  if (dSim >= 0 && pSim >= 0) return 0.25 * aSim + 0.35 * dSim + 0.40 * pSim;
  if (dSim >= 0) return 0.55 * aSim + 0.45 * dSim;
  return aSim;
}

/** Best cross-variant similarity (%) between a short frame and a movie frame.
 *  Compares short 'full' against every movie variant AND every short variant
 *  against movie 'full' — covers crops/zooms on either side without the full
 *  V×V blow-up. */
function frameSimPct(sSet: PreSet, si: number, mSet: PreSet, mi: number): number {
  const sFull = sSet.variantIdx.get('full') ?? 0;
  const mFull = mSet.variantIdx.get('full') ?? 0;
  let best = 0;
  for (let mVi = 0; mVi < mSet.numVariants; mVi++) {
    const s = pairSim(sSet, si, sFull, mSet, mi, mVi);
    if (s > best) best = s;
  }
  for (let sVi = 0; sVi < sSet.numVariants; sVi++) {
    if (sVi === sFull) continue;
    const s = pairSim(sSet, si, sVi, mSet, mi, mFull);
    if (s > best) best = s;
  }
  return best * 100;
}

/** Cheap probe: full-vs-full aHash only (mirror-aware). */
function probeSimPct(sSet: PreSet, si: number, mSet: PreSet, mi: number): number {
  const sFull = sSet.variantIdx.get('full') ?? 0;
  const mFull = mSet.variantIdx.get('full') ?? 0;
  const aWords = Math.min(sSet.aWords, mSet.aWords);
  const aBits = Math.max(1, Math.min(sSet.aBits, mSet.aBits));
  const sOff = (si * sSet.numVariants + sFull) * sSet.aWords;
  const mOff = (mi * mSet.numVariants + mFull) * mSet.aWords;
  let sim = 1 - ham(sSet.aFlat, sOff, mSet.aFlat, mOff, aWords) / aBits;
  if (mSet.faFlat) {
    const f = 1 - ham(sSet.aFlat, sOff, mSet.faFlat, mOff, aWords) / aBits;
    if (f > sim) sim = f;
  }
  return sim * 100;
}

// ---------------------------------------------------------------------------
// Stage 1 — probe every movie frame as a potential alignment
// ---------------------------------------------------------------------------

interface Alignment {
  /** movieTime - shortTime for this alignment. */
  delta: number;
  score: number;
}

function pickProbeIndexes(s0: number, s1: number): number[] {
  const mid = Math.round((s0 + s1) / 2);
  return Array.from(new Set([s0, mid, s1]));
}

async function probeAllAlignments(
  sSet: PreSet,
  mSet: PreSet,
  probeIdxs: number[],
  s0: number,
): Promise<Alignment[]> {
  const M = mSet.fps.length;
  const out: Alignment[] = [];
  const sT0 = sSet.fps[s0].timestamp;

  for (let m0 = 0; m0 < M; m0++) {
    const delta = mSet.fps[m0].timestamp - sT0;
    // Best of the probe frames — a mid-clip-only match must still survive.
    let best = 0;
    for (const si of probeIdxs) {
      const target = sSet.fps[si].timestamp + delta;
      const mi = nearestMovieIndex(mSet, target);
      if (mi < 0) continue;
      const s = probeSimPct(sSet, si, mSet, mi);
      if (s > best) best = s;
    }
    if (best >= PROBE_MIN_SIM) out.push({ delta, score: best });
    if (m0 % 20000 === 19999) await yieldLoop();
  }

  out.sort((a, b) => b.score - a.score);
  return out.slice(0, PROBE_KEEP);
}

// ---------------------------------------------------------------------------
// Stage 2 — consecutive-run detection along one time alignment
// ---------------------------------------------------------------------------

interface Run {
  /** Short-frame index range that matched consecutively. */
  sFrom: number;
  sTo: number;
  /** Matching movie frame index per short frame. */
  movieIdx: number[];
  /** Per-frame similarities (%) over the run. */
  sims: number[];
}

function detectRuns(
  sSet: PreSet,
  mSet: PreSet,
  s0: number,
  s1: number,
  delta: number,
): Run[] {
  const frameDur = estimateFrameDuration(mSet);
  const runs: Run[] = [];
  let cur: Run | null = null;

  for (let si = s0; si <= s1; si++) {
    const target = sSet.fps[si].timestamp + delta;
    const mi = nearestMovieIndex(mSet, target);
    const aligned = mi >= 0 && Math.abs(mSet.fps[mi].timestamp - target) <= frameDur * 0.75;
    const sim = aligned ? frameSimPct(sSet, si, mSet, mi) : 0;

    if (sim >= RUN_MIN_SIM) {
      if (!cur) cur = { sFrom: si, sTo: si, movieIdx: [], sims: [] };
      cur.sTo = si;
      cur.movieIdx.push(mi);
      cur.sims.push(sim);
    } else if (cur) {
      if (cur.sims.length >= RUN_MIN_FRAMES) runs.push(cur);
      cur = null;
    }
  }
  if (cur && cur.sims.length >= RUN_MIN_FRAMES) runs.push(cur);
  return runs;
}

// ---------------------------------------------------------------------------
// Candidate construction — duration-aware extension around the matched run
// ---------------------------------------------------------------------------

function buildCandidate(
  sSet: PreSet,
  mSet: PreSet,
  run: Run,
  shortStart: number,
  shortEnd: number,
  movieDuration: number,
): FreshCandidate | null {
  const matchedShortFrom = sSet.fps[run.sFrom].timestamp;
  const matchedShortTo = sSet.fps[run.sTo].timestamp;
  const movieFrom = mSet.fps[run.movieIdx[0]].timestamp;
  const movieTo = mSet.fps[run.movieIdx[run.movieIdx.length - 1]].timestamp;

  // Extend the movie range on BOTH sides by exactly the short-clip time left
  // before/after the matched frames, so the candidate spans the target clip's
  // full duration (a middle-of-clip run grows symmetrically outward).
  const leadIn = Math.max(0, matchedShortFrom - shortStart);
  const leadOut = Math.max(0, shortEnd - matchedShortTo);
  const movieStart = Math.max(0, movieFrom - leadIn);
  const movieEnd = Math.min(movieDuration, movieTo + leadOut);

  // Duration rule: the candidate's length must match the target clip's
  // duration. Edge-of-movie clamping can shorten it; below the floor it is
  // rejected (a 0.20s candidate for a 2s clip cannot exist at all).
  const targetDuration = Math.max(0.01, shortEnd - shortStart);
  if ((movieEnd - movieStart) / targetDuration < MIN_DURATION_RATIO) return null;

  const confidence = run.sims.reduce((a, b) => a + b, 0) / run.sims.length;
  const signals = computeSignals(sSet, mSet, run, confidence);

  const segment: MatchedSegment = {
    shortStart,
    shortEnd,
    movieStart,
    movieEnd,
    confidence,
    frameCount: run.sims.length,
    isApproximate: false,
    gapCount: 0,
    speedRatio: 1,
    matchSequence: run.movieIdx.map((mi, k) => ({
      shortTime: sSet.fps[run.sFrom + k].timestamp,
      movieTime: mSet.fps[mi].timestamp,
      similarity: run.sims[k],
    })),
  };

  return { segment, matchedShortFrom, matchedShortTo, signals };
}

// ---------------------------------------------------------------------------
// 4-signal computation over a run's matched frame pairs
// ---------------------------------------------------------------------------

/** Outer-ring cells of the 4x4 signature grid (everything but the center 2x2). */
const OUTER_CELLS = [0, 1, 2, 3, 4, 7, 8, 11, 12, 13, 14, 15];

function computeSignals(
  sSet: PreSet,
  mSet: PreSet,
  run: Run,
  hashConfidence: number,
): RetrySignals | null {
  let n = 0;
  let color = 0;
  let background = 0;
  let humanEdge = 0;

  for (let k = 0; k < run.movieIdx.length; k++) {
    const sSig = sSet.fps[run.sFrom + k]?.signature;
    const mSig = mSet.fps[run.movieIdx[k]]?.signature;
    if (!validSig(sSig) || !validSig(mSig)) continue;
    n++;
    color += colorGridSim(sSig!, mSig!);
    background += backgroundSim(sSig!, mSig!);
    humanEdge += humanEdgeSim(sSig!, mSig!);
  }

  if (n === 0) return null;
  return {
    structure: Math.max(0, Math.min(1, hashConfidence / 100)),
    color: color / n,
    background: background / n,
    humanEdge: humanEdge / n,
  };
}

function validSig(sig: FrameSignature | undefined): boolean {
  return !!sig &&
    sig.colorGrid?.length === 48 &&
    sig.skinScoreGrid?.length === 16 &&
    sig.detailGrid?.length === 16;
}

/** Whole-frame colorGrid agreement, 0..1. */
function colorGridSim(a: FrameSignature, b: FrameSignature): number {
  let d = 0;
  for (let i = 0; i < 48; i++) d += Math.abs(a.colorGrid[i] - b.colorGrid[i]);
  return 1 - d / (48 * 255);
}

/** Background: color + detail agreement over the OUTER ring cells only. */
function backgroundSim(a: FrameSignature, b: FrameSignature): number {
  let dc = 0;
  let dd = 0;
  for (const cell of OUTER_CELLS) {
    for (let ch = 0; ch < 3; ch++) {
      dc += Math.abs(a.colorGrid[cell * 3 + ch] - b.colorGrid[cell * 3 + ch]);
    }
    dd += Math.abs(a.detailGrid[cell] - b.detailGrid[cell]);
  }
  const colorPart = 1 - dc / (OUTER_CELLS.length * 3 * 255);
  const detailPart = 1 - dd / (OUTER_CELLS.length * 255);
  return 0.6 * colorPart + 0.4 * detailPart;
}

/** Human character / edge: skin-tone grid + detail (edge/texture) grid. */
function humanEdgeSim(a: FrameSignature, b: FrameSignature): number {
  let ds = 0;
  let dd = 0;
  for (let i = 0; i < 16; i++) {
    ds += Math.abs(a.skinScoreGrid[i] - b.skinScoreGrid[i]);
    dd += Math.abs(a.detailGrid[i] - b.detailGrid[i]);
  }
  const skinPart = 1 - ds / 16; // skin scores are 0..1 per cell
  const detailPart = 1 - dd / (16 * 255);
  return 0.5 * skinPart + 0.5 * detailPart;
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/** First index whose timestamp >= t. */
function lowerBound(fps: Array<{ timestamp: number }>, t: number): number {
  let lo = 0;
  let hi = fps.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (fps[mid].timestamp < t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Index of the movie frame whose timestamp is nearest to t, or -1. */
function nearestMovieIndex(mSet: PreSet, t: number): number {
  const fps = mSet.fps;
  if (fps.length === 0) return -1;
  const i = lowerBound(fps, t);
  if (i <= 0) return 0;
  if (i >= fps.length) return fps.length - 1;
  return (t - fps[i - 1].timestamp) <= (fps[i].timestamp - t) ? i - 1 : i;
}

function estimateFrameDuration(set: PreSet): number {
  const fps = set.fps;
  if (fps.length < 2) return 0.5;
  const span = fps[Math.min(fps.length - 1, 200)].timestamp - fps[0].timestamp;
  const n = Math.min(fps.length - 1, 200);
  const d = span / Math.max(1, n);
  return Number.isFinite(d) && d > 0 ? d : 0.5;
}

function yieldLoop(): Promise<void> {
  return new Promise(resolve => setImmediate(resolve));
}

function pct(v: number): string {
  return `${Math.round(v * 100)}%`;
}
