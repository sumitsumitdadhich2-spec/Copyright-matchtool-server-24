/**
 * Disk-backed verification records — one file per verified short-clip range.
 * ---------------------------------------------------------------------------
 * `<matchJobId>_verify_<rangeIndex>.json` under uploads/. Kept out of the main
 * match-result JSON on purpose: a record carries every candidate that was
 * considered for a range (each with its full MatchedSegment), which is large
 * and only ever needed when the user opens the compare/candidates UI or asks
 * for a re-check.
 *
 * These records are the single source of truth for the three product features
 * layered on verification:
 *   - the accept/reject verdict shown on a segment,
 *   - the list of alternate candidates for a range,
 *   - which candidate is currently the active match (`usedCandidateIndex`).
 */

import * as fs from 'fs';
import * as path from 'path';
import type { MatchedSegment } from '../matching-engine';

export type CandidateVerdict = 'accepted' | 'rejected' | 'unverifiable';

export interface CandidateRecord {
  /** The candidate segment itself, exactly as the matching engine produced it. */
  segment: MatchedSegment;
  /** True once Gemini was actually asked about this candidate. */
  checked: boolean;
  /** Gemini's answer. Absent while `checked` is false. */
  verdict?: CandidateVerdict;
  /** Gemini's self-reported certainty (0-100) for a checked candidate. */
  confidencePct?: number;
  /** Gemini's one-line justification. */
  reason?: string;
  /** Hash-matching confidence from the engine — the ordering signal only. */
  hashConfidence: number;
  /** Task 5 ranking: combined verification-order score. Absent for the
   *  engine's own pick (which always stays first) and when ranking is off. */
  rankScore?: number;
  /** Task 5 ranking: the individual signals that produced rankScore, kept
   *  for debugging. `timelineScore` is null for non-linear edits. */
  rankSignals?: {
    hashConfidence: number;
    /** 0..1 — speed-corrected candidate span vs the short range's span. */
    spanScore: number;
    /** 0..1 — agreeing-frame count relative to the best candidate's. */
    frameScore: number;
    /** 0..1 or null — agreement with the timeline the other segments form. */
    timelineScore: number | null;
  };
  /** RETRY-only multi-signal scores (0..1 each) computed by the fresh
   *  full-movie search over the candidate's consecutive-matching frame run.
   *  Optional extra field — the on-disk record format stays fully compatible
   *  with the main verification system, which simply ignores it. */
  retrySignals?: {
    structure: number;
    color: number;
    background: number;
    humanEdge: number;
  };
}

export interface VerificationRecord {
  /** Index of the short-clip range this record describes (stable per job). */
  segmentIndex: number;
  shortStart: number;
  shortEnd: number;
  recordedAt: number;
  candidates: CandidateRecord[];
  /** Index into `candidates` currently used as the active match for the range. */
  usedCandidateIndex?: number;
  /** Backwards/forwards-compatible alias the UI reads for the "★ Used" badge. */
  recoveredCandidateIndex?: number;
  /** True when Gemini accepted no candidate for this range. */
  dropped: boolean;
  /** Why verification could not run for this range, when it could not. */
  skippedReason?: string;
  /** Set while a manual re-check is in flight (added by the API layer). */
  retrying?: boolean;
  /** Task 1/5 debug metadata: detected fps + VFR flags from the ffprobe pass
   *  (server/video-metadata.ts), copied onto every record so a range can be
   *  debugged without hunting for the job-level metadata file. */
  videoMetadata?: {
    shortDeclaredFps: number | null;
    shortAverageFps: number | null;
    shortIsVFR: boolean;
    movieDeclaredFps: number | null;
    movieAverageFps: number | null;
    movieIsVFR: boolean;
  } | null;
}

function fileName(matchJobId: string, segmentIndex: number): string {
  return `${matchJobId}_verify_${segmentIndex}.json`;
}

function filePath(uploadDir: string, matchJobId: string, segmentIndex: number): string {
  return path.join(uploadDir, fileName(matchJobId, segmentIndex));
}

/** Keep the UI's "★ Used" alias in sync with the canonical field. */
function normalize(record: VerificationRecord): VerificationRecord {
  record.recoveredCandidateIndex = record.usedCandidateIndex;
  return record;
}

export function writeRecord(uploadDir: string, matchJobId: string, record: VerificationRecord): void {
  try {
    fs.writeFileSync(
      filePath(uploadDir, matchJobId, record.segmentIndex),
      JSON.stringify(normalize(record)),
    );
  } catch (e: any) {
    console.error(
      `[VerifyStore] Failed to write record for ${matchJobId} range ${record.segmentIndex}: ${e?.message || e}`,
    );
  }
}

export async function writeRecordAsync(
  uploadDir: string,
  matchJobId: string,
  record: VerificationRecord,
): Promise<void> {
  try {
    await fs.promises.writeFile(
      filePath(uploadDir, matchJobId, record.segmentIndex),
      JSON.stringify(normalize(record)),
    );
  } catch (e: any) {
    console.error(
      `[VerifyStore] Failed to write record for ${matchJobId} range ${record.segmentIndex}: ${e?.message || e}`,
    );
  }
}

export function readRecord(
  uploadDir: string,
  matchJobId: string,
  segmentIndex: number,
): VerificationRecord | null {
  const p = filePath(uploadDir, matchJobId, segmentIndex);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf-8')) as VerificationRecord;
  } catch (e: any) {
    console.warn(`[VerifyStore] Corrupt record ${fileName(matchJobId, segmentIndex)}: ${e?.message || e}`);
    return null;
  }
}

/** Range indexes that have a record for this job, ascending. */
export function listRecordIndexes(uploadDir: string, matchJobId: string): number[] {
  if (!fs.existsSync(uploadDir)) return [];
  const prefix = `${matchJobId}_verify_`;
  const out: number[] = [];
  try {
    for (const f of fs.readdirSync(uploadDir)) {
      if (!f.startsWith(prefix) || !f.endsWith('.json')) continue;
      const idx = Number(f.slice(prefix.length, -'.json'.length));
      if (Number.isFinite(idx)) out.push(idx);
    }
  } catch {
    return [];
  }
  return out.sort((a, b) => a - b);
}

export function readAllRecords(uploadDir: string, matchJobId: string): VerificationRecord[] {
  return listRecordIndexes(uploadDir, matchJobId)
    .map(idx => readRecord(uploadDir, matchJobId, idx))
    .filter((r): r is VerificationRecord => r !== null);
}

export function deleteRecordsForJob(uploadDir: string, matchJobId: string): number {
  let n = 0;
  for (const idx of listRecordIndexes(uploadDir, matchJobId)) {
    try {
      fs.unlinkSync(filePath(uploadDir, matchJobId, idx));
      n++;
    } catch {
      /* ignore */
    }
  }
  return n;
}
