/**
 * Display-only timeline sanity flag.
 * ---------------------------------------------------------------------------
 * A short clip normally walks forward through the reference film: as
 * shortStart increases, movieStart increases too. A segment that jumps
 * backwards against that trend is usually a mislocalised match, so the UI
 * marks it for human attention.
 *
 * This NEVER removes, re-orders, or rewrites a segment — it only sets a
 * boolean. It is also deliberately silent when the clip is genuinely
 * re-ordered (a supercut / non-linear edit), where "backwards" carries no
 * information: if fewer than 60% of consecutive pairs move forward, no flag is
 * set at all.
 */

import type { MatchedSegment } from '../matching-engine';

export function flagTimelineOutliers<T extends MatchedSegment>(segments: T[]): T[] {
  if (segments.length < 3) return segments.map(s => ({ ...s, timelineOutlier: false }));

  const ordered = [...segments].sort((a, b) => a.shortStart - b.shortStart);

  let forward = 0;
  for (let i = 1; i < ordered.length; i++) {
    if (ordered[i].movieStart >= ordered[i - 1].movieStart) forward++;
  }
  const forwardRatio = forward / (ordered.length - 1);

  // Non-linear edit — "backwards" is the norm here, so flag nothing.
  if (forwardRatio < 0.6) return segments.map(s => ({ ...s, timelineOutlier: false }));

  const outliers = new Set<T>();
  for (let i = 1; i < ordered.length; i++) {
    const prev = ordered[i - 1];
    const cur = ordered[i];
    // Only a real backwards jump counts: allow a small tolerance so a
    // slightly-overlapping neighbour is not flagged.
    if (cur.movieStart < prev.movieStart - 0.5) outliers.add(cur);
  }

  return segments.map(s => ({ ...s, timelineOutlier: outliers.has(s) }));
}

// ---------------------------------------------------------------------------
// Timeline-consistency scoring for candidate RANKING (Task 5).
// ---------------------------------------------------------------------------
// Same worldview as flagTimelineOutliers, fed forward into verification
// ordering instead of a display flag: when the clip's other matched segments
// walk forward through one continuous movie section, a candidate whose movie
// position agrees with that section deserves to be asked first. This function
// only SCORES — it never accepts, rejects, or rewrites anything; Gemini's
// verdict still decides every candidate.

/** Seconds of disagreement at which the consistency score decays to ~0.37.
 *  Overridable like every other pipeline tunable. */
const TIMELINE_SCORE_TAU_SECONDS = (() => {
  const n = Number(process.env.VERIFY_TIMELINE_TAU_SECONDS);
  if (!Number.isFinite(n)) return 60;
  return Math.max(5, Math.min(600, n));
})();

/**
 * Score how consistent a candidate movie position is with the timeline the
 * OTHER segments of the clip establish.
 *
 * Returns a value in [0, 1] (1 = exactly where the neighbours predict), or
 * `null` when no prediction is possible — fewer than 2 usable neighbours, or
 * the clip is a non-linear edit (< 60% forward pairs, same threshold as
 * flagTimelineOutliers), where "consistent" carries no information. Callers
 * must treat `null` as NEUTRAL, never as a penalty.
 *
 * @param neighbors  Segments of the same clip EXCLUDING the range being
 *                   ranked (pass the engine's accepted segments).
 * @param shortMid   Short-clip midpoint (seconds) of the range being ranked.
 * @param movieMid   Candidate's movie midpoint (seconds).
 */
export function timelineConsistencyScore(
  neighbors: MatchedSegment[],
  shortMid: number,
  movieMid: number,
): number | null {
  const usable = neighbors
    .filter(s => Number.isFinite(s.shortStart) && Number.isFinite(s.movieStart))
    .sort((a, b) => a.shortStart - b.shortStart);
  if (usable.length < 2) return null;

  let forward = 0;
  for (let i = 1; i < usable.length; i++) {
    if (usable[i].movieStart >= usable[i - 1].movieStart) forward++;
  }
  if (forward / (usable.length - 1) < 0.6) return null; // non-linear edit

  // Predict the movie midpoint at shortMid by interpolating between the two
  // straddling neighbours (or extrapolating from the nearest pair at the
  // clip's edges) — the same anchor math the expansion pass uses.
  const mids = usable.map(s => ({
    shortMid: (s.shortStart + s.shortEnd) / 2,
    movieMid: (s.movieStart + s.movieEnd) / 2,
  }));

  let prev = mids[0];
  let next = mids[mids.length - 1];
  for (const m of mids) {
    if (m.shortMid <= shortMid) prev = m;
    if (m.shortMid > shortMid) { next = m; break; }
  }
  if (prev === next) {
    // shortMid is outside the span — extrapolate from the closest two.
    if (shortMid < mids[0].shortMid) { prev = mids[0]; next = mids[1]; }
    else { prev = mids[mids.length - 2]; next = mids[mids.length - 1]; }
  }

  const ds = next.shortMid - prev.shortMid;
  const slope = Math.abs(ds) > 0.25 ? (next.movieMid - prev.movieMid) / ds : 1;
  const predicted = prev.movieMid + slope * (shortMid - prev.shortMid);
  if (!Number.isFinite(predicted)) return null;

  const err = Math.abs(movieMid - predicted);
  return Math.exp(-err / TIMELINE_SCORE_TAU_SECONDS);
}
