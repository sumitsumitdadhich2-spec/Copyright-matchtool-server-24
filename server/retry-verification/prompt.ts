/**
 * The single verification prompt.
 * ---------------------------------------------------------------------------
 * One prompt, one question: are these two clips the same footage?
 *
 * Design notes (these are the accuracy-relevant parts — change them carefully):
 *  - The model is told the clips may be re-encoded, cropped, letterboxed,
 *    colour-graded, mirrored, sped up/slowed down, or overlaid with text.
 *    Without this, near-duplicates get rejected for cosmetic reasons.
 *  - It is told to reject "same scene, different take/angle" and "same show,
 *    different moment", which is the dominant false-positive class coming out
 *    of perceptual-hash matching.
 *  - Evidence must be cited BEFORE the JSON verdict. Forcing the model to name
 *    concrete shared details first measurably reduces confident guessing.
 *  - The output contract is exactly what `parseVerdictJson` in gemini-vlm.ts
 *    already understands: { same, confidence, evidence, matchedTimeranges }.
 */

export const VIDEO1_LABEL =
  'VIDEO 1 — reference footage, cut from the full source movie:';
export const VIDEO2_LABEL =
  'VIDEO 2 — target footage, cut from the uploaded short clip:';

export function buildVerificationPrompt(opts: {
  movieStart: number;
  movieEnd: number;
  shortStart: number;
  shortEnd: number;
  speedRatio: number;
}): string {
  const speedNote =
    Math.abs(opts.speedRatio - 1) > 0.08
      ? `\nVIDEO 2 appears to run at roughly ${opts.speedRatio.toFixed(2)}x the speed of VIDEO 1. ` +
        `Different playback speed is NOT a reason to say they are different footage.`
      : '';

  return `TASK
Decide whether VIDEO 1 and VIDEO 2 show THE SAME UNDERLYING FOOTAGE — the same
recorded moment from the same production, even if one has been re-edited.

TREAT THESE AS THE SAME FOOTAGE (they are expected):
- re-encoding artifacts, blur, low resolution, or heavy compression
- crops, zooms, letterboxing, pillarboxing, or aspect-ratio changes
- colour grading, brightness/saturation shifts, filters, black-and-white
- horizontal mirroring / flips
- faster or slower playback, or a slightly different in/out point
- burned-in subtitles, captions, logos, watermarks, reaction-video borders,
  or picture-in-picture framing around the original footage

TREAT THESE AS DIFFERENT FOOTAGE (these are the mistakes to catch):
- the same actors or set, but a DIFFERENT take, angle, or camera setup
- the same film or show, but a DIFFERENT moment/scene
- visually similar but unrelated content (similar lighting, similar framing,
  generic talking heads, similar landscapes, similar crowds)
- only text, black frames, credits, or a static background in common
${speedNote}

HOW TO DECIDE
Look for frame-level correspondence: identical body positions and gestures at
the same relative time, identical background details, identical incidental
motion, identical wardrobe details, matching cuts. Shared subject matter alone
is NOT enough.

TIMING CONTEXT (informational only — do not use it as evidence)
VIDEO 1 covers ${opts.movieStart.toFixed(2)}s-${opts.movieEnd.toFixed(2)}s of the source movie.
VIDEO 2 covers ${opts.shortStart.toFixed(2)}s-${opts.shortEnd.toFixed(2)}s of the uploaded clip.

ANSWER FORMAT
First write 1-3 short lines naming the CONCRETE details you compared (what
specifically matches, or what specifically contradicts). Then output exactly one
JSON object and nothing after it:

{"same": true|false, "confidence": 0-100, "evidence": ["...", "..."], "matchedTimeranges": {"short": "0.0-2.0", "movie": "0.0-2.0"}}

- "same": true only if you believe it is the same underlying footage.
- "confidence": your certainty in the verdict you gave, 0-100.
- "evidence": the concrete details you cited above.
- "matchedTimeranges": the aligned windows you actually saw overlap, or null.
If the clips are too degraded, too short, or too dark to judge, answer
{"same": false, "confidence": 0, "evidence": ["insufficient visual information"], "matchedTimeranges": null}.`;
}
