/**
 * Clip extraction for the verification stage.
 * ---------------------------------------------------------------------------
 * Cuts one short MP4 out of an original uploaded video so it can be uploaded
 * to Gemini. This is the ONLY thing the verification system does with the
 * source video files — it never reads fingerprints, never re-hashes anything,
 * and never touches the matching engine or its pipeline.
 *
 * Sizing rules:
 *  - Clips are re-encoded (not stream-copied) so the output starts at the exact
 *    requested timestamp instead of the preceding keyframe — an off-by-a-second
 *    cut is the single easiest way to make a genuine match look wrong.
 *  - Very short ranges are padded to MIN_SECONDS so Gemini has enough frames to
 *    reason about; very long ranges are centre-cropped to MAX_SECONDS to keep
 *    upload size and token usage bounded.
 *  - Output is downscaled to 480p max and stripped of audio: the verdict is
 *    purely visual, and smaller files upload dramatically faster.
 */

import { spawn } from 'child_process';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { FFMPEG_BIN } from '../ffmpeg-path';

export const MIN_SECONDS = Number(process.env.VERIFY_CLIP_MIN_SECONDS) || 2;
export const MAX_SECONDS = Number(process.env.VERIFY_CLIP_MAX_SECONDS) || 12;

export interface CutWindow {
  start: number;
  duration: number;
}

/**
 * Turn a [start, end] match range into the window actually handed to ffmpeg.
 * Pads short ranges symmetrically (clamped at 0) and centre-crops long ones.
 */
export function planCut(start: number, end: number): CutWindow {
  const safeStart = Math.max(0, start);
  const raw = Math.max(0, end - safeStart);

  if (raw >= MIN_SECONDS && raw <= MAX_SECONDS) {
    return { start: safeStart, duration: raw };
  }
  if (raw < MIN_SECONDS) {
    const pad = (MIN_SECONDS - raw) / 2;
    return { start: Math.max(0, safeStart - pad), duration: MIN_SECONDS };
  }
  const centre = safeStart + raw / 2;
  return { start: Math.max(0, centre - MAX_SECONDS / 2), duration: MAX_SECONDS };
}

function tempPath(label: string): string {
  return path.join(
    os.tmpdir(),
    `verify-${label}-${Date.now()}-${Math.random().toString(36).slice(2)}.mp4`,
  );
}

/**
 * Cut [start, end] out of `videoPath` into a temp MP4.
 * Returns the path, or null (never throws) when ffmpeg produced nothing usable.
 * The caller owns the file and must pass it to `deleteClip` when done.
 */
export function cutClip(
  videoPath: string,
  start: number,
  end: number,
  label: string,
): Promise<string | null> {
  const { start: cutStart, duration } = planCut(start, end);
  const outPath = tempPath(label);

  const args = [
    '-hide_banner',
    '-loglevel', 'error',
    '-y',
    // -ss before -i keeps the seek fast; because we re-encode, ffmpeg decodes
    // from the preceding keyframe and discards, so the output starts exactly
    // at the requested timestamp.
    '-ss', cutStart.toFixed(3),
    '-i', videoPath,
    '-t', duration.toFixed(3),
    '-map', '0:v:0',
    '-an',
    '-vf', 'scale=-2:min(480\\,ih)',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-crf', '28',
    '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart',
    outPath,
  ];

  return new Promise<string | null>(resolve => {
    let stderr = '';
    const proc = spawn(FFMPEG_BIN, args, { stdio: ['ignore', 'ignore', 'pipe'] });
    proc.stderr?.on('data', d => { stderr += String(d); });

    const kill = setTimeout(() => {
      try { proc.kill('SIGKILL'); } catch { /* already gone */ }
    }, 120_000);

    proc.on('error', err => {
      clearTimeout(kill);
      console.warn(`[VerifyClip] ffmpeg could not be started (${err.message}) — cannot cut ${label}.`);
      resolve(null);
    });

    proc.on('close', code => {
      clearTimeout(kill);
      let size = 0;
      try { size = fs.statSync(outPath).size; } catch { /* no output */ }
      if (code !== 0 || size === 0) {
        console.warn(
          `[VerifyClip] Failed to cut ${label} ${cutStart.toFixed(2)}s+${duration.toFixed(2)}s ` +
          `from ${path.basename(videoPath)} (exit=${code}, bytes=${size})${stderr ? `: ${stderr.trim().slice(0, 200)}` : ''}`,
        );
        deleteClip(outPath);
        resolve(null);
        return;
      }
      resolve(outPath);
    });
  });
}

export function deleteClip(filePath: string | null | undefined): void {
  if (!filePath) return;
  try {
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
  } catch {
    /* temp dir cleanup is best effort */
  }
}
