// Load env files BEFORE any module reads process.env. tsx/node do not load
// .env files automatically (this is not Next.js), so without this the
// GPU_EMBED_SERVICE_URL / GEMINI_API_KEY etc. set in .env.development.local
// are silently invisible to the whole server.
import * as dotenv from 'dotenv';
dotenv.config({ path: ['.env.development.local', '.env.local', '.env'], quiet: true });

import express from 'express';
import multer from 'multer';
import cors from 'cors';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as http from 'http';
import { createServer as createViteServer } from 'vite';
import { extractFingerprints, NUM_WORKERS } from './server/pipeline';
import { matchVideosFromFiles, getAlternateCandidatesForRange, type MatchedSegment } from './server/matching-engine';
// Task 1 (FPS-aware alt pipeline): ffprobe metadata per match job, saved
// durably like verify records. Never read by the protected matching engine.
import { probeVideoMetadata, saveMatchVideoMetadata } from './server/video-metadata';
import { getGeminiStatus, geminiConfigured } from './server/gemini-vlm';
import {
  verifyMatchedSegments,
  readRecord,
  readAllRecords,
  deleteRecordsForJob,
  writeRecord,
  flagTimelineOutliers,
  type VerifySummary,
} from './server/verification';
// RETRY candidate system — dedicated 1:1 copy of server/verification/ used
// ONLY by the manual Retry endpoints (segment Retry + gap Retry), so future
// Retry-only changes can never affect the main matching/candidate system.
// Both copies share the same on-disk candidate records.
import { recheckSegment as retryRecheckSegment } from './server/retry-verification';

// ── Process-level safety net ────────────────────────────────────────────────
// Long background jobs (2-hour-movie fingerprinting/matching) touch a huge
// number of ffmpeg spawns, worker threads, and file writes over a long time —
// enough surface area that a single unexpected error anywhere in the process
// (not just inside the job that triggered it) would otherwise take down the
// whole server via Node's default "crash on unhandled rejection/exception"
// behavior, silently killing every OTHER in-flight job too. Log loudly and
// keep the process alive instead — each job already has its own try/catch
// that marks itself failed; this is only a last-resort net for anything that
// slips through, isolating one bad error from unrelated background jobs.
process.on('uncaughtException', (err) => {
  console.error('[FATAL] Uncaught exception (process kept alive):', err);
});
process.on('unhandledRejection', (reason) => {
  console.error('[FATAL] Unhandled promise rejection (process kept alive):', reason);
});

async function startServer() {
  // canvas.node needs libuuid.so.1 which lives in /lib/x86_64-linux-gnu on this host
  // but LD_LIBRARY_PATH starts empty in NixOS.  Setting it here (before any worker is
  // spawned) updates the real process-level environment via setenv(), so all worker
  // threads started later will find the library when they call dlopen('canvas.node').
  const SYSLIBS = '/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu';
  if (!process.env.LD_LIBRARY_PATH?.includes('/lib/x86_64-linux-gnu')) {
    process.env.LD_LIBRARY_PATH = process.env.LD_LIBRARY_PATH
      ? `${SYSLIBS}:${process.env.LD_LIBRARY_PATH}`
      : SYSLIBS;
  }

  const app = express();
  app.use(cors());
  app.use(express.json());
  const PORT = process.env.PORT
    ? parseInt(process.env.PORT)
    : process.env.DEV_PORT
      ? parseInt(process.env.DEV_PORT)
      : 3000;

  // Ensure upload directory exists
  const uploadDir = path.join(process.cwd(), 'uploads');
  if (!fs.existsSync(uploadDir)) {
    fs.mkdirSync(uploadDir, { recursive: true });
  }

  // Configure multer storage
  const storage = multer.diskStorage({
    destination: (req, file, cb) => {
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const ext = path.extname(file.originalname);
      const uniqueName = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}${ext}`;
      cb(null, uniqueName);
    }
  });

  const upload = multer({ storage });

  // In-memory job store
  interface Job {
    id: string;
    status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed' | 'stopped';
    totalFrames: number;
    processedFrames: number;
    error?: string;
    originalName?: string;
    fileSize?: number;
    // Bytes received so far while status === 'uploading'. Undefined once the
    // job moves past the upload phase (fingerprint extraction uses
    // totalFrames/processedFrames instead).
    uploadedBytes?: number;
    startedAt?: number;
    completedAt?: number;
  }
  const jobs = new Map<string, Job>();

  /**
   * Return the jobId of any job that is currently processing/pending for the
   * given filename + fileSize combination.  Used to detect duplicate uploads.
   */
  function findActiveJobForFile(filename: string, fileSize: number): string | null {
    for (const [id, job] of jobs) {
      if (
        (job.status === 'processing' || job.status === 'pending') &&
        job.originalName === filename &&
        job.fileSize === fileSize
      ) {
        return id;
      }
    }
    return null;
  }

  /** Cancel functions for currently-running jobs, keyed by jobId */
  const jobCancelFns = new Map<string, () => void>();

  // ── Match jobs: same background-job model as fingerprint jobs above ──────
  // /api/match used to hold one long-lived SSE connection open for the whole
  // match + VLM run. On mobile, backgrounding the tab kills that connection
  // after ~10-15s even though nothing on the server failed. Converting it to
  // a job (tracked here, polled by the client) means it survives backgrounding,
  // app switches, and full reconnects — exactly like fingerprint jobs already do.
  //
  // IMPORTANT: this is purely a transport change. The actual computation below
  // still calls the unmodified matchVideosFromFiles(...) / resolveSegmentsWithVLM(...)
  // with the exact same arguments as the old SSE handler did.
  interface MatchJobProgress {
    phase: string;
    pct: number;
    chunkIdx?: number;
    totalChunks?: number;
    shortStart?: number;
    shortEnd?: number;
    segmentsFound?: number;
    /** Verification stage: how many matched ranges have been verified so far. */
    verifyDone?: number;
    /** Verification stage: total matched ranges to verify. */
    verifyTotal?: number;
    /** Verification stage: human-readable progress line. */
    verifyMessage?: string;
  }

  interface MatchJob {
    id: string;
    type: 'match';
    status: 'processing' | 'completed' | 'failed' | 'stopped';
    movieJobId: string;
    shortJobId: string;
    originalName?: string; // display label, e.g. "clip.mp4 → movie.mp4"
    startedAt: number;
    completedAt?: number;
    error?: string;
    progress?: MatchJobProgress;
    // Populated on completion
    segments?: any[];
    unmatchedRanges?: any[];
    movieFrames?: number;
    shortFrames?: number;
    /** Outcome summary of the Gemini verification stage (see
     *  server/verification/verify.ts). `ran: false` + `reason` explains
     *  exactly why verification was skipped, so "why is nothing verified"
     *  is answerable from results, not just logs. */
    verifySummary?: VerifySummary;
  }

  const matchJobs = new Map<string, MatchJob>();
  /** Cancel functions for in-flight match jobs. Calling this flips the job to
   *  'stopped' immediately; because matchVideosFromFiles()/verifyMatchedSegments()
   *  must not be modified, the underlying computation is not preemptible
   *  mid-algorithm — it is simply left to finish in the background and its
   *  result is discarded (the job's status is checked before it is ever
   *  written), matching the "no mid-algorithm cancellation precision" note
   *  in the spec. The user sees the stop take effect immediately. */
  const matchJobCancelFns = new Map<string, () => void>();

  // Manual per-segment "Re-check" flow (preview UI). In-memory only, purely to
  // let the UI show a spinner while a re-check is in flight and to reject a
  // second concurrent re-check on the same segment — never candidate data
  // itself, which stays disk-backed via server/verification/store.ts. Losing
  // this set on a server restart is fine: the re-check itself couldn't have
  // survived the restart either, so "not retrying" is the correct state.
  const retryInFlight = new Set<string>();
  function retryKey(matchJobId: string, segmentIndex: number): string {
    return `${matchJobId}:${segmentIndex}`;
  }
  function gapRetryKey(matchJobId: string, shortStart: number): string {
    return `${matchJobId}:gap:${shortStart.toFixed(2)}`;
  }

  // Live step-by-step log lines for each in-flight (or just-finished) Retry,
  // keyed the same way as retryInFlight. In-memory only — purely a UI feature
  // so the user can watch which candidate is being verified right now. The
  // log of a key is reset when a new Retry starts on that key and survives
  // until then, so the UI can show the final outcome after completion.
  interface RetryLogEntry { t: number; message: string; }
  const retryLogs = new Map<string, RetryLogEntry[]>();
  const RETRY_LOG_MAX_LINES = 200;
  const RETRY_LOG_MAX_KEYS = 60;
  function logRetry(key: string, message: string): void {
    let log = retryLogs.get(key);
    if (!log) {
      // Cap total tracked keys — drop the oldest key when over budget.
      if (retryLogs.size >= RETRY_LOG_MAX_KEYS) {
        const oldest = retryLogs.keys().next().value;
        if (oldest !== undefined) retryLogs.delete(oldest);
      }
      log = [];
      retryLogs.set(key, log);
    }
    log.push({ t: Date.now(), message });
    if (log.length > RETRY_LOG_MAX_LINES) log.splice(0, log.length - RETRY_LOG_MAX_LINES);
    console.log(`[Re-check] ${key}: ${message}`);
  }
  function resetRetryLog(key: string): void {
    retryLogs.delete(key);
  }

  // ── Persisted candidate pool ────────────────────────────────────────────────
  // The engine's full pre-dedup candidate pool used to live only in memory for
  // the duration of the initial verify pass, so a later manual Retry could only
  // re-check candidates it already knew about. Persisting the pool per match
  // job lets Retry DISCOVER NEW candidates (and lets unmatched gaps be
  // retried at all). Best-effort on both sides: a missing/corrupt pool file
  // simply degrades Retry back to the old already-known-candidates behavior.
  function candidatePoolPath(matchJobId: string) {
    return path.join(uploadDir, `${matchJobId}_candidate_pool.json`);
  }
  async function saveCandidatePool(matchJobId: string, pool: MatchedSegment[] | undefined): Promise<void> {
    if (!pool || pool.length === 0) return;
    try {
      await fs.promises.writeFile(candidatePoolPath(matchJobId), JSON.stringify(pool));
      console.log(`[Match ${matchJobId}] Candidate pool persisted (${pool.length} candidate(s)) for future retries.`);
    } catch (e: any) {
      console.warn(`[Match ${matchJobId}] Could not persist candidate pool (non-fatal): ${e?.message || e}`);
    }
  }
  function loadCandidatePool(matchJobId: string): MatchedSegment[] {
    const p = candidatePoolPath(matchJobId);
    if (!fs.existsSync(p)) return [];
    try {
      const parsed = JSON.parse(fs.readFileSync(p, 'utf-8'));
      return Array.isArray(parsed) ? parsed : [];
    } catch (e: any) {
      console.warn(`[Match ${matchJobId}] Corrupt candidate pool file (ignored): ${e?.message || e}`);
      return [];
    }
  }

  function matchMetaPath(matchJobId: string) {
    return path.join(uploadDir, `${matchJobId}_matchmeta.json`);
  }
  function matchResultPath(matchJobId: string) {
    return path.join(uploadDir, `${matchJobId}_matchresult.json`);
  }

  interface MatchJobMeta {
    movieJobId: string;
    shortJobId: string;
    originalName?: string;
    status: MatchJob['status'];
    startedAt: number;
    completedAt?: number;
    error?: string;
  }

  function writeMatchJobMeta(matchJobId: string, meta: MatchJobMeta) {
    try {
      fs.writeFileSync(matchMetaPath(matchJobId), JSON.stringify(meta));
    } catch (e) {
      console.error(`[MatchMeta] Failed to write meta for ${matchJobId}:`, e);
    }
  }

  /** Reconstruct a match job from disk after a server restart. */
  function loadMatchJobFromDisk(matchJobId: string): MatchJob | null {
    const mp = matchMetaPath(matchJobId);
    if (!fs.existsSync(mp)) return null;
    let meta: MatchJobMeta;
    try {
      meta = JSON.parse(fs.readFileSync(mp, 'utf-8'));
    } catch {
      return null;
    }

    // A job whose meta still says 'processing' means the server died mid-run —
    // there is no checkpoint to resume from, so surface it as failed rather
    // than leaving a phantom "processing" job that will never finish.
    let status = meta.status;
    if (status === 'processing') status = 'failed';

    const job: MatchJob = {
      id: matchJobId,
      type: 'match',
      status,
      movieJobId: meta.movieJobId,
      shortJobId: meta.shortJobId,
      originalName: meta.originalName,
      startedAt: meta.startedAt,
      completedAt: meta.completedAt,
      error: status === 'failed' && !meta.error ? 'Interrupted by server restart' : meta.error,
    };

    if (status === 'completed') {
      const rp = matchResultPath(matchJobId);
      if (fs.existsSync(rp)) {
        try {
          const result = JSON.parse(fs.readFileSync(rp, 'utf-8'));
          job.segments = result.segments;
          job.unmatchedRanges = result.unmatchedRanges;
          job.movieFrames = result.movieFrames;
          job.shortFrames = result.shortFrames;
          job.verifySummary = result.verifySummary;
        } catch { /* corrupt result — treat as failed */ job.status = 'failed'; job.error = 'Result file corrupted'; }
      } else {
        job.status = 'failed';
        job.error = 'Result file missing';
      }
    }

    matchJobs.set(matchJobId, job);
    return job;
  }

  /** Scan uploads/ at startup to rebuild in-memory match job history. */
  function rebuildMatchJobsFromDisk() {
    if (!fs.existsSync(uploadDir)) return;
    let count = 0;
    for (const file of fs.readdirSync(uploadDir)) {
      if (!file.endsWith('_matchmeta.json')) continue;
      const id = file.replace('_matchmeta.json', '');
      if (loadMatchJobFromDisk(id)) count++;
    }
    if (count > 0) {
      console.log(`[Startup] Rebuilt ${count} match job(s) from disk.`);
    }
  }

  // Video identity registry: "filename:filesize" → jobId
  // Lets the frontend skip re-upload when the same file is selected again.
  interface JobMeta {
    originalName: string;
    fileSize: number;
    createdAt: number;
    totalFrames?: number;
    status?: 'completed' | 'failed' | 'stopped';
    startedAt?: number;
    completedAt?: number;
    /** Path to the original uploaded video file, kept on disk after fingerprinting
     *  so later passes (e.g. VLM scene verification) can extract frames on demand. */
    videoPath?: string;
  }

  /**
   * Resolve the on-disk path of the original uploaded video for a completed job,
   * if it still exists. Returns undefined for jobs predating video retention,
   * jobs whose video was cleaned up (failed/stopped), or missing files.
   */
  function getVideoPathForJob(jobId: string): string | undefined {
    try {
      const mp = metaPath(jobId);
      if (!fs.existsSync(mp)) return undefined;
      const meta: JobMeta = JSON.parse(fs.readFileSync(mp, 'utf-8'));
      if (meta.videoPath && fs.existsSync(meta.videoPath)) return meta.videoPath;
      return undefined;
    } catch {
      return undefined;
    }
  }
  /**
   * Size in bytes of the saved original video for a job, or 0 when no video is
   * kept on disk. Drives the "video saved (size)" chip and the total disk-usage
   * readout in the Job History panel.
   */
  function getVideoSizeForJob(jobId: string): number {
    const p = getVideoPathForJob(jobId);
    if (!p) return 0;
    try { return fs.statSync(p).size; } catch { return 0; }
  }
  const videoRegistry = new Map<string, string>();

  function metaPath(jobId: string) {
    return path.join(uploadDir, `${jobId}_meta.json`);
  }

  function checkpointFilePath(jobId: string) {
    return path.join(uploadDir, `${jobId}_checkpoint.json`);
  }

  /**
   * Scan uploads/ for a checkpoint file whose checkpointKey matches
   * "filename:filesize".  Returns the jobId of the incomplete job, or null.
   */
  function findCheckpoint(filename: string, fileSize: number): { jobId: string } | null {
    if (!fs.existsSync(uploadDir)) return null;
    const key = `${filename}:${fileSize}`;
    for (const f of fs.readdirSync(uploadDir)) {
      if (!f.endsWith('_checkpoint.json')) continue;
      try {
        const data = JSON.parse(fs.readFileSync(path.join(uploadDir, f), 'utf-8'));
        if (data.checkpointKey === key) return { jobId: data.jobId };
      } catch { /* corrupt — skip */ }
    }
    return null;
  }

  /**
   * Count the number of complete (newline-terminated) lines in a file.
   * Used to determine the exact resume frame index from a partial NDJSON result.
   */
  function countCompleteLines(filePath: string): number {
    if (!fs.existsSync(filePath)) return 0;
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      let count = 0;
      let pos = 0;
      while (true) {
        const nl = content.indexOf('\n', pos);
        if (nl === -1) break; // trailing incomplete line — not counted
        count++;
        pos = nl + 1;
      }
      return count;
    } catch { return 0; }
  }

  function writeJobMeta(jobId: string, meta: JobMeta) {
    try {
      fs.writeFileSync(metaPath(jobId), JSON.stringify(meta));
    } catch (e) {
      console.error(`[Meta] Failed to write meta for ${jobId}:`, e);
    }
  }

  function updateJobMetaFrames(jobId: string, totalFrames: number) {
    const mp = metaPath(jobId);
    try {
      if (fs.existsSync(mp)) {
        const meta: JobMeta = JSON.parse(fs.readFileSync(mp, 'utf-8'));
        meta.totalFrames = totalFrames;
        fs.writeFileSync(mp, JSON.stringify(meta));
      }
    } catch { /* non-fatal */ }
  }

  /** Reconstruct a job from disk after a server restart */
  function loadJobFromDisk(jobId: string): Job | null {
    const rp = path.join(uploadDir, `${jobId}_result.json`);
    const mp = metaPath(jobId);
    const cp = checkpointFilePath(jobId);

    // Read meta so we know filename, size, and persisted status
    let meta: JobMeta | null = null;
    if (fs.existsSync(mp)) {
      try { meta = JSON.parse(fs.readFileSync(mp, 'utf-8')); } catch { /* ignore */ }
    }

    const hasResult = fs.existsSync(rp);
    const hasCheckpoint = fs.existsSync(cp);

    let status: Job['status'];
    let frameCount = meta?.totalFrames ?? 0;

    if (meta?.status === 'stopped') {
      status = 'stopped';
    } else if (meta?.status === 'failed') {
      status = 'failed';
    } else if (hasResult && !hasCheckpoint) {
      status = 'completed';
    } else if (hasCheckpoint) {
      // Server restarted mid-job — job was interrupted (can be resumed by re-upload)
      status = 'failed';
    } else if (!hasResult && meta) {
      status = 'failed';
    } else {
      return null; // No meta and no result — nothing to reconstruct
    }

    const job: Job = {
      id: jobId,
      status,
      totalFrames: frameCount,
      processedFrames: frameCount,
      originalName: meta?.originalName,
      startedAt: meta?.startedAt ?? meta?.createdAt,
      completedAt: meta?.completedAt,
    };
    jobs.set(jobId, job);
    return job;
  }

  /** Scan uploads/ at startup to rebuild the video registry and full job history */
  async function rebuildJobsFromDisk() {
    if (!fs.existsSync(uploadDir)) return;
    let registryCount = 0;
    let historyCount = 0;
    for (const file of fs.readdirSync(uploadDir)) {
      if (!file.endsWith('_meta.json')) continue;
      const jobId = file.replace('_meta.json', '');
      const job = loadJobFromDisk(jobId); // populates jobs Map with all statuses
      if (!job) continue;
      historyCount++;
      if (job.status === 'completed') {
        // Also register in the video registry so /api/lookup-video works
        try {
          const meta: JobMeta = JSON.parse(
            fs.readFileSync(path.join(uploadDir, file), 'utf-8')
          );
          if (meta.originalName && meta.fileSize) {
            videoRegistry.set(`${meta.originalName}:${meta.fileSize}`, jobId);
            registryCount++;
          }
        } catch { /* corrupt meta — skip */ }
      }
    }
    if (historyCount > 0) {
      console.log(`[Startup] Rebuilt video registry: ${registryCount} cached job(s), ${historyCount} total in history`);
    }
  }

  await rebuildJobsFromDisk();
  rebuildMatchJobsFromDisk();

  // --- API ROUTES ---

  // 1. Health check
  app.get('/api/health', (req, res) => {
    res.json({ status: 'ok' });
  });

  // 1b. Runtime settings (Gemini API key + GPU embed service URL).
  // Persisted to runtime-settings.json so they survive restarts, and applied
  // to process.env immediately so all later-spawned worker threads inherit them.
  // UI-saved values are authoritative: they win over env vars both at boot and
  // via POST. Env vars are only a fallback when nothing was saved from the UI.
  const settingsPath = path.join(process.cwd(), 'runtime-settings.json');
  const loadRuntimeSettings = () => {
    try {
      if (fs.existsSync(settingsPath)) {
        const saved = JSON.parse(fs.readFileSync(settingsPath, 'utf-8'));
        // UI-saved settings are authoritative: they override env vars at boot.
        // Env vars only act as a fallback when nothing was ever saved from the UI.
        if (saved.geminiApiKey) {
          process.env.GEMINI_API_KEY = saved.geminiApiKey;
        }
        if (saved.gpuEmbedServiceUrl) {
          process.env.GPU_EMBED_SERVICE_URL = saved.gpuEmbedServiceUrl;
        }
        console.log('[settings] Loaded runtime settings from runtime-settings.json');
      }
    } catch (e) {
      console.warn('[settings] Failed to load runtime-settings.json:', e);
    }
  };
  loadRuntimeSettings();

  const maskKey = (k?: string) =>
    k && k.length > 8 ? `${k.slice(0, 4)}…${k.slice(-4)}` : k ? '••••' : '';

  app.get('/api/settings', (req, res) => {
    res.json({
      geminiConfigured: !!process.env.GEMINI_API_KEY,
      geminiKeyMasked: maskKey(process.env.GEMINI_API_KEY),
      gpuEmbedServiceUrl: process.env.GPU_EMBED_SERVICE_URL || '',
    });
  });

  app.post('/api/settings', (req, res) => {
    try {
      const { geminiApiKey, gpuEmbedServiceUrl } = req.body ?? {};

      if (geminiApiKey !== undefined) {
        const trimmed = String(geminiApiKey).trim();
        if (trimmed) process.env.GEMINI_API_KEY = trimmed;
        else delete process.env.GEMINI_API_KEY;
      }
      if (gpuEmbedServiceUrl !== undefined) {
        const trimmed = String(gpuEmbedServiceUrl).trim().replace(/\/+$/, '');
        if (trimmed && !/^https?:\/\//i.test(trimmed)) {
          return res.status(400).json({ error: 'GPU service URL must start with http:// or https://' });
        }
        if (trimmed) process.env.GPU_EMBED_SERVICE_URL = trimmed;
        else delete process.env.GPU_EMBED_SERVICE_URL;
      }

      // Persist (owner-only file perms — contains a secret)
      const toSave = {
        geminiApiKey: process.env.GEMINI_API_KEY || '',
        gpuEmbedServiceUrl: process.env.GPU_EMBED_SERVICE_URL || '',
      };
      fs.writeFileSync(settingsPath, JSON.stringify(toSave, null, 2), { mode: 0o600 });

      res.json({
        ok: true,
        geminiConfigured: !!process.env.GEMINI_API_KEY,
        geminiKeyMasked: maskKey(process.env.GEMINI_API_KEY),
        gpuEmbedServiceUrl: process.env.GPU_EMBED_SERVICE_URL || '',
      });
    } catch (e: any) {
      res.status(500).json({ error: e?.message || 'Failed to save settings' });
    }
  });

  // Test the Gemini key with a minimal models.list call
  app.post('/api/settings/test-gemini', async (req, res) => {
    const key = (req.body?.geminiApiKey ?? process.env.GEMINI_API_KEY ?? '').trim();
    if (!key) return res.json({ ok: false, error: 'No API key provided' });
    try {
      const r = await fetch(
        'https://generativelanguage.googleapis.com/v1beta/models?pageSize=1',
        { headers: { 'x-goog-api-key': key } },
      );
      if (r.ok) return res.json({ ok: true });
      const body = await r.text().catch(() => '');
      return res.json({ ok: false, error: `HTTP ${r.status}: ${body.slice(0, 200)}` });
    } catch (e: any) {
      return res.json({ ok: false, error: e?.message || 'Network error' });
    }
  });

  // 2. Upload chunk endpoint
  app.post('/api/upload-chunk', upload.single('chunk') as any, async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({ error: 'No chunk file uploaded' });
      }

      const { uploadId, chunkIndex, totalChunks, filename, totalSize } = req.body;
      const chunkPath = req.file.path;
      const finalPath = path.join(uploadDir, `${uploadId}-${filename}`);
      const idx = parseInt(chunkIndex);
      const total = parseInt(totalChunks);

      // Append chunk to final file
      await fs.promises.appendFile(finalPath, await fs.promises.readFile(chunkPath));
      await fs.promises.unlink(chunkPath);

      // Track the transfer itself as a job from the very first chunk, so a
      // large file (e.g. a 2-hour movie) shows up in Job History — and can be
      // reconnected to — during the upload, not only once extraction starts.
      // uploadId doubles as the eventual jobId (see the non-checkpoint branch
      // below), so this entry just flips in place once the upload finishes.
      if (idx === 0) {
        jobs.set(uploadId, {
          id: uploadId,
          status: 'uploading',
          totalFrames: 0,
          processedFrames: 0,
          originalName: filename,
          fileSize: parseInt(totalSize, 10) || 0,
          uploadedBytes: 0,
          startedAt: Date.now(),
        });
      }
      const uploadingJob = jobs.get(uploadId);
      if (uploadingJob && uploadingJob.status === 'uploading') {
        uploadingJob.uploadedBytes = Math.min(
          uploadingJob.fileSize || Infinity,
          (uploadingJob.uploadedBytes ?? 0) + req.file.size
        );
      }

      if (idx === total - 1) {
        // All chunks received — check for an existing incomplete checkpoint before
        // creating a new job so we can resume rather than start from scratch.
        const assembled = await fs.promises.stat(finalPath).catch(() => ({ size: 0 }));
        const checkpointKey = `${filename}:${assembled.size}`;

        // ── Duplicate-upload guard ────────────────────────────────────────────
        // If a job for this exact file is already running (e.g. user refreshed
        // mid-upload and re-uploaded the same file), do NOT start a second job.
        // Return the existing jobId so the frontend simply resumes polling it.
        const alreadyActiveId = findActiveJobForFile(filename, assembled.size);
        if (alreadyActiveId) {
          console.log(`[Job ${alreadyActiveId}] Duplicate upload detected for "${filename}" — active job ${alreadyActiveId} already running, skipping new job creation.`);
          // Clean up the redundant assembled file to avoid leaving orphaned data.
          await fs.promises.unlink(finalPath).catch(() => {});
          jobs.delete(uploadId); // discard this upload's temporary 'uploading' entry — resuming the existing job instead
          return res.json({ jobId: alreadyActiveId });
        }
        // ─────────────────────────────────────────────────────────────────────

        const existingCp = findCheckpoint(filename, assembled.size);

        let jobId: string;
        let resumeFrom = 0;

        if (existingCp) {
          // Resume the interrupted job using the same jobId (so the result file path
          // stays the same and we can append to it).
          jobId = existingCp.jobId;

          // Belt-and-suspenders: if this checkpoint jobId is somehow already
          // processing in memory (shouldn't happen after the guard above, but
          // protects against near-simultaneous requests in the same event loop
          // tick before findActiveJobForFile could see it).
          const cpJob = jobs.get(jobId);
          if (cpJob && (cpJob.status === 'processing' || cpJob.status === 'pending')) {
            console.log(`[Job ${jobId}] Duplicate upload detected for "${filename}" — active job ${jobId} already running via checkpoint, skipping new job creation.`);
            await fs.promises.unlink(finalPath).catch(() => {});
            jobs.delete(uploadId);
            return res.json({ jobId });
          }

          const partialResult = path.join(uploadDir, `${jobId}_result.json`);
          resumeFrom = countCompleteLines(partialResult);
          console.log(`[Job ${jobId}] Checkpoint found — resuming from frame ${resumeFrom} for "${filename}"`);
          // The resumed job lives under its own (older) id — drop this upload's
          // temporary 'uploading' entry so History doesn't show a duplicate.
          if (jobId !== uploadId) jobs.delete(uploadId);
        } else {
          // Common case: this upload's own id becomes the job id. The
          // 'uploading' entry created at chunk 0 simply flips in place to
          // 'pending'/'processing' below, so Job History shows one continuous
          // entry across the upload → extraction transition instead of two.
          jobId = uploadId;
        }

        const startedAt = Date.now();
        const job: Job = {
          id: jobId,
          status: 'pending',
          totalFrames: 0,
          processedFrames: resumeFrom, // show already-done frames immediately
          originalName: filename,
          fileSize: assembled.size,
          startedAt,
        };
        jobs.set(jobId, job);

        // Persist / refresh meta so we can recover after a future restart
        writeJobMeta(jobId, {
          originalName: filename,
          fileSize: assembled.size,
          createdAt: startedAt,
          startedAt,
        });

        res.json({ jobId });

        // Kick off processing in the background
        console.log(`[Job ${jobId}] Starting background processing for ${finalPath}...`);
        job.status = 'processing';

        const resultPath = path.join(uploadDir, `${jobId}_result.json`);
        const cpPath = checkpointFilePath(jobId);

        // Cancellation support — store the cancel fn so Stop endpoint can call it
        const controller = new AbortController();
        jobCancelFns.set(jobId, () => controller.abort());

        extractFingerprints(finalPath, resultPath, (decoded, processed) => {
          const j = jobs.get(jobId);
          if (j) {
            j.totalFrames = decoded;
            j.processedFrames = processed;
          }
        }, { resumeFrom, checkpointPath: cpPath, jobId, checkpointKey, abortSignal: controller.signal }).then((frameCount) => {
          console.log(`[Job ${jobId}] Finished processing ${frameCount} frames → ${resultPath}`);
          jobCancelFns.delete(jobId);

          const completedAt = Date.now();
          const j = jobs.get(jobId);
          if (j) {
            j.status = 'completed';
            j.processedFrames = frameCount;
            j.totalFrames = frameCount;
            j.completedAt = completedAt;
          }

          // Persist final status + frame count to meta, then register in video registry.
          // The original video file is kept on disk (not deleted) so a later pass
          // (e.g. VLM scene verification) can extract frames from it on demand.
          writeJobMeta(jobId, {
            originalName: filename,
            fileSize: assembled.size,
            createdAt: startedAt,
            startedAt,
            totalFrames: frameCount,
            status: 'completed',
            completedAt,
            videoPath: finalPath,
          });
          videoRegistry.set(checkpointKey, jobId);

          // Delete checkpoint — final result is now on disk
          fs.promises.unlink(cpPath).catch(() => {});
        }).catch((err) => {
          jobCancelFns.delete(jobId);
          const isStopped = err?.message === 'STOPPED';
          console.log(`[Job ${jobId}] ${isStopped ? 'Stopped by user' : `Failed: ${err.message}`}`);

          const completedAt = Date.now();
          const j = jobs.get(jobId);
          if (j) {
            j.status = isStopped ? 'stopped' : 'failed';
            j.error = isStopped ? undefined : err.message;
            j.completedAt = completedAt;
          }

          // Persist final status to meta so it survives server restart
          writeJobMeta(jobId, {
            originalName: filename,
            fileSize: assembled.size,
            createdAt: startedAt,
            startedAt,
            totalFrames: j?.totalFrames ?? 0,
            status: isStopped ? 'stopped' : 'failed',
            completedAt,
          });

          // On user-stop: clean up partial result and checkpoint to avoid
          // serving corrupted/incomplete fingerprints on a future lookup.
          if (isStopped) {
            fs.promises.unlink(resultPath).catch(() => {});
            fs.promises.unlink(cpPath).catch(() => {});
          }

          if (fs.existsSync(finalPath)) {
            fs.unlinkSync(finalPath);
          }
        });
      } else {
        res.json({ status: 'ok' });
      }
    } catch (err: any) {
      console.error('Upload chunk error:', err);
      res.status(500).json({ error: err.message });
    }
  });

  // 3. Upload endpoint (Legacy)
  app.post('/api/upload', upload.single('video') as any, (req, res) => {
    if (!req.file) {
      return res.status(400).json({ error: 'No video file uploaded' });
    }

    const jobId = `${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    const tempVideoPath = req.file.path;

    const job: Job = {
      id: jobId,
      status: 'pending',
      totalFrames: 0,
      processedFrames: 0
    };
    jobs.set(jobId, job);

    res.json({ jobId });

    console.log(`[Job ${jobId}] Starting background processing for ${tempVideoPath}...`);
    job.status = 'processing';

    const legacyResultPath = path.join(uploadDir, `${jobId}_result.json`);
    extractFingerprints(tempVideoPath, legacyResultPath, (decoded, processed) => {
      const j = jobs.get(jobId);
      if (j) {
        j.totalFrames = decoded;
        j.processedFrames = processed;
      }
    }).then((frameCount) => {
      console.log(`[Job ${jobId}] Finished processing ${frameCount} frames → ${legacyResultPath}`);

      try {
        if (fs.existsSync(tempVideoPath)) {
          fs.unlinkSync(tempVideoPath);
          console.log(`[Job ${jobId}] Cleaned up temporary video file`);
        }
      } catch (cleanupErr) {
        console.error(`[Job ${jobId}] Failed to clean up temp video file:`, cleanupErr);
      }

      const j = jobs.get(jobId);
      if (j) {
        j.status = 'completed';
        j.processedFrames = frameCount;
        j.totalFrames = frameCount;
      }
    }).catch((err) => {
      console.error(`[Job ${jobId}] Processing failed:`, err);
      
      try {
        if (fs.existsSync(tempVideoPath)) {
          fs.unlinkSync(tempVideoPath);
        }
      } catch (cleanupErr) {
        console.error(`[Job ${jobId}] Failed to clean up temp video file after error:`, cleanupErr);
      }

      const j = jobs.get(jobId);
      if (j) {
        j.status = 'failed';
        j.error = err.message || String(err);
      }
    });
  });

  // 4. Status endpoint — falls back to disk after a server restart
  app.get('/api/status/:jobId', (req, res) => {
    const { jobId } = req.params;
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    res.json(job);
  });

  // 5. Result retrieval endpoint
  // Always responds with a JSON array regardless of internal file format
  // (new files are NDJSON; old files are a JSON array).
  app.get('/api/result/:jobId', (req, res) => {
    const { jobId } = req.params;
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }

    if (job.status !== 'completed') {
      return res.status(400).json({ error: 'Job is not completed yet', status: job.status });
    }

    const resultPath = path.join(uploadDir, `${jobId}_result.json`);
    if (!fs.existsSync(resultPath)) {
      return res.status(404).json({ error: 'Result file not found' });
    }

    try {
      // Peek at first byte to detect format
      const fd = fs.openSync(resultPath, 'r');
      const peek = Buffer.alloc(1);
      fs.readSync(fd, peek, 0, 1, 0);
      fs.closeSync(fd);

      if (peek.toString('utf8') === '[') {
        // Legacy JSON array — serve directly
        res.sendFile(resultPath, (err) => {
          if (err) console.error(`Failed to send result file for job ${jobId}:`, err);
        });
      } else {
        // NDJSON — parse each line and send as a JSON array.
        // This endpoint is used by the browser mode (VideoProcessor.ts) which
        // needs a JSON array.  Short clips are small; movie files are only
        // fetched here if the user explicitly requests them in browser mode.
        const content = fs.readFileSync(resultPath, 'utf-8');
        const arr = content
          .split('\n')
          .filter(l => l.trim().length > 0)
          .map(l => JSON.parse(l));
        res.json(arr);
      }
    } catch (err: any) {
      console.error(`Failed to read result file for job ${jobId}:`, err);
      res.status(500).json({ error: err.message });
    }
  });

  // 6a. Video identity lookup — returns cached jobId if we've already processed this exact file
  app.get('/api/lookup-video', (req, res) => {
    const name = req.query.name as string;
    const size = parseInt(req.query.size as string, 10);
    if (!name || isNaN(size)) {
      return res.status(400).json({ error: 'name and size are required' });
    }
    const jobId = videoRegistry.get(`${name}:${size}`);
    if (!jobId) return res.status(404).json({ error: 'Not found' });

    // Verify result file still exists (user may have manually cleaned uploads/)
    const rp = path.join(uploadDir, `${jobId}_result.json`);
    if (!fs.existsSync(rp)) {
      videoRegistry.delete(`${name}:${size}`);
      return res.status(404).json({ error: 'Result file missing' });
    }

    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job || job.status !== 'completed') {
      return res.status(404).json({ error: 'Job not completed' });
    }
    res.json({ jobId, totalFrames: job.totalFrames });
  });

  // 6a-2. Stream the original uploaded video for a job (kept on disk after
  // fingerprinting). Supports HTTP Range requests so <video> seeking works.
  // This lets the frontend restore video previews after a page refresh —
  // the browser's local File object is gone, but the server copy survives.
  app.get('/api/video/:jobId', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });

    const videoPath = getVideoPathForJob(jobId);
    if (!videoPath) return res.status(404).json({ error: 'No saved video for this job' });

    let stat: fs.Stats;
    try { stat = fs.statSync(videoPath); } catch {
      return res.status(404).json({ error: 'Video file missing' });
    }

    const ext = path.extname(videoPath).toLowerCase();
    const mime: Record<string, string> = {
      '.mp4': 'video/mp4', '.webm': 'video/webm', '.mov': 'video/quicktime',
      '.mkv': 'video/x-matroska', '.avi': 'video/x-msvideo', '.m4v': 'video/x-m4v',
    };
    const contentType = mime[ext] ?? 'video/mp4';

    const range = req.headers.range;
    if (range) {
      const m = /bytes=(\d*)-(\d*)/.exec(range);
      let start = m && m[1] ? parseInt(m[1], 10) : 0;
      let end = m && m[2] ? parseInt(m[2], 10) : stat.size - 1;
      if (isNaN(start) || start < 0) start = 0;
      if (isNaN(end) || end >= stat.size) end = stat.size - 1;
      if (start > end) {
        return res.status(416).set('Content-Range', `bytes */${stat.size}`).end();
      }
      res.status(206).set({
        'Content-Range': `bytes ${start}-${end}/${stat.size}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': String(end - start + 1),
        'Content-Type': contentType,
      });
      fs.createReadStream(videoPath, { start, end }).pipe(res);
    } else {
      res.status(200).set({
        'Content-Length': String(stat.size),
        'Content-Type': contentType,
        'Accept-Ranges': 'bytes',
      });
      fs.createReadStream(videoPath).pipe(res);
    }
  });

  // 6a-3. Delete ONLY the saved video file for a job — fingerprints, results
  // and job history stay intact. Lets the user free disk space from the UI.
  app.delete('/api/video/:jobId', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });

    const videoPath = getVideoPathForJob(jobId);
    if (!videoPath) return res.status(404).json({ error: 'No saved video for this job' });

    try {
      fs.unlinkSync(videoPath);
    } catch (e: any) {
      return res.status(500).json({ error: e?.message || 'Failed to delete video' });
    }

    // Clear videoPath from meta so hasVideo reports false going forward
    try {
      const mp = metaPath(jobId);
      if (fs.existsSync(mp)) {
        const meta: JobMeta = JSON.parse(fs.readFileSync(mp, 'utf-8'));
        delete meta.videoPath;
        fs.writeFileSync(mp, JSON.stringify(meta));
      }
    } catch { /* non-fatal */ }

    console.log(`[Delete] Saved video for job ${jobId} removed (${videoPath})`);
    res.json({ deleted: true });
  });

  // 6b. List all jobs — for the Job History panel. Merges fingerprint jobs and
  // match jobs into one list, each tagged with `type` so the UI can render and
  // route stop/delete actions correctly.
  app.get('/api/jobs', (req, res) => {
    // Load any disk-only jobs (completed/failed/stopped before this process started)
    // that haven't been requested via /api/status yet.
    if (fs.existsSync(uploadDir)) {
      for (const file of fs.readdirSync(uploadDir)) {
        if (file.endsWith('_meta.json')) {
          const id = file.replace('_meta.json', '');
          if (!jobs.has(id)) loadJobFromDisk(id);
        } else if (file.endsWith('_matchmeta.json')) {
          const id = file.replace('_matchmeta.json', '');
          if (!matchJobs.has(id)) loadMatchJobFromDisk(id);
        }
      }
    }

    const STATUS_ORDER: Record<string, number> = {
      uploading: 0, processing: 1, pending: 2, completed: 3, stopped: 4, failed: 5
    };

    const fingerprintEntries = Array.from(jobs.values()).map(j => ({
      ...j,
      type: 'fingerprint' as const,
      // Whether the original uploaded video is still saved on the server's disk
      // (drives the "video saved" chip + preview restore + remove-video action).
      hasVideo: !!getVideoPathForJob(j.id),
      // Bytes the saved copy occupies (0 when no video is kept) — lets the UI
      // show per-job size and a total "video storage used" figure.
      videoSize: getVideoSizeForJob(j.id),
    }));
    const matchEntries = Array.from(matchJobs.values()).map(j => ({
      id: j.id,
      type: 'match' as const,
      status: j.status,
      // Source job ids so the frontend can restore video previews when
      // (re)opening a completed match job after a page refresh.
      movieJobId: j.movieJobId,
      shortJobId: j.shortJobId,
      // Reuse the fingerprint JobEntry shape for the frontend: totalFrames=100 /
      // processedFrames=pct lets the existing percentage math work unchanged;
      // `progress` carries the richer phase/segment detail for match-aware UI.
      totalFrames: 100,
      processedFrames: j.progress?.pct ?? 0,
      error: j.error,
      originalName: j.originalName ?? `Match ${j.id}`,
      startedAt: j.startedAt,
      completedAt: j.completedAt,
      progress: j.progress,
      segmentCount: j.segments?.length,
      // True only when BOTH source videos are still saved on disk — that is
      // what decides whether opening this match can actually show previews.
      hasVideo: !!(j.movieJobId && j.shortJobId
        && getVideoPathForJob(j.movieJobId) && getVideoPathForJob(j.shortJobId)),
      // Per-source availability so the UI can watch/remove each saved video
      // individually and explain exactly which one is missing (instead of an
      // all-or-nothing "videos removed" message).
      movieHasVideo: !!(j.movieJobId && getVideoPathForJob(j.movieJobId)),
      shortHasVideo: !!(j.shortJobId && getVideoPathForJob(j.shortJobId)),
      videoSize: (j.movieJobId ? getVideoSizeForJob(j.movieJobId) : 0)
        + (j.shortJobId ? getVideoSizeForJob(j.shortJobId) : 0),
    }));

    const list = [...fingerprintEntries, ...matchEntries].sort((a, b) => {
      const ao = STATUS_ORDER[a.status] ?? 5;
      const bo = STATUS_ORDER[b.status] ?? 5;
      if (ao !== bo) return ao - bo;
      return (b.startedAt ?? 0) - (a.startedAt ?? 0);
    });
    res.json(list);
  });

  // 6c. Single job detail
  app.get('/api/jobs/:jobId', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });
    const job = jobs.get(jobId) ?? loadJobFromDisk(jobId);
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json(job);
  });

  // 6d. Stop a running job cleanly
  app.post('/api/jobs/:jobId/stop', (req, res) => {
    const { jobId } = req.params;
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });
    const cancel = jobCancelFns.get(jobId);
    if (!cancel) {
      return res.status(400).json({ error: 'Job is not currently running or is not cancellable' });
    }
    cancel();
    res.json({ ok: true });
  });

  // 6e. Delete a job — stops it if running, then removes all files and clears registry
  app.delete('/api/job/:jobId', (req, res) => {
    const { jobId } = req.params;
    // Basic validation: only alphanumeric + dash
    if (!/^[\w-]+$/.test(jobId)) return res.status(400).json({ error: 'Invalid jobId' });

    // Stop if currently running (fire-and-forget — cleanup happens in the catch handler)
    const cancel = jobCancelFns.get(jobId);
    if (cancel) cancel();

    const rp = path.join(uploadDir, `${jobId}_result.json`);
    const mp = metaPath(jobId);
    const cp = checkpointFilePath(jobId);

    // Delete the saved original video too (must happen BEFORE the meta file is
    // removed — the video path lives inside the meta). Without this, deleting
    // a job leaves the multi-GB video orphaned on disk forever.
    const vp = getVideoPathForJob(jobId);
    if (vp) { try { fs.unlinkSync(vp); } catch { /* ignore */ } }

    let deleted = false;
    if (fs.existsSync(rp)) { try { fs.unlinkSync(rp); deleted = true; } catch { /* ignore */ } }
    if (fs.existsSync(mp)) { try { fs.unlinkSync(mp); } catch { /* ignore */ } }
    if (fs.existsSync(cp)) { try { fs.unlinkSync(cp); } catch { /* ignore */ } }

    // Remove from in-memory stores
    jobs.delete(jobId);
    jobCancelFns.delete(jobId);

    // Remove from video registry (search by value)
    for (const [k, v] of videoRegistry) {
      if (v === jobId) { videoRegistry.delete(k); break; }
    }

    console.log(`[Delete] Job ${jobId} removed (file existed: ${deleted})`);
    res.json({ deleted });
  });

  // 6. Match endpoint ��� background job, same model as fingerprint jobs.
  // Kicks off matching (+ optional VLM verification) asynchronously and
  // returns a matchJobId immediately; the client polls /api/match-status/:id
  // instead of holding one long-lived SSE connection open (which mobile
  // browsers kill after ~10-15s in the background).
  //
  // The core computation itself is UNCHANGED: matchVideosFromFiles(...) and
  // resolveSegmentsWithVLM(...) are called with the same accuracy-affecting
  // arguments, in the same order, as the previous SSE handler. The only
  // addition is an optional onSegmentResolved hook (see below) that persists
  // strictly-additive, preview-only candidate-comparison history and feeds
  // the deferred recovery pass — it never changes which segments the main
  // pass accepts, drops, or how fast it runs.
  app.post('/api/match', async (req, res) => {
    const {
      movieJobId,
      shortJobId,
      minSimilarity,
      minConsecutiveFrames,
      frameDrift
    } = req.body as {
      movieJobId: string;
      shortJobId: string;
      minSimilarity?: number;
      minConsecutiveFrames?: number;
      frameDrift?: number;
    };

    if (!movieJobId || !shortJobId) {
      return res.status(400).json({ error: 'movieJobId and shortJobId are required' });
    }

    // Allow confidence as low as 20 % (user-configurable)
    const resolvedMinSim    = (typeof minSimilarity    === 'number' && minSimilarity    >= 20 && minSimilarity    <= 99) ? minSimilarity    : 82;
    const resolvedMinFrames = (typeof minConsecutiveFrames === 'number' && minConsecutiveFrames >= 3 && minConsecutiveFrames <= 200) ? minConsecutiveFrames : 9;
    const resolvedDrift     = (typeof frameDrift === 'number' && frameDrift >= 0 && frameDrift <= 15) ? Math.round(frameDrift) : 3;

    const movieResultPath = path.join(uploadDir, `${movieJobId}_result.json`);
    const shortResultPath = path.join(uploadDir, `${shortJobId}_result.json`);

    if (!fs.existsSync(movieResultPath)) {
      return res.status(404).json({ error: `Movie result not found for job ${movieJobId}. Re-process the reference video.` });
    }
    if (!fs.existsSync(shortResultPath)) {
      return res.status(404).json({ error: `Short result not found for job ${shortJobId}. Re-process the target clip.` });
    }

    const matchJobId = `match-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    const startedAt = Date.now();

    // Best-effort display label using the source jobs' original filenames.
    let originalName = `Match ${matchJobId}`;
    try {
      const movieMeta = fs.existsSync(metaPath(movieJobId)) ? JSON.parse(fs.readFileSync(metaPath(movieJobId), 'utf-8')) : null;
      const shortMeta = fs.existsSync(metaPath(shortJobId)) ? JSON.parse(fs.readFileSync(metaPath(shortJobId), 'utf-8')) : null;
      if (shortMeta?.originalName && movieMeta?.originalName) {
        originalName = `${shortMeta.originalName} → ${movieMeta.originalName}`;
      }
    } catch { /* non-fatal — fall back to default label */ }

    const job: MatchJob = {
      id: matchJobId,
      type: 'match',
      status: 'processing',
      movieJobId,
      shortJobId,
      originalName,
      startedAt,
    };
    matchJobs.set(matchJobId, job);
    writeMatchJobMeta(matchJobId, { movieJobId, shortJobId, originalName, status: 'processing', startedAt });

    matchJobCancelFns.set(matchJobId, () => {
      const j = matchJobs.get(matchJobId);
      if (!j || j.status !== 'processing') return;
      j.status = 'stopped';
      j.completedAt = Date.now();
      writeMatchJobMeta(matchJobId, {
        movieJobId, shortJobId, originalName,
        status: 'stopped', startedAt, completedAt: j.completedAt,
      });
      matchJobCancelFns.delete(matchJobId);
    });

    res.json({ matchJobId });

    // ���─ Run the match in the background ─────────────────────────────────────
    console.log(`[Match ${matchJobId}] Starting: movie=${movieJobId} short=${shortJobId} drift=${resolvedDrift}`);

    (async () => {
      try {
        // Task 1 (FPS-aware alt pipeline): probe fps/VFR/duration for both
        // videos with ffprobe and persist it durably alongside the job, like
        // verify records. Best-effort and fully non-fatal — probeVideoMetadata
        // never throws, and a missing video file simply records null.
        try {
          const shortVideoForMeta = getVideoPathForJob(shortJobId);
          const movieVideoForMeta = getVideoPathForJob(movieJobId);
          const [shortProbe, movieProbe] = await Promise.all([
            shortVideoForMeta ? probeVideoMetadata(shortVideoForMeta) : Promise.resolve(null),
            movieVideoForMeta ? probeVideoMetadata(movieVideoForMeta) : Promise.resolve(null),
          ]);
          await saveMatchVideoMetadata(uploadDir, matchJobId, { short: shortProbe, movie: movieProbe });
          if (shortProbe || movieProbe) {
            console.log(
              `[Match ${matchJobId}] Video metadata: short fps=${shortProbe?.averageFps?.toFixed(2) ?? 'n/a'}` +
              `${shortProbe?.isVFR ? ' (VFR)' : ''}, movie fps=${movieProbe?.averageFps?.toFixed(2) ?? 'n/a'}` +
              `${movieProbe?.isVFR ? ' (VFR)' : ''}.`,
            );
          }
        } catch (metaErr: any) {
          console.warn(`[Match ${matchJobId}] Video metadata probe failed (non-fatal): ${metaErr?.message || metaErr}`);
        }

        // matchVideosFromFiles streams both files line-by-line and converts hash
        // strings directly into flat TypedArrays — never loads the full JSON into
        // memory.  Peak RAM drops from ~7 GB to ~400 MB for a 2-hour movie.
        const result = await matchVideosFromFiles(shortResultPath, movieResultPath, {
          minSimilarity:        resolvedMinSim,
          minConsecutiveFrames: resolvedMinFrames,
          frameDrift:           resolvedDrift,
          onProgress: (info) => {
            const j = matchJobs.get(matchJobId);
            if (j && j.status === 'processing') j.progress = { ...info };
          },
        });

        // If the job was stopped while matching ran, discard the result —
        // status was already flipped and persisted by the cancel function.
        if (matchJobs.get(matchJobId)?.status !== 'processing') return;

        console.log(`[Match ${matchJobId}] Done: ${result.segments.length} segments, ${result.unmatchedRanges.length} unmatched ranges.`);

        // Persist the engine's full pre-dedup candidate pool so a later manual
        // Retry (segments AND unmatched gaps) can search it for NEW candidates
        // instead of only re-checking already-known ones.
        await saveCandidatePool(matchJobId, result.candidatePool);


        // ── Gemini verification stage ───────────────────────────────────────
        // The new candidate/verification system (server/verification/). Takes
        // the engine's segments + the two original uploaded files, verifies
        // each range with Gemini (server/gemini-vlm.ts, used as-is), and
        // returns the finalised segments. It never throws: when Gemini is
        // unconfigured / out of daily quota, or the original videos are gone,
        // it skips LOUDLY and passes the segments through unverified, writing
        // a per-range record that says exactly why.
        const movieVideoPath = getVideoPathForJob(movieJobId);
        const shortVideoPath = getVideoPathForJob(shortJobId);

        const { segments: finalSegments, summary: verifySummary } = await verifyMatchedSegments({
          segments: result.segments,
          candidatePool: result.candidatePool,
          shortVideoPath,
          movieVideoPath,
          uploadDir,
          matchJobId,
          onProgress: (done, total, message) => {
            const j = matchJobs.get(matchJobId);
            if (!j || j.status !== 'processing') return;
            j.progress = {
              phase: 'verify',
              pct: Math.min(99, 97 + Math.round((done / Math.max(1, total)) * 3)),
              verifyDone: done,
              verifyTotal: total,
              verifyMessage: message,
            };
          },
        });

        // If stopped while verification ran, discard the result.
        if (matchJobs.get(matchJobId)?.status !== 'processing') return;

        const completedAt = Date.now();
        const finalJob = matchJobs.get(matchJobId);
        if (finalJob) {
          finalJob.status = 'completed';
          finalJob.completedAt = completedAt;
          finalJob.segments = finalSegments;
          finalJob.unmatchedRanges = result.unmatchedRanges;
          finalJob.movieFrames = result.movieFrames;
          finalJob.shortFrames = result.shortFrames;
          finalJob.verifySummary = verifySummary;
          finalJob.progress = { phase: 'finalizing', pct: 100 };
        }
        matchJobCancelFns.delete(matchJobId);
        writeMatchJobMeta(matchJobId, { movieJobId, shortJobId, originalName, status: 'completed', startedAt, completedAt });
        // Async write — a job with 100+ segments (each carrying a full
        // per-frame matchSequence) can serialize to a multi-MB JSON payload.
        // A synchronous write blocks the whole event loop while it happens,
        // which stalls every other in-flight /api/match-status poll and can
        // surface as "Failed to fetch" in the browser for unrelated jobs.
        await fs.promises.writeFile(matchResultPath(matchJobId), JSON.stringify({
          segments: finalSegments,
          unmatchedRanges: result.unmatchedRanges,
          movieFrames: result.movieFrames,
          shortFrames: result.shortFrames,
          verifySummary,
        }));
      } catch (err: any) {
        console.error(`[Match ${matchJobId}] Error:`, err);
        matchJobCancelFns.delete(matchJobId);
        if (matchJobs.get(matchJobId)?.status !== 'processing') return; // already stopped
        const completedAt = Date.now();
        const errJob = matchJobs.get(matchJobId);
        if (errJob) {
          errJob.status = 'failed';
          errJob.error = err.message || String(err);
          errJob.completedAt = completedAt;
        }
        writeMatchJobMeta(matchJobId, {
          movieJobId, shortJobId, originalName,
          status: 'failed', startedAt, completedAt, error: err.message || String(err),
        });
      }
    })();
  });

  // 6f. Match job status — polled by the client instead of an SSE stream.
  app.get('/api/match-status/:matchJobId', (req, res) => {
    const { matchJobId } = req.params;
    const job = matchJobs.get(matchJobId) ?? loadMatchJobFromDisk(matchJobId);
    if (!job) return res.status(404).json({ error: 'Match job not found' });
    res.json({
      matchJobId: job.id,
      status: job.status,
      movieJobId: job.movieJobId,
      shortJobId: job.shortJobId,
      originalName: job.originalName,
      progress: job.progress,
      error: job.error,
      segments: job.segments,
      unmatchedRanges: job.unmatchedRanges,
      movieFrames: job.movieFrames,
      shortFrames: job.shortFrames,
      verifySummary: job.verifySummary,
      startedAt: job.startedAt,
      completedAt: job.completedAt,
      gemini: getGeminiStatus(),
    });
  });

  // Standalone Gemini quota status — lets the UI poll the daily-limit flag
  // even when no match job is running.
  app.get('/api/gemini-status', (req, res) => {
    res.json(getGeminiStatus());
  });

  // 6g. Stop a running match job — mirrors /api/jobs/:jobId/stop for fingerprint jobs.
  app.post('/api/match-stop/:matchJobId', (req, res) => {
    const { matchJobId } = req.params;
    const cancel = matchJobCancelFns.get(matchJobId);
    if (!cancel) {
      return res.status(400).json({ error: 'Match job is not currently running or is not cancellable' });
    }
    cancel();
    res.json({ ok: true });
  });

  // 6h. Delete a match job — mirrors DELETE /api/job/:jobId for fingerprint jobs.
  app.delete('/api/match/:matchJobId', (req, res) => {
    const { matchJobId } = req.params;

    const cancel = matchJobCancelFns.get(matchJobId);
    if (cancel) cancel();

    const mp = matchMetaPath(matchJobId);
    const rp = matchResultPath(matchJobId);
    let deleted = false;
    if (fs.existsSync(mp)) { try { fs.unlinkSync(mp); deleted = true; } catch { /* ignore */ } }
    if (fs.existsSync(rp)) { try { fs.unlinkSync(rp); } catch { /* ignore */ } }
    deleteRecordsForJob(uploadDir, matchJobId);
    // Task 1 video-metadata file — cleaned up with the rest of the job.
    const vmPath = path.join(uploadDir, `${matchJobId}_videometa.json`);
    if (fs.existsSync(vmPath)) { try { fs.unlinkSync(vmPath); } catch { /* ignore */ } }

    matchJobs.delete(matchJobId);
    matchJobCancelFns.delete(matchJobId);

    console.log(`[Delete] Match job ${matchJobId} removed (file existed: ${deleted})`);
    res.json({ deleted });
  });

  // 6i. Per-range verification records — every candidate considered for each
  // short-clip range, its Gemini verdict, and which one is the active match.
  // Never part of the primary match result JSON; purely for the frontend's
  // compare-candidates preview and the manual re-check flow.
  app.get('/api/match/:matchJobId/candidates', (req, res) => {
    const { matchJobId } = req.params;
    const entries = readAllRecords(uploadDir, matchJobId)
      .map(e => ({ ...e, retrying: retryInFlight.has(retryKey(matchJobId, e.segmentIndex)) }));
    res.json({ matchJobId, segments: entries });
  });

  app.get('/api/match/:matchJobId/candidates/:segmentIndex', (req, res) => {
    const { matchJobId, segmentIndex } = req.params;
    const idx = Number(segmentIndex);
    if (!Number.isFinite(idx)) return res.status(400).json({ error: 'segmentIndex must be a number' });
    const entry = readRecord(uploadDir, matchJobId, idx);
    if (!entry) return res.status(404).json({ error: 'No candidate data for this segment' });
    const key = retryKey(matchJobId, idx);
    res.json({
      ...entry,
      retrying: retryInFlight.has(key),
      retryLog: retryLogs.get(key) ?? [],
    });
  });

  // Live Retry status + step-by-step logs — polled by the UI while any manual
  // Retry (segment or unmatched gap) is running, so the user can watch which
  // candidate is being verified right now. Logs persist after completion
  // (until the same key is retried again), so the final outcome stays visible.
  app.get('/api/match/:matchJobId/retry-status', (req, res) => {
    const { matchJobId } = req.params;
    const prefix = `${matchJobId}:`;
    const retries: Array<{ key: string; running: boolean; log: RetryLogEntry[] }> = [];
    for (const [key, log] of retryLogs) {
      if (!key.startsWith(prefix)) continue;
      retries.push({ key, running: retryInFlight.has(key), log });
    }
    // Keys that are in flight but have not logged anything yet.
    for (const key of retryInFlight) {
      if (!key.startsWith(prefix)) continue;
      if (!retryLogs.has(key)) retries.push({ key, running: true, log: [] });
    }
    res.json({ matchJobId, retries });
  });

  // 6j. Manual per-segment Re-check — a user-triggered action that re-runs the
  // exact same verification code path (server/verification/verify.ts →
  // recheckSegment) for ONE range and overwrites its record. Never touches
  // the matching engine. Returns immediately; the frontend polls the
  // candidates endpoints above (which include a `retrying` flag) to detect
  // completion, reusing the same fetch-and-poll pattern the rest of the app
  // already uses.
  app.post('/api/match/:matchJobId/segment/:segmentIndex/retry', async (req, res) => {
    const { matchJobId, segmentIndex: segmentIndexStr } = req.params;
    const segmentIndex = Number(segmentIndexStr);
    if (!Number.isFinite(segmentIndex)) return res.status(400).json({ error: 'segmentIndex must be a number' });

    const job = matchJobs.get(matchJobId) ?? loadMatchJobFromDisk(matchJobId);
    if (!job) return res.status(404).json({ error: 'Match job not found' });
    if (job.status !== 'completed') return res.status(400).json({ error: 'Match job is not completed yet' });

    const entry = readRecord(uploadDir, matchJobId, segmentIndex);
    if (!entry) return res.status(404).json({ error: 'No verification record for this segment' });

    const key = retryKey(matchJobId, segmentIndex);
    if (retryInFlight.has(key)) {
      return res.status(409).json({ error: 'Re-check already in progress for this segment' });
    }

    // Fail fast with a clear reason instead of kicking off a re-check that
    // recheckSegment would only refuse anyway.
    if (!geminiConfigured()) {
      return res.status(503).json({ error: 'No verification provider available — set GEMINI_API_KEY.' });
    }
    const gStatus = getGeminiStatus();
    if (gStatus.dailyLimitReached) {
      return res.status(503).json({ error: 'Gemini daily quota exhausted — add a new API key, then hit Re-check again.' });
    }

    const movieVideoPath = getVideoPathForJob(job.movieJobId);
    const shortVideoPath = getVideoPathForJob(job.shortJobId);
    if (!movieVideoPath || !shortVideoPath) {
      return res.status(400).json({ error: 'Original videos are no longer available for clip extraction.' });
    }

    retryInFlight.add(key);
    resetRetryLog(key);
    res.json({ ok: true, matchJobId, segmentIndex });

    const originalRange = { shortStart: entry.shortStart, shortEnd: entry.shortEnd };
    // The range's current active candidate is the primary; every candidate the
    // record already knows about is offered back as the alternate pool.
    const activeIdx = entry.usedCandidateIndex ?? entry.recoveredCandidateIndex ?? 0;
    const primarySegment = entry.candidates[activeIdx]?.segment ?? entry.candidates[0]?.segment;

    // Merge already-known candidates with the persisted engine pool, so this
    // Retry can DISCOVER candidates the original pass never offered (the pool
    // is pre-dedup and much deeper than the record). Known candidates go
    // first; collectCandidates dedups near-identical movie timestamps.
    const knownSegments = entry.candidates.map(c => c.segment);
    const diskPool = loadCandidatePool(matchJobId);
    const freshFromPool = getAlternateCandidatesForRange(
      diskPool, entry.shortStart, entry.shortEnd, [], 0.5,
    ).filter(cand =>
      !knownSegments.some(k => Math.abs(k.movieStart - cand.movieStart) < 1),
    );
    const mergedPool = [...knownSegments, ...freshFromPool];

    (async () => {
      try {
        console.log(`[Re-check] Match ${matchJobId} segment ${segmentIndex}: starting…`);
        logRetry(key,
          `Searching candidates: ${knownSegments.length} already known, ` +
          `${freshFromPool.length} NEW candidate(s) discovered in the engine's candidate pool` +
          `${diskPool.length === 0 ? ' (no persisted pool for this job — older match, using known candidates only)' : ''}.`,
        );
        const result = await retryRecheckSegment({
          segment: primarySegment,
          segmentIndex,
          candidatePool: mergedPool,
          shortVideoPath,
          movieVideoPath,
          uploadDir,
          matchJobId,
          onLog: (message) => logRetry(key, message),
        });
        console.log(`[Re-check] Match ${matchJobId} segment ${segmentIndex}: ${result.message}.`);
        logRetry(key, `Result: ${result.message}.`);

        // Swap the (possibly switched, possibly re-flagged) segment back into
        // the job's live segments + persisted result JSON. Matched by range,
        // not array index — a record's segmentIndex does not have to map 1:1
        // to a slot in the segments array after earlier swaps.
        const newSeg: any = result.segment;
        const liveJob = matchJobs.get(matchJobId);
        if (newSeg && liveJob?.segments) {
          let arrIdx = liveJob.segments.findIndex((s: any) =>
            Math.abs(s.shortStart - originalRange.shortStart) < 0.05 &&
            Math.abs(s.shortEnd - originalRange.shortEnd) < 0.05);
          // Fallback: largest overlap with the record's range — a previous
          // swap can have shifted the active range slightly; without this
          // the re-checked segment would be appended as a duplicate.
          if (arrIdx === -1) {
            let bestOverlap = 0;
            liveJob.segments.forEach((s: any, i: number) => {
              const overlap = Math.min(s.shortEnd, originalRange.shortEnd) - Math.max(s.shortStart, originalRange.shortStart);
              if (overlap > bestOverlap) { bestOverlap = overlap; arrIdx = i; }
            });
          }
          if (arrIdx !== -1) {
            liveJob.segments[arrIdx] = newSeg;
          } else {
            // This range had no active segment before — add it now.
            liveJob.segments = [...liveJob.segments, newSeg].sort((a: any, b: any) => a.shortStart - b.shortStart);
          }
          // Recompute the display-only timeline flags ��� swapping in a new
          // match can create OR resolve a backwards jump against the
          // neighbouring segments, so the badges must not stay stale.
          try { liveJob.segments = flagTimelineOutliers(liveJob.segments); } catch { /* display-only */ }
          await fs.promises.writeFile(matchResultPath(matchJobId), JSON.stringify({
            segments: liveJob.segments,
            unmatchedRanges: liveJob.unmatchedRanges,
            movieFrames: liveJob.movieFrames,
            shortFrames: liveJob.shortFrames,
            verifySummary: liveJob.verifySummary,
          }));
        }
      } catch (err: any) {
        console.error(`[Re-check] Match ${matchJobId} segment ${segmentIndex} failed:`, err?.message || err);
      } finally {
        retryInFlight.delete(key);
      }
    })();
  });

  // 6j-bis. Manual Retry for an UNMATCHED gap — searches the persisted engine
  // candidate pool for candidates overlapping the gap and runs the exact same
  // full-check verification (recheckSegment) on them. On acceptance the new
  // segment is added to the timeline and the gap is shrunk/removed. Never
  // touches the matching engine. Same fire-and-poll shape as the segment
  // Retry above; progress is watched via /retry-status (key `gap:<start>`).
  app.post('/api/match/:matchJobId/gap/retry', async (req, res) => {
    const { matchJobId } = req.params;
    const shortStart = Number(req.body?.shortStart);
    const shortEnd = Number(req.body?.shortEnd);
    if (!Number.isFinite(shortStart) || !Number.isFinite(shortEnd) || shortEnd <= shortStart) {
      return res.status(400).json({ error: 'shortStart and shortEnd must be numbers with shortEnd > shortStart' });
    }

    const job = matchJobs.get(matchJobId) ?? loadMatchJobFromDisk(matchJobId);
    if (!job) return res.status(404).json({ error: 'Match job not found' });
    if (job.status !== 'completed') return res.status(400).json({ error: 'Match job is not completed yet' });

    const key = gapRetryKey(matchJobId, shortStart);
    if (retryInFlight.has(key)) {
      return res.status(409).json({ error: 'A Retry is already in progress for this gap' });
    }

    if (!geminiConfigured()) {
      return res.status(503).json({ error: 'No verification provider available — set GEMINI_API_KEY.' });
    }
    const gStatus = getGeminiStatus();
    if (gStatus.dailyLimitReached) {
      return res.status(503).json({ error: 'Gemini daily quota exhausted — add a new API key, then hit Retry again.' });
    }

    const movieVideoPath = getVideoPathForJob(job.movieJobId);
    const shortVideoPath = getVideoPathForJob(job.shortJobId);
    if (!movieVideoPath || !shortVideoPath) {
      return res.status(400).json({ error: 'Original videos are no longer available for clip extraction.' });
    }

    // The ONLY candidate source for a gap is the persisted engine pool — the
    // gap has no verification record. No pool → nothing to try, fail fast.
    const diskPool = loadCandidatePool(matchJobId);
    if (diskPool.length === 0) {
      return res.status(404).json({ error: 'No persisted candidate pool for this match job (older match) — re-run matching to enable gap Retry.' });
    }
    const gapCandidates = getAlternateCandidatesForRange(diskPool, shortStart, shortEnd, [], 0.5);
    if (gapCandidates.length === 0) {
      return res.status(404).json({ error: 'The engine\'s candidate pool has no candidate overlapping this gap — nothing to verify.' });
    }

    retryInFlight.add(key);
    resetRetryLog(key);
    res.json({ ok: true, matchJobId, shortStart, shortEnd });

    // Reuse the record index of a previous gap Retry on the same range so
    // repeated attempts overwrite one record instead of piling up new ones;
    // otherwise allocate the next free index.
    const allRecords = readAllRecords(uploadDir, matchJobId);
    const prior = allRecords.find(r => {
      const overlap = Math.min(r.shortEnd, shortEnd) - Math.max(r.shortStart, shortStart);
      return overlap > 0.5 * (shortEnd - shortStart);
    });
    const segmentIndex = prior
      ? prior.segmentIndex
      : allRecords.length > 0 ? Math.max(...allRecords.map(r => r.segmentIndex)) + 1 : 0;

    (async () => {
      try {
        console.log(`[Re-check] Match ${matchJobId} gap ${shortStart.toFixed(2)}–${shortEnd.toFixed(2)}: starting…`);
        logRetry(key,
          `Gap Retry started for unmatched range ${shortStart.toFixed(2)}s–${shortEnd.toFixed(2)}s: ` +
          `${gapCandidates.length} candidate(s) discovered in the engine's saved candidate pool.`,
        );
        const result = await retryRecheckSegment({
          segment: gapCandidates[0],
          segmentIndex,
          candidatePool: gapCandidates,
          shortVideoPath,
          movieVideoPath,
          uploadDir,
          matchJobId,
          onLog: (message) => logRetry(key, message),
        });
        console.log(`[Re-check] Match ${matchJobId} gap ${shortStart.toFixed(2)}: ${result.message}.`);
        logRetry(key, `Result: ${result.message}.`);

        if (result.accepted && result.segment) {
          const liveJob = matchJobs.get(matchJobId) ?? job;
          const newSeg: any = result.segment;
          // Insert the newly-verified segment in short-clip order and subtract
          // its range from the unmatched gaps (shrinking or removing them).
          liveJob.segments = [...(liveJob.segments ?? []), newSeg]
            .sort((a: any, b: any) => a.shortStart - b.shortStart);
          const prevGaps: any[] = liveJob.unmatchedRanges ?? [];
          const nextGaps: any[] = [];
          for (const g of prevGaps) {
            if (newSeg.shortEnd <= g.shortStart + 0.05 || newSeg.shortStart >= g.shortEnd - 0.05) {
              nextGaps.push(g);
              continue;
            }
            // Keep any leftover sliver of at least 0.25 s on either side.
            if (newSeg.shortStart > g.shortStart + 0.25) nextGaps.push({ ...g, shortStart: g.shortStart, shortEnd: newSeg.shortStart });
            if (newSeg.shortEnd < g.shortEnd - 0.25) nextGaps.push({ ...g, shortStart: newSeg.shortEnd, shortEnd: g.shortEnd });
          }
          liveJob.unmatchedRanges = nextGaps;
          try { liveJob.segments = flagTimelineOutliers(liveJob.segments!); } catch { /* display-only */ }
          await fs.promises.writeFile(matchResultPath(matchJobId), JSON.stringify({
            segments: liveJob.segments,
            unmatchedRanges: liveJob.unmatchedRanges,
            movieFrames: liveJob.movieFrames,
            shortFrames: liveJob.shortFrames,
            verifySummary: liveJob.verifySummary,
          }));
          logRetry(key, `New segment added to the timeline (movie ${newSeg.movieStart.toFixed(2)}s–${newSeg.movieEnd.toFixed(2)}s); unmatched gap updated.`);
        } else {
          logRetry(key, 'No candidate was accepted — the gap stays unmatched. Its candidate data was saved so you can review it.');
        }
      } catch (err: any) {
        console.error(`[Re-check] Match ${matchJobId} gap ${shortStart.toFixed(2)} failed:`, err?.message || err);
        logRetry(key, `Gap Retry FAILED: ${err?.message || err}.`);
      } finally {
        retryInFlight.delete(key);
      }
    })();
  });

  // 6k. Manual candidate selection ("Make main segment") — purely additive,
  // user-triggered action. NEVER touches the matching engine, VLM resolver, or
  // recovery passes: it only swaps which already-discovered candidate is the
  // active match for a short-clip range, exactly the same segment-swap +
  // persist logic the Retry endpoint above already performs on acceptance.
  app.post('/api/match/:matchJobId/segment/:segmentIndex/select-candidate', async (req, res) => {
    const { matchJobId, segmentIndex: segmentIndexStr } = req.params;
    const segmentIndex = Number(segmentIndexStr);
    const candidateIndex = Number(req.body?.candidateIndex);
    if (!Number.isFinite(segmentIndex)) return res.status(400).json({ error: 'segmentIndex must be a number' });
    if (!Number.isFinite(candidateIndex)) return res.status(400).json({ error: 'candidateIndex must be a number' });

    const job = matchJobs.get(matchJobId) ?? loadMatchJobFromDisk(matchJobId);
    if (!job) return res.status(404).json({ error: 'Match job not found' });
    if (job.status !== 'completed') return res.status(400).json({ error: 'Match job is not completed yet' });

    // Don't fight a running Retry for the same segment — it would overwrite
    // this selection (or vice versa) when it finishes.
    if (retryInFlight.has(retryKey(matchJobId, segmentIndex))) {
      return res.status(409).json({ error: 'A Retry is currently running for this segment — wait for it to finish first.' });
    }

    const entry = readRecord(uploadDir, matchJobId, segmentIndex);
    if (!entry) return res.status(404).json({ error: 'No candidate data for this segment' });
    const candidate = entry.candidates[candidateIndex];
    if (!candidate) return res.status(404).json({ error: 'Candidate not found' });

    const newSeg = candidate.segment;
    const segsArr: any[] = job.segments ?? [];

    // Find the active segment for this candidate file's short-clip range —
    // exact range first (same 0.05s tolerance used everywhere), then largest
    // overlap as fallback (a previous swap can shift the active range slightly).
    let arrIdx = segsArr.findIndex((s: any) =>
      Math.abs(s.shortStart - entry.shortStart) < 0.05 &&
      Math.abs(s.shortEnd - entry.shortEnd) < 0.05);
    if (arrIdx === -1) {
      let bestOverlap = 0;
      segsArr.forEach((s: any, i: number) => {
        const overlap = Math.min(s.shortEnd, entry.shortEnd) - Math.max(s.shortStart, entry.shortStart);
        if (overlap > bestOverlap) { bestOverlap = overlap; arrIdx = i; }
      });
    }

    if (arrIdx !== -1) {
      segsArr[arrIdx] = newSeg;
      job.segments = segsArr;
    } else {
      // Range had no active segment (previously dropped) — add it now.
      job.segments = [...segsArr, newSeg].sort((a: any, b: any) => a.shortStart - b.shortStart);
    }

    // Recompute the display-only timeline flags — the promoted candidate can
    // sit at a different movie position than the segment it replaces, which
    // can create OR resolve a backwards jump against neighbouring segments.
    try { job.segments = flagTimelineOutliers(job.segments!); } catch { /* display-only */ }

    try {
      await fs.promises.writeFile(matchResultPath(matchJobId), JSON.stringify({
        segments: job.segments,
        unmatchedRanges: job.unmatchedRanges,
        movieFrames: job.movieFrames,
        shortFrames: job.shortFrames,
        verifySummary: job.verifySummary,
      }));
    } catch (e) {
      console.error(`[SelectCandidate] Failed to persist result for ${matchJobId}:`, e);
      return res.status(500).json({ error: 'Could not persist the selection to disk.' });
    }

    // Record the user's choice in the verification record so the ★ Used badge
    // follows the selection everywhere in the UI.
    try {
      entry.usedCandidateIndex = candidateIndex;
      writeRecord(uploadDir, matchJobId, entry);
    } catch (e) {
      console.error(`[SelectCandidate] Failed to update candidate file for ${matchJobId} seg ${segmentIndex}:`, e);
      // Segments were already persisted — not fatal for the selection itself.
    }

    console.log(`[SelectCandidate] Match ${matchJobId} segment ${segmentIndex}: candidate ${candidateIndex} promoted to main match by user.`);
    res.json({ ok: true, segments: job.segments, unmatchedRanges: job.unmatchedRanges });
  });

  // 6l. Manual candidate boundary adjustment — user-triggered, purely additive.
  // Nudges ONE edge (movieStart or movieEnd) of ONE stored candidate by ±1 s.
  // NEVER touches the matching engine, VLM resolver, or recovery passes: it
  // only edits the candidate file, and — iff that candidate is the currently
  // Used/main one — mirrors the change into the active segment with the exact
  // same range-find + persist logic select-candidate already uses.
  app.post('/api/match/:matchJobId/segment/:segmentIndex/adjust-candidate', async (req, res) => {
    const { matchJobId, segmentIndex: segmentIndexStr } = req.params;
    const segmentIndex = Number(segmentIndexStr);
    const candidateIndex = Number(req.body?.candidateIndex);
    const edge = req.body?.edge;
    const deltaSec = Number(req.body?.deltaSec);
    if (!Number.isFinite(segmentIndex)) return res.status(400).json({ error: 'segmentIndex must be a number' });
    if (!Number.isFinite(candidateIndex)) return res.status(400).json({ error: 'candidateIndex must be a number' });
    if (edge !== 'start' && edge !== 'end') return res.status(400).json({ error: "edge must be 'start' or 'end'" });
    if (deltaSec !== 1 && deltaSec !== -1) return res.status(400).json({ error: 'deltaSec must be +1 or -1' });

    const job = matchJobs.get(matchJobId) ?? loadMatchJobFromDisk(matchJobId);
    if (!job) return res.status(404).json({ error: 'Match job not found' });
    if (job.status !== 'completed') return res.status(400).json({ error: 'Match job is not completed yet' });

    // Don't fight a running Retry for the same segment — it could overwrite
    // this candidate file when it finishes.
    if (retryInFlight.has(retryKey(matchJobId, segmentIndex))) {
      return res.status(409).json({ error: 'A Retry is currently running for this segment — wait for it to finish first.' });
    }

    const entry = readRecord(uploadDir, matchJobId, segmentIndex);
    if (!entry) return res.status(404).json({ error: 'No candidate data for this segment' });
    const candidate = entry.candidates[candidateIndex];
    if (!candidate) return res.status(404).json({ error: 'Candidate not found' });

    const seg: any = candidate.segment;
    let newStart = seg.movieStart;
    let newEnd = seg.movieEnd;
    if (edge === 'start') newStart = seg.movieStart + deltaSec;
    else newEnd = seg.movieEnd + deltaSec;

    // Timeline is one-directional: clamp at 0 and keep at least 0.5 s of
    // segment so start can never cross end (or vice versa).
    if (newStart < 0) return res.status(400).json({ error: 'Cannot extend before 0:00 — movie start is already at the beginning.' });
    if (newEnd - newStart < 0.5) return res.status(400).json({ error: 'Segment would become too short (minimum 0.5 s).' });

    seg.movieStart = newStart;
    seg.movieEnd = newEnd;

    try {
      writeRecord(uploadDir, matchJobId, entry);
    } catch (e) {
      console.error(`[AdjustCandidate] Failed to persist verification record for ${matchJobId} seg ${segmentIndex}:`, e);
      return res.status(500).json({ error: 'Could not persist the adjustment to disk.' });
    }

    // If this candidate IS the active/main match for its range, mirror the
    // new bounds into the live segments + persisted result JSON.
    let segmentsUpdated = false;
    if ((entry.usedCandidateIndex ?? entry.recoveredCandidateIndex ?? 0) === candidateIndex) {
      const segsArr: any[] = job.segments ?? [];
      let arrIdx = segsArr.findIndex((s: any) =>
        Math.abs(s.shortStart - entry.shortStart) < 0.05 &&
        Math.abs(s.shortEnd - entry.shortEnd) < 0.05);
      if (arrIdx === -1) {
        let bestOverlap = 0;
        segsArr.forEach((s: any, i: number) => {
          const overlap = Math.min(s.shortEnd, entry.shortEnd) - Math.max(s.shortStart, entry.shortStart);
          if (overlap > bestOverlap) { bestOverlap = overlap; arrIdx = i; }
        });
      }
      if (arrIdx !== -1) {
        segsArr[arrIdx] = { ...segsArr[arrIdx], movieStart: newStart, movieEnd: newEnd };
        job.segments = segsArr;
        segmentsUpdated = true;
        // Recompute the display-only timeline flags — nudging movieStart can
        // move this segment onto or off the dominant forward movie timeline.
        try { job.segments = flagTimelineOutliers(job.segments); } catch { /* display-only */ }
        try {
          await fs.promises.writeFile(matchResultPath(matchJobId), JSON.stringify({
            segments: job.segments,
            unmatchedRanges: job.unmatchedRanges,
            movieFrames: job.movieFrames,
            shortFrames: job.shortFrames,
            verifySummary: job.verifySummary,
          }));
        } catch (e) {
          console.error(`[AdjustCandidate] Failed to persist result for ${matchJobId}:`, e);
          return res.status(500).json({ error: 'Adjusted the candidate but could not persist the main result to disk.' });
        }
      }
    }

    console.log(`[AdjustCandidate] Match ${matchJobId} segment ${segmentIndex}: candidate ${candidateIndex} ${edge} ${deltaSec > 0 ? '+' : ''}${deltaSec}s → ${newStart.toFixed(2)}–${newEnd.toFixed(2)} (mainUpdated=${segmentsUpdated}).`);
    res.json({ ok: true, entry, segmentsUpdated, segments: job.segments, unmatchedRanges: job.unmatchedRanges });
  });

  // 7. Worker Accuracy Calibration
  // Tests hash DETERMINISM: send each synthetic frame to the worker TWICE.
  // If both passes return identical 256-bit hashes → worker is stable & correct.
  // Also verifies: non-empty aHash, dHash, signature (colorGrid, skinScoreGrid,
  // detailGrid) — confirming that all 3 signal channels (structure, color/bg,
  // skin/character) are being computed.
  //
  // NOTE: We do NOT compare main-thread vs worker here because canvas (the native
  // addon used by the worker) cannot be loaded in the main tsx process on this
  // NixOS host (missing libuuid.so.1 in the main-thread LD path). Workers are
  // spawned as child processes that inherit the correct library environment.
  app.post('/api/sanity-test', async (req, res) => {
    try {
      const { Worker } = await import('worker_threads');
      const pathMod    = await import('path');

      const isProd = process.env.NODE_ENV === 'production';
      const workerFile = isProd
        ? pathMod.join(process.cwd(), 'dist/worker.cjs')
        : pathMod.join(process.cwd(), 'server/worker.ts');

      // Realistic frame size — worker downscales 320×240 → 160×120 (proper pipeline)
      const W = 320, H = 240, NUM_FRAMES = 10;

      function makeFakeData(fi: number): Uint8ClampedArray {
        const data = new Uint8ClampedArray(W * H * 4);
        for (let i = 0; i < W * H; i++) {
          const x = i % W;
          const y = Math.floor(i / W);
          data[i * 4]     = ((x * (fi + 1) * 31 + y * 97)  ^ (fi * 53))  & 255;
          data[i * 4 + 1] = ((y * (fi + 1) * 67 + x * 41)  ^ (fi * 29))  & 255;
          data[i * 4 + 2] = ((x * y * 7   + fi  * 113)     ^ 128)         & 255;
          data[i * 4 + 3] = 255;
        }
        return data;
      }

      interface WorkerResult {
        hash: string;
        dhash: string;
        signature?: { colorGrid?: number[]; skinScoreGrid?: number[]; detailGrid?: number[] };
      }

      // Send each frame to a fresh worker twice; compare pass-1 vs pass-2 results
      async function runPass(passIdx: 0 | 1): Promise<WorkerResult[]> {
        return new Promise<WorkerResult[]>((resolve, reject) => {
          // Bootstrap: require tsx/cjs then load the TypeScript worker file.
          // Using eval:true avoids the "Unknown file extension .ts" error that
          // occurs with --import tsx in worker_threads on this Node version.
          const bootstrapCode = isProd
            ? `require(${JSON.stringify(workerFile)})`
            : `require('tsx/cjs'); require(${JSON.stringify(workerFile)});`;
          // Same reason as in pipeline.ts: `require('tsx/cjs')` alone only
          // installs the CJS hook, which is bypassed when the worker's
          // imports go through the ESM loader chain. Registering tsx via
          // execArgv covers both module systems.
          const worker = new Worker(bootstrapCode, {
            eval: true,
            ...(isProd ? {} : { execArgv: ['--import', 'tsx'] }),
          });
          const results: WorkerResult[] = new Array(NUM_FRAMES).fill(null).map(() => ({
            hash: '', dhash: '', signature: undefined
          }));
          let sent = 0, received = 0;

          const sendNext = () => {
            if (sent >= NUM_FRAMES) return;
            const fi   = sent++;
            const data = makeFakeData(fi);
            worker.postMessage({
              id: fi,
              frameBuffer: Buffer.from(data.buffer),
              width: W, height: H
            });
          };

          worker.on('message', (msg: any) => {
            if (msg.error) { worker.terminate(); reject(new Error(msg.error)); return; }
            const v = msg.result?.variants?.full ?? {};
            results[msg.id] = {
              hash:      v.hash      ?? '',
              dhash:     v.dhash     ?? '',
              signature: msg.result?.signature ?? undefined,
            };
            received++;
            if (received >= NUM_FRAMES) { worker.terminate(); resolve(results); }
            else sendNext();
          });
          worker.on('error', (e: Error) => { worker.terminate(); reject(e); });
          setTimeout(() => { worker.terminate(); reject(new Error(`Worker pass ${passIdx} timed out`)); }, 45_000);
          // pipeline: send up to 4 at a time so the worker is always busy
          for (let i = 0; i < Math.min(4, NUM_FRAMES); i++) sendNext();
        });
      }

      let pass1: WorkerResult[] = [], pass2: WorkerResult[] = [];
      let workerError: string | null = null;
      try {
        // Run both passes in parallel (two separate workers)
        [pass1, pass2] = await Promise.all([runPass(0), runPass(1)]);
      } catch (e: any) {
        workerError = e.message || String(e);
        console.warn('[SanityTest] Worker error:', workerError);
      }

      const hasWorker = !workerError;

      const results = pass1.map((r1, fi) => {
        const r2 = pass2[fi];
        // all-zeros aHash is a valid edge case (uniform image → every pixel = mean → no pixel > mean)
        const hashOk      = r1.hash.length === 256;
        const dhashOk     = r1.dhash.length > 0;
        const deterOk     = hasWorker ? (r1.hash === r2?.hash) : false;
        const sigOk       = !!(r1.signature?.colorGrid?.length && r1.signature?.skinScoreGrid?.length && r1.signature?.detailGrid?.length);
        const pass        = hashOk && deterOk;
        return {
          frameIndex:      fi,
          pass,
          hashBits:        r1.hash.length,
          mainHashPrefix:  r1.hash.slice(0, 48),   // re-uses existing UI field: pass-1 hash
          workerHashPrefix: hasWorker ? (r2?.hash ?? '').slice(0, 48) : '(worker unavailable)', // pass-2 hash
          checks: { hashOk, dhashOk, deterministicOk: deterOk, signatureOk: sigOk },
        };
      });

      const allPass = results.every(r => r.pass);
      console.log(`[SanityTest] ${allPass ? 'PASS' : 'FAIL'} — worker=${hasWorker} frames=${results.filter(r => r.pass).length}/${NUM_FRAMES}`);
      res.json({
        pass: allPass,
        totalFrames: NUM_FRAMES,
        workerAvailable: hasWorker,
        workerError: workerError || undefined,
        results
      });
    } catch (err: any) {
      console.error('[SanityTest] Error:', err);
      res.status(500).json({ error: err.message || String(err) });
    }
  });

  // --- VITE MIDDLEWARE CONFIGURATION ---
  // Create the HTTP server up-front so Vite's HMR WebSocket can attach to the
  // SAME server/port that Express listens on. In middleware mode Vite would
  // otherwise spin up its own WS server on a separate port, which the hosted
  // preview proxy cannot reach — the browser's wss:// upgrade then fails with
  // "WebSocket closed without opened".
  const httpServer = http.createServer(app);

  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: {
        middlewareMode: true,
        allowedHosts: true,
        hmr: { server: httpServer },
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*all', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log('[v2] View-All-Candidates build active');
    console.log(`Server ready. Detected ${os.cpus().length} CPU cores. Worker pool sized to ${NUM_WORKERS} workers.`);
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
