/**
 * RETRY candidate system — a dedicated 1:1 copy of server/verification/.
 *
 * This copy exists so the manual Retry flow (segment Retry + gap Retry in
 * server.ts) can evolve independently WITHOUT touching the main matching /
 * candidate system in server/verification/, which the initial verify pass
 * keeps using untouched. Both copies read/write the SAME on-disk candidate
 * records, so results stay fully compatible.
 *
 * Make Retry-only behavior changes HERE, never in server/verification/.
 * The rest of the app should import from here and nowhere deeper.
 */

export {
  verifyMatchedSegments,
  recheckSegment,
  type VerifyRequest,
  type VerifyResult,
  type VerifySummary,
  type RecheckRequest,
  type RecheckResult,
} from './verify';

export {
  readRecord,
  readAllRecords,
  listRecordIndexes,
  deleteRecordsForJob,
  writeRecord,
  type CandidateRecord,
  type CandidateVerdict,
  type VerificationRecord,
} from './store';

export { flagTimelineOutliers } from './timeline';
