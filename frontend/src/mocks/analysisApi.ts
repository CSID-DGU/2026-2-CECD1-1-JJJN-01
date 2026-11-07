/**
 * 백엔드 없이 동작하는 가짜 API.
 * 실제 API 와 같은 함수 시그니처를 유지해서 api/analysis.ts 에서 그대로 바꿔 끼운다.
 * 작업 상태는 요청 시각 기준으로 시간이 흐르면서 단계가 진행되는 것처럼 보이게 한다.
 */

import type { AnalysisJob, CreateAnalysisRequest, CreateAnalysisResponse, JobStatus } from '@/types/analysis';
import type { ExerciseType } from '@/types/exercise';
import { readStorage, writeStorage } from '@/utils/storage';
import { buildMockResult } from './analysisResult';

interface MockJobRecord {
  jobId: string;
  exercise: ExerciseType;
  startedAt: number;
}

const STORAGE_KEY = 'mock-jobs';

/** 새로고침해도 polling 이 이어지도록 sessionStorage 에 둔다 */
function loadJobs(): Record<string, MockJobRecord> {
  return readStorage<Record<string, MockJobRecord>>(STORAGE_KEY) ?? {};
}

function saveJobs(jobs: Record<string, MockJobRecord>) {
  writeStorage(STORAGE_KEY, jobs);
}

/** [단계, 누적 소요 초] — 이 시각이 지나면 다음 단계로 */
const TIMELINE: Array<[JobStatus, number]> = [
  ['queued', 0.8],
  ['extracting', 3.0],
  ['retargeting', 5.0],
  ['scoring', 6.2],
  ['llm', 7.5],
];
const TOTAL_SEC = TIMELINE[TIMELINE.length - 1][1];

function delay(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

export async function mockCreateAnalysis(req: CreateAnalysisRequest): Promise<CreateAnalysisResponse> {
  await delay(400 + Math.min(req.video.size / (1024 * 1024), 20) * 30); // 용량 비례로 살짝 기다리는 척
  const jobId = `mock-${Date.now().toString(36)}`;
  const jobs = loadJobs();
  jobs[jobId] = { jobId, exercise: req.exercise, startedAt: Date.now() };
  saveJobs(jobs);
  return { jobId };
}

export async function mockGetAnalysisJob(jobId: string): Promise<AnalysisJob> {
  await delay(150);
  const record = loadJobs()[jobId];
  if (!record) {
    return { jobId, status: 'failed', error: '분석 작업을 찾을 수 없어요. 다시 시도해 주세요.' };
  }

  const elapsed = (Date.now() - record.startedAt) / 1000;
  for (const [status, until] of TIMELINE) {
    if (elapsed < until) {
      return { jobId, status, progress: Math.min(elapsed / TOTAL_SEC, 0.99) };
    }
  }

  return {
    jobId,
    status: 'done',
    progress: 1,
    result: buildMockResult(record.exercise, jobId),
  };
}
