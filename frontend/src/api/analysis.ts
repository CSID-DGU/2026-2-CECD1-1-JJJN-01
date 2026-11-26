import type { AnalysisJob, CreateAnalysisRequest, CreateAnalysisResponse } from '@/types/analysis';
import { mockCreateAnalysis, mockGetAnalysisJob } from '@/mocks/analysisApi';
import { API_BASE_URL, request, USE_MOCK } from './client';

/**
 * 분석 요청. 영상 원본을 전처리 없이 그대로 보낸다.
 * TODO(backend): 엔드포인트·필드 이름 확정 (지금: POST /analysis, multipart)
 */
export async function createAnalysis(req: CreateAnalysisRequest): Promise<CreateAnalysisResponse> {
  if (USE_MOCK) return mockCreateAnalysis(req);

  const form = new FormData();
  form.append('video', req.video);
  form.append('exercise', req.exercise);
  form.append('coachVideo', req.coachVideo);

  return request<CreateAnalysisResponse>('/analysis', { method: 'POST', body: form });
}

/**
 * 작업 상태 조회. done/failed 까지 폴링한다.
 * TODO(backend): GET /analysis/{jobId} 응답 형태 확정
 */
export async function getAnalysisJob(jobId: string): Promise<AnalysisJob> {
  if (USE_MOCK) return mockGetAnalysisJob(jobId);
  const job = await request<AnalysisJob>(`/analysis/${encodeURIComponent(jobId)}`);
  // 백엔드는 오버레이 영상을 상대 경로(/analysis/{jobId}/overlay.mp4)로 준다
  if (job.result?.overlayUrl?.startsWith('/')) job.result.overlayUrl = `${API_BASE_URL}${job.result.overlayUrl}`;
  return job;
}
