import { useEffect, useState } from 'react';
import { Navigate, useNavigate } from 'react-router';
import { createAnalysis } from '@/api/analysis';
import { fetchCoachVideoFile } from '@/api/coachVideo';
import { BottomBar } from '@/components/BottomBar';
import { Button } from '@/components/Button';
import { PageHeader } from '@/components/PageHeader';
import { SectionHeading } from '@/components/SectionHeading';
import { coachVideoFor } from '@/constants/coachVideos';
import { SKIP_INPUT_GATES } from '@/constants/dev';
import { EXERCISE_BY_ID } from '@/constants/exercises';
import { ROUTES } from '@/constants/routes';
import { useAnalysis } from '@/context/useAnalysis';
import { useVideoFile } from '@/hooks/useVideoFile';
import { CoachVideoPicker } from './components/CoachVideoPicker';
import { ShootingGuide } from './components/ShootingGuide';
import { VideoDropzone } from './components/VideoDropzone';

/** 2단계: 내 영상(필수)과 코치 영상(직접 올리기 또는 기본 제공)을 고른다. */
export function UploadPage() {
  const navigate = useNavigate();
  const { exercise, studentVideo, coachSelection, setStudentVideo, setCoachSelection, setJobId, setResult } =
    useAnalysis();

  const video = useVideoFile(studentVideo);
  // 직접 올린 코치 영상도 내 영상과 같은 미리보기 · 검사(길이 · 용량)를 거친다
  const coachVideo = useVideoFile(coachSelection?.kind === 'custom' ? coachSelection.file : null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  // 직접 올린 영상이 없으면 이 종목의 기본 영상을 골라 둔다. 저장돼 있던 선택이 이 종목의 것이 아니면
  // (종목을 바꿨거나 예전 id) 바꿔 끼우고, 이 종목에 기본 영상이 없으면 선택을 비운다.
  useEffect(() => {
    if (!exercise || coachSelection?.kind === 'custom') return;
    const first = coachVideoFor(exercise);
    if (coachSelection?.coachId === first?.id) return;
    setCoachSelection(first ? { kind: 'default', coachId: first.id } : null);
  }, [exercise, coachSelection, setCoachSelection]);

  if (!exercise) return <Navigate to={ROUTES.home} replace />;

  const meta = EXERCISE_BY_ID[exercise];
  const inputsReady =
    !!studentVideo && !video.error && !video.loading && !!coachSelection && !coachVideo.error && !coachVideo.loading;
  const canSubmit = (SKIP_INPUT_GATES || inputsReady) && !submitting;

  const handleSubmit = async () => {
    // 임시: 영상 없이도 넘어가도록 빈 파일로 대신한다 (SKIP_INPUT_GATES)
    const placeholder = () => (SKIP_INPUT_GATES ? new File([], 'placeholder.mp4', { type: 'video/mp4' }) : null);
    const videoToSend = studentVideo ?? placeholder();
    if (!videoToSend) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      // 코치 영상은 항상 파일로 보낸다. 기본 영상을 골랐으면 프론트에 들어 있는 그 파일을 읽어서 올린다
      const defaultCoach = coachSelection?.kind === 'default' ? coachVideoFor(exercise) : null;
      const coachToSend =
        coachSelection?.kind === 'custom'
          ? coachSelection.file
          : defaultCoach
            ? await fetchCoachVideoFile(defaultCoach)
            : placeholder();
      if (!coachToSend) return;
      const { jobId } = await createAnalysis({ video: videoToSend, exercise, coachVideo: coachToSend });
      setResult(null);
      setJobId(jobId);
      navigate(ROUTES.analyzing);
    } catch (e) {
      setSubmitError(e instanceof Error ? e.message : '분석 요청에 실패했어요. 다시 시도해 주세요.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <PageHeader title={`${meta.name} 영상 올리기`} backTo={ROUTES.home} />

      <main className="flex flex-1 flex-col gap-8 px-5 pt-2 pb-6">
        <section>
          <SectionHeading title="내 영상" description="비교할 내 동작이에요" />
          <VideoDropzone
            file={studentVideo}
            previewUrl={video.previewUrl}
            metadata={video.metadata}
            error={video.error}
            loading={video.loading}
            onPick={setStudentVideo}
            onClear={() => setStudentVideo(null)}
          />
          <div className="mt-3">
            <ShootingGuide lines={meta.shootingGuide} />
          </div>
        </section>

        <section>
          <SectionHeading title="코치 영상" description="이 동작을 내 체형에 맞게 옮겨서 비교해요" />
          <CoachVideoPicker
            exercise={exercise}
            value={coachSelection}
            onChange={setCoachSelection}
            customPreviewUrl={coachVideo.previewUrl}
            customMetadata={coachVideo.metadata}
            customError={coachVideo.error}
            customLoading={coachVideo.loading}
          />
        </section>
      </main>

      <BottomBar note={submitError ? <span className="text-bad">{submitError}</span> : undefined}>
        <Button size="lg" block disabled={!canSubmit} loading={submitting} onClick={handleSubmit}>
          {submitting ? '영상 보내는 중' : '분석 시작'}
        </Button>
      </BottomBar>
    </>
  );
}
