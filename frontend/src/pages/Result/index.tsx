import { useCallback, useMemo, useState } from 'react';
import { Navigate, useNavigate } from 'react-router';
import { BottomBar } from '@/components/BottomBar';
import { Button } from '@/components/Button';
import { PageHeader } from '@/components/PageHeader';
import { SectionHeading } from '@/components/SectionHeading';
import { EXERCISE_BY_ID } from '@/constants/exercises';
import { ROUTES } from '@/constants/routes';
import { useAnalysis } from '@/context/useAnalysis';
import { useFramePlayer } from '@/hooks/useFramePlayer';
import type { Segment } from '@/types/analysis';
import { segmentAtTime } from '@/utils/segments';
import { buildAlignmentLookup, secondsToFrame } from '@/utils/skeleton';
import { FeedbackPanel } from './components/FeedbackPanel';
import { ScoreCard } from './components/ScoreCard';
import { SegmentDetail } from './components/SegmentDetail';
import { SegmentTimeline } from './components/SegmentTimeline';
import { FrameControls } from './components/SkeletonViewer/FrameControls';
import { SkeletonViewer, type ViewerLayout } from './components/SkeletonViewer';

/**
 * 4단계: 결과.
 * 점수 → 3D 뷰어(나 vs 리타겟팅된 코치) → 구간 타임라인 → 구간 상세 → 피드백.
 * 타임라인·피드백에서 구간을 누르면 뷰어가 그 구간으로 간다.
 */
export function ResultPage() {
  const navigate = useNavigate();
  const { result, reset } = useAnalysis();

  const fps = result?.student.fps ?? 30;
  const frameCount = result?.student.frames.length ?? 0;
  const player = useFramePlayer({ frameCount, fps });

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layout, setLayout] = useState<ViewerLayout>('overlay');

  const lookup = useMemo(
    () => (result ? buildAlignmentLookup(result.alignment, frameCount) : null),
    [result, frameCount],
  );

  const currentSec = player.frame / fps;
  const playingSegment = useMemo(
    () => (result ? segmentAtTime(result.segments, currentSec) : null),
    [result, currentSec],
  );

  const jumpToSegment = useCallback(
    (segment: Segment) => {
      setSelectedId(segment.id);
      player.pause();
      if (segment.studentRange) player.seek(secondsToFrame(segment.studentRange[0], fps, frameCount));
    },
    [player, fps, frameCount],
  );

  if (!result) return <Navigate to={ROUTES.home} replace />;

  const { segments, student, coachRetargeted } = result;
  const exerciseName = EXERCISE_BY_ID[result.exercise].name;

  // 재생 중엔 재생 위치의 구간을, 멈췄을 땐 고른 구간(없으면 재생 위치의 구간)을 보여준다
  const selectedSegment = selectedId ? (segments.find((s) => s.id === selectedId) ?? null) : null;
  const activeSegment = player.isPlaying ? playingSegment : (selectedSegment ?? playingSegment);
  const highlightBones = new Set(activeSegment?.deviations.map((d) => d.bone) ?? []);

  const coachFrameCount = coachRetargeted.frames.length;
  const coachFrame = lookup ? Math.min(lookup[player.frame] ?? 0, Math.max(coachFrameCount - 1, 0)) : 0;
  const totalSec = Math.max(frameCount - 1, 0) / fps;

  return (
    <>
      <PageHeader
        title={`${exerciseName} 분석 결과`}
        backTo={ROUTES.upload}
        right={
          <button
            type="button"
            onClick={() => {
              reset();
              navigate(ROUTES.home);
            }}
            className="text-[13px] font-semibold text-ink-muted"
          >
            처음으로
          </button>
        }
      />

      <main className="flex flex-1 flex-col gap-8 px-5 pt-3 pb-6">
        <section>
          <ScoreCard score={result.score} />
        </section>

        <section className="flex flex-col gap-3">
          <SkeletonViewer
            student={student}
            coach={coachRetargeted}
            studentFrame={player.frame}
            coachFrame={coachFrame}
            highlightBones={highlightBones}
            layout={layout}
            onLayoutChange={setLayout}
          />
          <FrameControls player={player} fps={fps} frameCount={frameCount} />
          <p className="text-[12px] text-ink-faint">
            드래그해서 돌려 볼 수 있어요. 내 스켈레톤의 빨간 뼈가 코치와 차이가 큰 곳이에요.
          </p>
        </section>

        <section>
          <SectionHeading title="구간별로 보기" description="누르면 그 구간으로 이동해요" />
          <SegmentTimeline
            segments={segments}
            totalSec={totalSec}
            currentSec={currentSec}
            selectedId={activeSegment?.id ?? null}
            onSelect={jumpToSegment}
          />
          <div className="mt-4">
            {activeSegment ? (
              <SegmentDetail segment={activeSegment} score={result.score.bySegment?.[activeSegment.id]} />
            ) : (
              <p className="text-[13px] text-ink-faint">구간을 눌러 자세한 내용을 확인해 보세요.</p>
            )}
          </div>
        </section>

        <section>
          <SectionHeading title="코치 피드백" />
          <FeedbackPanel
            feedback={result.feedback}
            segments={segments}
            onJump={(id) => {
              const segment = segments.find((s) => s.id === id);
              if (segment) jumpToSegment(segment);
            }}
          />
        </section>
      </main>

      <BottomBar>
        <Button size="lg" block onClick={() => navigate(ROUTES.upload)}>
          다른 영상으로 다시 비교
        </Button>
      </BottomBar>
    </>
  );
}
