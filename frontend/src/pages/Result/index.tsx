import { useCallback, useMemo, useRef, useState } from 'react';
import { Navigate, useNavigate } from 'react-router';
import { SectionHeading } from '@/components/SectionHeading';
import { EXERCISE_BY_ID } from '@/constants/exercises';
import { ROUTES } from '@/constants/routes';
import { useAnalysis } from '@/context/useAnalysis';
import { useFramePlayer } from '@/hooks/useFramePlayer';
import type { Segment } from '@/types/analysis';
import { segmentAtTime } from '@/utils/segments';
import { secondsToFrame } from '@/utils/skeleton';
import { FeedbackPanel } from './components/FeedbackPanel';
import { OverlayVideo } from './components/OverlayVideo';
import { ResultHeader } from './components/ResultHeader';
import { SegmentTimeline } from './components/SegmentTimeline';
import { FrameControls } from './components/SkeletonViewer/FrameControls';
import { SkeletonViewer, type ViewerLayout } from './components/SkeletonViewer';
import { ViewerPanel } from './components/ViewerPanel';

/**
 * 4단계: 결과 (넓은 화면 기준).
 * 왼쪽: 코치를 내 영상 위에 겹친 오버레이 영상. 오른쪽: 같은 프레임의 3D 뷰어 + 지금 구간의 상세.
 * 구간을 누르면 둘 다 그 구간으로 간다.
 * 아래: 재생 컨트롤 → 구간 타임라인 → 피드백. 영상과 3D 는 재생 시계(player) 하나를 따른다.
 * 오버레이 영상이 없는 결과(overlayUrl 없음)는 왼쪽에 3D 뷰어를 그린다.
 */
export function ResultPage() {
  const navigate = useNavigate();
  const { result, reset } = useAnalysis();

  const fps = result?.student.fps ?? 30;
  const frameCount = result?.student.frames.length ?? 0;
  const player = useFramePlayer({ frameCount, fps });

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [layout, setLayout] = useState<ViewerLayout>('overlay');
  // 구간은 아래(타임라인 · 피드백)에서 누르므로, 누르면 영상과 3D 가 있는 위쪽으로 올라간다
  const stageRef = useRef<HTMLElement>(null);

  const currentSec = player.frame / fps;
  const playingSegment = useMemo(
    () => (result ? segmentAtTime(result.segments, currentSec) : null),
    [result, currentSec],
  );

  const jumpToSegment = useCallback(
    (segment: Segment) => {
      setSelectedId(segment.id);
      stageRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      player.pause();
      if (segment.studentRange) player.seek(secondsToFrame(segment.studentRange[0], fps, frameCount));
    },
    [player, fps, frameCount],
  );

  if (!result) return <Navigate to={ROUTES.home} replace />;

  const { segments, student, coachRetargeted, overlayUrl } = result;
  const exerciseName = EXERCISE_BY_ID[result.exercise].name;

  // 재생 중엔 재생 위치의 구간을, 멈췄을 땐 고른 구간(없으면 재생 위치의 구간)을 보여준다
  const selectedSegment = selectedId ? (segments.find((s) => s.id === selectedId) ?? null) : null;
  const activeSegment = player.isPlaying ? playingSegment : (selectedSegment ?? playingSegment);
  const highlightBones = new Set(activeSegment?.deviations.map((d) => d.bone) ?? []);

  // 두 스켈레톤은 같은 길이의 재생 순서로 온다. 재생 프레임마다 그릴 코치 프레임과
  // 누가 움직이는지는 백엔드가 frameMap 으로 정해 준다
  const coachFrame = result.frameMap.coachFrame[player.frame] ?? player.frame;
  const frameKind = result.frameMap.kind[player.frame] ?? 'together';
  // 타임라인의 축 길이. 프레임 k 는 [k, k + 1) / fps 를 차지하고, 마지막 구간은 frameCount / fps 에서 끝난다
  const totalSec = frameCount / fps;

  const viewer = (
    <SkeletonViewer
      student={student}
      coach={coachRetargeted}
      studentFrame={player.frame}
      coachFrame={coachFrame}
      frameKind={frameKind}
      highlightBones={highlightBones}
      layout={layout}
      onLayoutChange={setLayout}
      viewDirection={result.viewDirection}
      className="h-full"
    />
  );

  return (
    <>
      <ResultHeader
        title={`${exerciseName} 분석 결과`}
        onBack={() => navigate(ROUTES.upload)}
        onRetry={() => navigate(ROUTES.upload)}
        onHome={() => {
          reset();
          navigate(ROUTES.home);
        }}
      />

      <main className="flex flex-1 flex-col gap-10 pt-4 pb-16">
        <section ref={stageRef} className="flex scroll-mt-20 flex-col gap-4">
          {/* 영상(세로)은 높이에 맞춰 폭이 정해지고, 오른쪽 칸이 남은 폭을 채운다 */}
          <div className="flex h-[min(72vh,760px)] min-h-[480px] gap-6">
            {overlayUrl ? (
              <OverlayVideo src={overlayUrl} player={player} fps={fps} className="h-full w-auto max-w-[55%]" />
            ) : (
              <div className="aspect-square h-full">{viewer}</div>
            )}
            <div className="min-w-0 flex-1">
              <ViewerPanel
                segment={activeSegment}
                viewer={overlayUrl ? viewer : undefined}
                summary={result.feedback?.summary}
              />
            </div>
          </div>
          <FrameControls player={player} fps={fps} frameCount={frameCount} />
          <p className="text-[12px] text-ink-faint">
            {overlayUrl ? '영상 속 파란 몸이 내 체형으로 옮긴 코치예요. ' : ''}
            3D 뷰어에서 내 스켈레톤의 빨간 뼈가 코치와 차이가 큰 곳이에요.
          </p>
        </section>

        <section>
          <SectionHeading title="구간별로 보기" description="누르면 영상과 3D 가 그 구간으로 이동해요" />
          <SegmentTimeline
            segments={segments}
            totalSec={totalSec}
            currentSec={currentSec}
            selectedId={activeSegment?.id ?? null}
            onSelect={jumpToSegment}
          />
        </section>

        <section className="max-w-[820px]">
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
    </>
  );
}
