import { useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
import { Grid, OrbitControls } from '@react-three/drei';
import type { SkeletonSequence } from '@/types/analysis';
import { computeBounds, resolveBones } from '@/utils/skeleton';
import { cn } from '@/utils/cn';
import { SkeletonMesh } from './SkeletonMesh';

export type ViewerLayout = 'overlay' | 'side';

interface SkeletonViewerProps {
  student: SkeletonSequence;
  coach: SkeletonSequence;
  studentFrame: number;
  coachFrame: number;
  /** 사용자 스켈레톤에서 강조할 뼈 */
  highlightBones: Set<string>;
  layout: ViewerLayout;
  onLayoutChange: (layout: ViewerLayout) => void;
}

/** CSS 토큰과 같은 값. (three.js 재질에는 CSS 변수를 못 쓴다) */
const COLOR_COACH = '#2f5bea';
const COLOR_ME = '#f2653a';
const COLOR_BAD = '#d64545';

/**
 * 3D 뷰어. 사용자(주황)와 리타겟팅된 코치(파랑)를 한 공간에 그린다.
 * 겹쳐 보기 / 나란히 보기 전환, 드래그로 회전.
 */
export function SkeletonViewer({
  student,
  coach,
  studentFrame,
  coachFrame,
  highlightBones,
  layout,
  onLayoutChange,
}: SkeletonViewerProps) {
  const studentBones = useMemo(() => resolveBones(student.jointNames), [student.jointNames]);
  const coachBones = useMemo(() => resolveBones(coach.jointNames), [coach.jointNames]);
  const bounds = useMemo(() => computeBounds([student, coach]), [student, coach]);

  const sideOffset = layout === 'side' ? bounds.size * 0.45 : 0;
  const { center, size } = bounds;
  const cameraPosition: [number, number, number] = [
    center[0] + size * 1.2,
    center[1] + size * 0.35,
    center[2] + size * 1.9,
  ];
  // 겹쳐 보기에서는 코치를 얇고 반투명하게 그려서 내 스켈레톤(과 빨간 뼈)이 앞에 읽히게 한다
  const coachGhost = layout === 'overlay';

  const studentJoints = student.frames[studentFrame] ?? student.frames[0] ?? [];
  const coachJoints = coach.frames[coachFrame] ?? coach.frames[0] ?? [];

  return (
    <div className="overflow-hidden rounded-card bg-page">
      <div className="flex items-center justify-between px-3 pt-3">
        <div className="flex items-center gap-3 text-[12px] font-semibold">
          <span className="flex items-center gap-1.5 text-coach">
            <span className="size-2.5 rounded-full bg-coach" aria-hidden="true" />
            코치 (내 체형)
          </span>
          <span className="flex items-center gap-1.5 text-me">
            <span className="size-2.5 rounded-full bg-me" aria-hidden="true" />나
          </span>
        </div>
        <div
          role="radiogroup"
          aria-label="보기 방식"
          className="flex rounded-full bg-surface p-0.5 text-[12px] font-semibold"
        >
          {(
            [
              ['overlay', '겹쳐 보기'],
              ['side', '나란히'],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              role="radio"
              aria-checked={layout === value}
              onClick={() => onLayoutChange(value)}
              className={cn(
                'rounded-full px-2.5 py-1 transition-colors',
                layout === value ? 'bg-ink text-white' : 'text-ink-muted',
              )}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="aspect-square w-full touch-none">
        <Canvas camera={{ position: cameraPosition, fov: 35, near: 0.05, far: 50 }} dpr={[1, 2]}>
          <ambientLight intensity={0.9} />
          <directionalLight position={[3, 5, 4]} intensity={0.9} />
          <directionalLight position={[-3, 2, -2]} intensity={0.3} />

          <Grid
            position={[0, 0, 0]}
            args={[6, 6]}
            cellSize={0.25}
            cellThickness={0.6}
            cellColor="#d9dee4"
            sectionSize={1}
            sectionThickness={1}
            sectionColor="#c5ccd4"
            fadeDistance={9}
            fadeStrength={1}
          />

          <group position={[0, 0, -sideOffset]}>
            <SkeletonMesh
              joints={coachJoints}
              bones={coachBones}
              color={COLOR_COACH}
              opacity={coachGhost ? 0.45 : 1}
              jointRadius={coachGhost ? 0.02 : undefined}
              boneRadius={coachGhost ? 0.011 : undefined}
            />
          </group>
          <group position={[0, 0, sideOffset]}>
            <SkeletonMesh
              joints={studentJoints}
              bones={studentBones}
              color={COLOR_ME}
              highlight={highlightBones}
              highlightColor={COLOR_BAD}
            />
          </group>

          <OrbitControls
            makeDefault
            target={[center[0], center[1], center[2]]}
            enablePan={false}
            minDistance={size * 1.2}
            maxDistance={size * 5}
            maxPolarAngle={Math.PI * 0.55}
          />
        </Canvas>
      </div>
    </div>
  );
}
