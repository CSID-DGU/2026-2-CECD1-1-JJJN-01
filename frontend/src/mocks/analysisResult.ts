/**
 * 백엔드 없이 결과 화면을 확인하기 위한 가짜 결과.
 * 스쿼트 한 번을 간단한 2관절 다리 IK 로 만들어 낸다.
 * (데드리프트·벤치프레스도 지금은 같은 동작을 재활용한다 — 화면 확인용이므로)
 *
 * 타입(src/types/analysis.ts)이 바뀌면 이 파일만 같이 고치면 된다.
 */

import type { AlignmentPath, AnalysisResult, LlmFeedback, Segment, SkeletonSequence, Vec3 } from '@/types/analysis';
import type { ExerciseType } from '@/types/exercise';

export const MOCK_JOINT_NAMES = [
  'nose',
  'neck',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'pelvis',
  'left_hip',
  'right_hip',
  'left_knee',
  'right_knee',
  'left_ankle',
  'right_ankle',
];

/* ---------------------------------------------------------------- */
/* 동작 생성                                                          */
/* ---------------------------------------------------------------- */

const THIGH = 0.45;
const SHIN = 0.45;
const ANKLE_Y = 0.08;
const HIP_TOP = THIGH + SHIN + ANKLE_Y - 0.02; // 선 자세의 골반 높이
const HIP_BOTTOM = 0.5;
const TORSO = 0.5;

interface Phase {
  /** 구간 길이 (초) */
  durationSec: number;
  /** 0 = 선 자세, 1 = 바닥 */
  from: number;
  to: number;
}

interface MotionSpec {
  fps: number;
  phases: Phase[];
  /** 바닥에서의 최대 상체 기울기 (라디안) */
  maxLean: number;
  /** 바닥에서 무릎이 앞으로 나가는 정도 (0~1) */
  kneeForward: number;
  /** 좌우 비대칭 (양수면 왼쪽 무릎이 안으로) */
  kneeCave: number;
}

function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

/** 발목 고정, 골반 높이로 무릎 위치를 푸는 2관절 IK (x 앞, y 위) */
function solveKnee(hipX: number, hipY: number): [number, number] {
  const ax = 0;
  const ay = ANKLE_Y;
  const dx = hipX - ax;
  const dy = hipY - ay;
  const d = Math.min(Math.hypot(dx, dy), THIGH + SHIN - 1e-4);
  const a = (SHIN * SHIN - THIGH * THIGH + d * d) / (2 * d);
  const h = Math.sqrt(Math.max(SHIN * SHIN - a * a, 0));
  const ux = dx / Math.hypot(dx, dy);
  const uy = dy / Math.hypot(dx, dy);
  const mx = ax + a * ux;
  const my = ay + a * uy;
  // 앞쪽(+x) 해를 고른다
  const k1: [number, number] = [mx + h * uy, my - h * ux];
  const k2: [number, number] = [mx - h * uy, my + h * ux];
  return k1[0] >= k2[0] ? k1 : k2;
}

function poseAt(depth: number, spec: MotionSpec): Vec3[] {
  const hipY = HIP_TOP + (HIP_BOTTOM - HIP_TOP) * depth;
  const hipX = -0.12 * depth + 0.06 * spec.kneeForward * depth;
  const [kneeX, kneeY] = solveKnee(hipX, hipY);
  const lean = spec.maxLean * depth;

  const pelvis: Vec3 = [hipX, hipY, 0];
  const neck: Vec3 = [hipX + Math.sin(lean) * TORSO, hipY + Math.cos(lean) * TORSO, 0];
  const nose: Vec3 = [neck[0] + Math.sin(lean) * 0.16 + 0.05, neck[1] + Math.cos(lean) * 0.16, 0];

  const shoulderHalf = 0.19;
  const hipHalf = 0.11;
  const lShoulder: Vec3 = [neck[0], neck[1] - 0.02, shoulderHalf];
  const rShoulder: Vec3 = [neck[0], neck[1] - 0.02, -shoulderHalf];

  // 팔은 앞으로 뻗은 자세
  const armForward = 0.3;
  const lElbow: Vec3 = [lShoulder[0] + armForward * 0.5, lShoulder[1] - 0.08, shoulderHalf - 0.03];
  const rElbow: Vec3 = [rShoulder[0] + armForward * 0.5, rShoulder[1] - 0.08, -shoulderHalf + 0.03];
  const lWrist: Vec3 = [lElbow[0] + armForward * 0.6, lElbow[1] + 0.06, shoulderHalf - 0.08];
  const rWrist: Vec3 = [rElbow[0] + armForward * 0.6, rElbow[1] + 0.06, -shoulderHalf + 0.08];

  const cave = spec.kneeCave * depth;
  const lHip: Vec3 = [hipX, hipY, hipHalf];
  const rHip: Vec3 = [hipX, hipY, -hipHalf];
  const lKnee: Vec3 = [kneeX, kneeY, hipHalf - cave];
  const rKnee: Vec3 = [kneeX, kneeY, -hipHalf + cave * 0.3];
  const lAnkle: Vec3 = [0, ANKLE_Y, hipHalf];
  const rAnkle: Vec3 = [0, ANKLE_Y, -hipHalf];

  return [
    nose,
    neck,
    lShoulder,
    rShoulder,
    lElbow,
    rElbow,
    lWrist,
    rWrist,
    pelvis,
    lHip,
    rHip,
    lKnee,
    rKnee,
    lAnkle,
    rAnkle,
  ];
}

function buildSequence(spec: MotionSpec): { seq: SkeletonSequence; phaseFrames: Array<[number, number]> } {
  const frames: Vec3[][] = [];
  const phaseFrames: Array<[number, number]> = [];
  for (const phase of spec.phases) {
    const n = Math.max(1, Math.round(phase.durationSec * spec.fps));
    const start = frames.length;
    for (let i = 0; i < n; i++) {
      const t = n === 1 ? 0 : i / (n - 1);
      const depth = phase.from + (phase.to - phase.from) * easeInOut(t);
      frames.push(poseAt(depth, spec));
    }
    phaseFrames.push([start, frames.length]);
  }
  return { seq: { jointNames: MOCK_JOINT_NAMES, fps: spec.fps, frames }, phaseFrames };
}

/** 구간별로 사용자 프레임 ↔ 코치 프레임을 선형으로 잇는다. */
function buildAlignment(student: Array<[number, number]>, coach: Array<[number, number]>): AlignmentPath {
  const path: AlignmentPath = [];
  for (let p = 0; p < student.length; p++) {
    const [s0, s1] = student[p];
    const [c0, c1] = coach[p];
    const sn = s1 - s0;
    const cn = c1 - c0;
    for (let i = 0; i < sn; i++) {
      const c = c0 + Math.min(cn - 1, Math.round((i / Math.max(sn - 1, 1)) * (cn - 1)));
      path.push([s0 + i, c]);
    }
  }
  return path;
}

/* ---------------------------------------------------------------- */
/* 결과 조립                                                          */
/* ---------------------------------------------------------------- */

const SEGMENT_LABELS: Record<ExerciseType, string[]> = {
  squat: ['시작 자세', '하강', '바닥 유지', '상승', '마무리 자세'],
  deadlift: ['셋업', '리프트', '락아웃', '내리기', '마무리 자세'],
  bench_press: ['시작 자세', '하강', '가슴 터치', '프레스', '마무리 자세'],
};

export function buildMockResult(exercise: ExerciseType, jobId: string): AnalysisResult {
  const fps = 30;

  // 코치(리타겟팅): 하강 1.0s · 바닥 0.5s · 상승 1.0s
  const coachSpec: MotionSpec = {
    fps,
    phases: [
      { durationSec: 0.3, from: 0, to: 0 },
      { durationSec: 1.0, from: 0, to: 1 },
      { durationSec: 0.5, from: 1, to: 1 },
      { durationSec: 1.0, from: 1, to: 0 },
      { durationSec: 0.3, from: 0, to: 0 },
    ],
    maxLean: (35 * Math.PI) / 180,
    kneeForward: 0.6,
    kneeCave: 0,
  };

  // 사용자: 하강 1.2s(slow) · 바닥 0.15s(거의 없음) · 상승 1.5s(slow), 상체가 더 숙여짐
  const studentSpec: MotionSpec = {
    fps,
    phases: [
      { durationSec: 0.3, from: 0, to: 0 },
      { durationSec: 1.2, from: 0, to: 0.92 },
      { durationSec: 0.15, from: 0.92, to: 0.92 },
      { durationSec: 1.5, from: 0.92, to: 0 },
      { durationSec: 0.3, from: 0, to: 0 },
    ],
    maxLean: (51 * Math.PI) / 180,
    kneeForward: 0.9,
    kneeCave: 0.05,
  };

  const coach = buildSequence(coachSpec);
  const student = buildSequence(studentSpec);
  const alignment = buildAlignment(student.phaseFrames, coach.phaseFrames);

  const toSec = (range: [number, number]): [number, number] => [range[0] / fps, range[1] / fps];
  const labels = SEGMENT_LABELS[exercise];

  const segments: Segment[] = [
    {
      id: 'seg-0',
      label: labels[0],
      verdict: 'match',
      studentRange: toSec(student.phaseFrames[0]),
      coachRange: toSec(coach.phaseFrames[0]),
      timing: { label: 'ok', ratio: 1.0 },
      deviations: [],
    },
    {
      id: 'seg-1',
      label: labels[1],
      verdict: 'match',
      studentRange: toSec(student.phaseFrames[1]),
      coachRange: toSec(coach.phaseFrames[1]),
      timing: { label: 'slow', ratio: 1.2 },
      deviations: [
        { bone: 'torso', angleDiffDeg: 15.9 },
        { bone: 'left_shin', angleDiffDeg: 6.1 },
      ],
    },
    {
      id: 'seg-2',
      label: labels[2],
      verdict: 'missing',
      studentRange: toSec(student.phaseFrames[2]),
      coachRange: toSec(coach.phaseFrames[2]),
      deviations: [],
    },
    {
      id: 'seg-3',
      label: labels[3],
      verdict: 'match',
      studentRange: toSec(student.phaseFrames[3]),
      coachRange: toSec(coach.phaseFrames[3]),
      timing: { label: 'slow', ratio: 1.5 },
      deviations: [
        { bone: 'torso', angleDiffDeg: 12.4 },
        { bone: 'left_thigh', angleDiffDeg: 8.2 },
      ],
    },
    {
      id: 'seg-4',
      label: labels[4],
      verdict: 'match',
      studentRange: toSec(student.phaseFrames[4]),
      coachRange: toSec(coach.phaseFrames[4]),
      timing: { label: 'ok', ratio: 1.0 },
      deviations: [],
    },
  ];

  const feedback: LlmFeedback = {
    summary: '전체 동작 흐름은 좋아요. 바닥에서 잠깐 멈추는 구간이 빠졌고, 내려갈 때 상체가 코치보다 많이 숙여져요.',
    items: [
      {
        segmentId: 'seg-1',
        text: '하강이 코치보다 1.2배 느려요. 속도 자체는 괜찮지만 상체가 16° 더 기울어요. 가슴을 들고 시선을 앞에 두면 몸통 각도가 잡혀요.',
      },
      {
        segmentId: 'seg-2',
        text: '바닥에서 바로 튕기듯 올라와요. 코치는 바닥에서 0.5초 정도 멈춰요. 한 박자 멈췄다가 올라오는 연습을 해 보세요.',
      },
      {
        segmentId: 'seg-3',
        text: '상승이 코치보다 1.5배 느리고, 왼쪽 허벅지 각도가 8° 차이 나요. 양쪽 무릎을 같은 방향으로 밀면서 올라오세요.',
      },
    ],
  };

  return {
    jobId,
    exercise,
    score: {
      total: 72,
      bySegment: { 'seg-0': 96, 'seg-1': 71, 'seg-2': 38, 'seg-3': 68, 'seg-4': 93 },
    },
    segments,
    student: student.seq,
    coachRetargeted: coach.seq,
    alignment,
    feedback,
  };
}
