import type { SkeletonSequence, Vec3 } from '@/types/analysis';

/**
 * 뼈 연결 정의.
 * 관절 "이름" 기준이라 백엔드가 어떤 순서·개수로 관절을 보내든, 이름만 아래와 맞으면 그려진다.
 * 이름이 없는 뼈는 그냥 건너뛴다(점만 찍힌다).
 *
 * TODO(backend): 관절 세트 확정되면 이름을 맞춘다. (MediaPipe 는 left_shoulder 등 같은 이름을 쓴다)
 */
export interface BoneDef {
  /** 결과의 deviations[].bone 과 맞추는 이름 */
  name: string;
  from: string;
  to: string;
}

export const BONES: BoneDef[] = [
  { name: 'torso', from: 'pelvis', to: 'neck' },
  { name: 'head', from: 'neck', to: 'nose' },
  { name: 'left_clavicle', from: 'neck', to: 'left_shoulder' },
  { name: 'right_clavicle', from: 'neck', to: 'right_shoulder' },
  { name: 'left_upper_arm', from: 'left_shoulder', to: 'left_elbow' },
  { name: 'right_upper_arm', from: 'right_shoulder', to: 'right_elbow' },
  { name: 'left_forearm', from: 'left_elbow', to: 'left_wrist' },
  { name: 'right_forearm', from: 'right_elbow', to: 'right_wrist' },
  { name: 'left_pelvis', from: 'pelvis', to: 'left_hip' },
  { name: 'right_pelvis', from: 'pelvis', to: 'right_hip' },
  { name: 'left_thigh', from: 'left_hip', to: 'left_knee' },
  { name: 'right_thigh', from: 'right_hip', to: 'right_knee' },
  { name: 'left_shin', from: 'left_knee', to: 'left_ankle' },
  { name: 'right_shin', from: 'right_knee', to: 'right_ankle' },
];

/** 화면에 보여줄 뼈 이름 */
export const BONE_LABELS: Record<string, string> = {
  torso: '몸통',
  head: '머리',
  left_clavicle: '왼쪽 어깨선',
  right_clavicle: '오른쪽 어깨선',
  left_upper_arm: '왼쪽 위팔',
  right_upper_arm: '오른쪽 위팔',
  left_forearm: '왼쪽 아래팔',
  right_forearm: '오른쪽 아래팔',
  left_pelvis: '왼쪽 골반',
  right_pelvis: '오른쪽 골반',
  left_thigh: '왼쪽 허벅지',
  right_thigh: '오른쪽 허벅지',
  left_shin: '왼쪽 정강이',
  right_shin: '오른쪽 정강이',
};

export function boneLabel(name: string): string {
  return BONE_LABELS[name] ?? name;
}

export interface ResolvedBone {
  name: string;
  /** jointNames 안의 인덱스 */
  a: number;
  b: number;
}

/** 관절 이름 배열을 받아 실제로 그릴 수 있는 뼈만 인덱스 쌍으로 돌려준다. */
export function resolveBones(jointNames: string[]): ResolvedBone[] {
  const index = new Map(jointNames.map((n, i) => [n, i]));
  const out: ResolvedBone[] = [];
  for (const bone of BONES) {
    const a = index.get(bone.from);
    const b = index.get(bone.to);
    if (a !== undefined && b !== undefined) out.push({ name: bone.name, a, b });
  }
  return out;
}

export interface Bounds {
  min: Vec3;
  max: Vec3;
  center: Vec3;
  /** 가장 긴 축의 길이 */
  size: number;
}

/** 시퀀스 전체의 바운딩 박스. 카메라를 맞추는 데 쓴다. */
export function computeBounds(sequences: SkeletonSequence[]): Bounds {
  const min: Vec3 = [Infinity, Infinity, Infinity];
  const max: Vec3 = [-Infinity, -Infinity, -Infinity];
  for (const seq of sequences) {
    for (const frame of seq.frames) {
      for (const [x, y, z] of frame) {
        if (x < min[0]) min[0] = x;
        if (y < min[1]) min[1] = y;
        if (z < min[2]) min[2] = z;
        if (x > max[0]) max[0] = x;
        if (y > max[1]) max[1] = y;
        if (z > max[2]) max[2] = z;
      }
    }
  }
  if (!Number.isFinite(min[0])) {
    return { min: [0, 0, 0], max: [0, 0, 0], center: [0, 0, 0], size: 1 };
  }
  const center: Vec3 = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
  const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.1);
  return { min, max, center, size };
}

/**
 * 소수 프레임 위치의 관절 좌표. 앞뒤 프레임(floor / ceil)을 선형 보간한다.
 * 지금 백엔드는 frameMap.coachFrame 을 정수로 보내서 보간 없이 그 프레임이 그대로 나온다.
 * 소수가 와도 그려지도록 남겨 둔다. 범위를 벗어나면 양 끝 프레임으로 자른다.
 */
export function sampleFrame(seq: SkeletonSequence, frame: number): Vec3[] {
  const last = seq.frames.length - 1;
  if (last < 0) return [];
  const f = Number.isFinite(frame) ? Math.min(Math.max(frame, 0), last) : 0;
  const lo = Math.floor(f);
  const t = f - lo;
  const a = seq.frames[lo];
  if (t === 0) return a;
  const b = seq.frames[lo + 1];
  return a.map((p, j) => {
    const q = b[j] ?? p;
    return [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, p[2] + (q[2] - p[2]) * t];
  });
}

/** 초 → 프레임 인덱스 */
export function secondsToFrame(sec: number, fps: number, frameCount: number): number {
  const f = Math.round(sec * fps);
  return Math.min(Math.max(f, 0), Math.max(frameCount - 1, 0));
}
