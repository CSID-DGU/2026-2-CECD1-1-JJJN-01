import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import type { Vec3 } from '@/types/analysis';
import type { ResolvedBone } from '@/utils/skeleton';

interface SkeletonMeshProps {
  /** 이 프레임의 관절 위치 */
  joints: Vec3[];
  bones: ResolvedBone[];
  color: string;
  /** 강조할 뼈 이름 (차이가 큰 뼈) */
  highlight?: Set<string>;
  highlightColor?: string;
  jointRadius?: number;
  boneRadius?: number;
  opacity?: number;
}

const UP = new THREE.Vector3(0, 1, 0);

/**
 * 스켈레톤 한 사람분.
 * 관절은 구, 뼈는 길이 1 짜리 원기둥을 두 관절 사이로 늘려서 그린다.
 * 지오메트리·재질은 한 번만 만들고 프레임마다 위치만 바꾼다.
 */
export function SkeletonMesh({
  joints,
  bones,
  color,
  highlight,
  highlightColor = '#d64545',
  jointRadius = 0.028,
  boneRadius = 0.016,
  opacity = 1,
}: SkeletonMeshProps) {
  const sphere = useMemo(() => new THREE.SphereGeometry(jointRadius, 14, 14), [jointRadius]);
  const cylinder = useMemo(() => new THREE.CylinderGeometry(boneRadius, boneRadius, 1, 10), [boneRadius]);
  const thickCylinder = useMemo(
    () => new THREE.CylinderGeometry(boneRadius * 1.7, boneRadius * 1.7, 1, 10),
    [boneRadius],
  );
  const material = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color,
        transparent: opacity < 1,
        opacity,
        // 반투명(겹쳐 보기의 코치)일 때 깊이를 안 쓰면 뒤에 있는 사용자 뼈가 비쳐 보인다
        depthWrite: opacity >= 1,
        roughness: 0.55,
      }),
    [color, opacity],
  );
  const highlightMaterial = useMemo(
    () => new THREE.MeshStandardMaterial({ color: highlightColor, roughness: 0.45 }),
    [highlightColor],
  );

  useEffect(
    () => () => {
      sphere.dispose();
      cylinder.dispose();
      thickCylinder.dispose();
      material.dispose();
      highlightMaterial.dispose();
    },
    [sphere, cylinder, thickCylinder, material, highlightMaterial],
  );

  return (
    <group>
      {joints.map((p, i) => (
        <mesh key={i} geometry={sphere} material={material} position={p} />
      ))}
      {bones.map((bone) => {
        const a = joints[bone.a];
        const b = joints[bone.b];
        if (!a || !b) return null;
        const isHighlighted = highlight?.has(bone.name) ?? false;
        return (
          <Bone
            key={bone.name}
            a={a}
            b={b}
            geometry={isHighlighted ? thickCylinder : cylinder}
            material={isHighlighted ? highlightMaterial : material}
          />
        );
      })}
    </group>
  );
}

interface BoneProps {
  a: Vec3;
  b: Vec3;
  geometry: THREE.BufferGeometry;
  material: THREE.Material;
}

function Bone({ a, b, geometry, material }: BoneProps) {
  const { position, quaternion, length } = useMemo(() => {
    const va = new THREE.Vector3(a[0], a[1], a[2]);
    const vb = new THREE.Vector3(b[0], b[1], b[2]);
    const dir = vb.clone().sub(va);
    const length = Math.max(dir.length(), 1e-4);
    const position = va.clone().add(vb).multiplyScalar(0.5);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(UP, dir.normalize());
    return { position, quaternion, length };
  }, [a, b]);

  return (
    <mesh geometry={geometry} material={material} position={position} quaternion={quaternion} scale={[1, length, 1]} />
  );
}
