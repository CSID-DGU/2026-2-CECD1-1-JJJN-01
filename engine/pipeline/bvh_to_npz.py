"""BVH (BigVegas topology, pipeline output) -> joint-position NPZ for motion_scoring.

Positions come from the official FK of the BVH file (outer_utils.Animation.positions_global),
which applies every joint's rotation, including the `*_virtual` joints that carry PAN's
separate rotations of the three edges leaving the pelvis. (Skeleton.raw_to_joint collapses
those three into one and is not used here.)

Output NPZ (motion_scoring.load_npz contract, read with up="y"):
  joints       (T, 22, 3) float32  SMPL-22 order, cm, Y up (Mixamo frame)
  joint_names  (22,)               SMPL names
  fps          ()                  from the BVH frame time (pipeline: 30)

  python -m pipeline.bvh_to_npz IN.bvh OUT.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import REPO  # noqa: E402

# SMPL-22 name -> BigVegas joint (same joint, left/right as seen by the person)
SMPL_FROM_BIGVEGAS = {
    'pelvis': 'Pelvis', 'left_hip': 'LeftUpLeg', 'right_hip': 'RightUpLeg', 'spine1': 'Spine',
    'left_knee': 'LeftLeg', 'right_knee': 'RightLeg', 'spine2': 'Spine1',
    'left_ankle': 'LeftFoot', 'right_ankle': 'RightFoot', 'spine3': 'Spine2',
    'left_foot': 'LeftToeBase', 'right_foot': 'RightToeBase', 'neck': 'Neck',
    'left_collar': 'LeftShoulder', 'right_collar': 'RightShoulder', 'head': 'Head',
    'left_shoulder': 'LeftArm', 'right_shoulder': 'RightArm', 'left_elbow': 'LeftForeArm',
    'right_elbow': 'RightForeArm', 'left_wrist': 'LeftHand', 'right_wrist': 'RightHand',
}


def bvh_joints(path):
    """-> joints (T, 22, 3) in SMPL-22 order, names, fps."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import outer_utils.Animation as Animation
    import outer_utils.BVH as BVH
    anim, names, frametime = BVH.load(str(path))
    pos = Animation.positions_global(anim)
    names = list(names)
    missing = [b for b in SMPL_FROM_BIGVEGAS.values() if b not in names]
    if missing:
        raise ValueError(f'{path}: joints missing for SMPL mapping: {missing}')
    idx = [names.index(b) for b in SMPL_FROM_BIGVEGAS.values()]
    return pos[:, idx].astype(np.float32), list(SMPL_FROM_BIGVEGAS), 1. / frametime


def convert(bvh, out):
    joints, names, fps = bvh_joints(bvh)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, joints=joints, joint_names=np.array(names), fps=np.float64(fps))
    return joints.shape, fps


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('bvh')
    ap.add_argument('out')
    opt = ap.parse_args(argv)
    shape, fps = convert(opt.bvh, opt.out)
    print(f'{opt.bvh} -> {opt.out}: joints {shape}, {fps:.2f} fps')


if __name__ == '__main__':
    main()
