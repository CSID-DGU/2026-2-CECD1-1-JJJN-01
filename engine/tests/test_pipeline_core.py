"""Consistency checks of pipeline/core against the official PAN code.

Run: python tests/test_pipeline_core.py   (from anywhere; needs CUDA for the PAN check)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import MIXAMO_DIR, enter_repo  # noqa: E402

enter_repo()
import numpy as np  # noqa: E402
import torch  # noqa: E402

import outer_utils.Animation as Animation  # noqa: E402
import outer_utils.BVH as BVH  # noqa: E402
from data_preprocess.Mixamo.bvh_parser import BVH_file  # noqa: E402
from pipeline.core import quat as Q  # noqa: E402
from pipeline.core.skeleton import Skeleton  # noqa: E402

CLIP = MIXAMO_DIR / 'BigVegas/Standing To Crouch.bvh'


def test_quat_ops():
    g = torch.Generator().manual_seed(0)
    a = Q.normalize(torch.randn(100, 4, generator=g, dtype=torch.float64))
    b = Q.normalize(torch.randn(100, 4, generator=g, dtype=torch.float64))
    v = torch.randn(100, 3, generator=g, dtype=torch.float64)
    assert torch.allclose(Q.rotate(Q.mul(a, b), v), Q.rotate(a, Q.rotate(b, v)), atol=1e-10)
    assert torch.allclose(Q.angle_between(a, -a), torch.zeros(100, dtype=torch.float64), atol=1e-6)
    # log is the inverse of axis-angle
    r = Q.log(a)
    ang = r.norm(dim=-1, keepdim=True)
    back = torch.cat([torch.cos(ang / 2), torch.sin(ang / 2) * r / ang], -1)
    assert torch.allclose(Q.angle_between(back, a), torch.zeros(100, dtype=torch.float64), atol=1e-6)
    # Horn recovers a known rotation
    src = torch.randn(100, 6, 3, generator=g, dtype=torch.float64)
    est = Q.horn(src, Q.rotate(a[:, None].expand(-1, 6, -1), src))
    assert Q.angle_between(est, a).max() < 1e-6
    # twist about an axis is recovered, swing is zero
    axis = torch.tensor([0., 0.6, 0.8], dtype=torch.float64)
    t = torch.cat([torch.cos(torch.tensor([0.35])), torch.sin(torch.tensor([0.35])) * axis], -1)
    tw, sw = Q.twist_swing_angles(t[None].double(), axis)
    assert abs(float(tw) - 0.7) < 1e-6 and float(sw) < 1e-6


def test_fk_and_raw_roundtrip():
    f = BVH_file(str(CLIP))
    skel = Skeleton.from_bvh(CLIP)
    raw = f.to_tensor(quater=True).double()
    q, root = skel.raw_to_joint(raw)
    assert torch.equal(skel.joint_to_raw(q, root), raw)
    pos = skel.fk(Q.normalize(q), root)
    # Reference: the official PAN FK on the same raw (the 83-joint source BVH is not a
    # valid reference because BVH_file simplifies the hierarchy).
    from models.Kinematics import ForwardKinematics

    class A:
        fk_world, pos_repr, rotation = True, '3d', 'quaternion'
    pan = ForwardKinematics(A, f.edges).forward_from_raw(
        raw[None].float(), torch.tensor(f.offset, dtype=torch.float)[None])[0].double()
    err = float((pan - pos).abs().max())
    assert err < 1e-3, err
    # Our raw -> BVH_writer -> BVH.load path gives the same positions (frame 0..T)
    from data_preprocess.Mixamo.bvh_writer import BVH_writer
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        BVH_writer(f.edges, f.names).write_raw(raw.float(), 'quaternion', f'{d}/x.bvh')
        anim, names, _ = BVH.load(f'{d}/x.bvh')
    ref = Animation.positions_global(anim)
    keep = [i for i, n in enumerate(names) if 'virtual' not in n]
    ref = torch.as_tensor(ref[:, keep])
    names = [names[i] for i in keep]
    order = [names.index(n) for n in skel.names[1:]]
    err = float((ref[:, order] - pos[:, 1:]).abs().max())
    assert err < 1e-2, err


def test_training_structure():
    """Edges sharing a parent carry identical rotations in the data (joint-space is exact)."""
    from pipeline.core.priors import TRAIN_NPY_DIR
    skel = Skeleton.from_character('BigVegas')
    clips = np.load(TRAIN_NPY_DIR / 'BigVegas.npy', allow_pickle=True)
    m = np.concatenate([np.asarray(c) for c in clips[:50]])
    e = m[:, :-3].reshape(len(m), -1, 3)
    for a in set(skel.edge_parent):
        group = [i for i, (p, _) in enumerate(skel.edges) if p == a]
        for i in group[1:]:
            assert np.abs(e[:, group[0]] - e[:, i]).max() < 1e-5


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
