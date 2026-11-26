"""SMPL body model and the rotation helpers the overlay needs (numpy in, numpy out).

The model is WHAM's neutral SMPL (<WHAM_DIR>/dataset/body_models/smpl), run through `smplx` (WHAM venv).
smplx puts a mesh at  v = R_go · (v_local − J0) + J0 + transl  (J0 = the pelvis of the rest pose, which
depends on betas): a rigid move of the whole body therefore changes transl by more than the move itself,
see `move`.
"""
import numpy as np
from scipy.spatial.transform import Rotation

FEET = [7, 8, 10, 11]          # SMPL ankles + feet: their centre is the "feet" anchor
PELVIS = 0
# SMPL-22 names in SMPL joint order (the response's jointNames use these, pipeline/bvh_to_npz.py)
SMPL22 = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle',
          'right_ankle', 'spine3', 'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar', 'head',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']


# ---------- rotations ----------
def aa_to_mat(aa):
    """(..., 3) axis-angle -> (..., 3, 3)."""
    aa = np.asarray(aa, np.float64)
    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(aa.shape[:-1] + (3, 3))


def mat_to_aa(R):
    """(..., 3, 3) -> (..., 3) axis-angle."""
    R = np.asarray(R, np.float64)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def rot_y(theta):
    """(...,) angle -> (..., 3, 3) rotation about +Y (world up)."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.zeros(np.shape(theta) + (3, 3))
    R[..., 0, 0], R[..., 0, 2], R[..., 1, 1], R[..., 2, 0], R[..., 2, 2] = c, s, 1., -s, c
    return R


def split_yaw(R):
    """Root rotation (..., 3, 3), world Y up -> (yaw angle, tilt) with R = rot_y(yaw) · tilt.

    The heading is taken from the body's left-right axis (SMPL local +x, the hip axis), which stays close
    to horizontal in a squat, a deadlift (trunk bent forward: its forward axis turns vertical) and lying on a
    bench, so the yaw is well defined in all three."""
    x = R[..., :, 0]
    yaw = np.arctan2(-x[..., 2], x[..., 0])
    return yaw, np.swapaxes(rot_y(yaw), -1, -2) @ R


# ---------- body model ----------
class Body:
    def __init__(self, model_dir=None, device='cpu'):
        import smplx
        import torch
        from pathlib import Path
        if model_dir is None:
            import os
            project = Path(__file__).resolve().parents[3]
            model_dir = Path(os.environ.get('WHAM_DIR', project / 'third_party/WHAM')) / 'dataset/body_models/smpl'
        self.torch, self.device = torch, device
        self.model = smplx.SMPL(str(model_dir), gender='neutral').to(device)
        self.faces = np.asarray(self.model.faces, np.int64)

    def __call__(self, global_orient, body_pose, betas, transl=None):
        """global_orient (N,3) axis-angle, body_pose (N,69), betas (10,) or (N,10), transl (N,3) -> verts (N,6890,3),
        joints (N,24,3)."""
        t = lambda a: self.torch.as_tensor(np.array(a, np.float32), device=self.device)
        go = np.atleast_2d(global_orient)
        n = len(go)
        b = np.broadcast_to(np.asarray(betas, np.float32).reshape(-1, 10), (n, 10))
        tr = np.zeros((n, 3)) if transl is None else np.atleast_2d(transl)
        with self.torch.no_grad():
            out = self.model(global_orient=t(go), body_pose=t(np.atleast_2d(body_pose)), betas=t(b), transl=t(tr))
        return out.vertices.cpu().numpy(), out.joints[:, :24].cpu().numpy()

    def rest_pelvis(self, betas):
        """J0: the pelvis of the zero pose for these betas (the centre smplx rotates global_orient about)."""
        _, j = self(np.zeros(3), np.zeros(69), betas)
        return j[0, PELVIS]


def move(global_orient, transl, R, t, j0):
    """SMPL params after moving the whole body rigidly by x -> R x + t. -> (global_orient, transl)."""
    go = mat_to_aa(R @ aa_to_mat(global_orient))
    tr = (R @ (j0 + np.asarray(transl))[..., None])[..., 0] + t - j0
    return go, tr
