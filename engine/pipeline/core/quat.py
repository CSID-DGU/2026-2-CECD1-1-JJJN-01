"""Quaternion helpers (torch, w-x-y-z, Hamilton product, same convention as PAN)."""
import torch


def normalize(q, eps=1e-8):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(eps)


def conj(q):
    return q * q.new_tensor([1., -1., -1., -1.])


def mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw], -1)


def rotate(q, v):
    """Rotate vectors v (...,3) by unit quaternions q (...,4)."""
    w, u = q[..., :1], q[..., 1:]
    t = 2 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def positive_w(q):
    return torch.where(q[..., :1] < 0, -q, q)


def log(q):
    """Rotation vector (axis * angle, radians) of unit quaternions, angle in [0, pi]."""
    q = positive_w(q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(s, q[..., :1])
    scale = torch.where(s > 1e-7, angle / s.clamp_min(1e-12), torch.full_like(s, 2.))
    return v * scale


def angle_between(a, b):
    """Geodesic angle (radians) between rotations; sign-agnostic (q == -q)."""
    d = (normalize(a) * normalize(b)).sum(-1).abs().clamp(max=1.)
    return 2 * torch.acos(d)


def twist_swing_angles(d, axis):
    """Split rotation d into twist about unit `axis` (local frame) and the remaining swing.

    d = swing * twist. Returns (twist_angle, swing_angle) in radians.
    """
    d = positive_w(normalize(d))
    proj = (d[..., 1:] * axis).sum(-1, keepdim=True)
    twist = normalize(torch.cat([d[..., :1], proj * axis], -1))
    swing = mul(d, conj(twist))
    twist_angle = 2 * torch.atan2(proj.abs()[..., 0], twist[..., 0].abs().clamp_min(1e-12))
    twist_angle = torch.where(twist[..., 0].abs() < 1e-12, torch.full_like(twist_angle, torch.pi), twist_angle)
    swing_angle = 2 * torch.acos(swing[..., 0].abs().clamp(max=1.))
    return twist_angle, swing_angle


def horn(src, dst, weight=None):
    """Least-squares rotation q with dst ~= rotate(q, src) (Horn 1987).

    src, dst: (..., N, 3); weight: (..., N) or None. Returns (..., 4).
    """
    if weight is not None:
        src = src * weight[..., None]
    S = torch.einsum('...ni,...nj->...ij', src, dst)
    xx, xy, xz = S[..., 0, 0], S[..., 0, 1], S[..., 0, 2]
    yx, yy, yz = S[..., 1, 0], S[..., 1, 1], S[..., 1, 2]
    zx, zy, zz = S[..., 2, 0], S[..., 2, 1], S[..., 2, 2]
    N = torch.stack([
        torch.stack([xx + yy + zz, yz - zy, zx - xz, xy - yx], -1),
        torch.stack([yz - zy, xx - yy - zz, xy + yx, zx + xz], -1),
        torch.stack([zx - xz, xy + yx, -xx + yy - zz, yz + zy], -1),
        torch.stack([xy - yx, zx + xz, yz + zy, -xx - yy + zz], -1)], -2)
    _, vec = torch.linalg.eigh(N)
    return positive_w(vec[..., :, -1])


def continuous(q, dim=0):
    """Flip signs along `dim` so consecutive quaternions stay in the same hemisphere."""
    q = q.clone().movedim(dim, 0)
    for t in range(1, q.shape[0]):
        flip = (q[t] * q[t - 1]).sum(-1, keepdim=True) < 0
        q[t] = torch.where(flip, -q[t], q[t])
    return q.movedim(0, dim)
