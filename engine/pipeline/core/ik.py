"""Position -> local rotation IK with optional data-driven constraints.

loss = fit                                   positions (normalised by skeleton height)
     + w_limit  * soft joint limits           percentile box of training rotation vectors
     + w_prior  * PAN-space prior             ||(q - mu) / sigma||^2 with PAN's own stats
     + w_smooth * temporal smoothness         1 - <q_t, q_t-1>^2 inside each clip
     + w_root_acc * root acceleration          only with free_root (root position optimised)

Structural facts taken from the data (always on): one rotation per joint, so edges that
share a parent share a rotation. `fix_constant` additionally pins joints that never move
in the training data (Mixamo `Hips`). The root position is the observed root unless
`free_root`, in which case it is optimised (initialised at the observation) so noisy root
observations are averaged with the other joints and the trajectory acceleration prior.
"""
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch

from . import quat as Q


@dataclass(frozen=True)
class IKConfig:
    init: str = 'data'            # 'identity' | 'data' (training mean pose)
    root_init: str = 'horn'       # 'none' | 'horn' (closed-form root orientation)
    fix_constant: bool = True
    w_limit: float = 1e-3
    w_prior: float = 1e-5
    w_smooth: float = 1e-4
    free_root: bool = False
    w_root_acc: float = 0.
    steps: int = 1500
    lr: float = 0.03


PRESETS = {
    # Unconstrained baseline (identity init, no constraints).
    'free': IKConfig(init='identity', root_init='none', fix_constant=False, w_limit=0., w_prior=0.),
    # Same objective, only a data-driven starting point.
    'init': IKConfig(fix_constant=False, w_limit=0., w_prior=0.),
    # + fixed joints and soft joint limits.
    'limits': IKConfig(w_prior=0.),
    # + prior in PAN's normalised input space.
    'full': IKConfig(),
    # + root position optimised with an acceleration prior (for noisy observations).
    'full_root': IKConfig(free_root=True, w_root_acc=0.1),
}


def _rest_positions(skel, base):
    with torch.no_grad():
        return skel.fk(base[:1], torch.zeros_like(base[:1, 0, :3]))[0]


def horn_root(skel, obs, mask, base):
    """Root orientation and position per frame from rigid-ish torso points.

    Uses both hips plus the first fully observed spine-chain joint, else both shoulders
    (`LeftArm`/`RightArm`, approximate because the spine bends). Point sets are centred,
    so an unobserved root (e.g. MediaPipe, where only the hip centre is known) is fine.
    """
    rest = _rest_positions(skel, base)
    names = ['LeftUpLeg', 'RightUpLeg']
    spine = [n for n in ('Spine', 'Spine1', 'Spine2', 'Neck') if bool(mask[:, skel.index(n)].all())]
    names += spine[:1] or ['LeftArm', 'RightArm']
    idx = [skel.index(n) for n in names]
    if not bool(mask[:, idx].all()):
        raise ValueError(f'Root init needs {names} observed in every frame')
    src = rest[idx] - rest[idx].mean(0)
    dst = obs[:, idx] - obs[:, idx].mean(1, keepdim=True)
    q = Q.horn(src.expand(obs.shape[0], -1, -1), dst)
    root = obs[:, idx].mean(1) - Q.rotate(q, (rest[idx].mean(0) - rest[0]).expand_as(obs[:, 0]))
    return q, root


def solve(obs, mask, skel, priors, pan_stats, segments=None, cfg=IKConfig(),
          device='cuda:0', log_every=250, log=print):
    """obs: (T, J, 3) cm; mask: (T, J) or (J,) bool, or float confidence weights (0 =
    unobserved); segments: (T,) clip ids. An unobserved root requires cfg.free_root.

    Returns dict with q (T, J, 4), root (T, 3), fitted positions and diagnostics.
    """
    started = time.monotonic()
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    T, J, _ = obs.shape
    weight = torch.as_tensor(mask, device=device).float()
    if weight.ndim == 1:
        weight = weight.expand(T, J)
    mask = weight > 0
    if not bool(mask[:, 0].all()) and not cfg.free_root:
        raise ValueError('The root joint must be observed in every frame (or use free_root)')
    seg = torch.zeros(T, dtype=torch.long, device=device) if segments is None else \
        torch.as_tensor(segments, dtype=torch.long, device=device)
    same = seg[1:] == seg[:-1]

    pj = priors['joints']
    rot_joints = skel.rotating_joints
    fixed = [j for j in rot_joints if cfg.fix_constant and pj[skel.names[j]]['fixed']]
    free = [j for j in rot_joints if j not in fixed]
    limited = [j for j in free if j != 0]                    # root orientation is observed

    base = torch.zeros((T, J, 4), device=device)
    base[..., 0] = 1
    for j in fixed:
        base[:, j] = torch.tensor(pj[skel.names[j]]['mean'], device=device)
    init = base[:, free].clone()
    if cfg.init == 'data':
        for k, j in enumerate(free):
            init[:, k] = torch.tensor(pj[skel.names[j]]['mean'], device=device)
    elif cfg.init != 'identity':
        raise ValueError(cfg.init)
    root_init = obs[:, 0].clone()
    if cfg.root_init == 'horn':
        init[:, free.index(0)], horn_pos = horn_root(skel, obs, mask, base)
        root_init = torch.where(mask[:, :1], root_init, horn_pos)
    elif cfg.root_init != 'none':
        raise ValueError(cfg.root_init)

    lo = torch.tensor(np.radians([pj[skel.names[j]]['limit_lo_deg'] for j in limited]),
                      dtype=torch.float32, device=device)
    hi = torch.tensor(np.radians([pj[skel.names[j]]['limit_hi_deg'] for j in limited]),
                      dtype=torch.float32, device=device)
    mu, sigma = (s.to(device) for s in pan_stats)
    prior_edges = [e for e, (a, _) in enumerate(skel.edges) if a in limited]
    ch = torch.tensor([4 * e + k for e in prior_edges for k in range(4)], device=device)
    mu_e, sig_e = mu[ch].view(-1, 4), sigma[ch].view(-1, 4)
    prior_parent = [skel.edges[e][0] for e in prior_edges]

    height = skel.height()
    offsets = torch.tensor(skel.offsets, dtype=torch.float32, device=device)
    seg_ok2 = same[1:] & same[:-1]                            # t-1, t, t+1 in one clip
    free_idx = torch.tensor(free, device=device)
    limited_idx = torch.tensor([free.index(j) for j in limited], device=device)
    w = weight

    param = torch.nn.Parameter(init.clone())
    root_param = torch.nn.Parameter(root_init, requires_grad=cfg.free_root)
    opt = torch.optim.Adam([param] + ([root_param] if cfg.free_root else []), lr=cfg.lr)

    def terms(p, root):
        qn = Q.normalize(p)
        q = base.clone()
        q[:, free_idx] = qn
        pred = skel.fk(q, root, offsets)
        fit = ((pred - obs) / height).square().sum(-1).mul(w).sum() / w.sum()
        out = dict(fit=fit)
        if cfg.w_limit and limited:
            r = Q.log(qn[:, limited_idx])
            out['limit'] = (torch.relu(lo - r) + torch.relu(r - hi)).square().mean()
        if cfg.w_prior and prior_edges:
            qe = q[:, prior_parent]
            zp = ((qe - mu_e) / sig_e).square().sum(-1)
            zn = ((-qe - mu_e) / sig_e).square().sum(-1)
            out['prior'] = torch.minimum(zp, zn).mean() / 4
        if cfg.w_smooth and T > 1 and bool(same.any()):
            dot = (qn[1:] * qn[:-1]).sum(-1).square()
            out['smooth'] = (1 - dot)[same].clamp_min(0).mean()
        if cfg.free_root and cfg.w_root_acc and bool(seg_ok2.any()):
            acc = (root[2:] - 2 * root[1:-1] + root[:-2]) / height
            out['root_acc'] = acc.square().sum(-1)[seg_ok2].mean()
        total = fit + sum(getattr(cfg, f'w_{k}') * v for k, v in out.items() if k != 'fit')
        return total, out, q, pred

    best, best_p, best_r = float('inf'), param.detach().clone(), root_param.detach().clone()
    history = []
    for step in range(cfg.steps):
        opt.zero_grad()
        total, parts, _, _ = terms(param, root_param)
        if not torch.isfinite(total):
            raise RuntimeError(f'Non-finite IK loss at step {step}')
        if total.item() < best:
            best, best_p, best_r = total.item(), param.detach().clone(), root_param.detach().clone()
        total.backward()
        torch.nn.utils.clip_grad_norm_([param], 10.)
        opt.step()
        with torch.no_grad():
            param.copy_(Q.normalize(param))
        if log and (step == 0 or (step + 1) % log_every == 0 or step + 1 == cfg.steps):
            rec = {k: float(v.detach()) for k, v in parts.items()}
            rec['step'] = step + 1
            history.append(rec)
            log('  IK %4d/%d ' % (step + 1, cfg.steps) +
                ' '.join(f'{k}={v:.3e}' for k, v in rec.items() if k != 'step'))
    with torch.no_grad():
        _, parts, q, pred = terms(best_p, best_r)
    return dict(q=q.cpu(), root=best_r.cpu(), positions=pred.cpu(),
                final={k: float(v) for k, v in parts.items()}, history=history,
                fixed=[skel.names[j] for j in fixed], config=asdict(cfg),
                seconds=time.monotonic() - started)
