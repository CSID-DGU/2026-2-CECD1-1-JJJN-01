"""Same-topology (intra-structural) retargeting check: BigVegas -> group-A test characters.

This is the coach -> user step: both bodies share BigVegas' topology and differ in bone
lengths. Aj, Kaya and SportyGranny have paired ground truth for the test motions, so the
step can be scored before running it on a real user.

Conditions (input = BigVegas ground-truth rotations of each test clip):
  copy       rotations copied, root position scaled by the leg (hip height) ratio
  pan_own    PAN decoder on the target offsets, target's own mean/var (official-like)
  pan_user   PAN decoder on the target offsets, BigVegas mean/var with the root channels
             scaled by the leg ratio -- what is possible for a real user (no stats)

Metrics per clip and target:
  pan_err    mean(dpos^2) / height^2 x1e3 against the ground truth positions
  rot_deg    mean joint rotation error against the ground truth (rotating joints)
  foot_y     lowest toe height over the clip minus the ground truth's (cm)

  python -m pipeline.intra_check --name v1 [--max-clips 20]
Outputs runs/intra/<name>/{results.json,summary.md}.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import MEAN_VAR_DIR, REPO, enter_repo  # noqa: E402

TARGETS = ('Aj', 'Kaya', 'SportyGranny')
CONDITIONS = ('copy', 'pan_own', 'pan_user')


def positions(skel, raw):
    from pipeline.core import quat as Q
    q, root = skel.raw_to_joint(torch.as_tensor(raw, dtype=torch.float64).cpu())
    return skel.fk(Q.normalize(q), root).numpy(), Q.normalize(q)


def rot_deg(skel, qa, qb):
    from pipeline.core import quat as Q
    j = skel.rotating_joints
    return float(np.degrees(Q.angle_between(qa[:, j], qb[:, j]).mean()))


def main(argv=None):
    from pipeline.core.pan_runtime import PanRuntime
    from pipeline.core.skeleton import Skeleton
    from pipeline.harness import load_clips

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--name', default='intra')
    ap.add_argument('--targets', nargs='+', default=list(TARGETS))
    ap.add_argument('--max-clips', type=int, default=None)
    ap.add_argument('--device', default='cuda:0')
    opt = ap.parse_args(argv)
    enter_repo()
    started = time.time()

    runtime = PanRuntime(source='BigVegas', device=opt.device)
    clips, skipped = load_clips('BigVegas', opt.targets, opt.max_clips)
    skels = {t: Skeleton.from_character(t) for t in opt.targets}
    stats = {t: tuple(torch.tensor(np.load(MEAN_VAR_DIR / f'{t}_{k}.npy')).reshape(-1) for k in ('mean', 'var'))
             for t in opt.targets}
    toes = {t: [s.index(n) for n in ('LeftToeBase', 'RightToeBase')] for t, s in skels.items()}

    results = []
    for c in clips:
        src = c['src'].float()
        row = dict(name=c['name'], frames=int(src.shape[1]), targets={})
        for t in opt.targets:
            skel = skels[t]
            h = skel.height()
            ratio = runtime.root_ratio(skel.offsets)
            outs = {'copy': torch.cat([src[:-3], src[-3:] * ratio])}
            outs['pan_own'] = runtime.retarget_intra(src, skel.offsets, *stats[t]).cpu()
            outs['pan_user'] = runtime.retarget_intra(src, skel.offsets).cpu()
            gt_pos, gt_q = positions(skel, c['gt'][t])
            gt_foot = gt_pos[:, toes[t], 1].min()
            row['targets'][t] = {}
            for k, raw in outs.items():
                pos, q = positions(skel, raw)
                row['targets'][t][k] = dict(pan_err=float(((pos - gt_pos) ** 2).mean() / h ** 2 * 1e3),
                                            rot_deg=rot_deg(skel, q, gt_q),
                                            foot_y=float(pos[:, toes[t], 1].min() - gt_foot))
        results.append(row)

    out = REPO / 'runs/intra' / opt.name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(dict(clips=results, skipped=skipped), indent=1))

    def agg(t, k, m, f=np.mean):
        return float(f([r['targets'][t][k][m] for r in results]))

    lines = [f'# Intra-structural check `{opt.name}`\n',
             f'{len(results)} BigVegas test clips -> {", ".join(opt.targets)}, {time.time() - started:.0f}s. '
             'Input: ground-truth BigVegas rotations.\n',
             '| target | leg ratio | condition | pan_err×1e3 mean | median | rot_deg | foot_y cm (mean) |',
             '|---|---|---|---|---|---|---|']
    for t in opt.targets:
        ratio = runtime.root_ratio(skels[t].offsets)
        for k in CONDITIONS:
            lines.append(f'| {t} | {ratio:.3f} | {k} | {agg(t, k, "pan_err"):.3f} | '
                         f'{agg(t, k, "pan_err", np.median):.3f} | {agg(t, k, "rot_deg"):.2f} | '
                         f'{agg(t, k, "foot_y"):+.2f} |')
    lines.append('')
    for k in CONDITIONS:
        lines.append(f'* all targets, {k}: pan_err {np.mean([agg(t, k, "pan_err") for t in opt.targets]):.3f}, '
                     f'rot_deg {np.mean([agg(t, k, "rot_deg") for t in opt.targets]):.2f}')
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
