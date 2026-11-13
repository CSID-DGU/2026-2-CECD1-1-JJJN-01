"""End to end: coach video + user video -> coach retargeted to the user's body -> motion_scoring JSON.

  python -m pipeline.score_user --coach COACH --user USER [--name NAME] [--compare-original]

COACH / USER are each either
  * a video file  -> pipeline.run_video (WHAM, IK, --ground frame) into runs/video_wham/<stem>/,
                     reused if that run already exists (--rerun [user|coach] processes from scratch)
  * a run_video output directory (e.g. runs/video_wham/squat_diag_gframe)

Steps
  1  video -> person skeleton + motion          runs/video_wham/<stem>/source_person.bvh
  2  coach -> user body (PAN)                   runs/coach_user/<name>/coach_on_user_pan.bvh, user.bvh
  3  BVH -> joint NPZ (official FK, SMPL-22)    runs/scoring/<name>/{coach_retargeted,user}.npz
  4  motion_scoring (real-video preset: auto tau, pose_trim 0.3, mirror check)
                                                runs/scoring/<name>/score.json  <- final result
  --compare-original also scores the coach's own skeleton (coach.bvh) -> score_coach_original.json,
  to see what the retargeting changes in the score (scoring uses bone directions only).
"""
import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import REPO  # noqa: E402

SCORING_DIR = Path(__file__).resolve().parent / 'motion_scoring'
VIDEO_EXT = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v'}


def run(cmd):
    print('$', ' '.join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def ensure_run(arg, ground, rerun, device):
    p = (Path.cwd() / arg).resolve()
    if p.is_dir():
        if not (p / 'source_person.bvh').exists():
            raise SystemExit(f'{p} is not a run_video output (no source_person.bvh)')
        return p
    if p.suffix.lower() not in VIDEO_EXT or not p.exists():
        raise SystemExit(f'{arg}: neither a run directory nor a video file')
    out = REPO / 'runs/video_wham' / p.stem
    if (out / 'report.json').exists() and \
            json.loads((out / 'report.json').read_text()).get('ground_mode', 'clip') != ground:
        out = out.with_name(f'{p.stem}_g{ground}')     # keep a run made with the other floor mode
    if (out / 'source_person.bvh').exists() and not rerun:
        print(f'reusing {out} (pass --rerun to process from scratch)')
        return out
    if rerun:                    # from scratch: WHAM reuses wham_work/ (2D detection, features) if present
        for cached in (out / 'wham_work', out / 'video_wham.mp4', out / 'wham.npz'):
            if cached.is_dir():
                shutil.rmtree(cached)
            elif cached.exists():
                cached.unlink()
    run([sys.executable, '-m', 'pipeline.run_video', p, '--estimator', 'wham', '--ground', ground,
         '--name', out.name, '--device', device])
    return out


def score_result(coach_npz, user_npz, mirror=True):
    """motion_scoring Result object (to_dict() is score.json; .debug['path'] the frame alignment)."""
    if str(SCORING_DIR) not in sys.path:
        sys.path.insert(0, str(SCORING_DIR))
    from motion_scoring import Config, load_npz, score
    cfg = replace(Config(), auto_tau=True, pose_trim=0.3)            # real-video preset (as the CLI)
    return score(load_npz(coach_npz, up='y'), load_npz(user_npz, up='y'), cfg, try_mirror=mirror)


def score_pair(coach_npz, user_npz, out_json, mirror=True):
    res = score_result(coach_npz, user_npz, mirror)
    d = json.loads(json.dumps(res.to_dict(), ensure_ascii=False))
    Path(out_json).write_text(json.dumps(d, indent=2, ensure_ascii=False))
    return d


def main(argv=None):
    from pipeline.bvh_to_npz import convert
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--coach', required=True, help='coach video or run_video output directory')
    ap.add_argument('--user', required=True, help='user video or run_video output directory')
    ap.add_argument('--name', help='output name (default <coach>_to_<user>)')
    ap.add_argument('--ground', default='frame', choices=['frame', 'clip'],
                    help='floor handling for new runs (frame: feet always on the floor, e.g. squats; clip: jumps)')
    ap.add_argument('--rerun', nargs='?', const='both', choices=['both', 'coach', 'user'],
                    help='process the video(s) from scratch as if never seen (clears WHAM caches): '
                         '--rerun = both, --rerun user / --rerun coach = only that one')
    ap.add_argument('--compare-original', action='store_true', help="also score the coach's own skeleton")
    ap.add_argument('--no-mirror', action='store_true', help='do not try the mirrored user')
    ap.add_argument('--no-viz', action='store_true', help='skip comparison.html')
    ap.add_argument('--device', default='cuda:0')
    opt = ap.parse_args(argv)

    coach = ensure_run(opt.coach, opt.ground, opt.rerun in ('both', 'coach'), opt.device)
    user = ensure_run(opt.user, opt.ground, opt.rerun in ('both', 'user'), opt.device)
    name = opt.name or f'{coach.name}_to_{user.name}'
    run([sys.executable, '-m', 'pipeline.coach_to_user', '--coach', coach, '--user', user, '--name', name,
         '--device', opt.device] + (['--no-viz'] if opt.no_viz else []))
    retarget = REPO / 'runs/coach_user' / name

    out = REPO / 'runs/scoring' / name
    out.mkdir(parents=True, exist_ok=True)
    convert(retarget / 'coach_on_user_pan.bvh', out / 'coach_retargeted.npz')
    convert(retarget / 'user.bvh', out / 'user.npz')
    result = score_pair(out / 'coach_retargeted.npz', out / 'user.npz', out / 'score.json', not opt.no_mirror)
    files = {'score.json': result}
    if opt.compare_original:
        convert(retarget / 'coach.bvh', out / 'coach_original.npz')
        files['score_coach_original.json'] = score_pair(out / 'coach_original.npz', out / 'user.npz',
                                                        out / 'score_coach_original.json', not opt.no_mirror)
    (out / 'inputs.json').write_text(json.dumps(dict(
        coach_run=str(coach), user_run=str(user), retarget_dir=str(retarget),
        coach_source='coach_on_user_pan.bvh', user_source='user.bvh', up='y',
        preset=dict(auto_tau=True, pose_trim=0.3, try_mirror=not opt.no_mirror)), indent=1))

    for fname, r in files.items():
        p = r['params']
        print(f'\n{fname}: {len(r["segments"])} segments, tempo x{r["global_tempo"]}, coverage {r["coverage"]}, '
              f'mirrored {r["mirrored"]}, tau {p["tau_deg"]:.2f} deg, kappa {p["kappa"]:.2f}')
        for s in r['segments']:
            t = s['timing']
            timing = f'{t["label"]} x{t["ratio"]}' if t and t.get('label') else '-'
            pose = ', '.join(f'{w["bone"]} {w["diff_deg"]}°' for w in s['pose']['worst']) if s['pose'] else '-'
            print(f'  {s["structure"]:<13} user_t {str(s["student_t"]):<16} {timing:<16} {pose}')
    print(f'\nWrote {out}')


if __name__ == '__main__':
    main()
