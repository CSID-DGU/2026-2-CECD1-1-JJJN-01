"""The retargeted coach as a SMPL body over the user's video, in the result screen's playback order (the service's
overlay; PLAN.md 8). Run with the WHAM venv (smplx) from engine/:

  ../.venv-wham/bin/python -m pipeline.overlay.coach_on_video --user-run runs/video_wham/<user> \
      --coach-run runs/video_wham/<coach> --retarget runs/coach_user/<name> --exercise squat --out <dir>

The coach is method 2 (compare.py): the PAN result coach_on_user_pan.bvh turned into SMPL rotations, calibrated on
the coach's own IK BVH against the coach's WHAM (coach_pose.calibrate_from_wham), then placed on the user frame by
frame (the user's shape, heading, feet and camera: place.py) and mirrored when the scorer matched the mirrored user.
Frame k of the video is playback frame k of the API result (runs.api_playback = backend build_result), so the
frontend plays it on the same clock as the 3D viewer: video time k / fps.

The user run's wham.npz needs the camera fields of tools/pose_estimation/wham_extract.py; the coach run only its
IK BVH, wham.npz pose_world and report.json.

Writes <out>/:
  overlay.mp4     H.264, playback order, fps 1 / pb dt, long side <= --long-side
  playback.json   pb (student / coach BVH frame per playback frame, segments, dt), mirrored, frames, fps
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pipeline.overlay import coach_pose as C  # noqa: E402
from pipeline.overlay import place as P  # noqa: E402
from pipeline.overlay import render as R  # noqa: E402
from pipeline.overlay.runs import Run, api_playback, jsonable  # noqa: E402
from pipeline.overlay.smpl_np import Body  # noqa: E402

BATCH = 64


def coach_pose(coach, retarget, mirrored):
    """Method 2 coach pose (tilt, body_pose) per coach BVH frame, calibrated, mirrored if the scorer did."""
    calib, _ = C.calibrate_from_wham(coach.dir / 'source_pan_input.bvh', coach.npz, coach.report)
    pose = C.from_pan_bvh(Path(retarget) / 'coach_on_user_pan.bvh', None, calib=calib)
    return C.mirror(pose) if mirrored else pose


def coach_meshes(body, user, pose, pb, anchor):
    """Per playback frame: (user npz frame, coach vertices in the user's camera frame or None)."""
    placed, keys, cache = [], [], {}
    st = None
    for si, ci in zip(pb['student'], pb['coach']):
        f = user.frame(int(si))
        if st is None or st.f != f:
            st = P.StudentFrame(body, user.npz, f)
            st.f = f
        key = (f, int(ci))
        if key not in cache:
            cache[key] = None
            if st.ok:
                prm, _ = P.place_world(body, pose['tilt'][int(ci)], pose['body_pose'][int(ci)], st, anchor)
                cache[key] = len(placed)
                placed.append(P.to_camera(prm, st))
        keys.append((f, cache[key]))
    verts = []
    for a in range(0, len(placed), BATCH):                  # one SMPL call per batch (GPU when there is one)
        chunk = placed[a:a + BATCH]
        stack = lambda k: np.stack([p[k] for p in chunk])
        verts.append(body(stack('global_orient'), stack('body_pose'), stack('betas'), stack('transl'))[0])
    verts = np.concatenate(verts) if verts else np.zeros((0, 6890, 3))
    return [(f, None if i is None else verts[i]) for f, i in keys]


def write_video(path, user, frames, faces, fps, long_side, alpha):
    w, h = (float(x) for x in user.npz['img_size'])
    s = min(1., long_side / max(w, h))
    size = (int(round(w * s / 2) * 2), int(round(h * s / 2) * 2))      # even, for yuv420p
    cam = R.Camera(size[0], size[1], float(user.npz['focal']) * size[0] / w)
    video = R.FrameReader(user.video)
    wr = R.writer(path, fps)
    for f, verts in frames:
        bg = cv2.resize(video.get(f), size, interpolation=cv2.INTER_AREA)
        meshes = [] if verts is None else [(verts, R.COLORS['coach'])]
        wr.append_data(R.composite(bg, meshes, faces, cam, alpha)[..., ::-1])
    wr.close()
    return size


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--user-run', required=True, help='run_video output of the user video')
    ap.add_argument('--coach-run', required=True, help='run_video output of the coach video')
    ap.add_argument('--retarget', required=True, help='coach_to_user output (coach_on_user_pan.bvh, user.bvh)')
    ap.add_argument('--exercise', default='squat', help='for scorer_setup, as in the API')
    ap.add_argument('--out', required=True)
    ap.add_argument('--anchor', default='feet', choices=['feet', 'pelvis'])
    ap.add_argument('--alpha', type=float, default=0.6, help='mesh opacity')
    ap.add_argument('--long-side', type=int, default=1280, help='output size: longer side in pixels')
    ap.add_argument('--device', default=None, help='smplx device (default: cuda when available)')
    opt = ap.parse_args(argv)

    t0 = time.time()
    out = Path(opt.out)
    out.mkdir(parents=True, exist_ok=True)
    if opt.device is None:
        import torch
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    user, coach = Run(opt.user_run), Run(opt.coach_run, camera=False)
    pb, mirrored = api_playback(opt.retarget, opt.exercise)
    fps = 1. / pb['dt']
    pose = coach_pose(coach, opt.retarget, mirrored)
    body = Body(device=opt.device)
    frames = coach_meshes(body, user, pose, pb, opt.anchor)
    t1 = time.time()
    size = write_video(out / 'overlay.mp4', user, frames, body.faces, fps, opt.long_side, opt.alpha)
    (out / 'playback.json').write_text(json.dumps(jsonable(dict(pb, mirrored=mirrored, frames=len(frames),
                                                                fps=fps)), indent=1))
    print(f'overlay: {len(frames)} frames {size[0]}x{size[1]} @ {fps:.2f} fps, mirrored {mirrored}, '
          f'{opt.device}: SMPL {t1 - t0:.0f} s, video {time.time() - t1:.0f} s')
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
