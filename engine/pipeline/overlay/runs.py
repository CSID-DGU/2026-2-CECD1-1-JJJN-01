"""What the overlay reads from the pipeline's outputs: run_video runs (wham.npz, report.json, the video) and the
playback order of a coach_to_user retarget, made the way the backend makes it.

Used by coach_on_video.py (the service) and compare.py (the method comparison).
"""
import json
import sys
from pathlib import Path

import numpy as np

from pipeline.common.paths import PROJECT

from . import place as P

CAMERA_KEYS = ('pose_cam', 'trans_cam', 'focal', 'img_size')


def api_playback(retarget_dir, exercise):
    """res.playback() and res.to_dict()['mirrored'] exactly as backend build_result makes them (same scorer
    setup, same BVHs), so that overlay frame k is the result screen's playback frame k."""
    if str(PROJECT / 'backend') not in sys.path:
        sys.path.insert(0, str(PROJECT / 'backend'))
    from app.result.result_builder import scorer_setup
    from pipeline.bvh_to_npz import bvh_joints
    cfg, weights = scorer_setup(exercise)
    from motion_scoring import score, track_from_joints
    coach, names, fps_c = bvh_joints(Path(retarget_dir) / 'coach_on_user_pan.bvh')
    user, _, fps_u = bvh_joints(Path(retarget_dir) / 'user.bvh')
    track = lambda j, fps: track_from_joints(j, np.arange(len(j)) / fps, 'y', None, names)
    res = score(track(coach, fps_c), track(user, fps_u), cfg, weights, try_mirror=True)
    return res.playback(), bool(res.to_dict()['mirrored'])


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


class Run:
    """A run_video output: wham.npz, report.json (frames_used, fps_in, the video WHAM read).
    camera=True: the run is drawn on its own video, so wham.npz must carry the camera fields of
    tools/pose_estimation/wham_extract.py (runs made before that change: rerun WHAM)."""

    def __init__(self, path, camera=True):
        self.dir = Path(path)
        self.report = json.loads((self.dir / 'report.json').read_text())
        self.npz = dict(np.load(self.dir / 'wham.npz'))
        missing = [k for k in CAMERA_KEYS if k not in self.npz]
        if camera and missing:
            raise SystemExit(f'{self.dir}/wham.npz has no {missing}: rerun WHAM (run_video / score_user --rerun)')
        self.video = Path(self.report['video'])          # video_wham.mp4 when converted, else the source

    def frame(self, bvh_i):
        return min(P.npz_frame(bvh_i, self.report), len(self.npz['detected']) - 1)


def bvh_frames(path):
    for line in open(path):
        if line.startswith('Frames:'):
            return int(line.split()[1])
    raise ValueError(f'{path}: no Frames line')
