"""S0: video -> MediaPipe Pose landmarks (.npz). Run with the separate pose venv:

  ../.venv-pose/bin/python tools/pose_estimation/mediapipe_extract.py VIDEO OUT.npz     # from engine/

Output contract (consumed by pipeline/core/mediapipe_format.py):
  world       (T, 33, 3) float32  metric world landmarks, metres, origin at hip centre,
                                  MediaPipe axes (x right, y down, z away from camera)
  image       (T, 33, 3) float32  normalised image landmarks (x, y in [0,1], z relative)
  visibility  (T, 33)    float32
  detected    (T,)       bool     False where no person was found (landmarks are NaN)
  fps, width, height, source, model
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

MODEL = Path(__file__).resolve().parent / 'models/pose_landmarker_heavy.task'


def extract(video, model=MODEL, min_conf=0.5):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    opts = PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(model)),
                                 running_mode=RunningMode.VIDEO, num_poses=1,
                                 min_pose_detection_confidence=min_conf,
                                 min_pose_presence_confidence=min_conf, min_tracking_confidence=min_conf)
    world, image, vis, det = [], [], [], []
    with PoseLandmarker.create_from_options(opts) as lm:
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                      int(round(t * 1000 / fps)))
            t += 1
            if res.pose_world_landmarks:
                wl, il = res.pose_world_landmarks[0], res.pose_landmarks[0]
                world.append([[p.x, p.y, p.z] for p in wl])
                image.append([[p.x, p.y, p.z] for p in il])
                vis.append([p.visibility for p in wl])
                det.append(True)
            else:
                world.append(np.full((33, 3), np.nan))
                image.append(np.full((33, 3), np.nan))
                vis.append(np.zeros(33))
                det.append(False)
    cap.release()
    if not world:
        raise RuntimeError(f'No frames decoded from {video}')
    return dict(world=np.asarray(world, np.float32), image=np.asarray(image, np.float32),
                visibility=np.asarray(vis, np.float32), detected=np.asarray(det),
                fps=float(fps), width=w, height=h, source=str(video), model=str(model))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('video')
    ap.add_argument('out')
    ap.add_argument('--min-conf', type=float, default=0.5)
    opt = ap.parse_args()
    data = extract(opt.video, min_conf=opt.min_conf)
    Path(opt.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(opt.out, **data)
    print(f'{len(data["detected"])} frames @ {data["fps"]:.2f} fps, person in '
          f'{int(data["detected"].sum())}; wrote {opt.out}')


if __name__ == '__main__':
    main()
