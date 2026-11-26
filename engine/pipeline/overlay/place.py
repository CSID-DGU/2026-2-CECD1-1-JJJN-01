"""Put a body (coach pose or the student's own) on the student's video: world placement first, camera last.

A pose to place is the common format of coach_pose: per frame `tilt` (3, 3), the root rotation without the
heading (yaw), and `body_pose` (69,) local rotations. Placing it:
  1  place_world  shape = student betas, heading = the student's at that frame, position = feet (or pelvis)
                  on the student's -> SMPL params in WHAM's world frame (Y up, metres). The 3D viewer reads these.
  2  to_camera    the student's world -> camera move at that frame (camera_from_student) -> camera-frame
                  params, what the overlay video draws.
The student is placed by the same function (its own pose, heading and feet: it lands on itself), so
student_self.mp4 checks this whole path.

Frames: pb (motion_scoring playback) counts BVH frames. A run_video BVH starts at the first frame WHAM saw
the person (report.json frames_used[0]) and is resampled to 30 fps; wham.npz and the video WHAM read count
the video's frames (`npz_frame`).
"""
import numpy as np

from .smpl_np import FEET, PELVIS, aa_to_mat, mat_to_aa, move, rot_y, split_yaw

BVH_FPS = 30.


def npz_frame(i, report):
    """BVH frame i of a run_video run -> frame of its wham.npz / of the video WHAM read."""
    return int(report['frames_used'][0] + round(i * report['fps_in'] / BVH_FPS))


def pose_from_npz(npz, key='pose_world'):
    """wham.npz -> the common pose format {tilt (T,3,3), body_pose (T,69)} (frames WHAM missed are NaN)."""
    p = np.asarray(npz[key], np.float64)
    _, tilt = split_yaw(aa_to_mat(np.nan_to_num(p[:, :3])))
    tilt[np.isnan(p[:, 0])] = np.nan
    return dict(tilt=tilt, body_pose=p[:, 3:])


class StudentFrame:
    """What the student gives a placement at one npz frame f: shape, heading, feet / pelvis, camera."""

    def __init__(self, body, npz, f):
        self.betas = np.asarray(npz['betas'], np.float64)
        self.j0 = body.rest_pelvis(self.betas)
        gw, tw = npz['pose_world'][f, :3], npz['trans_world'][f]
        gc, tc = npz['pose_cam'][f, :3], npz['trans_cam'][f]
        self.ok = bool(np.isfinite(gw).all() and np.isfinite(gc).all())
        if not self.ok:
            return
        Rw, Rc = aa_to_mat(gw), aa_to_mat(gc)
        self.yaw, _ = split_yaw(Rw)
        _, j = body(gw, npz['pose_world'][f, 3:], self.betas, tw)
        self.anchors = dict(feet=j[0, FEET].mean(0), pelvis=j[0, PELVIS])
        # world -> camera at this frame: the student's world body onto its camera body (exact for the student).
        # The translation is per frame: WHAM's world path is integrated from velocities and drifts from the
        # camera-frame position, so only the student's own frame ties the two together.
        self.R = Rc @ Rw.T
        self.t = (self.j0 + tc) - self.R @ (self.j0 + tw)


def place_world(body, tilt, body_pose, st, anchor='feet'):
    """One pose frame on student frame st -> world params dict(global_orient, body_pose, transl, betas)
    and the 24 SMPL joints in world."""
    go = mat_to_aa(rot_y(st.yaw) @ tilt)
    _, j = body(go, body_pose, st.betas)
    p = j[0, FEET].mean(0) if anchor == 'feet' else j[0, PELVIS]
    transl = st.anchors[anchor] - p
    return dict(global_orient=go, body_pose=np.asarray(body_pose, np.float64), transl=transl,
                betas=st.betas), j[0] + transl


def to_camera(params, st):
    """World params -> the same body in the student's camera frame (OpenCV axes)."""
    go, tr = move(params['global_orient'], params['transl'], st.R, st.t, st.j0)
    return dict(params, global_orient=go, transl=tr)
