"""Draw SMPL meshes over video frames (OpenCV only) -> mp4.

The camera is WHAM's (lib/vis/renderer.py, lib/utils/imutils.compute_cam_intrinsics): pinhole, focal
sqrt(w^2 + h^2) px unless given, principal point at the image centre, OpenCV axes (x right, y down,
z forward). WHAM's own renderer needs pytorch3d, which the WHAM venv does not have; this one is a painter's
algorithm (back faces dropped, the rest filled far to near with Lambert shading from the camera), drawn on a
layer of its own and blended with the frame so the body is see-through.
"""
import cv2
import numpy as np

# RGB, as the frontend (styles/index.css): student = --color-me, coach = --color-coach; method 2 apart from both
COLORS = dict(student=(242, 101, 58), coach=(47, 91, 234), coach_m2=(20, 184, 166))


class Camera:
    def __init__(self, w, h, focal=None):
        self.w, self.h = int(w), int(h)
        self.f = float(focal) if focal is not None else float(np.hypot(w, h))
        self.c = np.array([w / 2., h / 2.])

    def project(self, v):
        return v[:, :2] / v[:, 2:3] * self.f + self.c


def draw_mesh(layer, mask, verts, faces, cam, rgb):
    """Fill the visible faces of one mesh (camera frame) into layer (BGR) and mask, in place."""
    tri = verts[faces]                                            # (F, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    centre = tri.mean(1)
    front = ((n * centre).sum(1) < 0) & (tri[:, :, 2] > 0.05).all(1)
    if not front.any():
        return
    n, centre, tri = n[front], centre[front], tri[front]
    shade = (-(n * centre).sum(1) / (np.linalg.norm(n, axis=1) * np.linalg.norm(centre, axis=1) + 1e-12))
    shade = 0.35 + 0.65 * np.clip(shade, 0, 1)
    px = np.round(cam.project(tri.reshape(-1, 3)) * 16).astype(np.int32).reshape(-1, 3, 2)   # 4-bit subpixel
    bgr = np.asarray(rgb[::-1], np.float64)
    for k in np.argsort(-centre[:, 2]):                            # far to near
        c = tuple(float(x) for x in bgr * shade[k])
        cv2.fillConvexPoly(layer, px[k], c, lineType=cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(mask, px[k], 255, lineType=cv2.LINE_AA, shift=4)


def composite(frame, meshes, faces, cam, alpha=0.6):
    """frame (BGR uint8) + meshes [(verts_cam, rgb)] -> BGR uint8. Meshes are drawn in the given order,
    each blended over what is under it."""
    out = frame.astype(np.float32)
    for verts, rgb in meshes:
        layer = np.zeros_like(frame)
        mask = np.zeros(frame.shape[:2], np.uint8)
        draw_mesh(layer, mask, verts, faces, cam, rgb)
        a = (mask.astype(np.float32) / 255. * alpha)[..., None]
        out = out * (1 - a) + layer.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


class FrameReader:
    """Frames of a video by index, read forwards (pb indices never go back)."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        assert self.cap.isOpened(), f'cannot open {path}'
        self.i, self.frame = -1, None

    def get(self, i):
        if i < self.i:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            self.i = i - 1
        while self.i < i:
            ok, frame = self.cap.read()
            if not ok:
                break                                              # past the end: keep the last frame
            self.frame, self.i = frame, self.i + 1
        return self.frame


def writer(path, fps):
    import imageio
    return imageio.get_writer(str(path), fps=fps, codec='libx264', quality=8, macro_block_size=1,
                              pixelformat='yuv420p')
