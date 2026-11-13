"""CP-DTW motion scoring. Input: student skeleton + coach skeleton retargeted to the student body.

    from motion_scoring import score, load_npz
    result = score(load_npz("coach.npz"), load_npz("student.npz"))
    result.to_dict()["segments"]   # structure / timing / pose per student segment
"""
from .config import Config
from .step1_preprocess import Track, load_npz, track_from_joints
from .pipeline import Result, score

__all__ = ["Config", "Track", "Result", "score", "load_npz", "track_from_joints"]
