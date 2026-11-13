"""CLI:  python -m motion_scoring coach.npz student.npz [--up=-y] [--tau 4.5 --kappa 5] [--json out.json]"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json

from .config import Config
from .step1_preprocess import load_npz
from .pipeline import score


def main():
    ap = argparse.ArgumentParser(prog="motion_scoring")
    ap.add_argument("coach")
    ap.add_argument("student")
    ap.add_argument("--up", default="y", help="up axis of the NPZ joints: x, y, z, -x, -y, -z")
    ap.add_argument("--tau", type=float)
    ap.add_argument("--kappa", type=float)
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args()
    cfg = Config()
    if a.tau: cfg = replace(cfg, tau_deg=a.tau)
    if a.kappa: cfg = replace(cfg, kappa=a.kappa)
    coach, student = load_npz(a.coach, a.up), load_npz(a.student, a.up)
    # real video preset: tau from the coach's own repetitions, robust (trimmed) pose distance
    cfg = replace(cfg, auto_tau=not a.tau, pose_trim=0.3)
    res = score(coach, student, cfg, try_mirror=not a.no_mirror)
    text = json.dumps(res.to_dict(), indent=2, ensure_ascii=False)
    if a.json:
        open(a.json, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
