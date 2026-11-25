"""예제 실행: 합성 스쿼트(같은 폴더의 synthetic.py)로 전체 로직을 돌리고 결과를 표로 보여 준다. 데이터 파일이 필요 없다.

    python validation/run_example.py          # engine/pipeline/motion_scoring 에서

코치는 스쿼트 4회(바닥에서 바로 올라옴). 학생은 같은 동작인데 네 가지가 다르다.
    2번째 바닥에서 1초 멈춤              -> extra
    3번째 하강이 1.5배 느림               -> match + slow
    3번째 반복 뒤 선 자세로 쉬지 않음      -> missing
    바닥에서 몸통을 10도 더 숙임          -> pose 라벨이 붙을 만큼은 아니지만 bones 에 크기와 방향이 나온다
실제 스켈레톤(NPZ)은 `python -m motion_scoring coach.npz student.npz` 로 돌린다.
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]                   # synthetic.py (이 폴더), motion_scoring 패키지
import synthetic as syn  # noqa: E402
from motion_scoring import Config, main_weights, score, track_from_joints  # noqa: E402

# 실제 영상 설정 (백엔드가 스쿼트를 채점할 때와 같음): tau 와 kappa 는 코치 영상에서, 자세 비교는 다리 · 몸통 위주로
cfg = replace(Config(), auto_tau=True)
weights = main_weights(("trunk", "l_thigh", "r_thigh", "l_shank", "r_shank"), cfg)

coach = syn.phases(reps=4, top=0.8)
student = list(coach)
bottoms = [i for i, (_, depth) in enumerate(student) if depth == 1.0]      # 하강 구간의 위치
student[bottoms[2]] = (1.5, 1.0)                                            # 3번째 하강을 1.0초 -> 1.5초
student.insert(bottoms[1] + 1, (1.0, 1.0))                                  # 2번째 바닥에서 1초 멈춤
del student[bottoms[2] + 3]                                                 # 3번째 반복 뒤의 선 자세 쉼


def track(phases, seed, lean=1.0):
    joints, fps = syn.skeleton(phases, noise_cm=0.2, seed=seed, lean=lean)  # 영상마다 다른 관절 노이즈
    return track_from_joints(joints, np.arange(len(joints)) / fps, "y")


out = score(track(coach, 1), track(student, 2, lean=1.25), cfg, weights).to_dict()   # lean 1.25: 바닥에서 40도 -> 50도
p = out["params"]
print(f"tau = {p['tau_deg']:.2f} deg ({p['tau_from']}), kappa = {p['kappa']:.2f}, global_tempo = {out['global_tempo']}")
row = "{:<13} {:<16} {:<16} {:<12} {:<12} {}"
print(row.format("structure", "student_t", "coach_t", "timing", "pose / hold", "가장 크게 벌어졌을 때 5도 넘게 다른 뼈 (bones)"))
for s in out["segments"]:
    t = s["timing"]
    timing = f"{t['label']} x{t['ratio']}" if t else "-"
    note = ", ".join(f"{w['bone']} {w['diff_deg']}°" for w in s["pose"]["worst"]) if s["pose"] else ""
    if "hold" in s:
        note += f"hold x{s['hold']['ratio']}"
    bones = ", ".join(f"{b['bone']} {b['deg']}° {b['dir']}" for b in s.get("bones", []) if b["deg"] > 5)
    print(row.format(s["structure"], str(s["student_t"]), str(s["coach_t"]), timing, note or "-", bones or "-"))
