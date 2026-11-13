"""예제 실행: data/의 코치·학생 스켈레톤으로 전체 로직을 돌리고, 기대 결과와 비교한다.

    python run_example.py
"""
import json
from dataclasses import replace
from pathlib import Path

from motion_scoring import Config, load_npz, score

HERE = Path(__file__).parent
cfg = replace(Config(), auto_tau=True, pose_trim=0.3)          # 실제 영상 설정
res = score(load_npz(HERE / "data/coach.npz", up="-y"), load_npz(HERE / "data/student.npz", up="-y"), cfg)
out = json.loads(json.dumps(res.to_dict(), ensure_ascii=False))

print(f"tau = {res.cfg.tau_deg:.2f} deg, kappa = {res.cfg.kappa:.2f}, global_tempo = {out['global_tempo']}")
print(f"{'structure':<13} {'student_t':<16} {'timing':<18} pose")
for s in out["segments"]:
    t = s["timing"]
    timing = f"{t['label']} x{t['ratio']}" if t and t.get("label") else "-"
    pose = ", ".join(f"{w['bone']} {w['diff_deg']}°" for w in s["pose"]["worst"]) if s["pose"] else "-"
    print(f"{s['structure']:<13} {str(s['student_t']):<16} {timing:<18} {pose}")

exp = json.loads((HERE / "data/expected/result.json").read_text())
print("\n기대 결과와 같음:", out["segments"] == exp["segments"])
