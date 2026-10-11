"""Run protected regressions while redirecting their audit report side effect."""
import sys,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from examples import m5_coordinated_eval
new=ROOT/'outputs/m5_1_motion_redesign/regression_audit';new.mkdir(exist_ok=True)
(new/'preservation_before.json').write_bytes((m5_coordinated_eval.OUT/'preservation_before.json').read_bytes())
m5_coordinated_eval.OUT=new
runpy.run_path(str(ROOT/'tests/run_shooting_grip_regressions.py'),run_name='__main__')
