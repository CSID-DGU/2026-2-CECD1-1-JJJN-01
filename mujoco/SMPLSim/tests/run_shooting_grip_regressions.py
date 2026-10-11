"""Read original local four-body XMLs for legacy tests without copying assets."""
import sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
import test_smplh_motion
# Runtime test path override only. No protected source, model, or NPZ is changed.
test_smplh_motion.ROOT=ROOT.parents[2]/'basketball-r1/SMPLSim'
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
