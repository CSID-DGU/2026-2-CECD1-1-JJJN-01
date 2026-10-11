"""Shared Unicode-safe loading for the generated standalone MJCF."""
from pathlib import Path
import mujoco

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'assets/smplh_humanoid.xml'


def load_model(path=MODEL_PATH):
    # MuJoCo's native Windows file API may not accept non-ASCII paths.
    # This generated MJCF has no external assets/includes to resolve.
    return mujoco.MjModel.from_xml_string(Path(path).read_text(encoding='utf-8'))


def reset_pose(model, data, pose='standing'):
    mujoco.mj_resetDataKeyframe(model, data, model.key(pose).id)
    mujoco.mj_forward(model, data)
