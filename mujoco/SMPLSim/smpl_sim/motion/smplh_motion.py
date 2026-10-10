"""Explicit SMPL+H axis-angle -> generated XYZ-hinge MJCF mapping."""
from dataclasses import dataclass
import warnings
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation, Slerp
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES

# M2 body geometry remains in SMPL axes; change only the floating root frame.
SMPL_TO_WORLD = Rotation.from_quat([.5, .5, .5, .5])  # scipy xyzw


class SMPLHMapping:
    def __init__(self, model):
        self.model = model
        if model.nq != 160 or model.nv != 159 or model.nu != 153:
            raise ValueError('Expected the M2 52-body, 153 XYZ-hinge model')
        self.root = model.joint('Pelvis').id
        if model.jnt_type[self.root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError('Pelvis must be a freejoint')
        self.root_adr = int(model.jnt_qposadr[self.root])
        if self.root_adr != 0:
            raise ValueError("Expected floating root at qpos address 0")
        self.addresses = []
        for name in SMPLH_BONE_ORDER_NAMES[1:]:
            ids = [model.joint(f'{name}_{axis}').id for axis in 'xyz']
            if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in ids):
                raise ValueError(f'{name}: expected XYZ hinges')
            if not np.allclose(model.jnt_axis[ids], np.eye(3)):
                raise ValueError(f'{name}: unexpected hinge axes')
            adrs = model.jnt_qposadr[ids]
            if not np.array_equal(adrs, np.arange(adrs[0], adrs[0]+3)):
                raise ValueError(f'{name}: unexpected rotation composition order')
            self.addresses.append(adrs.copy())
        self.actuator_joint_ids = model.actuator_trnid[:, 0].copy()
        if len(np.unique(self.actuator_joint_ids)) != 153:
            raise ValueError('Expected one motor per hinge')
        if not np.allclose(model.actuator_gear[:, 0], 1):
            raise ValueError('This controller expects unit motor gears')
        self.actuator_qpos = model.jnt_qposadr[self.actuator_joint_ids]
        self.actuator_dof = model.jnt_dofadr[self.actuator_joint_ids]

    def convert(self, poses, root_position, coordinate_system='smpl', translation_kind='root_position'):
        poses = np.asarray(poses, dtype=float)
        root_position = np.asarray(root_position, dtype=float)
        if poses.ndim != 2 or poses.shape[1] != 156 or root_position.shape != (len(poses), 3):
            raise ValueError('Expected poses (N,156) and root_position (N,3)')
        if len(poses) == 0 or not np.isfinite(poses).all() or not np.isfinite(root_position).all():
            raise ValueError('Motion must be nonempty and finite')
        if coordinate_system not in ('smpl', 'mujoco'):
            raise ValueError('coordinate_system must be smpl or mujoco')
        if translation_kind not in ('root_position', 'smpl_transl'):
            raise ValueError('translation_kind must be root_position or smpl_transl')
        if translation_kind == 'smpl_transl' and coordinate_system != 'smpl':
            raise ValueError('smpl_transl must be expressed in SMPL coordinates')
        rotations = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 52, 3, 3)
        q = np.tile(self.model.qpos0, (len(poses), 1))
        positions = root_position.copy()
        if translation_kind == 'smpl_transl':
            positions += self.model.body_pos[self.model.jnt_bodyid[self.root]]
        root_rot = Rotation.from_matrix(rotations[:, 0])
        if coordinate_system == 'smpl':
            root_rot = SMPL_TO_WORLD * root_rot
            positions = SMPL_TO_WORLD.apply(positions)
        q[:, self.root_adr:self.root_adr+3] = positions
        q[:, self.root_adr+3:self.root_adr+7] = root_rot.as_quat()[:, [3, 0, 1, 2]]
        for i, adr in enumerate(self.addresses, 1):
            with warnings.catch_warnings():
                warnings.simplefilter('error', UserWarning)
                try:
                    angles = Rotation.from_matrix(rotations[:, i]).as_euler('XYZ')
                except UserWarning as exc:
                    raise ValueError(f'Gimbal lock in {SMPLH_BONE_ORDER_NAMES[i]}: change the motion or use continuous IK') from exc
            angles = np.unwrap(angles, axis=0)
            low, high = self.model.jnt_range[[self.model.joint(f'{SMPLH_BONE_ORDER_NAMES[i]}_{a}').id for a in 'xyz']].T
            if np.any(angles < low - 1e-6) or np.any(angles > high + 1e-6):
                raise ValueError(f'Motion exceeds joint range: {SMPLH_BONE_ORDER_NAMES[i]}')
            q[:, adr] = angles
        return q


@dataclass
class SMPLHMotion:
    poses: np.ndarray
    root_position: np.ndarray
    fps: float
    coordinate_system: str = 'smpl'
    translation_kind: str = 'root_position'

    def __post_init__(self):
        self.poses = np.asarray(self.poses, dtype=float)
        self.root_position = np.asarray(self.root_position, dtype=float)
        if self.poses.ndim != 2 or self.poses.shape[1:] != (156,) or len(self.poses) < 2:
            raise ValueError('poses must have shape (N>=2,156), radians axis-angle')
        if self.root_position.shape != (len(self.poses), 3):
            raise ValueError('root_position must have shape (N,3), meters')
        if not np.isfinite(self.poses).all() or not np.isfinite(self.root_position).all():
            raise ValueError('Motion contains nonfinite values')
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError('fps must be positive and finite')
        if self.coordinate_system not in ('smpl', 'mujoco') or self.translation_kind not in ('root_position', 'smpl_transl'):
            raise ValueError('Unknown motion convention')
        if self.translation_kind == 'smpl_transl' and self.coordinate_system != 'smpl':
            raise ValueError('smpl_transl requires SMPL coordinates')

    @property
    def duration(self):
        return (len(self.poses)-1) / self.fps

    def save(self, path):
        np.savez_compressed(path, poses=self.poses, root_position=self.root_position,
                            fps=self.fps, coordinate_system=self.coordinate_system,
                            translation_kind=self.translation_kind, schema_version=1)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as f:
            if int(f['schema_version']) != 1:
                raise ValueError('Unsupported motion schema version')
            return cls(f['poses'], f['root_position'], float(f['fps']),
                       str(f['coordinate_system']), str(f['translation_kind']))

    def bind(self, model):
        return BoundMotion(self, SMPLHMapping(model))


class BoundMotion:
    def __init__(self, motion, mapping):
        self.motion, self.mapping = motion, mapping
        self.qpos = mapping.convert(motion.poses, motion.root_position,
                                    motion.coordinate_system, motion.translation_kind)
        self.times = np.arange(len(self.qpos)) / motion.fps
        self.root_slerp = Slerp(self.times, Rotation.from_quat(self.qpos[:, 3:7][:, [1,2,3,0]]))

    def sample(self, t):
        t = float(np.clip(t, 0, self.motion.duration))
        frame = min(int(t*self.motion.fps), len(self.qpos)-2)
        a = (t-self.times[frame])*self.motion.fps
        q = (1-a)*self.qpos[frame] + a*self.qpos[frame+1]
        q[3:7] = self.root_slerp(t).as_quat()[[3,0,1,2]]
        return q
