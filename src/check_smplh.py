"""Validate a local SMPL+H model and save a headless visual preview.

Run from any directory: python src/check_smplh.py
This checks geometry and articulation, not MuJoCo dynamics or motion tracking.
"""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import smplx
import torch

ROOT = Path(__file__).resolve().parents[1]


def draw(ax, vertices, faces, title, azim=-90, center=None, span=None):
    # Display SMPL's Y-up coordinates as plot Z-up, preserving handedness.
    points = vertices[:, [0, 2, 1]].copy()
    points[:, 1] *= -1
    mesh = Poly3DCollection(points[faces], linewidths=0.05,
                            edgecolors=(0.15, 0.25, 0.35, 0.12),
                            facecolors=(0.48, 0.70, 0.85, 1))
    ax.add_collection3d(mesh)
    if center is None:
        center = (points.max(axis=0) + points.min(axis=0)) / 2
    if span is None:
        span = np.ptp(points, axis=0).max() * 0.55
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=10, azim=azim)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path,
                        default=ROOT / 'data/models/smplh/SMPLH_MALE.pkl')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/smplh_check')
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    model = smplx.SMPLH(str(args.model), gender='male', use_pca=False,
                       flat_hand_mean=True, num_betas=10)
    body = torch.zeros(1, model.NUM_BODY_JOINTS * 3)
    body[0, 17 * 3 + 1] = -0.9  # Left elbow, excluding root in body_pose.
    body[0, 18 * 3 + 1] = 0.9   # Right elbow.
    left = torch.zeros(1, 45)
    right = torch.zeros(1, 45)
    left.reshape(1, 15, 3)[0, :, 2] = 0.5
    right.reshape(1, 15, 3)[0, :, 2] = -0.5
    with torch.no_grad():
        outputs = {
            'rest': model(),
            'body': model(body_pose=body),
            'left_hand': model(left_hand_pose=left),
            'right_hand': model(right_hand_pose=right),
        }
    vertices = {key: out.vertices[0].cpu().numpy()
                for key, out in outputs.items()}
    faces = np.asarray(model.faces)
    rest = vertices['rest']
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= len(rest):
        raise ValueError('Invalid mesh face indices')
    changes = {}
    for key, out in outputs.items():
        if not torch.isfinite(out.vertices).all() or not torch.isfinite(out.joints).all():
            raise ValueError(f'Non-finite geometry in {key}')
        if key != 'rest':
            changes[key] = float(np.linalg.norm(vertices[key] - rest, axis=1).max())
            if changes[key] < 1e-4:
                raise ValueError(f'Articulation has no measurable effect: {key}')
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    draw(fig.add_subplot(2, 3, 1, projection='3d'), rest, faces, 'Rest pose: front')
    draw(fig.add_subplot(2, 3, 2, projection='3d'), rest, faces, 'Rest pose: side', azim=0)
    draw(fig.add_subplot(2, 3, 3, projection='3d'), vertices['body'], faces,
         'Elbow articulation (synthetic test)', azim=-55)
    # Hand subsets are selected from skinning weights, including each wrist.
    weights = model.lbs_weights.detach().cpu().numpy()
    for panel, key, joints, title in [
        (4, 'rest', [21] + list(range(37, 52)), 'Right hand: rest'),
        (5, 'right_hand', [21] + list(range(37, 52)), 'Right hand: synthetic flexion'),
        (6, 'left_hand', [20] + list(range(22, 37)), 'Left hand: synthetic flexion'),
    ]:
        mask = weights[:, joints].sum(axis=1) > 0.5
        hand_faces = faces[np.all(mask[faces], axis=1)]
        if not len(hand_faces):
            raise ValueError('No hand faces found')
        points = rest[mask][:, [0, 2, 1]].copy()
        points[:, 1] *= -1
        center = (points.max(axis=0) + points.min(axis=0)) / 2
        span = float(np.ptp(points, axis=0).max() * 0.7)
        draw(fig.add_subplot(2, 3, panel, projection='3d'), vertices[key], hand_faces,
             title, azim=-60, center=center, span=span)
    fig.suptitle('SMPL+H male | zero shape coefficients | geometry check only', fontsize=15)
    fig.tight_layout()
    preview = args.output / 'preview.png'
    fig.savefig(preview, dpi=150)
    plt.close(fig)
    report = {
        'status': 'passed',
        'model_file': str(args.model.resolve()),
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'num_vertices': len(rest), 'num_faces': len(faces),
        'output_joint_count': int(outputs['rest'].joints.shape[1]),
        'skeletal_joint_count': int(model.parents.numel()),
        'num_betas_used': model.num_betas,
        'shape': 'zero betas; model mean shape',
        'extent_xyz_m': np.ptp(rest, axis=0).tolist(),
        'max_vertex_displacement_m': changes,
        'checks': ['finite vertices and joints in all four poses',
                   'valid face indices', 'body and each hand change geometry independently'],
        'scope': 'Geometry and articulation only; no physics, balance, or shooting validation.',
        'preview': str(preview.resolve()),
    }
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == '__main__':
    main()
