"""Joint-space skeleton for PAN's Mixamo topologies.

PAN's raw motion is edge-major: for edge e = (a -> b) it stores the local rotation of
joint a, followed by the root (joint 0) position. Edges sharing a parent therefore carry
the same rotation (verified on the training data), so solving in joint space keeps that
structure by construction.
"""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from . import quat as Q
from .paths import STD_BVH_DIR


@dataclass(frozen=True)
class Skeleton:
    names: tuple
    parents: tuple          # parents[0] == -1, parent index < child index
    offsets: np.ndarray     # (J, 3) rest offsets in the parent's frame (cm)
    edges: tuple            # ((parent, child), ...) in PAN channel order
    ee_ids: tuple

    @classmethod
    def from_bvh(cls, path):
        from data_preprocess.Mixamo.bvh_parser import BVH_file
        f = BVH_file(str(path))
        parents = (-1,) + tuple(int(p) for p in f.topology[1:])   # BVH_file stores root parent as 0
        if any(not 0 <= p < j for j, p in enumerate(parents) if j):
            raise ValueError(f'Unsupported joint order in {path}')
        return cls(tuple(f.names), parents, np.asarray(f.offset, dtype=np.float64),
                   tuple((int(e[0]), int(e[1])) for e in f.edges), tuple(int(i) for i in f.get_ee_id()))

    @classmethod
    def from_character(cls, name):
        return cls.from_bvh(Path(STD_BVH_DIR) / f'{name}.bvh')

    def with_offsets(self, offsets):
        offsets = np.asarray(offsets, dtype=np.float64)
        if offsets.shape != self.offsets.shape:
            raise ValueError('Offset shape mismatch')
        return replace(self, offsets=offsets)

    @property
    def n_joints(self):
        return len(self.names)

    @property
    def n_channels(self):
        return len(self.edges) * 4 + 3

    def index(self, name):
        return self.names.index(name)

    def children(self, j):
        return [c for c, p in enumerate(self.parents) if p == j]

    @property
    def rotating_joints(self):
        """Joints whose rotation appears in PAN's raw (parents of at least one edge)."""
        return sorted({a for a, _ in self.edges})

    @property
    def edge_parent(self):
        return [a for a, _ in self.edges]

    def height(self, offsets=None):
        """Same definition as BVH_file.get_height (toe chain + head chain to the root)."""
        off = self.offsets if offsets is None else offsets
        total = 0.
        for ee in (self.ee_ids[0], self.ee_ids[2]):
            p = ee
            while p != 0:
                total += float(np.linalg.norm(off[p]))
                p = self.parents[p]
        return total

    def bone_axis(self, j):
        """Unit direction of joint j's single non-zero child bone (local frame), else None."""
        kids = [c for c in self.children(j) if np.linalg.norm(self.offsets[c]) > 1e-6]
        if len(kids) != 1:
            return None
        o = self.offsets[kids[0]]
        return o / np.linalg.norm(o)

    def fk(self, q, root, offsets=None):
        """Global joint positions. q: (..., J, 4) local unit quats, root: (..., 3)."""
        off = torch.as_tensor(self.offsets if offsets is None else offsets, dtype=q.dtype, device=q.device)
        g = [q[..., 0, :]]
        p = [root]
        for j in range(1, self.n_joints):
            a = self.parents[j]
            p.append(p[a] + Q.rotate(g[a], off[j].expand_as(root)))
            g.append(Q.mul(g[a], q[..., j, :]))
        return torch.stack(p, -2)

    def joint_to_raw(self, q, root):
        """(T, J, 4), (T, 3) -> PAN raw (C, T)."""
        rot = q[:, self.edge_parent, :].reshape(q.shape[0], -1)
        return torch.cat([rot, root], -1).T.contiguous()

    def raw_to_joint(self, raw):
        """PAN raw (C, T) -> (T, J, 4) local quats (identity for leaves), (T, 3) root."""
        raw = torch.as_tensor(raw)
        T = raw.shape[1]
        rot = raw[:-3].T.reshape(T, len(self.edges), 4)
        q = torch.zeros((T, self.n_joints, 4), dtype=raw.dtype, device=raw.device)
        q[..., 0] = 1
        for e, (a, _) in enumerate(self.edges):
            q[:, a] = rot[:, e]
        return q, raw[-3:].T.contiguous()
