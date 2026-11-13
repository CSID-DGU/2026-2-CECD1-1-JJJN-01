"""Thin inference wrapper around the official pretrained Mixamo PAN.

Uses only encoder(source) -> decoder(latent, target skeleton), i.e. the retargeting
branch of the official forward(), so no dummy target motion is needed. Source is group A
(e.g. BigVegas, 22 edges), targets are group B characters (27 edges).
"""
import math
import shlex
from pathlib import Path

import numpy as np
import torch

from .paths import MEAN_VAR_DIR, REPO, STD_BVH_DIR, enter_repo


def canonical_sign(raw, mean, std):
    """Pick q or -q per (frame, edge) to be closest to PAN's training stats.

    q and -q are the same rotation but different network inputs; training data used
    whatever sign from_euler produced, which this reproduces in distribution.
    """
    rot = raw[:-3].T.reshape(raw.shape[1], -1, 4)
    mu = mean[:-3].to(raw).view(1, -1, 4)
    sd = std[:-3].to(raw).view(1, -1, 4)
    zp = ((rot - mu) / sd).square().sum(-1, keepdim=True)
    zn = ((-rot - mu) / sd).square().sum(-1, keepdim=True)
    rot = torch.where(zn < zp, -rot, rot)
    return torch.cat([rot.reshape(raw.shape[1], -1).T, raw[-3:]], 0)


def ood_stats(normalized, threshold=4.):
    z = normalized[:-3].abs()
    return dict(mean_abs_z=float(z.mean()), frac_abs_z_gt=float((z > threshold).float().mean()),
                threshold=threshold)


class PanRuntime:
    def __init__(self, model_dir='pretrained_mixamo', source='BigVegas', targets=('Mousey_m',),
                 epoch=1000, device='cuda:0'):
        enter_repo()
        from data_preprocess.Mixamo import create_dataset
        from models import create_model_mixamo
        from parser.parser_mixamo import get_parser
        targets = list(targets)
        for name in [source] + targets:
            for p in (STD_BVH_DIR / f'{name}.bvh', MEAN_VAR_DIR / f'{name}_mean.npy',
                      MEAN_VAR_DIR / f'{name}_var.npy'):
                if not p.exists():
                    raise FileNotFoundError(p)
        model_dir = (REPO / model_dir).resolve()
        args = get_parser().parse_args(shlex.split((model_dir / 'para.txt').read_text().strip())[1:])
        args.save_dir, args.cuda_device = str(model_dir), device
        args.is_train, args.model, args.rotation = False, 'pan', 'quaternion'
        # Weights were saved from DataParallel ('module.' keys); keep use_parallel from
        # para.txt so the official loader matches them, and unwrap at call time.
        self.characters = [[source], targets]
        self.data = create_dataset(args, self.characters)
        self.model = create_model_mixamo(args, self.characters, self.data)
        self.model.load(epoch=epoch)
        for net in self.model.models:
            net.auto_encoder.eval()
            net.static_encoder.eval()
        self.device, self.source, self.targets = device, source, targets
        with torch.no_grad():
            self.target_repr = self._unwrap(self.model.models[1].static_encoder)(self.data.offsets[1])

    @staticmethod
    def _unwrap(m):
        return getattr(m, 'module', m)

    @property
    def source_stats(self):
        return self.data.mean[0][0].reshape(-1), self.data.var[0][0].reshape(-1)

    def normalize_source(self, raw):
        mean, std = self.source_stats
        return (raw.to(self.device) - mean[:, None]) / std[:, None]

    @torch.no_grad()
    def retarget(self, raw):
        """raw: (C_src, T) denormalised source motion -> {target: (C_tgt, T)}."""
        raw = raw.to(self.device, torch.float32)
        T = raw.shape[1]
        N = max(64, math.ceil(T / 4) * 4)
        padded = torch.cat([raw, raw[:, -1:].repeat(1, N - T)], 1)
        latent = self._unwrap(self.model.models[0].auto_encoder).enc(self.normalize_source(padded)[None])
        dec = self._unwrap(self.model.models[1].auto_encoder).dec
        out = {}
        for i, name in enumerate(self.targets):
            res = dec(latent, self.target_repr[i:i + 1])
            out[name] = self.data.denorm(1, [i], res)[0, :, :T]
        return out

    @torch.no_grad()
    def retarget_intra(self, raw, offsets, mean=None, std=None):
        """Same-topology retargeting onto an arbitrary skeleton of the source's group (A).

        raw: (C_src, T) denormalised source motion on the source character's scale.
        offsets: (J, 3) rest offsets of the new skeleton (source topology, cm).
        mean/std: output statistics of the new skeleton. A real user has none, so by default
        the source statistics are used with the root channels scaled by `root_ratio`
        (rotations are scale-free; root position statistics follow the hip height: on the
        Mixamo group-A characters the leg ratio matches the real root stats, total height
        does not, e.g. Kaya 0.73 vs 0.88).
        """
        raw = raw.to(self.device, torch.float32)
        T = raw.shape[1]
        N = max(64, math.ceil(T / 4) * 4)
        padded = torch.cat([raw, raw[:, -1:].repeat(1, N - T)], 1)
        latent = self._unwrap(self.model.models[0].auto_encoder).enc(self.normalize_source(padded)[None])
        off = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)[None]
        repr_ = self._unwrap(self.model.models[0].static_encoder)(off)
        res = self._unwrap(self.model.models[0].auto_encoder).dec(latent, repr_)[0, :, :T]
        if mean is None:
            mean, std = (s.clone() for s in self.source_stats)
            ratio = self.root_ratio(offsets)
            mean[-3:] *= ratio
            std[-3:] *= ratio
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.device).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32, device=self.device).reshape(-1)
        return res * std[:, None] + mean[:, None]

    def root_ratio(self, offsets):
        """Rest-pose hip height (root above the lowest foot/toe joint) relative to the source."""
        from .person import rest_positions
        from .skeleton import Skeleton
        skel = Skeleton.from_character(self.source)
        feet = [skel.index(n) for n in ('LeftFoot', 'RightFoot', 'LeftToeBase', 'RightToeBase')]
        hip = lambda off: -rest_positions(skel, np.asarray(off, dtype=np.float64))[feet, 1].min()
        return float(hip(offsets) / hip(skel.offsets))

    @torch.no_grad()
    def positions(self, raw, group, index):
        """Global positions (T, J, 3) of a raw motion on a dataset character."""
        raw = raw.to(self.device, torch.float32)
        fk = self.model.models[group].fk
        return fk.forward_from_raw(raw[None], self.data.offsets[group][[index]])[0]

    def writer(self, group, index):
        return self.model.writer[group][index]
