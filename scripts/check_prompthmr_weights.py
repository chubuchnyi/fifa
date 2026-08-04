"""Does the released PromptHMR checkpoint fit the released code, key for key?

Evidence for docs/findings/occlusion-pose-research-2026-08-04.md — run it instead of trusting
the write-up. CPU only, ~1 min, needs the gitignored checkout at backends/PromptHMR (see that
doc for what to clone and fetch).

    .venv/bin/python scripts/check_prompthmr_weights.py             # key coverage, ~1 min
    .venv/bin/python scripts/check_prompthmr_weights.py --forward   # + a real forward, ~4 min

Three things are stubbed, and none of them is part of the network under test: the gated classic
SMPL (buffers taken from the checkpoint, which carries the whole neutral model, so no MPI
account is needed), pytorch_lightning (-> nn.Module, so no training stack), and xformers
(-> torch SDPA, same maths in [B,M,H,K] layout, so no CUDA build). SMPL-X and smplx2smpl.pkl
are the real files.

--forward feeds noise through the image model and reports the four tensors pitch3d's HMRBackend
seam consumes, then re-runs with mask_prompt=False. The delta between the two only shows the
mask path is live -- on noise it is not a quality measure.
"""
import argparse
import collections
import os
import sys
import types

import torch
import torch.nn as nn

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--backend-dir', default='backends/PromptHMR',
                    help='PromptHMR checkout (its config paths are relative, so we chdir into it)')
parser.add_argument('--forward', action='store_true',
                    help='also run a forward pass and A/B the mask prompt')
args = parser.parse_args()

os.chdir(args.backend_dir)
sys.path.insert(0, '.')

pl = types.ModuleType('pytorch_lightning')
pl.LightningModule = nn.Module
sys.modules['pytorch_lightning'] = pl


def _memory_efficient_attention(q, k, v, attn_bias=None, p=0.0, scale=None):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=scale)
    return out.transpose(1, 2)


xf = types.ModuleType('xformers')
xf_ops = types.ModuleType('xformers.ops')
xf_ops.memory_efficient_attention = _memory_efficient_attention
xf_ops.unbind = lambda x, dim=0: torch.unbind(x, dim)
xf.ops = xf_ops
sys.modules['xformers'] = xf
sys.modules['xformers.ops'] = xf_ops

sd = torch.load('data/pretrain/phmr/checkpoint.ckpt', map_location='cpu',
                weights_only=True)['state_dict']

# Drop `pretrained=`: proves the MetaCLIP download at construction is redundant, because the
# checkpoint carries the CLIP tower itself and the load below still reports 0 missing.
import open_clip  # noqa: E402, I001  (every import below must follow the stubbing above)

_real_create = open_clip.create_model
open_clip.create_model = lambda name, pretrained=None, **kw: _real_create(name, **kw)

import prompt_hmr.models.phmr as phmr_mod  # noqa: E402


class _StubSMPL(nn.Module):
    """Stands in for the gated classic SMPL. Its buffers are the checkpoint's own."""

    def __init__(self, *a, **k):
        super().__init__()
        self.vertex_joint_selector = nn.Module()
        for key, v in sd.items():
            if not key.startswith('smpl.'):
                continue
            name = key[5:]
            owner, _, leaf = name.rpartition('.')
            (getattr(self, owner) if owner else self).register_buffer(leaf, v.clone())

    def joints_from_vertices(self, verts):
        return torch.einsum('jv,bvc->bjc', self.J_regressor, verts)


phmr_mod.SMPL = _StubSMPL

from prompt_hmr.core.config import parse_args  # noqa: E402
from prompt_hmr.models import build_phmr  # noqa: E402

model = build_phmr(parse_args(['--cfg', 'data/pretrain/phmr/config.yaml']))
missing, unexpected = model.load_state_dict(sd, strict=False)

real_missing = [k for k in missing if k.split('.')[0] != 'smpl']
real_unexpected = [k for k in unexpected if k.split('.')[0] != 'smpl']
landed = [k for k in sd if k not in unexpected and k.split('.')[0] != 'smpl']

print(f'checkpoint tensors        : {len(sd)}')
print(f'model tensors             : {len(model.state_dict())}')
print(f'MISSING (real modules)    : {len(real_missing)} {real_missing[:6]}')
print(f'UNEXPECTED (real modules) : {len(real_unexpected)} {real_unexpected[:6]}')
print('loaded groups             :',
      dict(collections.Counter(k.split('.')[0] for k in landed)))
print(f'params landed             : {sum(sd[k].numel() for k in landed) / 1e6:.1f}M')

mw = model.state_dict()['prompt_encoder.mask_downscaling.6.weight']
match = torch.equal(mw, sd['prompt_encoder.mask_downscaling.6.weight'])
print(f'mask_downscaling.6 loaded : match={match} std={mw.std().item():.4f}')

gated = [k for k in sd if k.split('.')[0] == 'smpl']
print(f'gated classic-SMPL buffers: {len(gated)}, '
      f'{sum(sd[k].numel() for k in gated) / 1e6:.1f}M params (data, not learned weights)')

if not args.forward:
    sys.exit(1 if real_missing or real_unexpected else 0)

print(f'\nSMPL_NEUTRAL.pkl present   : {os.path.isdir("data/body_models/smpl")}')
model = model.eval()
model.is_train = False
torch.manual_seed(0)

K = torch.eye(3)[None]
K[0, 0, 0] = K[0, 1, 1] = 1200.0
K[0, 0, 2] = K[0, 1, 2] = 448.0
masks = torch.zeros(2, 1, 256, 256)
masks[:, :, 60:200, 80:170] = 1.0
batch = [{
    'image': torch.randn(3, 896, 896),
    'cam_int': K,
    'boxes': torch.tensor([[300., 200., 420., 620., 1.], [500., 240., 610., 650., 1.]]),
    'kpts': torch.rand(2, 25, 3),
    'masks': masks,
}]

with torch.no_grad():
    out = model(batch, mask_prompt=True)[0]
    out_nomask = model(batch, mask_prompt=False)[0]

print('forward pass               : OK')
for key in ('rotmat', 'transl', 'betas', 'features'):
    shape, finite = str(tuple(out[key].shape)), bool(torch.isfinite(out[key]).all())
    print(f'  {key:9s} {shape:18s} finite={finite}')

delta = (out['rotmat'] - out_nomask['rotmat']).abs().max().item()
print(f'mask on vs off             : max |d rotmat| = {delta:.4f} '
      f'({"live" if delta > 0 else "INERT"}; noise input, so not a quality measure)')

sys.exit(1 if real_missing or real_unexpected or delta == 0 else 0)
