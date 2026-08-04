"""Does the released PromptHMR checkpoint fit the released code, key for key?

Evidence for docs/findings/occlusion-pose-research-2026-08-04.md — run it instead of trusting
the write-up. CPU only, ~1 min, needs the gitignored checkout at backends/PromptHMR (see that
doc for what to clone and fetch).

    .venv/bin/python scripts/check_prompthmr_weights.py

Three things are stubbed, and none of them is part of the network under test: the gated classic
SMPL (buffers shaped from the checkpoint, since SMPL_NEUTRAL.pkl needs an MPI account),
pytorch_lightning (-> nn.Module, so no training stack), and xformers (-> torch SDPA, same maths
in [B,M,H,K] layout, so no CUDA build). SMPL-X and smplx2smpl.pkl are the real files.
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
    def __init__(self, *a, **k):
        super().__init__()
        for key, v in sd.items():
            if key.startswith('smpl.'):
                self.register_buffer(key[5:].replace('.', '__'), torch.zeros_like(v))


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

sys.exit(1 if real_missing or real_unexpected else 0)
