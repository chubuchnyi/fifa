"""Two A/Bs on the real clip, at the frame where players cross hardest (#132).

Both run the same PromptHMR checkpoint on the same frame and score each player by the IoU
between his projected mesh silhouette and his own SAM mask -- higher means the mesh stays on
its own player rather than drifting onto the neighbour. Masks come from box-prompted SAM and
identity from our own ByteTrack ids, so nothing here measures a different tracker.

    .venv/bin/python scripts/prompthmr_find_crossing.py --frames 60       # writes tracks.npz
    .venv/bin/python scripts/prompthmr_mask_ab.py --frame 29              # whole frame
    .venv/bin/python scripts/prompthmr_mask_ab.py --frame 29 --crop-pair  # crossing pair only
    .venv/bin/python scripts/prompthmr_mask_ab.py --frame 29 --crop-pair --joint-vs-solo

**Default arms, masks vs boxes** (`mask_prompt` is one flag): does a segmentation prompt keep
the mesh off the neighbour? Answered 2026-08-05, and the answer is no -- null whole-frame
(mean 0.495 vs 0.477) and still null cropped (0.630 vs 0.625, panels alike by eye). Keep the
whole-frame run as the honest control, but read it knowing a broadcast player is only ~30-50 px
tall once 1920 is letterboxed to 896; --crop-pair puts the pair at the size the model expects.

**--joint-vs-solo**: the follow-up that null pointed at. Arm 1 co-decodes every player in one
forward pass; arm 2 runs one pass per player, so no query can attend to the other. Same weights,
same image, same boxes -- the only variable is the cross-person attention. Arm 2 is the regime
our own path runs in (GVHMRPoseEstimator calls HMRBackend.estimate_bodies per track), so this is
the mechanism question #132 actually poses, without needing SMPLest-X's 8 GB pod checkpoint.

Writes <out>/ab_f<N>[_crop][_solo].jpg plus a 4x zoom on the overlapping pair.
"""
import argparse
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
# SAM is pulled from the hub; keep it with the rest of the weights, not in ~/.cache.
os.environ.setdefault('HF_HOME', str(REPO / 'models/hf'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--tracks', default='out/phmr_ab/tracks.npz')
parser.add_argument('--calib', default='calib/Colombia-1-0-Congo-DR1080p.npz')
parser.add_argument('--backend-dir', default='backends/PromptHMR', help='the code checkout')
parser.add_argument('--weights', default='models/prompthmr',
                    help='the weights bundle (holds data/)')
parser.add_argument('--out-dir', default='out/phmr_ab')
parser.add_argument('--frame', type=int, default=None,
                    help='frame to run; default = the strongest crossing in tracks.npz')
parser.add_argument('--sam', default='facebook/sam-vit-base')
parser.add_argument('--crop-pair', action='store_true',
                    help='crop to the crossing pair first, so the players fill the 896 input')
parser.add_argument('--crop-margin', type=float, default=1.0,
                    help='margin around the pair, as a fraction of their union box')
parser.add_argument('--joint-vs-solo', action='store_true',
                    help='swap the arms: co-decode all players in one pass vs one pass each')
args = parser.parse_args()

import cv2  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

tracks_npz = np.load(REPO / args.tracks, allow_pickle=True)
ranking = tracks_npz['ranking']
frame_idx = args.frame if args.frame is not None else int(ranking[0][0])
track_ids = tracks_npz['track_ids']
t_frames, t_boxes = tracks_npz['frames'], tracks_npz['boxes']

present = []
for tid, frames, boxes in zip(track_ids, t_frames, t_boxes, strict=True):
    hit = np.where(np.asarray(frames) == frame_idx)[0]
    if hit.size:
        present.append((int(tid), np.asarray(boxes)[hit[0]].astype(float)))
if not present:
    sys.exit(f'no tracks on frame {frame_idx}')
print(f'frame {frame_idx}: {len(present)} tracked players')

cap = cv2.VideoCapture(str(REPO / args.clip))
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ok, frame_bgr = cap.read()
cap.release()
if not ok:
    sys.exit(f'could not read frame {frame_idx}')
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
H, W = frame_rgb.shape[:2]
principal = (W / 2.0, H / 2.0)  # the solve stores no principal point; the frame centre is it


def box_iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    ab = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return float(inter / (ab - inter))


if args.crop_pair:
    # Full-frame is the wrong regime for this question: letterboxing 1920 -> 896 leaves a
    # broadcast player ~30-50 px tall and his 256-px mask prompt ~10 px, so the prompt carries
    # almost no shape. Cropping to the crossing pair first restores the scale the model expects.
    pair = max(((i, j) for i in range(len(present)) for j in range(i + 1, len(present))),
               key=lambda p: box_iou(present[p[0]][1], present[p[1]][1]))
    a, b = present[pair[0]][1], present[pair[1]][1]
    ux0, uy0 = min(a[0], b[0]), min(a[1], b[1])
    ux1, uy1 = max(a[2], b[2]), max(a[3], b[3])
    mx, my = (ux1 - ux0) * args.crop_margin, (uy1 - uy0) * args.crop_margin
    cx0, cy0 = int(max(0, ux0 - mx)), int(max(0, uy0 - my))
    cx1, cy1 = int(min(W, ux1 + mx)), int(min(H, uy1 + my))

    frame_rgb = frame_rgb[cy0:cy1, cx0:cx1]
    principal = (principal[0] - cx0, principal[1] - cy0)  # NOT the crop centre
    kept = []
    for tid, bx in present:
        shifted = bx.copy()
        shifted[[0, 2]] -= cx0
        shifted[[1, 3]] -= cy0
        inside = shifted[0] >= 0 and shifted[1] >= 0
        if inside and shifted[2] <= cx1 - cx0 and shifted[3] <= cy1 - cy0:
            kept.append((tid, shifted))
    present = kept
    H, W = frame_rgb.shape[:2]
    print(f'crop [{cx0},{cy0}]-[{cx1},{cy1}] = {W}x{H}, {len(present)} whole players; '
          f'tallest {max(b[3] - b[1] for _t, b in present):.0f} px '
          f'-> {max(b[3] - b[1] for _t, b in present) * 896 / max(W, H):.0f} px at 896')

# ---- masks: SAM prompted by each tracked box, so identity comes from ByteTrack ----------
print(f'segmenting with {args.sam} ...', flush=True)
from transformers import SamModel, SamProcessor  # noqa: E402

sam_proc = SamProcessor.from_pretrained(args.sam)
sam = SamModel.from_pretrained(args.sam).eval()

# All boxes go in one call so the image encoder runs once, not once per player.
inputs = sam_proc(Image.fromarray(frame_rgb),
                  input_boxes=[[b[:4].tolist() for _t, b in present]], return_tensors='pt')
with torch.no_grad():
    sam_out = sam(**inputs, multimask_output=False)
masks_t = sam_proc.image_processor.post_process_masks(
    sam_out.pred_masks.cpu(), inputs['original_sizes'].cpu(),
    inputs['reshaped_input_sizes'].cpu())[0]
full_masks = [masks_t[i, 0].numpy().astype(bool) for i in range(len(present))]
print(f'  {sum(int(m.sum()) for m in full_masks)} mask px over {len(full_masks)} players')

# ---- PromptHMR image model -------------------------------------------------------------
# Upstream hardcodes its asset paths relative to cwd ('data/pretrain/...', 'data/body_models/...'),
# so run from the weights bundle and import the code from wherever the checkout lives.
sys.path.insert(0, str(REPO / args.backend_dir))
os.chdir(REPO / args.weights)

pl = types.ModuleType('pytorch_lightning')
pl.LightningModule = nn.Module
sys.modules['pytorch_lightning'] = pl


def _mea(q, k, v, attn_bias=None, p=0.0, scale=None):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, scale=scale)
    return o.transpose(1, 2)


xf = types.ModuleType('xformers')
xo = types.ModuleType('xformers.ops')
xo.memory_efficient_attention = _mea
xo.unbind = lambda x, dim=0: torch.unbind(x, dim)
xf.ops = xo
sys.modules['xformers'] = xf
sys.modules['xformers.ops'] = xo

sd = torch.load('data/pretrain/phmr/checkpoint.ckpt', map_location='cpu',
                weights_only=True)['state_dict']

import open_clip  # noqa: E402, I001
_rc = open_clip.create_model
open_clip.create_model = lambda name, pretrained=None, **kw: _rc(name, **kw)

import prompt_hmr.models.phmr as phmr_mod  # noqa: E402


class _StubSMPL(nn.Module):
    def __init__(self, *a, **k):
        super().__init__()
        self.vertex_joint_selector = nn.Module()
        for key, v in sd.items():
            if not key.startswith('smpl.'):
                continue
            owner, _, leaf = key[5:].rpartition('.')
            (getattr(self, owner) if owner else self).register_buffer(leaf, v.clone())

    def joints_from_vertices(self, verts):
        return torch.einsum('jv,bvc->bjc', self.J_regressor, verts)


phmr_mod.SMPL = _StubSMPL

from prompt_hmr.core.config import parse_args  # noqa: E402
from prompt_hmr.models import build_phmr  # noqa: E402

print('building PromptHMR ...', flush=True)
model = build_phmr(parse_args(['--cfg', 'data/pretrain/phmr/config.yaml'])).eval()
model.load_state_dict(sd, strict=False)
model.is_train = False

calib = np.load(REPO / args.calib)
focal = float(np.asarray(calib['focal']).reshape(-1)[0])
if args.crop_pair:
    # The solved 4169 px focal describes the 1920-wide frame; on a 370 px window that is a ~5
    # degree FOV, and the model -- which has never seen one -- puts every body off-canvas.
    # Substitute a plain ~53 degree pinhole centred on the crop. Depth from this run is
    # therefore not metric, which costs nothing: world placement comes from our own field
    # homography, so all this A/B asks is whether the mesh sits on its own player's pixels.
    print(f'crop intrinsics: focal {max(W, H)} px replaces the solved {focal:.1f} px '
          '-- depth is not metric here, by design')
    focal, principal = float(max(W, H)), (W / 2.0, H / 2.0)
print(f'camera: focal {focal:.1f} px, principal point {principal}')

K = torch.eye(3)
K[0, 0] = K[1, 1] = focal
K[0, 2], K[1, 2] = principal

IMG = 896
MSK = int(IMG / 14 * 4)


def to_padded(mask_bool, size=MSK):
    """Contain+pad exactly as pad_image does to the frame, so mask and image stay registered."""
    im = Image.fromarray(mask_bool.astype(np.uint8) * 255)
    im = ImageOps.pad(ImageOps.contain(im, (size, size)), size=(size, size))
    return np.array(im).astype(np.float32) / 255.0


from torchvision.transforms import Compose, Normalize, ToTensor  # noqa: E402

norm = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


def pad_image(item, IMG_SIZE=IMG):
    """Verbatim port of pipeline/phmr_vid.py:pad_image, so importing their package (and its
    omegaconf/antlr4 chain) is not needed just to letterbox a frame."""
    img = item['image_cv']
    size = np.array([img.shape[1], img.shape[0]])
    scale = IMG_SIZE / max(size)
    offset = (IMG_SIZE - scale * size) / 2

    pil = ImageOps.pad(ImageOps.contain(Image.fromarray(img), (IMG_SIZE, IMG_SIZE)),
                       size=(IMG_SIZE, IMG_SIZE))
    item['image_cv'] = np.array(pil)
    item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
    item['cam_int'][:, :2] *= scale
    item['cam_int'][:, :2, -1] += torch.from_numpy(offset).float()
    item['boxes'] *= scale
    item['boxes'][:, :2] += torch.from_numpy(offset).float()
    item['boxes'][:, 2:4] += torch.from_numpy(offset).float()
    return item

boxes = np.stack([np.concatenate([b[:4], [1.0]]) for _t, b in present])
item = {
    'boxes': torch.from_numpy(boxes).float(),
    'cam_int': K[None].repeat(len(present), 1, 1).float(),
    'image_cv': frame_rgb.copy(),
    'track_ids': [t for t, _b in present],
    'masks': torch.from_numpy(np.stack([to_padded(m) for m in full_masks])).float()[:, None],
}
item = pad_image(item, IMG_SIZE=IMG)
item['image'] = norm(item['image_cv'])
padded_rgb = np.array(item['image_cv'])
item['image_cv'] = torch.tensor(item['image_cv'])

def forward(it, use_mask):
    with torch.no_grad():
        return model([it], mask_prompt=use_mask, kpt_prompt=False, text_prompt=False)[0]


def solo(it):
    """One forward pass per player, so no query can attend to the other one.

    Same image, same boxes, same weights, same `interaction` flag as the joint pass -- the only
    thing removed is who else is in the batch. That is the regime our own pose path runs in:
    GVHMRPoseEstimator calls HMRBackend.estimate_bodies per track, so crossing players are
    never co-decoded.
    """
    outs = []
    for i in range(len(present)):
        one = dict(it)
        one['boxes'] = it['boxes'][i:i + 1]
        one['masks'] = it['masks'][i:i + 1]
        one['track_ids'] = [it['track_ids'][i]]
        outs.append(forward(one, False))
    return {'vertices': torch.cat([o['vertices'] for o in outs])}


if args.joint_vs_solo:
    # SMPLDecoder's BUDDI-style cross-person attention is gated by batch['interaction'], and
    # upstream's own inference leaves it unset -- so a plain multi-person pass is N independent
    # decodes and joint-vs-solo comes out bit-identical. Turn it on for BOTH arms.
    item['interaction'] = True
    arms = (('joint', lambda: forward(item, False)), ('solo', lambda: solo(item)))
else:
    arms = (('masks', lambda: forward(item, True)), ('boxes', lambda: forward(item, False)))
left, right = arms[0][0], arms[1][0]

runs = {}
for label, run_arm in arms:
    print(f'forward, {label} ...', flush=True)
    runs[label] = run_arm()

# ---- overlay + silhouette agreement ----------------------------------------------------
Kp = item['cam_int'][0].numpy()
palette = [(255, 80, 80), (80, 200, 255), (120, 255, 120), (255, 220, 80),
           (255, 120, 255), (120, 255, 230), (255, 160, 60), (200, 160, 255)]

ref_masks = [to_padded(m, IMG) > 0.5 for m in full_masks]

panels, scores = {}, {}
for label, out in runs.items():
    canvas = padded_rgb.copy()
    verts = out['vertices'].numpy()
    sil_scores = []
    for i, (_tid, _b) in enumerate(present):
        v = verts[i]
        uv = (Kp @ (v / np.maximum(v[:, 2:3], 1e-6)).T).T[:, :2]
        u, vv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        keep = (u >= 0) & (u < IMG) & (vv >= 0) & (vv < IMG)
        sil = np.zeros((IMG, IMG), bool)
        sil[vv[keep], u[keep]] = True
        sil = cv2.dilate(sil.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        colour = palette[i % len(palette)]
        canvas[sil] = (0.45 * np.array(colour) + 0.55 * canvas[sil]).astype(np.uint8)
        inter = float((sil & ref_masks[i]).sum())
        union = float((sil | ref_masks[i]).sum())
        sil_scores.append(inter / union if union else 0.0)
    for i, (tid, _b) in enumerate(present):
        bx = item['boxes'][i].numpy()
        cv2.rectangle(canvas, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])),
                      palette[i % len(palette)], 1)
        cv2.putText(canvas, str(tid), (int(bx[0]), max(10, int(bx[1]) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, palette[i % len(palette)], 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA)
    panels[label], scores[label] = canvas, sil_scores

print('\nmesh-vs-own-mask IoU (higher = mesh stays on its own player)')
print(f'  track   {left:>5}   {right:>5}   delta')
for i, (tid, _b) in enumerate(present):
    m, b = scores[left][i], scores[right][i]
    print(f'  {tid:5d}   {m:.3f}   {b:.3f}   {m - b:+.3f}')
mm, bb = float(np.mean(scores[left])), float(np.mean(scores[right]))
print(f'  mean    {mm:.3f}   {bb:.3f}   {mm - bb:+.3f}')

out_dir = REPO / args.out_dir
out_dir.mkdir(parents=True, exist_ok=True)
side = np.concatenate([panels[left], panels[right]], axis=1)
tag = f'f{frame_idx}' + ('_crop' if args.crop_pair else '') \
    + ('_solo' if args.joint_vs_solo else '')
dst = out_dir / f'ab_{tag}.jpg'
cv2.imwrite(str(dst), cv2.cvtColor(side, cv2.COLOR_RGB2BGR))
print(f'\nwrote {dst}')

# Zoom on the overlapping pair: at broadcast scale a full-frame verdict is not trustworthy.
pb = item['boxes'].numpy()
best, pair = -1.0, None
for i in range(len(pb)):
    for j in range(i + 1, len(pb)):
        x0, y0 = max(pb[i][0], pb[j][0]), max(pb[i][1], pb[j][1])
        x1, y1 = min(pb[i][2], pb[j][2]), min(pb[i][3], pb[j][3])
        inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if inter > best:
            best, pair = inter, (i, j)

if pair is not None and best > 0:
    i, j = pair
    x0 = int(max(0, min(pb[i][0], pb[j][0]) - 40))
    y0 = int(max(0, min(pb[i][1], pb[j][1]) - 40))
    x1 = int(min(IMG, max(pb[i][2], pb[j][2]) + 40))
    y1 = int(min(IMG, max(pb[i][3], pb[j][3]) + 40))
    crops = [cv2.resize(p[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
             for p in (panels[left], panels[right])]
    zoom = np.concatenate(crops, axis=1)
    zdst = out_dir / f'ab_{tag}_zoom.jpg'
    cv2.imwrite(str(zdst), cv2.cvtColor(zoom, cv2.COLOR_RGB2BGR))
    ids = (present[i][0], present[j][0])
    print(f'wrote {zdst}  (tracks {ids[0]} & {ids[1]}, overlap {best:.0f} px^2)')
