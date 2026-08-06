"""Does per-crop pose fuse on overlapping players? — A2 again, over a SAMPLE instead of one crop.

`smplestx_occlusion.py` scores one pair and draws a panel. It was used on 2026-08-05 to conclude
"the pose half of #132 does not reproduce" — from frame 87, tracks 15 and 85, a case I had myself
measured as mild. Counted afterwards, **10.7 % of this clip's player crops carry >20 % of another
player's box and 76 % of frames hold at least one**, so that was one sample out of 423 and the
verdict did not survive (see `occlusion-pose-research-2026-08-04.md`, 2026-08-06 note).

    .venv/bin/python scripts/smplestx_occlusion_sweep.py --pairs 40 --min-cover 0.20

This runs the same production path over a stratified sample and reports a **distribution**:

* **own-mask IoU** — is the mesh on the right person at all.
* **cross-contamination** — fraction of player A's mesh silhouette landing inside B's mask. This is
  the fusion #132 names; one number per pair, so it can be summarised honestly.
* **fused** — a crisp per-pair verdict: a mesh lands *more* on the other player's mask than on its
  own. That is what "спекаются" means, made countable.

Masks come from SAM and are **checked against kit colour before being trusted**: opposing kits mean
the two masks are verifiably different men. Same-kit pairs are scored but reported apart, because
there the mask check cannot tell a fused pair from a correct one — the trap that made frame 124
look like a crossing when it was one player boxed twice.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--tracks', default='out/phmr_ab/tracks_split.npz')
parser.add_argument('--last-frame', type=int, default=235, help='shot 1 ends at 235')
parser.add_argument('--pairs', type=int, default=40)
parser.add_argument('--min-cover', type=float, default=0.20,
                    help='fraction of the BACK player\'s box the front one must cover')
parser.add_argument('--repo', default='backends/SMPLest-X')
parser.add_argument('--ckpt', default='smplest_x_h_slim')
parser.add_argument('--device', default='cpu')
parser.add_argument('--threads', type=int, default=6)
parser.add_argument('--sam', default='facebook/sam-vit-base')
parser.add_argument('--out', default='out/phmr_ab/occlusion_sweep.json')
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

import torch  # noqa: E402

torch.set_num_threads(args.threads)
if args.device == 'cpu':
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

z = np.load(REPO / args.tracks, allow_pickle=True)
W, H = int(z['width']), int(z['height'])
per_frame: dict[int, list] = {}
for tid, frames, boxes in zip(z['track_ids'], z['frames'], z['boxes'], strict=True):
    for f, b in zip(np.asarray(frames).tolist(), np.asarray(boxes), strict=True):
        b = np.asarray(b, float)[:4]
        if f <= args.last_frame and 25 < b[3] - b[1] < 0.45 * H and b[2] - b[0] < 0.30 * W:
            per_frame.setdefault(int(f), []).append((int(tid), b))


def overlap(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


# Candidate pairs, ranked by how much of the BACK player the front one covers. Feet lower in the
# frame = nearer the raised camera, so that one is in front.
cands = []
for f, items in sorted(per_frame.items()):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            (ti, bi), (tj, bj) = items[i], items[j]
            inter = overlap(bi, bj)
            if inter <= 0:
                continue
            back, front = ((ti, bi), (tj, bj)) if bi[3] < bj[3] else ((tj, bj), (ti, bi))
            area = (back[1][2] - back[1][0]) * (back[1][3] - back[1][1])
            cov = inter / area if area > 0 else 0.0
            if cov >= args.min_cover:
                cands.append((f, front[0], back[0], float(cov)))
print(f'{len(cands)} candidate pairs at cover >= {args.min_cover} in frames 0-{args.last_frame}')
if not cands:
    raise SystemExit('nothing to measure')

# Stratify across the cover range rather than taking the worst N: the question is what happens on
# a typical contaminated crop, and the extremes are already known to be duplicates or near-misses.
rng = np.random.default_rng(args.seed)
order = np.argsort([c[3] for c in cands])
picks = sorted({int(i) for i in np.linspace(0, len(order) - 1, min(args.pairs, len(order)))})
sample = [cands[order[i]] for i in picks]
print(f'sampling {len(sample)} of them, spread over cover '
      f'{sample[0][3]:.2f}-{sample[-1][3]:.2f}')

print(f'\nloading SMPLest-X ({args.device}) ...', flush=True)
from pitch3d.adapters.models.smplestx_backend import SMPLestXBackend  # noqa: E402

backend = SMPLestXBackend(repo_dir=str(REPO / args.repo), ckpt_name=args.ckpt, device=args.device)
backend._load()
cfg = backend._cfg


class Tap:
    """Snoop the model's full output while `_infer_crop` — the production path — drives it."""

    def __init__(self, model):
        self.model, self.last = model, None

    def __call__(self, *a, **k):
        self.last = self.model(*a, **k)
        return self.last


tap = Tap(backend._model)
backend._model = tap
sys.path.insert(0, str(REPO / args.repo))
from PIL import Image  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402
from utils.data_utils import process_bbox  # noqa: E402

proc = SamProcessor.from_pretrained(args.sam)
sam = SamModel.from_pretrained(args.sam).eval()
FACES = backend._smplx_layer().faces.astype(np.int32)


def silhouette(rgb, box):
    """Run the production crop path and rasterise the posed mesh into full-frame pixels.

    SMPLest-X predicts in a virtual 5000 px camera over ``input_body_shape``; mapping it back onto
    the *processed* box is what puts the mesh in the frame's own pixels (upstream
    `main/inference.py:139-143`). Getting this wrong silently moves every mesh, so it is written
    once, here.
    """
    out = backend._infer_crop(torch, rgb, box, W, H)
    if out is None:
        return None
    mesh = tap.last['smplx_mesh_cam'].detach().cpu().numpy()[0]
    x1, y1, x2, y2 = box
    bb = process_bbox(np.array([x1, y1, abs(x2 - x1), abs(y2 - y1)], dtype=np.float32),
                      W, H, cfg.model.input_img_shape, getattr(cfg.data, 'bbox_ratio', 1.25))
    if bb is None:
        return None
    fx = cfg.model.focal[0] / cfg.model.input_body_shape[1] * bb[2]
    fy = cfg.model.focal[1] / cfg.model.input_body_shape[0] * bb[3]
    cx = cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bb[2] + bb[0]
    cy = cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bb[3] + bb[1]
    uv = np.stack([mesh[:, 0] / mesh[:, 2] * fx + cx, mesh[:, 1] / mesh[:, 2] * fy + cy], 1)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, np.round(uv[FACES]).astype(np.int32), 1)
    return sil.astype(bool)


def sam_mask(rgb, box):
    inp = proc(Image.fromarray(rgb), input_boxes=[[box.tolist()]], return_tensors='pt')
    with torch.no_grad():
        o = sam(**inp, multimask_output=False)
    return proc.image_processor.post_process_masks(
        o.pred_masks.cpu(), inp['original_sizes'].cpu(),
        inp['reshaped_input_sizes'].cpu())[0][0, 0].numpy().astype(bool)


def kit_of(frame_bgr, region):
    px = frame_bgr[region] if region.dtype == bool else None
    if px is None or px.size == 0:
        return '?'
    bgr = px.reshape(-1, 3).mean(0)
    if bgr[0] > 110 and bgr[0] > bgr[2]:
        return 'BLU'
    if bgr[2] > 120 and bgr[1] > 110:
        return 'YEL'
    return '?'


cap = cv2.VideoCapture(str(REPO / args.clip))
rows = []
out_path = REPO / args.out
out_path.parent.mkdir(parents=True, exist_ok=True)
print('\n frame  front  back  cover   IoU_f  IoU_b   cross_f  cross_b  kits      verdict',
      flush=True)
for f, tf, tb, cov in sample:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
    ok, frame_bgr = cap.read()
    if not ok:
        continue
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    bf = next(b for t, b in per_frame[f] if t == tf)
    bb_ = next(b for t, b in per_frame[f] if t == tb)
    sf, sb = silhouette(rgb, bf), silhouette(rgb, bb_)
    if sf is None or sb is None:
        continue
    mf, mb = sam_mask(rgb, bf), sam_mask(rgb, bb_)
    kf, kb = kit_of(frame_bgr, mf), kit_of(frame_bgr, mb)

    def iou(a, b):
        u = int((a | b).sum())
        return float((a & b).sum() / u) if u else 0.0

    nf, nb = int(sf.sum()), int(sb.sum())
    own_f = float((sf & mf).sum() / nf) if nf else 0.0
    oth_f = float((sf & mb).sum() / nf) if nf else 0.0
    own_b = float((sb & mb).sum() / nb) if nb else 0.0
    oth_b = float((sb & mf).sum() / nb) if nb else 0.0
    fused = (oth_f > own_f) or (oth_b > own_b)
    same_kit = kf == kb
    row = dict(frame=int(f), front=int(tf), back=int(tb), cover=cov,
               iou_front=iou(sf, mf), iou_back=iou(sb, mb),
               own_front=own_f, cross_front=oth_f, own_back=own_b, cross_back=oth_b,
               kit_front=kf, kit_back=kb, same_kit=bool(same_kit), fused=bool(fused))
    rows.append(row)
    verdict = 'FUSED' if fused else 'separate'
    if same_kit:
        verdict += ' (same kit — mask check void)'
    print(f'{f:6d} {tf:6d} {tb:5d}  {cov:.2f}  {row["iou_front"]:6.3f} {row["iou_back"]:6.3f}   '
          f'{oth_f:6.3f}  {oth_b:6.3f}  {kf}/{kb}  {verdict}', flush=True)
    out_path.write_text(json.dumps(rows, indent=1))
cap.release()

if not rows:
    raise SystemExit('no pairs scored')
opp = [r for r in rows if not r['same_kit'] and '?' not in (r['kit_front'], r['kit_back'])]
print(f'\nscored {len(rows)} pairs; {len(opp)} with verifiable opposing kits')
for label, sel in (('all pairs', rows), ('opposing kits only', opp)):
    if not sel:
        continue
    cross = np.array([max(r['cross_front'], r['cross_back']) for r in sel])
    own = np.array([min(r['own_front'], r['own_back']) for r in sel])
    print(f'  {label:20} cross-contamination median {np.median(cross):.3f}  p90 '
          f'{np.percentile(cross, 90):.3f}  max {cross.max():.3f}')
    print(f'  {" " * 20} worst own-mask share  median {np.median(own):.3f}  '
          f'min {own.min():.3f}')
    print(f'  {" " * 20} FUSED: {sum(r["fused"] for r in sel)}/{len(sel)} pairs')
print(f'\nwrote {out_path}')
