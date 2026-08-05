"""What does OUR per-crop pose path do when one player is hidden behind another? — A2 of #132.

`track_continuity.py --kit-scan` says where two *different* men genuinely overlap. This runs the
production estimator there and measures whether the two meshes stay on their own players.

    .venv/bin/python scripts/smplestx_occlusion.py --frame 87 --pair 15 85

It is the control arm: no PromptHMR, no masks, nothing new — SMPLest-X Huge fed our ByteTrack
boxes, exactly as `--pose gvhmr --pose-backend ...smplestx_backend:make` runs it. Any later arm
has to beat these numbers to be worth its dependencies.

Three metrics, because own-mask IoU alone cannot tell a *bad fit* from a *fused* one:

* **own-mask IoU** — mesh silhouette against that player's own mask. "Is the mesh on the right
  person at all?"
* **cross-contamination** — the fraction of player A's mesh that lands inside player B's mask.
  This is the fusion #132 names: two crops both collapsing onto the front player barely dents
  metric 1 but sends this one towards 1.0.
* **depth order** — does the back player's solved root stay behind the front player's? A per-crop
  estimator has no reason to preserve it, and a flipped pair is visible in the render.

The masks are SAM's, box-prompted, and are **checked before they are trusted**: the two teams wear
light blue and yellow, so a mask whose own mean colour is not its player's kit is a mask of the
wrong man, and this script says so instead of quietly scoring against it.

SMPLest-X hardcodes `.cuda()` in its model, its transforms and its Tester. Rather than patch a
gitignored checkout, `--device cpu` makes those calls identity for the run.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / 'src'))

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--clip', default='samples/video/Colombia-1-0-Congo-DR1080p.mp4')
parser.add_argument('--tracks', default='out/phmr_ab/tracks.npz')
parser.add_argument('--frame', type=int, default=87)
parser.add_argument('--pair', type=int, nargs=2, default=(15, 85), metavar=('FRONT', 'BACK'))
parser.add_argument('--repo', default='backends/SMPLest-X')
parser.add_argument('--ckpt', default='smplest_x_h_slim',
                    help='the shipped 8.2 GB file is 2/3 optimizer state; _slim is network-only')
parser.add_argument('--device', default='cpu')
parser.add_argument('--threads', type=int, default=6,
                    help='a ViT-H on all 16 cores makes the machine unusable; cap it')
parser.add_argument('--sam', default='facebook/sam-vit-base')
parser.add_argument('--out', default='out/phmr_ab')
parser.add_argument('--zoom', type=int, default=5)
args = parser.parse_args()

import torch  # noqa: E402

torch.set_num_threads(args.threads)
if args.device == 'cpu':
    # Every blocking call is a `.cuda()` *method*, so neutralising the two methods covers the
    # model, utils/transforms.py and Tester alike. DataParallel already no-ops with no devices.
    torch.Tensor.cuda = lambda self, *a, **k: self
    torch.nn.Module.cuda = lambda self, *a, **k: self

z = np.load(REPO / args.tracks, allow_pickle=True)
box = {}
for tid, frames, boxes in zip(z['track_ids'], z['frames'], z['boxes'], strict=True):
    for f, b in zip(np.asarray(frames).tolist(), np.asarray(boxes), strict=True):
        box[(int(tid), int(f))] = np.asarray(b, float)[:4]

front_id, back_id = args.pair
boxes = {t: box[(t, args.frame)] for t in args.pair}

cap = cv2.VideoCapture(str(REPO / args.clip))
cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
ok, frame_bgr = cap.read()
cap.release()
if not ok:
    raise SystemExit(f'cannot read frame {args.frame}')
H, W = frame_bgr.shape[:2]
rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
print(f'frame {args.frame}  {W}x{H}   front {front_id}  back {back_id}')


def kit_of(mask_or_box):
    """Mean BGR over a region -> team. Light blue (Colombia) vs yellow (Congo DR)."""
    if mask_or_box.dtype == bool:
        px = frame_bgr[mask_or_box]
    else:
        x0, y0, x1, y1 = (int(v) for v in mask_or_box)
        h, w = y1 - y0, x1 - x0
        px = frame_bgr[y0 + int(0.20 * h):y0 + int(0.50 * h),
                       x0 + int(0.25 * w):x0 + int(0.75 * w)].reshape(-1, 3)
    if px.size == 0:
        return '?', np.zeros(3)
    bgr = px.reshape(-1, 3).mean(0)
    if bgr[0] > 110 and bgr[0] > bgr[2]:
        return 'BLU', bgr
    if bgr[2] > 120 and bgr[1] > 110:
        return 'YEL', bgr
    return '?', bgr


print('\nwho is who, from kit colour inside the box')
expect = {}
for t in args.pair:
    k, bgr = kit_of(boxes[t])
    expect[t] = k
    print(f'  track {t:3d}  box {np.round(boxes[t], 1)}  kit {k}  BGR {np.round(bgr, 0)}')
if expect[front_id] == expect[back_id]:
    print('  !! same kit -- this pair may be one player boxed twice; check before trusting scores')

print(f'\nloading SMPLest-X ({args.device}) ...', flush=True)
from pitch3d.adapters.models.smplestx_backend import SMPLestXBackend  # noqa: E402

backend = SMPLestXBackend(repo_dir=str(REPO / args.repo), ckpt_name=args.ckpt,
                          device=args.device)
backend._load()
cfg = backend._cfg


class Tap:
    """Snoop the model's full output while `_infer_crop` -- the production path -- drives it."""

    def __init__(self, model):
        self.model, self.last = model, None

    def __call__(self, *a, **k):
        self.last = self.model(*a, **k)
        return self.last


tap = Tap(backend._model)
backend._model = tap

sys.path.insert(0, str(REPO / args.repo))
from utils.data_utils import process_bbox  # noqa: E402

faces = None
result = {}
for t in args.pair:
    print(f'  inferring track {t} ...', flush=True)
    out = backend._infer_crop(torch, rgb, boxes[t], W, H)
    if out is None:
        raise SystemExit(f'degenerate box for track {t}')
    root, body, betas, pelvis_h = out
    mesh = tap.last['smplx_mesh_cam'].detach().cpu().numpy()[0]  # (10475,3), cam_trans applied

    x1, y1, x2, y2 = boxes[t]
    bb = process_bbox(np.array([x1, y1, abs(x2 - x1), abs(y2 - y1)], dtype=np.float32),
                      W, H, cfg.model.input_img_shape, getattr(cfg.data, 'bbox_ratio', 1.25))
    # SMPLest-X predicts in a virtual 5000 px camera over input_body_shape; map it back onto the
    # processed box, which is the frame's own pixels (main/inference.py:139-143).
    fx = cfg.model.focal[0] / cfg.model.input_body_shape[1] * bb[2]
    fy = cfg.model.focal[1] / cfg.model.input_body_shape[0] * bb[3]
    cx = cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bb[2] + bb[0]
    cy = cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bb[3] + bb[1]
    uv = np.stack([mesh[:, 0] / mesh[:, 2] * fx + cx, mesh[:, 1] / mesh[:, 2] * fy + cy], 1)

    if faces is None:
        faces = backend._smplx_layer().faces.astype(np.int32)
    sil = np.zeros((H, W), np.uint8)
    tri = np.round(uv[faces]).astype(np.int32)
    cv2.fillPoly(sil, tri, 1)
    result[t] = {'mesh': mesh, 'uv': uv, 'sil': sil.astype(bool), 'root': root,
                 'depth': float(np.median(mesh[:, 2])), 'pelvis_h': pelvis_h, 'body': body}
    print(f'    mesh depth {result[t]["depth"]:.2f} m, silhouette {int(sil.sum())} px')

print(f'\nsegmenting with {args.sam} ...', flush=True)
from PIL import Image  # noqa: E402
from transformers import SamModel, SamProcessor  # noqa: E402

proc = SamProcessor.from_pretrained(args.sam)
sam = SamModel.from_pretrained(args.sam).eval()
for t in args.pair:
    inp = proc(Image.fromarray(rgb), input_boxes=[[boxes[t].tolist()]], return_tensors='pt')
    with torch.no_grad():
        o = sam(**inp, multimask_output=False)
    m = proc.image_processor.post_process_masks(
        o.pred_masks.cpu(), inp['original_sizes'].cpu(),
        inp['reshaped_input_sizes'].cpu())[0][0, 0].numpy().astype(bool)
    k, bgr = kit_of(m)
    result[t]['mask'] = m
    result[t]['mask_kit'] = k
    verdict = 'ok' if k == expect[t] else f'WRONG MAN (expected {expect[t]})'
    print(f'  track {t:3d}  mask {int(m.sum()):6d} px  kit {k}  BGR {np.round(bgr, 0)}  {verdict}')

trust = all(result[t]['mask_kit'] == expect[t] for t in args.pair)
if not trust:
    print('  -> at least one mask is of the wrong player, so own-mask IoU below is NOT evidence.')


def iou(a, b):
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else 0.0


print('\n1. own-mask IoU -- is the mesh on the right person?')
for t in args.pair:
    print(f'  track {t:3d}  IoU {iou(result[t]["sil"], result[t]["mask"]):.3f}')

print('\n2. cross-contamination -- fraction of this mesh landing on the OTHER player')
for a, b in ((front_id, back_id), (back_id, front_id)):
    sil, other = result[a]['sil'], result[b]['mask']
    own = result[a]['mask']
    n = int(sil.sum())
    onto_other = float((sil & other).sum() / n) if n else 0.0
    onto_own = float((sil & own).sum() / n) if n else 0.0
    print(f'  mesh {a:3d} -> mask {b:3d}: {onto_other:.3f}   (onto its own mask {onto_own:.3f})')

print('\n3. depth order -- the back player must solve FARTHER than the front one')
d_front, d_back = result[front_id]['depth'], result[back_id]['depth']
ok_depth = d_back > d_front
print(f'  front {front_id} {d_front:6.2f} m   back {back_id} {d_back:6.2f} m   '
      f'{"held" if ok_depth else "FLIPPED"}  (gap {d_back - d_front:+.2f} m)')

pad = 30
x0 = max(0, int(min(boxes[t][0] for t in args.pair)) - pad)
y0 = max(0, int(min(boxes[t][1] for t in args.pair)) - pad)
x1 = min(W, int(max(boxes[t][2] for t in args.pair)) + pad)
y1 = min(H, int(max(boxes[t][3] for t in args.pair)) + pad)
panels = []
colour = {front_id: (0, 255, 0), back_id: (0, 165, 255)}
for title, draw_sil in (('frame', False), ('meshes', True)):
    tile = frame_bgr[y0:y1, x0:x1].copy()
    if draw_sil:
        for t in args.pair:
            edge = result[t]['sil'][y0:y1, x0:x1]
            tile[edge] = (0.45 * np.array(colour[t]) + 0.55 * tile[edge]).astype(np.uint8)
    for t in args.pair:
        b = boxes[t]
        cv2.rectangle(tile, (int(b[0]) - x0, int(b[1]) - y0),
                      (int(b[2]) - x0, int(b[3]) - y0), colour[t], 1)
    tile = cv2.resize(tile, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
    cv2.putText(tile, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    panels.append(tile)
out_png = Path(args.out) / f'a2_smplestx_f{args.frame}_{front_id}_{back_id}.png'
out_png.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(out_png), np.hstack(panels))
print(f'\nwrote {out_png}')
