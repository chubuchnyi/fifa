"""Run the SMPLest-X HMRBackend on REAL broadcast frames through our seam.

Decodes a few frames of a real clip, gets real person boxes via our RF-DETR adapter (COCO
base, person=1), builds one pseudo-tracklet (largest person/frame), then calls
SMPLestXBackend.estimate_bodies — exercising _iter_frames decode + the per-tracklet loop +
RawBodyMotion assembly on real pixels. No GT, so this validates WIRING, not MPJPE.

Clip comes from $PITCH3D_CLIP (default /workspace/clip.mp4). Run on the pod:
    python scripts/validate_smplestx_real.py
"""

import os

import cv2
import numpy as np

os.environ.setdefault("PITCH3D_SMPLESTX_REPO", "/workspace/repos/SMPLest-X")
os.environ.setdefault("PITCH3D_DEVICE", "cuda")

from pitch3d.adapters.models.detection import RFDETRDetector  # noqa: E402
from pitch3d.adapters.models.smplestx_backend import make  # noqa: E402
from pitch3d.core.ports.io import ClipRef  # noqa: E402
from pitch3d.core.ports.perception import Tracklet, Tracks  # noqa: E402

VIDEO = os.environ.get("PITCH3D_CLIP", "/workspace/clip.mp4")

cap = cv2.VideoCapture(VIDEO)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

frames = np.arange(0, min(8, n_total))
clip = ClipRef(source_id="iniesta", uri=VIDEO, frames=frames, width=width, height=height, fps=fps)
print(f"clip: {width}x{height} @ {fps:.1f}fps, {n_total} frames; processing {frames.tolist()}")

# Real person boxes via RF-DETR COCO base (person class id = 1).
det = RFDETRDetector(class_map={1: "player"}, score_threshold=0.4)
dets = det.detect(clip)
fr, bb = [], []
for fd in dets.frames:
    persons = [d for d in fd.items if d.cls == "player"]
    if not persons:
        continue
    big = max(
        persons,
        key=lambda d: (d.bbox_xyxy[2] - d.bbox_xyxy[0]) * (d.bbox_xyxy[3] - d.bbox_xyxy[1]),
    )
    fr.append(fd.frame)
    bb.append(big.bbox_xyxy)
print(f"frames with a detected person: {fr}")
if not fr:
    raise SystemExit("no persons detected — cannot validate pose on real frames")

tracks = Tracks(
    tracklets=[Tracklet(track_id=1, frames=np.array(fr), bboxes_xyxy=np.array(bb), cls="player")],
    teams=[],
)

backend = make()
out = backend.estimate_bodies(clip, tracks)
for tid, rbm in out.items():
    print(f"track {tid}: {rbm.frames.shape[0]} frames {rbm.frames.tolist()}")
    print(f"  global_orient {rbm.global_orient.shape}  body_pose {rbm.body_pose.shape}  betas {rbm.betas.shape}")
    print(f"  betas[:5]       {np.round(rbm.betas[:5], 3).tolist()}")
    print(f"  global_orient[0] {np.round(rbm.global_orient[0], 3).tolist()}")
    print(f"  body_pose[0,:2]  {np.round(rbm.body_pose[0, :2].ravel(), 3).tolist()}")
print("REAL_RUN_OK")
