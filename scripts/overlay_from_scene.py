"""Draw a scene.json's SMPL-X poses onto the real video frames it came from.

Reuses the *exact* GUI path so the standalone PNGs match what poseannot shows:
    pitch3d load_scene → poseannot FK (SMPL-X) → poseannot.camera projection
    (with the validated 180°-roll / camera-X-mirror auto-detect).

This is the honest apples-to-apples overlay for the A/B pose bake-off: feed it
scene_A.json (SMPLest-X) and scene_B.json (SAM 3D Body) with the same video and
frames, and compare the two skeleton overlays on identical pixels.

Usage:
    .venv/bin/python scripts/overlay_from_scene.py \
        --scene out/bakeoff/scene_A.json \
        --video samples/video/Colombia-1-0-Congo-DR1080p.mp4 \
        --frames 0,20,40 --label "A: SMPLest-X" --out out/bakeoff
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from poseannot.camera import frame_projector, project_points
from poseannot.config import PoseAnnotConfig
from poseannot.config import load as _load_cfg
from poseannot.scene_state import build_scene_state
from poseannot.video import frame_size

# SMPL-X body (pelvis + 21) — identical to poseannot/static/index.html BONES.
BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (9, 14), (14, 17), (17, 19), (19, 21),
]


def _read_frame(video: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {idx} from {video}")
    return bgr


def _cfg_for(scene: str, video: str) -> PoseAnnotConfig:
    from dataclasses import replace
    base = _load_cfg()
    return replace(base, scene_json=Path(scene).resolve(),
                   source_video=Path(video).resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="0,20,40")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="out/bakeoff")
    ap.add_argument("--prefix", default=None, help="output filename prefix (default: scene stem)")
    args = ap.parse_args()

    frames = [int(x) for x in args.frames.split(",") if x.strip() != ""]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or Path(args.scene).stem

    cfg = _cfg_for(args.scene, args.video)
    st = build_scene_state(cfg)
    W, H = frame_size(str(cfg.source_video))
    print(f"scene={args.scene} subjects={len(st.subjects)} video={W}x{H} frames={frames}")

    for f in frames:
        proj = frame_projector(st.scene.camera, f, video_size=(W, H))
        bgr = _read_frame(str(cfg.source_video), f)
        drawn = 0
        for tid, sub in st.subjects.items():
            hit = np.where(sub.frames == f)[0]
            if hit.size == 0:
                continue
            k = int(hit[0])
            pts = project_points(sub.joints[k], proj)  # (22, 2)
            valid = np.isfinite(pts).all(axis=1)
            for a, b in BONES:
                if valid[a] and valid[b]:
                    pa = (int(round(pts[a, 0])), int(round(pts[a, 1])))
                    pb = (int(round(pts[b, 0])), int(round(pts[b, 1])))
                    cv2.line(bgr, pa, pb, (0, 220, 0), 2)
            for j in range(pts.shape[0]):
                if valid[j]:
                    c = (int(round(pts[j, 0])), int(round(pts[j, 1])))
                    cv2.circle(bgr, c, 3, (0, 0, 255), -1)
            if valid[0]:
                cv2.putText(bgr, str(tid), (int(pts[0, 0]) + 4, int(pts[0, 1]) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            drawn += 1
        banner = f"{args.label}   frame {f}   {drawn} subjects   (flip={proj.frame_flipped})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(bgr, banner, (16, 34), font, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(bgr, banner, (16, 34), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        fn = out_dir / f"overlay_{prefix}_f{f}.png"
        cv2.imwrite(str(fn), bgr)
        print(f"  frame {f}: {drawn} subjects -> {fn}")


if __name__ == "__main__":
    main()
