"""Camera-plane sanity check — the camera-vs-pose bug separator.

Projects the STANDARD pitch line markings (known world geometry, completely
independent of any pose) through the *exact same* frame_projector /
project_points path the pose overlay uses, and draws them on the real video
frame.  This cleanly forks the debugging:

  * pitch lines land ON the painted white lines  ->  camera / flip / intrinsics
    are correct, so any overlay misalignment is a POSE bug (transl/orient/scale).
  * pitch lines land wrong                        ->  the camera path is the bug;
    fix that before touching pose.

It also prints where the centre mark (0,0,0) and the four pitch corners land,
and how many markings fall inside the frame, so the verdict is objective and
not only visual.

Usage:
    python scripts/debug/camera_plane_check.py --scene out/bakeoff/scene_A.json \
        --frames 0 16 --out out/bakeoff
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from poseannot.camera import frame_projector, project_points
from poseannot.config import load as load_cfg
from poseannot.scene_state import build_scene_state
from poseannot.video import frame_size, read_frame

from pitch3d.core.scene.pitch import pitch_line_world_points

DEFAULT_VIDEO = "samples/video/Colombia-1-0-Congo-DR1080p.mp4"


def _load(scene: str, video: str):
    base = load_cfg()
    cfg = replace(
        base,
        scene_json=Path(scene).resolve(),
        source_video=Path(video).resolve(),
        corrections_out=Path("/tmp/pose_probe_noedits.json").resolve(),
    )
    st = build_scene_state(cfg)
    W, H = frame_size(str(cfg.source_video))
    return st, cfg, W, H


def check(scene: str, video: str, frame: int, out_dir: str) -> None:
    st, cfg, W, H = _load(scene, video)
    field = st.scene.field
    proj = frame_projector(st.scene.camera, frame, video_size=(W, H))

    lines = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5)
    uv = project_points(lines, proj)
    good = np.isfinite(uv).all(axis=1)
    inb = good & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)

    # Named anchors so the verdict is readable, not just a dot cloud.
    L = float(field.dimensions.length) / 2.0
    Wd = float(field.dimensions.width) / 2.0
    anchors = {
        "centre": (0.0, 0.0),
        "corner+ +": (L, Wd), "corner+ -": (L, -Wd),
        "corner- +": (-L, Wd), "corner- -": (-L, -Wd),
    }
    apts = np.array([[x, y, field.plane_z] for x, y in anchors.values()], dtype=float)
    auv = project_points(apts, proj)

    print(
        f"\n=== camera-plane  {Path(scene).name}  frame {frame}  {W}x{H}  "
        f"flip={proj.frame_flipped} ==="
    )
    print(f"  pitch markings: {int(good.sum())}/{len(lines)} in front, {int(inb.sum())} inside frame")
    for (name, _), (u, v) in zip(anchors.items(), auv):
        tag = "" if (np.isfinite(u) and 0 <= u < W and 0 <= v < H) else "  <-- off/behind"
        print(f"    {name:<9} -> ({u:8.0f},{v:8.0f}){tag}")

    bgr = read_frame(video, frame)
    for (u, v), ok in zip(uv, inb):
        if ok:
            cv2.circle(bgr, (int(u), int(v)), 2, (0, 255, 0), -1)
    for (u, v) in auv:
        if np.isfinite(u):
            cv2.drawMarker(bgr, (int(u), int(v)), (0, 0, 255),
                           cv2.MARKER_TILTED_CROSS, 22, 2)
    banner = f"{Path(scene).stem} f{frame} pitch-lines inframe={int(inb.sum())} flip={proj.frame_flipped}"
    cv2.putText(bgr, banner, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(bgr, banner, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    fn = outp / f"camplane_{Path(scene).stem}_f{frame}.png"
    cv2.imwrite(str(fn), bgr)
    print(f"  wrote {fn}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="out/bakeoff")
    args = ap.parse_args()
    for fr in args.frames:
        check(args.scene, args.video, fr, args.out)


if __name__ == "__main__":
    main()
