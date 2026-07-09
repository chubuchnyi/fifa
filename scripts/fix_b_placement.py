#!/usr/bin/env python
"""Repair variant B (SAM 3D Body) root placement — task #59.

Root cause (measured 2026-07-09): B's ``pose.transl`` was stored as
``(foot_pixel_x, foot_pixel_y, pelvis_height_m)`` — the foot XY never went
through the pitch homography, so the GUI (which projects world→pixel through
the camera) sends every body off-screen. B's stored ``camera`` was also a bad
solve (translation ~[867,-281,-392] vs A's sane [-34,2.9,73]).

Since A and B are the SAME video, they share one camera + calibration. This
script rebuilds B's scene to be A-consistent:

  1. adopt A's ``camera`` and ``field`` (same clip → same calibration),
  2. re-ground every B subject's feet the way A does — the stored foot pixels
     back through ``calibration.image_to_world`` (identical path to
     ``GVHMRPoseEstimator._ground_root``), keeping B's per-frame pelvis height.

This repairs the *artifact* losslessly and locally (no pod). The durable
pipeline fix is to have the SAM-3D backend feed ``_ground_root`` with the good
camera on the next pod run; documented in docs/STATUS.md.

Idempotent: foot pixels are always read from a pristine ``scene.json.orig``.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
A_PATH = ROOT / "poseannot/clips/A_smplestx/scene.json"
B_PATH = ROOT / "poseannot/clips/B_sam3dbody/scene.json"

# pitch half-extents for the sanity report only
HALF_L, HALF_W = 55.0, 38.0


def main() -> None:
    from pitch3d.core.scene.serialization import load_scene, save_scene

    a_scene = load_scene(str(A_PATH))

    orig = B_PATH.with_suffix(".json.orig")
    if not orig.exists():
        shutil.copy(B_PATH, orig)
        print(f"backed up pristine B scene → {orig.name}")
    b_scene = load_scene(str(orig))  # always re-ground from pristine foot pixels

    # 1. adopt A's camera + calibration (same video → same real camera)
    b_scene.camera = copy.deepcopy(a_scene.camera)
    b_scene.field = copy.deepcopy(a_scene.field)
    cal = a_scene.field.calibration

    # 2. re-ground each subject's feet through the homography (A's _ground_root path)
    off = 0
    total = 0
    for sub in b_scene.subjects:
        pose = sub.proposal.pose
        frames = np.asarray(pose.frames).reshape(-1)
        foot_px = np.asarray(pose.transl, dtype=float)[:, :2].copy()
        world_xy = np.stack(
            [cal.image_to_world(int(f), foot_px[i])[0] for i, f in enumerate(frames)]
        )
        pose.transl[:, 0] = world_xy[:, 0]
        pose.transl[:, 1] = world_xy[:, 1]
        # transl[:, 2] (pelvis height) kept as B computed it
        on = (np.abs(world_xy[:, 0]) < HALF_L) & (np.abs(world_xy[:, 1]) < HALF_W)
        total += on.size
        off += int((~on).sum())
        print(
            f"  tid={sub.track_id:>3}  world x[{world_xy[:,0].min():6.1f},{world_xy[:,0].max():6.1f}] "
            f"y[{world_xy[:,1].min():6.1f},{world_xy[:,1].max():6.1f}]  on-pitch {int(on.sum())}/{on.size}"
        )

    save_scene(b_scene, str(B_PATH))
    print(f"\nwrote repaired B scene → {B_PATH}")
    print(f"on-pitch foot samples: {total - off}/{total} ({100*(total-off)/total:.0f}%)")


if __name__ == "__main__":
    main()
