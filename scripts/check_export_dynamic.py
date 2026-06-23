"""Assert a `smplx_npz` export holds genuine *dynamic* poses, not T-pose / stub fills.

Reads every `subject_*.npz` in a dir and checks, per subject:
  - betas are non-zero            -> a person-specific shape was actually estimated
  - body_pose varies across frames -> real motion, not one frozen pose repeated
  - transl moves across frames     -> the subject travels across the pitch

Exits non-zero if any subject looks degenerate, so it can gate a real-model E2E run
(local or on a GPU pod). Single-frame exports skip the across-frame checks.

  python scripts/check_export_dynamic.py out/run/export/scene.smplx_npz
"""

import glob
import os
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "out/run/export/scene.smplx_npz"
files = sorted(glob.glob(os.path.join(d, "subject_*.npz")))
if not files:
    print(f"NO subject_*.npz in {d}")
    sys.exit(2)

bad = 0
for f in files:
    z = np.load(f)
    betas, body_pose, transl = z["betas"], z["body_pose"], z["transl"]
    n = int(body_pose.shape[0])
    betas_norm = float(np.linalg.norm(betas))
    pose_std = float(body_pose.reshape(n, -1).std(0).mean()) if n > 1 else 0.0
    transl_span = float(np.linalg.norm(transl.max(0) - transl.min(0))) if n > 1 else 0.0
    ok = betas_norm > 1e-3 and (n == 1 or (pose_std > 1e-4 and transl_span > 1e-3))
    bad += 0 if ok else 1
    print(
        f"  [{'OK ' if ok else 'BAD'}] {os.path.basename(f)}  frames={n}  "
        f"|betas|={betas_norm:.3f}  body_pose_std={pose_std:.4f}  transl_span={transl_span:.3f}m"
    )

print(f"{'ALL DYNAMIC' if bad == 0 else f'{bad} DEGENERATE'}  ({len(files)} subjects in {d})")
sys.exit(1 if bad else 0)
