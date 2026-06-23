"""Run the pose bake-off on a dataset and print the Global/Local MPJPE grid (metres) as JSON.

This is the command-line front door to :mod:`pitch3d.eval` — the harness made executable so the
"first honest accuracy number" is one command away once a real backend + GT are in hand.

Two datasets:

* ``--dataset synthetic`` — runnable **now**, no asset/GPU. Scores a backend against the
  deterministic synthetic GT (``pitch3d.eval.generate_scene``). With the built-in ``--backend
  oracle`` it is a self-test (Condition A ≈ 0); ``--backend zero`` prints the Local-MPJPE floor.
* ``--dataset 3dpw`` — the real-GT benchmark (``pitch3d.eval.datasets_3dpw``). Needs the
  downloaded 3DPW sequence pickle (``--pkl``) and, for a meaningful number, a real SMPL-X
  ``--backend`` (dotted path, ADR-0006) on a box with the SMPL-X asset (``--joint-model smplx``).
  3DPW has a moving camera and no pitch plane → **Condition A only**.

Examples::

    # self-test the whole path, here and now:
    PYTHONPATH=src python scripts/run_pose_eval.py --dataset synthetic --backend oracle

    # the real number (on a box with the data + SMPL-X asset + the wired backend):
    PYTHONPATH=src python scripts/run_pose_eval.py --dataset 3dpw \
        --pkl /data/3dpw/sequenceFiles/test/downtown_walking_00.pkl \
        --images /data/3dpw/imageFiles/downtown_walking_00 \
        --joint-model smplx \
        --backend pitch3d.adapters.models.smplestx_backend:make
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from pitch3d.adapters.models.pose import HMRBackend, RawBodyMotion
from pitch3d.app.wiring import _resolve_backend
from pitch3d.eval.bodymodel import PlaceholderJointModel, SmplxJointModel
from pitch3d.eval.dataset import evaluate_dataset
from pitch3d.eval.datasets_3dpw import diagnose_3dpw_scene, load_3dpw_sequence
from pitch3d.eval.synthetic import generate_scene


class _OracleBackend:
    """Built-in oracle: replay the scene's GT articulation (synthetic → Condition A ≈ 0)."""

    def __init__(self, scene):
        self.scene = scene

    def estimate_bodies(self, clip, tracks):
        s = self.scene
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id, frames=s.frames,
                global_orient=s.gt_global_orient[:, n], body_pose=s.gt_body_pose[:, n],
                betas=s.gt_betas[n],
            )
            for n, tl in enumerate(tracks.tracklets)
        }


class _ZeroBackend:
    """Built-in floor: zero articulation (the Local-MPJPE sanity baseline)."""

    def __init__(self, scene):
        self.scene = scene

    def estimate_bodies(self, clip, tracks):
        s = self.scene
        p = s.joint_model.n_pose_joints
        return {
            tl.track_id: RawBodyMotion(
                track_id=tl.track_id, frames=s.frames,
                global_orient=np.zeros((s.n_frames, 3)), body_pose=np.zeros((s.n_frames, p, 3)),
                betas=np.zeros(10),
            )
            for tl in tracks.tracklets
        }


def _make_backend(spec: str, scene) -> HMRBackend:
    if spec == "oracle":
        return _OracleBackend(scene)
    if spec == "zero":
        return _ZeroBackend(scene)
    return _resolve_backend(spec, HMRBackend)


def _parse(argv):
    p = argparse.ArgumentParser(description="Pose bake-off → Global/Local MPJPE grid (JSON).")
    p.add_argument("--dataset", choices=["synthetic", "3dpw"], default="synthetic")
    p.add_argument("--backend", default="oracle",
                   help="'oracle'/'zero' (built-in) or a dotted path 'pkg.mod:Factory' (ADR-0006)")
    p.add_argument("--no-visible-only", dest="visible_only", action="store_false",
                   help="score ALL joints (default: visible-only, mirroring official evaluators)")
    # synthetic knobs
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, default=3)
    p.add_argument("--condition-b", action="store_true",
                   help="synthetic only: also score Condition B via the GT homography")
    # 3dpw knobs
    p.add_argument("--pkl", help="3DPW sequenceFiles/<seq>.pkl (required for --dataset 3dpw)")
    p.add_argument("--images", help="3DPW imageFiles/<seq> dir → the backend's clip URI")
    p.add_argument("--joint-model", choices=["placeholder", "smplx"], default="placeholder",
                   help="FK seam for the backend's params (use 'smplx' to score a SMPL-X backend)")
    p.add_argument("--stride", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse(argv)
    calibration = None
    if args.dataset == "synthetic":
        scene = generate_scene(seed=args.seed, n_frames=args.frames, n_subjects=args.subjects)
        if args.condition_b:
            calibration = scene.field_calibration()
    else:
        if not args.pkl:
            raise SystemExit("--pkl is required for --dataset 3dpw")
        jm = SmplxJointModel() if args.joint_model == "smplx" else PlaceholderJointModel()
        scene = load_3dpw_sequence(args.pkl, args.images, joint_model=jm, stride=args.stride)
        print(f"[3dpw] {scene.source_id}: {diagnose_3dpw_scene(scene)}", file=sys.stderr)

    grid = evaluate_dataset(scene, _make_backend(args.backend, scene),
                            calibration=calibration, visible_only=args.visible_only)
    print(json.dumps({
        "dataset": args.dataset,
        "source_id": getattr(scene, "source_id", "synthetic"),
        "backend": args.backend,
        "n_frames": scene.n_frames, "n_subjects": scene.n_subjects,
        "visible_only": args.visible_only,
        "grid": grid,
    }, indent=2))
    return grid


if __name__ == "__main__":
    main()
