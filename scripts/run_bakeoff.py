#!/usr/bin/env python3
"""Bake-off driver — candidate × condition → Global/Local MPJPE table.

Runs the procedure in ``docs/pose-bakeoff-runbook.md`` on the in-house synthetic oracle, so the
whole harness (FK → place → score) is exercised with **no GPU and no frames**. Real candidates
(SMPLest-X, SAM 3D Body) implement the same ``HMRBackend`` seam and slot into :data:`CANDIDATES`
on the box; until then the candidates are the GT oracle (methodology self-check — must score ~0)
and the zero-pose floor (the finite Local-MPJPE sanity baseline). Two conditions per candidate:

    A — GT camera; isolates pose-net / articulation quality.
    B — root grounded via our calibration. On synthetic the GT homography is the perfect-calib
        stand-in, so the A→B gap measured here is the *methodology floor* of foot-point grounding,
        NOT PnLCalib error (and it shows in Global MPJPE only — Local is root-relative).

The headline WorldPose numbers come from the challenge's official evaluator on the box (runbook §1);
this driver is the harness and the synthetic sanity pass.

Usage:
    python scripts/run_bakeoff.py [--seed S] [--subjects N] [--frames T] [--fk placeholder|smplx]
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence

from pitch3d.adapters.models.pose import HMRBackend
from pitch3d.eval.backends import GtOracleBackend, ZeroPoseBackend
from pitch3d.eval.bodymodel import JointModel, PlaceholderJointModel
from pitch3d.eval.harness import run_conditions
from pitch3d.eval.synthetic import CAMERA_VIEWS, SyntheticScene, generate_scene

#: name → factory(scene) → HMRBackend. Register real candidates here on the box (same seam).
CANDIDATES: dict[str, Callable[[SyntheticScene], HMRBackend]] = {
    "gt-oracle": GtOracleBackend,
    "zero-pose": ZeroPoseBackend,
}


def _fk(name: str) -> JointModel:
    """Resolve ``--fk`` to a JointModel (placeholder is asset-free; smplx needs the .npz)."""
    if name == "placeholder":
        return PlaceholderJointModel()
    if name == "smplx":
        from pitch3d.eval.bodymodel import SmplxJointModel

        return SmplxJointModel()
    raise SystemExit(f"unknown --fk {name!r} (expected 'placeholder' or 'smplx')")


def run_bakeoff(
    scene: SyntheticScene,
    candidates: Mapping[str, Callable[[SyntheticScene], HMRBackend]] = CANDIDATES,
    visible_only: bool = False,
) -> dict[str, dict[str, dict[str, float] | None]]:
    """candidate → ``{'A': grid, 'B': grid}`` over ``scene`` (B via the GT-homography stand-in).

    ``visible_only`` scores only joints the scene marks visible (occlusion-aware), mirroring the
    official evaluator's masking — most useful with a multi-subject / stacked-camera scene.
    """
    calibration = scene.field_calibration()
    return {
        name: run_conditions(
            scene, make(scene), calibration=calibration, visible_only=visible_only
        )
        for name, make in candidates.items()
    }


def format_table(grid: Mapping[str, Mapping[str, Mapping[str, float] | None]]) -> str:
    """Render the candidate × condition grid as a fixed-width metres table."""
    header = (
        f"{'candidate':<12} | {'A Global':>9} {'A Local':>9} | "
        f"{'B Global':>9} {'B Local':>9}   (metres)"
    )
    lines = [header, "-" * len(header)]

    def cell(g: Mapping[str, float] | None, key: str) -> str:
        return f"{g[key]:>9.4f}" if g is not None else f"{'—':>9}"

    for name, conds in grid.items():
        a, b = conds["A"], conds["B"]
        lines.append(
            f"{name:<12} | {cell(a, 'global_mpjpe_m')} {cell(a, 'local_mpjpe_m')} | "
            f"{cell(b, 'global_mpjpe_m')} {cell(b, 'local_mpjpe_m')}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pose bake-off driver (synthetic conditions A/B).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, default=3)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--fk", default="placeholder", choices=["placeholder", "smplx"])
    ap.add_argument("--camera", default="main_sideline", choices=list(CAMERA_VIEWS))
    ap.add_argument(
        "--visible-only",
        action="store_true",
        help="score only joints the scene marks visible (occlusion-aware)",
    )
    args = ap.parse_args(argv)

    scene = generate_scene(
        n_subjects=args.subjects,
        n_frames=args.frames,
        seed=args.seed,
        joint_model=_fk(args.fk),
        camera=CAMERA_VIEWS[args.camera],
    )
    grid = run_bakeoff(scene, visible_only=args.visible_only)
    print(
        f"# pose bake-off — synthetic (seed={args.seed}, N={args.subjects}, T={args.frames}, "
        f"FK={args.fk}, cam={args.camera}, visible_only={args.visible_only}); metres, no Procrustes"
    )
    print(format_table(grid))
    print(
        "\nA = GT camera (pose-net) · B = foot-point grounding via GT homography (calib stand-in)."
        "\nReal SMPLest-X / SAM-3D-Body register into CANDIDATES on the box — see "
        "docs/pose-bakeoff-runbook.md."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
