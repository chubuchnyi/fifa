#!/usr/bin/env python3
"""What does Global MPJPE actually tell us about a video judged by eye? (R7, #99)

The research briefs spec Global MPJPE 0.35-0.45 m and say "do not spec below the broadcast
envelope". This measures whether that number can rank the errors *we* care about. It cannot:
below, three error fields with a Global MPJPE identical to five decimals range from invisible
at a novel viewpoint to ruinous.

Run::

    PYTHONPATH=src .venv/bin/python scripts/bench_novel_view_metric.py

Three experiments:

``envelope``
    Three error fields pinned to the same Global MPJPE — a wrong camera, a wobbling camera, and
    per-player scatter — scored with :mod:`pitch3d.eval.novel_view`. The headline result.

``instrument``
    The metric's own bias, measured rather than assumed. A per-frame rigid fit has 6 degrees of
    freedom; with few players it can absorb genuine per-player error and flatter us. Sweeps the
    subject count to find where the number becomes trustworthy, and checks our real clip against
    that (``poseannot/clips/A_smplestx/scene.json`` carries 21 subjects).

``smoother``
    The baseline any future temporal smoother must beat. Sweeps Gaussian sigma and reports mean
    local MPJPE against the top-speed-decile value, so the trade the yaw low-pass made — 90% of
    the jitter gone, real 100-degree turns flattened with it — is visible as a number instead of
    being rediscovered by eye.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as np  # noqa: E402

from pitch3d.core.correction.engine import smooth_vector  # noqa: E402
from pitch3d.eval.metrics import mpjpe_global  # noqa: E402
from pitch3d.eval.novel_view import (  # noqa: E402
    decompose_global_error,
    local_mpjpe_by_speed,
)
from pitch3d.eval.synthetic import CAMERA_VIEWS, generate_scene  # noqa: E402

TARGET_M = 0.40  # mid-point of the briefs' 0.35-0.45 m envelope
FPS = 29.97      # the target clip
CLIP_SUBJECTS = 21


def _scene(n_subjects: int = 12, n_frames: int = 60, seed: int = 0):
    return generate_scene(
        n_subjects=n_subjects, n_frames=n_frames, seed=seed, camera=CAMERA_VIEWS["main_sideline"]
    )


def _scaled_to(pred: np.ndarray, gt: np.ndarray, target: float) -> np.ndarray:
    """Rescale an error field so its Global MPJPE is exactly ``target`` — the fair comparison."""
    err = pred - gt
    return gt + err * (target / mpjpe_global(pred, gt))


def _yaw(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _fields(gt: np.ndarray, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Three error fields, each normalised to the same Global MPJPE."""
    t = gt.shape[0]

    wrong_camera = gt @ _yaw(2.0).T + np.array([0.6, -0.3, 0.02])

    wobble = np.zeros((t, 1, 1, 3))
    wobble[:, 0, 0, :2] = np.column_stack(
        [np.sin(np.linspace(0, 6 * np.pi, t)), np.cos(np.linspace(0, 4 * np.pi, t))]
    )
    wobbling_camera = gt + wobble

    direction = rng.normal(size=(1, gt.shape[1], 1, 3))
    scatter = gt + direction / np.linalg.norm(direction, axis=-1, keepdims=True)

    return {
        name: _scaled_to(pred, gt, TARGET_M)
        for name, pred in [
            ("wrong camera (static)", wrong_camera),
            ("wobbling camera", wobbling_camera),
            ("per-player scatter", scatter),
        ]
    }


def envelope() -> None:
    print("== three error fields the briefs' envelope cannot tell apart ==")
    gt = _scene().joints_world
    rng = np.random.default_rng(0)

    rows = []
    for name, pred in _fields(gt, rng).items():
        g = decompose_global_error(pred, gt)
        rows.append((name, g))
        assert abs(g["global_mpjpe_m"] - TARGET_M) < 1e-9

    print(f"  every row below has Global MPJPE = {TARGET_M:.3f} m, inside the briefs' envelope\n")
    head = f"  {'error field':<24}{'static fit':>11}{'per-frame':>11}{'swim':>9}{'absorbed':>10}"
    print(head)
    print("  " + "-" * (len(head) - 2))
    for name, g in rows:
        print(
            f"  {name:<24}{g['after_static_camera_m']:>11.3f}{g['after_perframe_camera_m']:>11.3f}"
            f"{g['scene_swim_m']:>9.3f}{g['camera_absorbed_frac'] * 100:>9.1f}%"
        )
    print("\n  static fit = best single camera re-placement; per-frame = best camera per frame")
    print("  (R7 headline); swim = the difference, i.e. common-mode error that MOVES, which a")
    print("  viewer sees as the scene sliding under a locked-off shot.")
    print("  Read: one number, three verdicts — free, visible-as-swim, and fatal.")


def instrument(trials: int = 40) -> None:
    print("\n== the metric's own bias: when does the fit start absorbing REAL player error? ==")
    print(f"  pure per-player scatter at {TARGET_M:.2f} m — an honest metric must absorb 0%.")
    print(f"  {trials} random scatter draws per row; one draw is far too noisy to read.\n")
    print(f"  {'subjects':>9}{'falsely absorbed':>18}{'  (mean +- sd)':<16}{'worst draw':>12}")
    print("  " + "-" * 57)

    for n in (2, 3, 5, 8, 12, CLIP_SUBJECTS):
        gt = _scene(n_subjects=n, n_frames=40, seed=2).joints_world
        rng = np.random.default_rng(100 + n)
        leaked = []
        for _ in range(trials):
            direction = rng.normal(size=(1, n, 1, 3))
            pred = _scaled_to(
                gt + direction / np.linalg.norm(direction, axis=-1, keepdims=True), gt, TARGET_M
            )
            leaked.append(decompose_global_error(pred, gt)["camera_absorbed_frac"] * 100)
        flag = "   <- our clip" if n == CLIP_SUBJECTS else ""
        print(
            f"  {n:>9}{np.mean(leaked):>13.1f}% +- {np.std(leaked):<12.1f}"
            f"{max(leaked):>11.1f}%{flag}"
        )
    print("\n  The fit has 6 DOF, so with few bodies it passes real scatter off as camera error.")
    print("  The leak falls with subject count but does NOT vanish, and the worst-draw column is")
    print("  the one to plan against: at our clip's 21 subjects a single bad configuration can")
    print("  still hide a fifth of a genuine per-player error. Read the R7 residual as a LOWER")
    print("  bound on what a viewer sees, never as an unbiased estimate.")


def _with_turns(gt: np.ndarray, rng, degrees: float = 120.0, span: int = 8) -> np.ndarray:
    """Give each subject one fast turn — the motion a low-pass silently removes.

    The synthetic fixture's articulation is a pure sinusoid, which no smoother can damage; real
    football is sinusoid *plus* transients, and the transients are the whole argument.
    """
    t = gt.shape[0]
    out = gt.copy()
    for n in range(gt.shape[1]):
        start = int(rng.integers(span, t - 2 * span))
        ramp = np.clip((np.arange(t) - start) / span, 0.0, 1.0) * np.radians(degrees)
        c, s = np.cos(ramp), np.sin(ramp)
        rot = np.zeros((t, 3, 3))
        rot[:, 0, 0] = rot[:, 1, 1] = c
        rot[:, 0, 1], rot[:, 1, 0], rot[:, 2, 2] = -s, s, 1.0
        root = gt[:, n, :1, :]
        out[:, n] = root + np.einsum("tij,tkj->tki", rot, gt[:, n] - root)
    return out


def _sweep(gt: np.ndarray, noisy: np.ndarray, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'sigma':>7}{'local MPJPE':>14}{'top decile':>13}{'penalty':>10}")
    print("  " + "-" * 44)
    shape = gt.shape
    for s in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
        pred = (
            noisy
            if s == 0.0
            else smooth_vector(noisy.reshape(shape[0], -1), 9, "gaussian", s).reshape(shape)
        )
        g = local_mpjpe_by_speed(pred, gt, fps=FPS)
        print(
            f"  {s:>7.1f}{g['local_mpjpe_m']:>14.4f}{g['local_mpjpe_top_decile_m']:>13.4f}"
            f"{g['top_decile_penalty']:>10.2f}"
        )


def smoother() -> None:
    print("\n== the baseline a future temporal smoother has to beat ==")
    print("  per-frame jitter sigma=0.03 m, Gaussian smoothing over a 9-frame window")
    rng = np.random.default_rng(4)
    smooth_gt = _scene(n_subjects=8, n_frames=120, seed=3).joints_world
    turning_gt = _with_turns(smooth_gt, rng)

    for gt, label in (
        (smooth_gt, "A. smooth motion only (the fixture's sinusoid)"),
        (turning_gt, "B. same motion plus one 120-degree turn per player over 8 frames"),
    ):
        _sweep(gt, gt + rng.normal(0, 0.03, gt.shape), label)

    print("\n  A says smoothing is free; B says it is not, and only the top-decile column can")
    print("  tell them apart. Gate: a proposed smoother must beat the best B row on the TOP")
    print("  DECILE. Comparing on the mean is what let the yaw low-pass ship.")


if __name__ == "__main__":
    envelope()
    instrument()
    smoother()
