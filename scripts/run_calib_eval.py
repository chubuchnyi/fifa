"""Run the calibration bake-off → pitch-line reprojection grid (JSON).

Command-line front door to :mod:`pitch3d.eval` calibration scoring — the calibration analogue of
``scripts/run_pose_eval.py``. It answers "how well does our :class:`FieldCalibrator` register the
pitch on *independent* real broadcast frames?", the B1 question that ``samples/video/clip.mp4``
(faint markings, zero landmarks) could not.

Two datasets:

* ``--dataset synthetic`` — runnable **now**, no asset/GPU. Generates synthetic frames plus their
  *true* image→world homographies (:func:`pitch3d.eval.datasets_soccernet.synthetic_calib_frames`)
  and scores either the true homography (``--backend oracle`` → reprojection ≈ 0, a self-test) or a
  perturbed one (``--backend perturb`` → error grows with ``--perturb-sigma``). Proves the harness +
  metric end to end with one command.
* ``--dataset soccernet`` — the real benchmark. Loads a SoccerNet ``calibration-2023`` split
  (``--frames-dir``; openly downloadable via ``scripts/get_soccernet_calibration.py``), builds a
  directory clip, runs a real ``FieldCalibrator`` (``--backend`` dotted path ``pkg.mod:make``,
  ADR-0006) on a CUDA box, and scores its per-frame homographies against the pitch-line GT. Each
  SoccerNet image is an independent view, so run the calibrator with temporal smoothing off.

Examples::

    # self-test the whole path, here and now:
    PYTHONPATH=src python scripts/run_calib_eval.py --dataset synthetic --backend oracle

    # the real number (on a box with the data + the wired calibrator):
    PYTHONPATH=src python scripts/run_calib_eval.py --dataset soccernet \
        --frames-dir /workspace/SoccerNet/calibration-2023/test --limit 200 \
        --backend pitch3d.adapters.models.pnlcalib_backend:make
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from pitch3d.eval.calib_metrics import (
    evaluate_calibration,
    format_sweep_table,
    summarize_threshold_sweep,
)
from pitch3d.eval.datasets_soccernet import (
    as_clip,
    load_calib_dir,
    synthetic_calib_frames,
)


def _perturb(homographies: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Add relative Gaussian noise to each homography (renormalising ``H[2, 2]``) — a probe."""
    rng = np.random.default_rng(seed)
    out = homographies * (1.0 + sigma * rng.standard_normal(homographies.shape))
    return np.stack([h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h for h in out])


def _parse(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibration bake-off → reprojection grid (JSON).")
    p.add_argument("--dataset", choices=["synthetic", "soccernet"], default="synthetic")
    p.add_argument("--backend", default="oracle",
                   help="synthetic: 'oracle'/'perturb'; soccernet: dotted 'pkg.mod:make'")
    p.add_argument("--thresholds", default="5,10",
                   help="comma-separated pixel thresholds for line-accuracy (default 5,10)")
    # synthetic knobs
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--perturb-sigma", type=float, default=0.02)
    # soccernet knobs
    p.add_argument("--frames-dir", help="SoccerNet calibration split dir (<id>.jpg + <id>.json)")
    p.add_argument("--limit", type=int, default=None, help="cap number of frames (quick smoke)")
    p.add_argument("--min-lines", type=int, default=4,
                   help="keep frames with >= this many straight pitch lines (SoccerNet uses > 4)")
    p.add_argument("--device", default="cuda", help="calibrator inference device (soccernet mode)")
    # PnLCalib heatmap-gate overrides (the completeness lever). Single values override the backend's
    # env defaults; --threshold-sweep runs several kp gates and tabulates completeness vs accuracy.
    p.add_argument("--kp-threshold", type=float, default=None,
                   help="override PnLCalib keypoint heatmap gate (lower → more frames, noisier)")
    p.add_argument("--line-threshold", type=float, default=None,
                   help="override PnLCalib line heatmap gate")
    p.add_argument("--threshold-sweep", default=None,
                   help="comma-separated kp thresholds to sweep, e.g. '0.3434,0.25,0.15'")
    return p.parse_args(argv)


def _apply_threshold_env(kp_threshold: float | None, line_threshold: float | None) -> None:
    """Push explicit threshold overrides into the env the dotted backend reads (PnLCalib seam)."""
    if kp_threshold is not None:
        os.environ["PNLCALIB_KP_THRESHOLD"] = str(kp_threshold)
    if line_threshold is not None:
        os.environ["PNLCALIB_LINE_THRESHOLD"] = str(line_threshold)


def _calibrate_and_score(
    frames: list, frames_dir: str, spec: str, device: str, thresholds: tuple[float, ...]
) -> dict:
    """Resolve the dotted calibrator, run it over the SoccerNet clip, and score the homographies."""
    clip = as_clip(frames, frames_dir)
    calibrator = _resolve_calibrator(spec, device)
    print(f"[soccernet] calibrating {clip.n_frames} frames from {frames_dir}", file=sys.stderr)
    fc = calibrator.calibrate(clip)
    return evaluate_calibration(
        frames, fc.homographies, confidence=fc.confidence, thresholds_px=thresholds
    )


def main(argv: list[str] | None = None) -> dict:
    args = _parse(argv)
    thresholds = tuple(float(t) for t in args.thresholds.split(",") if t.strip())

    if args.dataset == "synthetic":
        if args.threshold_sweep:
            raise SystemExit("--threshold-sweep needs --dataset soccernet (no backend to sweep)")
        frames, true_h = synthetic_calib_frames(n_frames=args.frames, seed=args.seed)
        if args.backend == "oracle":
            homographies = true_h
        elif args.backend == "perturb":
            homographies = _perturb(true_h, args.perturb_sigma, args.seed)
        else:
            raise SystemExit("synthetic dataset takes --backend oracle|perturb (no real images)")
        grid = evaluate_calibration(frames, homographies, confidence=None, thresholds_px=thresholds)
        print(json.dumps({
            "dataset": args.dataset, "backend": args.backend,
            "n_frames": grid["n_frames"], "grid": grid,
        }, indent=2))
        return grid

    if not args.frames_dir:
        raise SystemExit("--frames-dir is required for --dataset soccernet")
    frames = load_calib_dir(args.frames_dir, min_lines=args.min_lines, limit=args.limit)
    if not frames:
        raise SystemExit(f"no annotated frames with >= {args.min_lines} lines in {args.frames_dir}")

    if args.threshold_sweep:
        kp_values = [float(t) for t in args.threshold_sweep.split(",") if t.strip()]
        if not kp_values:
            raise SystemExit("--threshold-sweep needs comma-separated kp thresholds")
        rows: list[tuple[float, dict]] = []
        for kp_th in kp_values:
            _apply_threshold_env(kp_th, args.line_threshold)
            rows.append(
                (kp_th, _calibrate_and_score(frames, args.frames_dir, args.backend,
                                             args.device, thresholds))
            )
        summary = summarize_threshold_sweep(rows)
        print(format_sweep_table(summary), file=sys.stderr)
        payload = {
            "dataset": "soccernet", "backend": args.backend,
            "threshold_sweep": summary, "grids": [g for _, g in rows],
        }
        print(json.dumps(payload, indent=2))
        return payload

    _apply_threshold_env(args.kp_threshold, args.line_threshold)
    grid = _calibrate_and_score(frames, args.frames_dir, args.backend, args.device, thresholds)
    print(json.dumps({
        "dataset": "soccernet", "backend": args.backend,
        "n_frames": grid["n_frames"], "grid": grid,
    }, indent=2))
    return grid


def _resolve_calibrator(spec: str, device: str):
    """Wrap a dotted ``KeypointBackend`` (e.g. PnLCalib) in the robust DLT calibrator.

    Mirrors the pipeline's ``--calibrator keypoints --calibrator-backend <spec>`` seam (ADR-0006):
    the dotted path resolves to a :class:`KeypointBackend` (per-frame pitch landmarks), which
    :class:`KeypointFieldCalibrator` turns into the per-frame image→world homography via RANSAC +
    confidence-weighted DLT. Temporal smoothing is **off** (``smooth_window=1``) because each
    SoccerNet image is an independent broadcast view, not a contiguous clip.
    """
    from pitch3d.adapters.models import KeypointFieldCalibrator
    from pitch3d.adapters.models.calibration import KeypointBackend
    from pitch3d.app.wiring import _resolve_backend

    return KeypointFieldCalibrator(
        backend=_resolve_backend(spec, KeypointBackend), smooth_window=1, device=device
    )


if __name__ == "__main__":
    main()
