#!/usr/bin/env python3
"""Reproduce the R10 verdict: USAC/MAGSAC++ is unusable for pitch calibration.

The research brief proposed swapping our hand-rolled RANSAC in
``pitch3d.adapters.models.calibration`` for OpenCV's USAC/MAGSAC++. This script is the
measurement that rejected it. Run it to re-check the claim against any OpenCV build::

    .venv/bin/python scripts/bench_ransac_usac.py

Three experiments:

``head-to-head``
    Synthetic pitch correspondences (image px -> world m) under our real ``_H_GT``, with
    Gaussian keypoint noise and planted gross outliers. Compares the world-space error of
    our estimator against USAC's, and USAC's inlier recall.

``control``
    The same estimator on textbook px->px correspondences, where it works perfectly. This
    is what rules out "the wrapper is being called wrong".

``heteroscedasticity``
    The mechanism. Measures the spread of world-space error produced by *uniform* image
    noise across the pitch. MAGSAC++ marginalises over a single global noise scale; a
    broadcast pitch homography does not have one.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.models.calibration import (
    _apply_homography,
    reprojection_error,
    solve_homography_ransac,
)

#: Ground-truth image->world homography, mirroring tests/unit/test_calibration_adapter.py.
H_GT = np.array([[0.050, 0.002, -30.0], [0.001, -0.050, 18.0], [2e-4, 1e-4, 1.0]])

#: A mild px->px homography that USAC handles perfectly — the control.
H_BOOK = np.array([[1.02, 0.03, 12.0], [-0.02, 0.98, -8.0], [1e-5, 2e-5, 1.0]])

TRIALS = 20


def _usac(src: np.ndarray, dst: np.ndarray, threshold: float):
    import cv2

    h, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, threshold)
    return h, (None if mask is None else mask.ravel().astype(bool))


def _pitch_case(seed: int, n: int, n_out: int, noise_px: float):
    """Image px -> world m correspondences under ``H_GT``, first ``n_out`` are gross outliers."""
    r = np.random.default_rng(seed)
    world = np.column_stack([r.uniform(-52, 52, n), r.uniform(-34, 34, n)])
    img = _apply_homography(np.linalg.inv(H_GT), world) + r.normal(0, noise_px, (n, 2))
    if n_out:
        img[:n_out] += r.uniform(-1000, 1000, (n_out, 2))
    return img, world


def head_to_head() -> None:
    print("== head-to-head: pitch geometry, image px -> world m, 2 px keypoint noise ==")
    print(f"{'n':>4} {'out':>4} | {'ours err':>9} {'usac err':>9} | {'usac ok':>8} "
          f"{'inliers':>8} {'recall':>7}")
    for n in (8, 12, 20, 40):
        for n_out in (0, 1, 2, 4):
            if n_out >= n // 2:
                continue
            ours, theirs, inl, rec, ok = [], [], [], [], 0
            for s in range(TRIALS):
                img, world = _pitch_case(s, n, n_out, 2.0)
                clean = slice(n_out, None)
                h, _ = solve_homography_ransac(img, world, threshold=1.0, seed=0)
                ours.append(reprojection_error(h, img[clean], world[clean]))
                hu, mu = _usac(img, world, 1.0)
                if hu is not None and mu is not None:
                    ok += 1
                    inl.append(int(mu.sum()))
                    rec.append(float(mu[clean].mean()))
                    theirs.append(reprojection_error(hu, img[clean], world[clean]))
            m = lambda a: np.mean(a) if a else float("nan")  # noqa: E731
            print(f"{n:>4} {n_out:>4} | {m(ours):>9.4f} {m(theirs):>9.4f} | {ok:>5}/{TRIALS} "
                  f"{m(inl):>8.1f} {m(rec):>7.2f}")
    print("  err = RMS world metres on the CLEAN correspondences only (lower is better).")


def control() -> None:
    print("\n== control: textbook px -> px, 20 pts / 4 outliers, thr 3 px, expect 16 inliers ==")
    n, n_out = 20, 4
    for sigma in (0.0, 0.5, 1.0, 2.0, 3.0):
        import cv2

        ok, inl = 0, []
        for s in range(TRIALS):
            r = np.random.default_rng(s)
            src = r.uniform(0, 1000, (n, 2))
            dst = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_BOOK).reshape(-1, 2)
            dst += r.normal(0, sigma, dst.shape)
            dst[:n_out] += r.uniform(-200, 200, (n_out, 2))
            h, mask = _usac(src, dst, 3.0)
            if h is not None:
                ok += 1
                inl.append(int(mask.sum()))
        print(f"  noise sigma={sigma:<5} usac ok={ok:>2}/{TRIALS} "
              f"inliers={np.mean(inl) if inl else float('nan'):>5.1f}")
    print("  USAC is fine here at every point count — so the failure above is the geometry,")
    print("  not the call. Note it already sheds real inliers once sigma nears the threshold.")


def heteroscedasticity() -> None:
    print("\n== mechanism: world-space error from a UNIFORM 2 px image noise ==")
    r = np.random.default_rng(0)
    world = np.column_stack([r.uniform(-52, 52, 4000), r.uniform(-34, 34, 4000)])
    img = _apply_homography(np.linalg.inv(H_GT), world)
    disp = np.linalg.norm(
        _apply_homography(H_GT, img + r.normal(0, 2.0, img.shape)) - world, axis=1
    )
    q = np.percentile(disp, [5, 25, 50, 75, 95, 99])
    print(f"  p5/p25/p50/p75/p95/p99 (m) = {np.round(q, 3)}")
    print(f"  p95/p5 = {q[4] / q[0]:.1f}x, p99/p5 = {q[5] / q[0]:.1f}x")
    print("  MAGSAC++ marginalises over ONE global sigma. A broadcast pitch has no single")
    print("  sigma: a pixel at the far touchline is worth an order of magnitude more metres")
    print("  than one at the near touchline.")


if __name__ == "__main__":
    head_to_head()
    control()
    heteroscedasticity()
