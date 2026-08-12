"""Pitch-keypoint field calibrator — third real adapter (M1, FR-7).

Self-hosted default for `FieldCalibrator`: a pitch-keypoint model + homography solve +
temporal smoothing, behind the optional ``cv`` extra. Split like the detector so the *logic*
is testable with **no model, no cv2, no GPU**:

* :class:`KeypointFieldCalibrator` — the **pure** half: given the backend's per-frame
  ``image↔world`` landmark correspondences, it solves the image→world homography with a
  numpy **normalized DLT** (Hartley normalisation + SVD), scores each frame by reprojection
  error, carries the last good homography through under-detected frames, and applies optional
  temporal smoothing. Numpy only; unit-tested via an injected backend.
* :class:`PitchKeypointBackend` — the **heavy** half: runs the landmark model on the frames.
  All model/cv2 imports are lazy, so importing this module never pulls the heavy stack; it
  raises an actionable error if the ``cv`` extra is missing.

Swap it in via ``default_ports(calibrator="keypoints")`` (wiring) — one fake replaced at a
time, satisfying the very same ``FieldCalibrator`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import FieldCalibrator
from ...core.scene.field import FieldCalibration
from ...core.scene.provenance import Backend, ModelInfo, impl_name


@dataclass
class FrameKeypoints:
    """Backend output for one frame: matched pitch landmarks in image and world coords.

    Optionally also carries **point-on-line** observations: image points known to lie on a named
    pitch line but not at any identifiable position along it. A line detector produces these on
    frames where the keypoint head finds few intersections, so they are exactly the extra evidence
    that matters when the correspondence count is thin (see :func:`solve_homography`).

    Attributes:
        frame: Source frame index.
        image_uv: ``(K, 2)`` detected landmark positions, image px.
        world_xy: ``(K, 2)`` the landmarks' known pitch-plane coords (metres, ``Z = 0``).
        confidence: ``(K,)`` per-landmark detection confidence in ``[0, 1]`` (defaults to ones).
        line_uv: ``(M, 2)`` image points lying on a known pitch line, or ``None``.
        line_abc: ``(M, 3)`` each point's world line ``(a, b, c)``, ``a² + b² == 1``, or ``None``.
        line_confidence: ``(M,)`` per-observation confidence (defaults to ones when lines present).
    """

    frame: int
    image_uv: np.ndarray
    world_xy: np.ndarray
    confidence: np.ndarray | None = None
    line_uv: np.ndarray | None = None
    line_abc: np.ndarray | None = None
    line_confidence: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.image_uv = np.asarray(self.image_uv, dtype=float).reshape(-1, 2)
        self.world_xy = np.asarray(self.world_xy, dtype=float).reshape(-1, 2)
        k = self.image_uv.shape[0]
        if self.world_xy.shape[0] != k:
            raise ValueError(
                f"ragged keypoints at frame {self.frame}: "
                f"{k} image points, {self.world_xy.shape[0]} world points"
            )
        if self.confidence is None:
            self.confidence = np.ones(k, dtype=float)
        else:
            self.confidence = np.asarray(self.confidence, dtype=float).reshape(-1)
            if self.confidence.shape[0] != k:
                raise ValueError(f"ragged keypoint confidence at frame {self.frame}")

        if self.line_uv is None and self.line_abc is None:
            return
        if self.line_uv is None or self.line_abc is None:
            raise ValueError(f"line_uv and line_abc must be given together at frame {self.frame}")
        self.line_uv = np.asarray(self.line_uv, dtype=float).reshape(-1, 2)
        self.line_abc = np.asarray(self.line_abc, dtype=float).reshape(-1, 3)
        m = self.line_uv.shape[0]
        if self.line_abc.shape[0] != m:
            raise ValueError(
                f"ragged line observations at frame {self.frame}: "
                f"{m} image points, {self.line_abc.shape[0]} world lines"
            )
        if self.line_confidence is None:
            self.line_confidence = np.ones(m, dtype=float)
        else:
            self.line_confidence = np.asarray(self.line_confidence, dtype=float).reshape(-1)
            if self.line_confidence.shape[0] != m:
                raise ValueError(f"ragged line confidence at frame {self.frame}")

    @property
    def n_lines(self) -> int:
        """Point-on-line observations attached to this frame (0 when the backend supplies none)."""
        return 0 if self.line_uv is None else int(self.line_uv.shape[0])


@dataclass
class FrameHomography:
    """Backend output for one frame: a full-camera-module image→world homography (or ``None``).

    The camera-module counterpart of :class:`FrameKeypoints`: instead of raw landmark
    correspondences (which a downstream DLT would then fit), the backend has *already* run a full
    field-calibration solve (e.g. PnLCalib's ``FramebyFrameCalib`` — points **and** lines) and emits
    the resulting image→world homography directly.

    Attributes:
        frame: Source frame index.
        homography: ``(3, 3)`` image→world homography (pitch-plane metres, centre-origin), or
            ``None`` when the solve failed for this frame (too few landmarks / degenerate view).
        rep_err_px: The solve's own reprojection error in **pixels** (lower is better), used to
            score confidence downstream; ``None`` when unavailable / unsolved.
        n_landmarks: Detected pitch landmarks that fed the solve (diagnostic only).
    """

    frame: int
    homography: np.ndarray | None
    rep_err_px: float | None = None
    n_landmarks: int = 0

    def __post_init__(self) -> None:
        if self.homography is not None:
            self.homography = np.asarray(self.homography, dtype=float).reshape(3, 3)


@runtime_checkable
class KeypointBackend(Protocol):
    """The heavy half: detect matched pitch landmarks per frame.

    Kept behind this protocol so :class:`KeypointFieldCalibrator`'s homography solve + scoring +
    smoothing can be tested with a stub returning canned :class:`FrameKeypoints` — no GPU.
    """

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        """Return one :class:`FrameKeypoints` per processed frame of ``clip``."""
        ...


@runtime_checkable
class HomographyBackend(Protocol):
    """The heavy half for the camera-module solver: emit a per-frame image→world homography.

    The camera-module counterpart of :class:`KeypointBackend`. Where that protocol stops at raw
    landmark correspondences (leaving the homography fit to the pure DLT calibrator), this one runs
    a *full* field-calibration solve on the box (points **and** pitch lines, e.g. PnLCalib's
    ``FramebyFrameCalib`` + heuristic voting) and returns the homography directly. Kept behind this
    protocol so :class:`CameraModuleFieldCalibrator`'s scoring + smoothing stays testable with a
    stub returning canned :class:`FrameHomography` — no GPU.
    """

    def calibrate_frames(self, clip: ClipRef) -> list[FrameHomography]:
        """Return one :class:`FrameHomography` per processed frame of ``clip``."""
        ...


@runtime_checkable
class FrameMotionBackend(Protocol):
    """The heavy half for camera propagation: how the image itself moved between frames.

    A broadcast main camera is on a tripod, so consecutive frames are related by **one**
    homography whatever the scene depth — no dense flow is needed, and no GPU (#104). Kept behind
    this protocol so :func:`carry_on_motion` stays testable with a stub returning canned matrices.
    """

    def frame_motion(self, clip: ClipRef) -> np.ndarray:
        """Return ``(T-1, 3, 3)`` pixel homographies mapping frame ``k``'s pixels onto ``k+1``'s."""
        ...


def _normalization_matrix(pts: np.ndarray) -> np.ndarray:
    """Hartley normalisation: translate centroid to origin, scale mean distance to ``sqrt(2)``."""
    centroid = pts.mean(axis=0)
    shifted = pts - centroid
    mean_dist = float(np.sqrt((shifted ** 2).sum(axis=1)).mean())
    s = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    return np.array(
        [[s, 0.0, -s * centroid[0]], [0.0, s, -s * centroid[1]], [0.0, 0.0, 1.0]], dtype=float
    )


def _apply_homography(h: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Project ``pts`` (N,2) through ``h`` (3,3) → (N,2), dividing out the homogeneous coord."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ h.T
    return hom[:, :2] / hom[:, 2:3]


def _check_line_observations(
    line_uv: np.ndarray | None, line_abc: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Validate a point-on-line observation set → ``(uv (M,2), abc (M,3))``; ``(0,·)`` if absent."""
    if line_uv is None or line_abc is None:
        if line_uv is not None or line_abc is not None:
            raise ValueError("line_uv and line_abc must be given together")
        return np.empty((0, 2)), np.empty((0, 3))
    uv = np.asarray(line_uv, dtype=float).reshape(-1, 2)
    abc = np.asarray(line_abc, dtype=float).reshape(-1, 3)
    if uv.shape[0] != abc.shape[0]:
        raise ValueError(f"ragged line observations: {uv.shape[0]} points, {abc.shape[0]} lines")
    return uv, abc


def point_line_residual(h: np.ndarray, line_uv: np.ndarray, line_abc: np.ndarray) -> np.ndarray:
    """Per-observation distance (world metres) from ``H @ uv`` to its known pitch line.

    Directly comparable to a point-to-point reprojection residual, because
    :func:`~pitch3d.core.scene.pitch.world_line_from_segment` scales each ``(a, b, c)`` to
    ``a² + b² == 1``.
    """
    uv, abc = _check_line_observations(line_uv, line_abc)
    if uv.shape[0] == 0:
        return np.empty(0)
    world = _apply_homography(h, uv)
    return np.abs(abc[:, 0] * world[:, 0] + abc[:, 1] * world[:, 1] + abc[:, 2])


def solve_homography(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    line_uv: np.ndarray | None = None,
    line_abc: np.ndarray | None = None,
    line_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Normalized-DLT homography ``H`` with ``dst ~ H @ src`` (image→world), shape ``(3, 3)``.

    Hartley-normalises both point sets for numerical conditioning, solves by SVD, denormalises, and
    scales so ``H[2, 2] == 1``. If ``weights`` is given (one non-negative weight per correspondence,
    e.g. per-landmark detection confidence), each correspondence's two DLT rows are scaled by it, so
    uncertain landmarks pull the fit less. ``weights=None`` reproduces the plain unweighted DLT.

    ``line_uv``/``line_abc`` add **point-on-line** observations: an image point known only to lie
    *somewhere* on a named pitch line, whose world line is ``(a, b, c)`` with ``a² + b² == 1``. Such
    an observation fixes one coordinate instead of two, so it contributes a single DLT row
    (``lᵀ H x = 0``) against a correspondence's two — but it needs no identifiable *point*, which is
    what makes it available on frames where the keypoint head finds few intersections. Both kinds
    live in the same 9 unknowns and the same SVD, and their residuals share units (metres), so
    weights are comparable across them.

    Needs 8 independent rows in total: ``2·len(src) + len(line_uv) ≥ 8``. Points alone therefore
    still need ≥ 4; with lines present ≥ 2 correspondences suffice, but never zero — the world-side
    Hartley normalisation is derived from ``dst``.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    l_uv, l_abc = _check_line_observations(line_uv, line_abc)
    n, m = src.shape[0], l_uv.shape[0]
    if dst.shape[0] != n:
        raise ValueError(f"ragged correspondences: {n} src / {dst.shape[0]} dst")
    if 2 * n + m < 8 or n < (2 if m else 4):
        raise ValueError(
            f"under-determined: {n} correspondences + {m} point-on-line observations "
            f"give {2 * n + m} DLT rows, need ≥8 from ≥{2 if m else 4} correspondences"
        )

    t_src, t_dst = _normalization_matrix(src), _normalization_matrix(dst)
    src_n = _apply_homography(t_src, src)
    dst_n = _apply_homography(t_dst, dst)

    rows = []
    for (sx, sy), (dx, dy) in zip(src_n, dst_n, strict=True):
        rows.append([0.0, 0.0, 0.0, -sx, -sy, -1.0, dy * sx, dy * sy, dy])
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, -dx])
    a = np.asarray(rows, dtype=float).reshape(-1, 9)
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"need one weight per correspondence, got {w.shape[0]} for {n}")
        a = a * np.repeat(np.clip(w, 0.0, None), 2)[:, None]  # two DLT rows per correspondence

    if m:
        # A world line transforms contragrediently to a world point, so the normalised line is
        # T_dst⁻ᵀ·l; rescaling it back to a²+b²=1 keeps its row on the same metric scale as the
        # point rows above, which is what lets one weight vector govern both.
        lines_n = l_abc @ np.linalg.inv(t_dst)
        lines_n = lines_n / np.hypot(lines_n[:, 0], lines_n[:, 1]).clip(1e-12)[:, None]
        uv_n = _apply_homography(t_src, l_uv)
        b_rows = np.einsum("mk,mj->mkj", lines_n, np.hstack([uv_n, np.ones((m, 1))]))
        b = b_rows.reshape(m, 9)
        if line_weights is not None:
            lw = np.asarray(line_weights, dtype=float).reshape(-1)
            if lw.shape[0] != m:
                raise ValueError(f"need one weight per line observation, got {lw.shape[0]} for {m}")
            b = b * np.clip(lw, 0.0, None)[:, None]
        a = np.vstack([a, b])

    _, _, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)

    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    return h / h[2, 2] if abs(h[2, 2]) > 1e-12 else h


def reprojection_error(h: np.ndarray, src: np.ndarray, dst: np.ndarray) -> float:
    """RMS distance (world units) between ``H @ src`` and ``dst`` over all correspondences."""
    pred = _apply_homography(h, src)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    return float(np.sqrt(((pred - dst) ** 2).sum(axis=1).mean()))


def solve_homography_ransac(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    threshold: float = 1.0,
    max_iters: int = 200,
    seed: int = 0,
    line_uv: np.ndarray | None = None,
    line_abc: np.ndarray | None = None,
    line_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Robust image→world homography by RANSAC, returning ``(H, inlier_mask)``.

    A plain least-squares DLT is wrecked by even one mislocalised landmark — the dominant failure
    on real broadcast frames, where the keypoint net emits the odd gross outlier. This repeatedly
    fits ``H`` from a random minimal 4-point sample, counts inliers (reprojection residual under
    ``threshold`` world units), keeps the largest consensus set, then refits a confidence-weighted
    DLT on those inliers. With exactly 4 points there is nothing to reject (single weighted solve,
    all-true mask); if no ≥4 consensus emerges it falls back to fitting all points. Deterministic
    for a given ``seed``. On clean correspondences every sample agrees, so it reproduces the plain
    DLT.

    Point-on-line observations (see :func:`solve_homography`) join the **refit**, not the sampling:
    consensus is still decided by the identifiable correspondences, then any line observation
    further than ``threshold`` metres from the winning hypothesis is dropped and the rest are added
    to the final weighted solve. Lines therefore stiffen a fit the points already agree on rather
    than voting on which points to trust — a mislabelled line class would otherwise carry a whole
    consensus set with it. The returned mask covers ``src`` only.

    Threshold and residuals are in **world metres** throughout, deliberately: a pitch homography is
    strongly heteroscedastic (a uniform 2 px image error is ~0.03 m at the near touchline and
    ~0.23 m at the far one, an 8.4x spread we measured in ``scripts/bench_ransac_usac.py``), so a
    single pixel-domain inlier scale — what a black-box robust estimator marginalises over — does
    not exist here. See ADR-0012.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    l_uv, l_abc = _check_line_observations(line_uv, line_abc)
    lines = {"line_uv": l_uv, "line_abc": l_abc, "line_weights": line_weights} if l_uv.size else {}
    n = src.shape[0]
    if dst.shape[0] != n:
        raise ValueError(f"ragged correspondences: {n} src / {dst.shape[0]} dst")
    if n <= 4:  # nothing to reject: every point must be used, lines just add rows
        return solve_homography(src, dst, weights, **lines), np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_key = (3, -np.inf)  # (inlier count, -mean residual); any ≥4-inlier hypothesis beats it
    best_h: np.ndarray | None = None
    for _ in range(max_iters):
        idx = rng.choice(n, size=4, replace=False)
        try:
            h = solve_homography(src[idx], dst[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            # a degenerate minimal sample can map points to infinity → non-finite residuals,
            # which we explicitly reject below; don't let that benign case warn.
            resid = np.sqrt(((_apply_homography(h, src) - dst) ** 2).sum(axis=1))
        if not np.all(np.isfinite(resid)):
            continue
        mask = resid < threshold
        count = int(mask.sum())
        if count < 4:
            continue
        key = (count, -float(resid[mask].mean()))
        if key > best_key:
            best_key, best_mask, best_h = key, mask, h

    if best_mask is None:  # no consensus — best-effort fit over everything
        return solve_homography(src, dst, weights, **lines), np.ones(n, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[best_mask]
    if lines:
        assert best_h is not None
        with np.errstate(divide="ignore", invalid="ignore"):
            keep = point_line_residual(best_h, l_uv, l_abc) < threshold
        lines = {"line_uv": l_uv[keep], "line_abc": l_abc[keep]} if keep.any() else {}
        if lines and line_weights is not None:
            lines["line_weights"] = np.asarray(line_weights, dtype=float).reshape(-1)[keep]
    return solve_homography(src[best_mask], dst[best_mask], w, **lines), best_mask


def _confidence_from_error(err: float, scale_m: float) -> float:
    """Map a reprojection error (m) to ``(0, 1]``: 1 at zero error, 0.5 at ``scale_m``."""
    return 1.0 / (1.0 + max(err, 0.0) / max(scale_m, 1e-9))


def _temporal_smooth(homographies: np.ndarray, window: int) -> np.ndarray:
    """Box-average homographies over a centred frame window, renormalising each ``H[2,2]``.

    Superseded by :func:`carry_on_motion` wherever inter-frame motion is available: averaging
    homography *coefficients* has no geometric meaning (a homography is defined only up to scale
    and its entries are not commensurate), and carrying measurably dominates it on both axes
    (#104). Kept for :class:`CameraModuleFieldCalibrator`, which has no motion source.
    """
    t = homographies.shape[0]
    if window <= 1 or t <= 2:
        return homographies
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(homographies)
    for i in range(t):
        m = homographies[max(0, i - half):min(t, i + half + 1)].mean(axis=0)
        out[i] = m / m[2, 2] if abs(m[2, 2]) > 1e-12 else m
    return out


#: Probe pixels as (width, height) fractions — where players' feet actually are. The camera track
#: is scored and fused here rather than over the whole frame because error at the horizon is both
#: enormous and invisible: nobody is standing there (#104).
_PROBE_FRAC = np.array(
    [[1 / 3, 0.741], [1 / 2, 0.741], [2 / 3, 0.741], [0.396, 0.880], [0.604, 0.880]]
)


def probe_pixels(width: int, height: int) -> np.ndarray:
    """The ``(5, 2)`` probe points for a frame of this size (see :data:`_PROBE_FRAC`)."""
    return _PROBE_FRAC * np.array([[float(width), float(height)]])


def _chain_motion(motion: np.ndarray, k: int, j: int) -> np.ndarray:
    """Map frame ``k``'s pixels onto frame ``j``'s, composed from the measured inter-frame fits."""
    g = np.eye(3)
    if j > k:
        for i in range(k, j):
            g = motion[i] @ g
    else:
        for i in range(j, k):
            g = motion[i] @ g
        g = np.linalg.inv(g)
    return g


def carry_on_motion(
    homographies: np.ndarray, motion: np.ndarray, window: int, probe_uv: np.ndarray
) -> np.ndarray:
    """Re-estimate each frame's homography from its neighbours, carried by the measured motion.

    The per-frame calibration *swims*: consecutive frames disagree by median 0.119 m about where
    the same physical point sits, while the camera pans smoothly (#104). Each neighbour therefore
    carries real information about this frame, reachable by composing the inter-frame pixel motion.

    The vote happens where the quantity is **physical**. Homography coefficients cannot be averaged
    — only defined up to scale, entries not commensurate — so instead every neighbour predicts
    *where a probe pixel lands on the pitch*, those world points are combined with a per-coordinate
    median (robust to a neighbour whose own solve failed), and a homography is re-fitted to the
    result.

    This is a **trade, not a free win**: over the target clip it removes 92 % of the swim while
    giving up ~0.0035 m of paint accuracy — favourable by 31×, but it must be reported as a trade
    (#104). ``window=0`` disables it.
    """
    n = homographies.shape[0]
    if window <= 0 or n < 2:
        return homographies
    if motion.shape[0] != n - 1:
        raise ValueError(f"need {n - 1} inter-frame motions for {n} frames, got {motion.shape[0]}")
    out = np.empty_like(homographies)
    for k in range(n):
        preds = []
        for j in range(max(0, k - window), min(n - 1, k + window) + 1):
            uv = probe_uv if j == k else _apply_homography(_chain_motion(motion, k, j), probe_uv)
            preds.append(_apply_homography(homographies[j], uv))
        out[k] = solve_homography(probe_uv, np.median(np.stack(preds), axis=0))
    return out


@dataclass
class KeypointFieldCalibrator(FieldCalibrator):
    """Pitch-keypoint homography calibrator (FR-7) — pure robust solve over an injected backend.

    Each frame's landmarks are fitted with a **RANSAC + confidence-weighted DLT** so a few
    mislocalised pitch points (the dominant failure on real broadcast frames) are rejected instead
    of dragging the whole homography — and confidence is scored on the inliers, downweighted by how
    many landmarks actually agreed (R-6 honesty).

    Attributes:
        backend: The landmark-detection backend. If ``None``, a real :class:`PitchKeypointBackend`
            is constructed lazily on first use (needs the ``cv`` extra + weights + GPU).
        min_keypoints: Evidence floor, counted in DLT rows as ``2 · min_keypoints``: a
            correspondence supplies two rows, a point-on-line observation one. Frames below it
            reuse the last good homography at confidence 0 (drift is surfaced honestly, R-6), or
            identity if none yet. With no line observations this is simply "≥ this many landmarks".
        smooth_window: Centred temporal-smoothing window in frames (1 disables smoothing).
        conf_scale_m: Reprojection error (metres) that maps to confidence 0.5.
        ransac_threshold_m: Max reprojection residual (metres) for a landmark to count as an inlier.
        ransac_iters: RANSAC hypothesis count per frame.
        seed: RANSAC RNG seed — makes calibration reproducible.
        device: Inference device for the default backend.
    """

    backend: KeypointBackend | None = None
    motion: FrameMotionBackend | None = None
    carry_window: int = 8
    min_keypoints: int = 4
    smooth_window: int = 1
    conf_scale_m: float = 0.5
    ransac_threshold_m: float = 1.0
    ransac_iters: int = 200
    seed: int = 0
    device: str = "cuda"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="PitchKeypoints+DLT",
            backend=Backend.LOCAL,
            params={
                "smooth_window": self.smooth_window,
                "carry_window": self.carry_window if self.motion is not None else 0,
                "ransac_threshold_m": self.ransac_threshold_m,
                "device": self.device,
                "impl": impl_name(self.backend, "PitchKeypointBackend"),
            },
        )

    def calibrate(self, clip: ClipRef) -> FieldCalibration:
        per = self._backend().detect_keypoints(clip)
        if not per:
            raise ValueError("keypoint backend returned no frames")

        homs: list[np.ndarray] = []
        confs: list[float] = []
        frames: list[int] = []
        last_good: np.ndarray | None = None
        for fk in per:
            conf_kp = fk.confidence
            assert conf_kp is not None  # FrameKeypoints.__post_init__ fills this (ones if unset)
            k, m = fk.image_uv.shape[0], fk.n_lines
            # A correspondence is worth two DLT rows and a point-on-line observation one, so the
            # solvability test is on *rows*; ≥2 points because the world-side Hartley normalisation
            # is derived from them. With no lines this is exactly `k >= min_keypoints`.
            if k >= 2 and 2 * k + m >= max(8, 2 * self.min_keypoints):
                h, inliers = solve_homography_ransac(
                    fk.image_uv,
                    fk.world_xy,
                    weights=conf_kp,
                    threshold=self.ransac_threshold_m,
                    max_iters=self.ransac_iters,
                    seed=self.seed,
                    line_uv=fk.line_uv,
                    line_abc=fk.line_abc,
                    line_weights=fk.line_confidence,
                )
                resid = np.linalg.norm(
                    _apply_homography(h, fk.image_uv[inliers]) - fk.world_xy[inliers], axis=1
                )
                agree, mean_conf = inliers.astype(float), conf_kp[inliers]
                rows = 2 * int(inliers.sum())
                if m:
                    assert fk.line_confidence is not None
                    line_resid = point_line_residual(h, fk.line_uv, fk.line_abc)
                    line_in = line_resid < self.ransac_threshold_m
                    # Points alone can be too few to over-constrain the fit (k=2 reprojects
                    # exactly), so the lines that made the frame solvable must also score it —
                    # otherwise a thin frame would report false confidence (R-6).
                    resid = np.concatenate([resid, line_resid[line_in]])
                    agree = np.concatenate([agree, line_in.astype(float)])
                    mean_conf = np.concatenate([mean_conf, fk.line_confidence[line_in]])
                    rows += int(line_in.sum())
                # Normalise by residual *degrees of freedom*, not by observation count: a
                # homography has 8, so only `rows - 8` of the residual is free to be non-zero.
                # At zero redundancy the fit reproduces its own agreeing observations exactly and
                # a count-normalised score reads a perfect 0 error however wrong the homography
                # is — which is why the frames with the least evidence used to score *highest*
                # (#105). Undefined redundancy means unverifiable, which is confidence 0, not 1.
                dof = rows - 8
                err = float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else float("inf")
                conf = (
                    _confidence_from_error(err, self.conf_scale_m)
                    * float(mean_conf.mean() if mean_conf.size else 0.0)
                    * float(agree.mean())  # honest down-weight by the agreeing-evidence fraction
                )
                last_good = h
            else:
                h = last_good if last_good is not None else np.eye(3)
                conf = 0.0  # under-detected: carry last good, but flag zero confidence
            homs.append(h)
            confs.append(conf)
            frames.append(int(fk.frame))

        track = np.stack(homs)
        if self.motion is not None and self.carry_window > 0 and track.shape[0] > 1:
            track = carry_on_motion(
                track,
                self.motion.frame_motion(clip),
                self.carry_window,
                probe_pixels(clip.width, clip.height),
            )
        else:
            track = _temporal_smooth(track, self.smooth_window)
        return FieldCalibration(
            homographies=track,
            frames=np.asarray(frames, dtype=int),
            confidence=np.asarray(confs, dtype=float),
        )

    def _backend(self) -> KeypointBackend:
        return self.backend or PitchKeypointBackend(device=self.device)


# ── PnLCalib's world is a template, not a world (#118) ──────────────────────────────────────────
# Its keypoint table is a top-down pitch TEMPLATE: X across it, Y *down* it, as in any image. Read
# those two axes as our X and Y and then call the third one "up", and the labelling is left-handed
# — so a homography that maps the lawn perfectly still decomposes to a camera looking UPWARD from
# under the grass. Measured on the target clip at every candidate focal and on all 60 frames:
# optical axis +0.175 (up), centre 18 m below the pitch. Turning the world 180° about X — Y and Z
# both negated, a proper rotation — makes it a real broadcast gantry (-0.175, 18 m up, 72 m beyond
# the touchline) and is what the rest of this codebase means by "world" (`core.scene.units`: Z-up,
# right-handed, metres; the same frame SMPL-X, Blender and the novel-view cameras live in).
#
# On the pitch plane Z = 0 that rotation is just the Y negation below. It costs nothing to get
# right and everything to get wrong: the pitch is symmetric about Y = 0, so the flip moves not one
# pixel and no marking metric can ever catch it. Only something with height can — which is why it
# survived until a goalpost had to be drawn. `poseannot.camera.plane_orientation` measures the
# handedness of a homography in hand rather than trusting a label, so calibrations solved before
# this fix are still read correctly instead of silently.
TEMPLATE_TO_WORLD = np.diag([1.0, -1.0, 1.0])


def image_to_world_from_cam_params(cam_params: dict) -> np.ndarray:
    """Convert a PnLCalib camera-module ``cam_params`` dict to a ``(3, 3)`` image→world homography.

    The camera module solves a full pinhole camera ``P = K · [R | t]`` (3×4), mapping centre-origin
    world metres ``[X, Y, Z, 1]ᵀ`` to image pixels. On the pitch plane ``Z = 0`` the third column
    drops out, so the world→image map is the ``(3, 3)`` ``H_w→i = P[:, (0, 1, 3)]``; we invert and
    renormalise it to get image→world in the **same** centre-origin metric frame the DLT path uses
    (``keypoint_world_coords_2D``), then turn that template frame into ours with
    :data:`TEMPLATE_TO_WORLD`, making it a drop-in for :class:`FieldCalibration`.

    Args:
        cam_params: PnLCalib ``cam_params`` with keys ``x_focal_length``, ``y_focal_length``,
            ``principal_point`` (2,), ``position_meters`` (3,), ``rotation_matrix`` (3×3).

    Returns:
        ``(3, 3)`` image→world homography, normalised so ``H[2, 2] == 1``.

    Raises:
        numpy.linalg.LinAlgError: If the world→image plane map is singular (degenerate view) — the
            caller turns this into a ``None`` :class:`FrameHomography` for that frame.
    """
    fx = float(cam_params["x_focal_length"])
    fy = float(cam_params["y_focal_length"])
    px, py = (float(v) for v in cam_params["principal_point"])
    position = np.asarray(cam_params["position_meters"], dtype=float).reshape(3)
    rotation = np.asarray(cam_params["rotation_matrix"], dtype=float).reshape(3, 3)

    extrinsic = np.eye(4, dtype=float)[:-1]  # (3, 4): [I | 0]
    extrinsic[:, -1] = -position  # camera at `position`, so translate world by −position
    intrinsic = np.array([[fx, 0.0, px], [0.0, fy, py], [0.0, 0.0, 1.0]], dtype=float)
    projection = intrinsic @ (rotation @ extrinsic)  # (3, 4) world→image

    h_w2i = projection[:, [0, 1, 3]]  # drop Z column (pitch plane Z=0) → (3, 3) world→image
    h_i2w = TEMPLATE_TO_WORLD @ np.linalg.inv(h_w2i)  # LinAlgError on a singular/degenerate view
    return h_i2w / h_i2w[2, 2] if abs(h_i2w[2, 2]) > 1e-12 else h_i2w


@dataclass
class CameraModuleFieldCalibrator(FieldCalibrator):
    """Full-camera-module field calibrator (FR-7) — scores + smooths a backend's per-frame solve.

    The camera-module sibling of :class:`KeypointFieldCalibrator`. Where that class fits a planar
    DLT from *points only* in-process, this one delegates the whole solve to a
    :class:`HomographyBackend` that runs PnLCalib's ``FramebyFrameCalib`` on the box — points
    **and** pitch lines, RANSAC/mode sweep, L/R plane disambiguation — and emits a ready image→world
    homography per frame. The pure half here is the *honest bookkeeping*: score each solved frame by
    its reprojection error (pixels), carry the last good homography through unsolved frames at
    confidence 0 (drift surfaced, never hidden — R-6), and apply optional temporal smoothing.

    Attributes:
        backend: The full-solve homography backend. Required — there is no in-process fallback (the
            camera module lives on the GPU box); ``calibrate`` raises if it is ``None``.
        smooth_window: Centred temporal-smoothing window in frames (1 disables smoothing).
        conf_scale_px: Reprojection error (**pixels**) that maps to confidence 0.5.
        device: Inference device, forwarded for provenance only (the backend owns the model).
    """

    backend: HomographyBackend | None = None
    smooth_window: int = 1
    conf_scale_px: float = 5.0
    device: str = "cuda"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="PnLCalib-Camera",
            backend=Backend.LOCAL,
            params={
                "smooth_window": self.smooth_window,
                "conf_scale_px": self.conf_scale_px,
                "device": self.device,
            },
        )

    def calibrate(self, clip: ClipRef) -> FieldCalibration:
        if self.backend is None:
            raise ValueError(
                "CameraModuleFieldCalibrator needs a HomographyBackend (the full camera-module "
                "solve runs on the GPU box); inject one or use KeypointFieldCalibrator (DLT)."
            )
        per = self.backend.calibrate_frames(clip)
        if not per:
            raise ValueError("homography backend returned no frames")

        homs: list[np.ndarray] = []
        confs: list[float] = []
        frames: list[int] = []
        last_good: np.ndarray | None = None
        for fh in per:
            if fh.homography is not None:
                h = fh.homography
                rep = fh.rep_err_px
                err = rep if rep is not None and np.isfinite(rep) else 0.0
                conf = _confidence_from_error(err, self.conf_scale_px)
                last_good = h
            else:
                h = last_good if last_good is not None else np.eye(3)
                conf = 0.0  # unsolved frame: carry last good, but flag zero confidence (R-6)
            homs.append(h)
            confs.append(conf)
            frames.append(int(fh.frame))

        smoothed = _temporal_smooth(np.stack(homs), self.smooth_window)
        return FieldCalibration(
            homographies=smoothed,
            frames=np.asarray(frames, dtype=int),
            confidence=np.asarray(confs, dtype=float),
        )


@dataclass
class LucasKanadeMotion:
    """Inter-frame camera motion from corner tracking — CPU, no weights, no GPU.

    R2 originally booked a GPU for RAFT-small dense optical flow. It is not needed: a broadcast
    main camera is on a tripod, so the whole image is related frame-to-frame by **one** homography
    whatever the scene depth. ``goodFeaturesToTrack`` + ``calcOpticalFlowPyrLK`` + RANSAC recovers
    it at 4000/4000 corners in ~1 min for the target clip (#104).

    RANSAC is doing real work here, not tidying: players move independently of the camera, so the
    correct fit is the *majority* motion — the stadium.

    Attributes:
        max_corners: Corner budget per frame.
        quality: ``goodFeaturesToTrack`` quality level.
        min_distance: Minimum corner separation, px.
        win_size: Lucas-Kanade search window, px.
        max_level: Pyramid levels for large inter-frame displacement.
        ransac_px: Reprojection threshold separating camera motion from independent movers.
    """

    max_corners: int = 4000
    quality: float = 0.01
    min_distance: int = 8
    win_size: int = 21
    max_level: int = 4
    ransac_px: float = 2.0

    def frame_motion(self, clip: ClipRef) -> np.ndarray:  # pragma: no cover - heavy decode path
        import cv2

        from ..io.frames import iter_clip_frames

        out: list[np.ndarray] = []
        prev: np.ndarray | None = None
        for _, bgr in iter_clip_frames(clip.uri, clip.frames.tolist(), crop=clip.crop):
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                out.append(self._between(cv2, prev, gray))
            prev = gray
        if len(out) != clip.n_frames - 1:
            raise ValueError(f"decoded {len(out) + 1} frames, clip asked for {clip.n_frames}")
        return np.stack(out) if out else np.empty((0, 3, 3))

    def _between(self, cv2, a: np.ndarray, b: np.ndarray) -> np.ndarray:  # pragma: no cover
        p0 = cv2.goodFeaturesToTrack(
            a, maxCorners=self.max_corners, qualityLevel=self.quality,
            minDistance=self.min_distance,
        )
        if p0 is None or len(p0) < 4:
            return np.eye(3)
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            a, b, p0, None, winSize=(self.win_size, self.win_size), maxLevel=self.max_level
        )
        ok = st.ravel().astype(bool)
        if ok.sum() < 4:
            return np.eye(3)
        g, _ = cv2.findHomography(
            p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2), cv2.RANSAC, self.ransac_px
        )
        # A failed fit must not silently corrupt the chain every later frame is carried through:
        # identity says "no measured motion", which the window median can outvote (R-6).
        return np.eye(3) if g is None else np.asarray(g, dtype=float)


@dataclass
class PitchKeypointBackend:
    """Real pitch-landmark detection: lazy model/cv2, no import cost.

    Detects known pitch landmarks (lines, corners, marks) per frame and pairs them with their
    fixed world coordinates on the pitch model. Imports the heavy stack only when
    :meth:`detect_keypoints` is first called, so this module stays import-safe without ``cv``.
    """

    weights: str | None = None
    device: str = "cuda"
    _model: object = None

    def detect_keypoints(  # pragma: no cover - heavy path
        self, clip: ClipRef
    ) -> list[FrameKeypoints]:
        self._load()
        raise NotImplementedError(
            "wire the concrete pitch-keypoint model's output into FrameKeypoints "
            "(image_uv ↔ world_xy) here; the homography solve is already done in the adapter."
        )

    def _load(self) -> object:  # pragma: no cover - exercised only without the extra
        if self._model is None:
            try:
                import cv2  # noqa: F401  (stand-in for the concrete landmark model's deps)
            except ImportError as exc:
                raise RuntimeError(
                    "the pitch-keypoint model is not installed. Install the detection extra: "
                    "`pip install 'pitch3d[cv]'`, or inject a KeypointBackend."
                ) from exc
            self._model = object()
        return self._model
