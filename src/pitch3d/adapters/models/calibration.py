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
from ...core.scene.provenance import Backend, ModelInfo


@dataclass
class FrameKeypoints:
    """Backend output for one frame: matched pitch landmarks in image and world coords.

    Attributes:
        frame: Source frame index.
        image_uv: ``(K, 2)`` detected landmark positions, image px.
        world_xy: ``(K, 2)`` the landmarks' known pitch-plane coords (metres, ``Z = 0``).
        confidence: ``(K,)`` per-landmark detection confidence in ``[0, 1]`` (defaults to ones).
    """

    frame: int
    image_uv: np.ndarray
    world_xy: np.ndarray
    confidence: np.ndarray | None = None

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


@runtime_checkable
class KeypointBackend(Protocol):
    """The heavy half: detect matched pitch landmarks per frame.

    Kept behind this protocol so :class:`KeypointFieldCalibrator`'s homography solve + scoring +
    smoothing can be tested with a stub returning canned :class:`FrameKeypoints` — no GPU.
    """

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        """Return one :class:`FrameKeypoints` per processed frame of ``clip``."""
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


def solve_homography(
    src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Normalized-DLT homography ``H`` with ``dst ~ H @ src`` (image→world), shape ``(3, 3)``.

    Needs ≥ 4 non-collinear correspondences. Hartley-normalises both point sets for numerical
    conditioning, solves by SVD, denormalises, and scales so ``H[2, 2] == 1``. If ``weights`` is
    given (one non-negative weight per correspondence, e.g. per-landmark detection confidence),
    each correspondence's two DLT rows are scaled by it, so uncertain landmarks pull the fit less.
    ``weights=None`` reproduces the plain unweighted DLT exactly.
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    n = src.shape[0]
    if n < 4 or dst.shape[0] != n:
        raise ValueError(f"need ≥4 matched correspondences, got {n} src / {dst.shape[0]} dst")

    t_src, t_dst = _normalization_matrix(src), _normalization_matrix(dst)
    src_n = _apply_homography(t_src, src)
    dst_n = _apply_homography(t_dst, dst)

    rows = []
    for (sx, sy), (dx, dy) in zip(src_n, dst_n, strict=True):
        rows.append([0.0, 0.0, 0.0, -sx, -sy, -1.0, dy * sx, dy * sy, dy])
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy, -dx])
    a = np.asarray(rows, dtype=float)
    if weights is not None:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n:
            raise ValueError(f"need one weight per correspondence, got {w.shape[0]} for {n}")
        a = a * np.repeat(np.clip(w, 0.0, None), 2)[:, None]  # two DLT rows per correspondence
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
    """
    src = np.asarray(src, dtype=float).reshape(-1, 2)
    dst = np.asarray(dst, dtype=float).reshape(-1, 2)
    n = src.shape[0]
    if n < 4 or dst.shape[0] != n:
        raise ValueError(f"need ≥4 matched correspondences, got {n} src / {dst.shape[0]} dst")
    if n == 4:
        return solve_homography(src, dst, weights), np.ones(n, dtype=bool)

    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    best_key = (3, -np.inf)  # (inlier count, -mean residual); any ≥4-inlier hypothesis beats it
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
            best_key, best_mask = key, mask

    if best_mask is None:  # no consensus — best-effort fit over everything
        return solve_homography(src, dst, weights), np.ones(n, dtype=bool)
    w = None if weights is None else np.asarray(weights, dtype=float).reshape(-1)[best_mask]
    return solve_homography(src[best_mask], dst[best_mask], w), best_mask


def _confidence_from_error(err: float, scale_m: float) -> float:
    """Map a reprojection error (m) to ``(0, 1]``: 1 at zero error, 0.5 at ``scale_m``."""
    return 1.0 / (1.0 + max(err, 0.0) / max(scale_m, 1e-9))


def _temporal_smooth(homographies: np.ndarray, window: int) -> np.ndarray:
    """Box-average homographies over a centred frame window, renormalising each ``H[2,2]``."""
    t = homographies.shape[0]
    if window <= 1 or t <= 2:
        return homographies
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(homographies)
    for i in range(t):
        m = homographies[max(0, i - half):min(t, i + half + 1)].mean(axis=0)
        out[i] = m / m[2, 2] if abs(m[2, 2]) > 1e-12 else m
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
        min_keypoints: Frames with fewer matched landmarks reuse the last good homography at
            confidence 0 (drift is surfaced honestly, R-6), or identity if none yet.
        smooth_window: Centred temporal-smoothing window in frames (1 disables smoothing).
        conf_scale_m: Reprojection error (metres) that maps to confidence 0.5.
        ransac_threshold_m: Max reprojection residual (metres) for a landmark to count as an inlier.
        ransac_iters: RANSAC hypothesis count per frame.
        seed: RANSAC RNG seed — makes calibration reproducible.
        device: Inference device for the default backend.
    """

    backend: KeypointBackend | None = None
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
                "ransac_threshold_m": self.ransac_threshold_m,
                "device": self.device,
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
            if fk.image_uv.shape[0] >= self.min_keypoints:
                h, inliers = solve_homography_ransac(
                    fk.image_uv,
                    fk.world_xy,
                    weights=conf_kp,
                    threshold=self.ransac_threshold_m,
                    max_iters=self.ransac_iters,
                    seed=self.seed,
                )
                err = reprojection_error(h, fk.image_uv[inliers], fk.world_xy[inliers])
                conf = (
                    _confidence_from_error(err, self.conf_scale_m)
                    * float(conf_kp[inliers].mean())
                    * float(inliers.mean())  # honest down-weight by the agreeing-landmark fraction
                )
                last_good = h
            else:
                h = last_good if last_good is not None else np.eye(3)
                conf = 0.0  # under-detected: carry last good, but flag zero confidence
            homs.append(h)
            confs.append(conf)
            frames.append(int(fk.frame))

        smoothed = _temporal_smooth(np.stack(homs), self.smooth_window)
        return FieldCalibration(
            homographies=smoothed,
            frames=np.asarray(frames, dtype=int),
            confidence=np.asarray(confs, dtype=float),
        )

    def _backend(self) -> KeypointBackend:
        return self.backend or PitchKeypointBackend(device=self.device)


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
