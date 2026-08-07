"""GVHMR SMPL-X pose estimator — fourth real adapter, the central one (M1, FR-8, FR-22c).

Self-hosted default for `PoseEstimator`: a gravity-view HMR network (GVHMR / WHAM / TRAM),
behind the optional ``hmr`` extra. Split so the *model-independent* logic is testable with
**no torch, no GPU**:

* :class:`GVHMRPoseEstimator` — the **pure** half. The network only knows *articulation*
  (per-frame ``global_orient`` + ``body_pose`` + shape ``betas``); turning that into a scene
  subject means **anchoring the world root on the pitch** — exactly FR-8's "root from
  homography". This half projects each tracklet's foot point to world metres through the field
  calibration, stacks the metric root, assembles the canonical :class:`SubjectMotion`, and
  applies geometric :meth:`refit` constraints (root nudges/locks/floor, plus the M3-2
  measured-homography-anchor lock) deterministically. Numpy only; unit-tested via an injected
  backend. The cluster-occlusion completion option (:class:`OcclusionBackend`, Diffusion-VAS +
  SAM-3, M3-2) is gated the same way as the network and validated against the homography anchor.
* :class:`GVHMRBackend` — the **heavy** half: *not wired yet* (roadmap M1/P2.4) and GPU-bound.
  GVHMR is a research repo, not a pip package, so the ``hmr`` extra ships only its substrate
  (torch/smplx/chumpy) — not the network or its weights; :meth:`GVHMRBackend.estimate_bodies`
  raises an actionable ``NotImplementedError`` pointing at ``--pose fake`` or an injected backend.
  The pure half above is complete and tested, so the pose path runs end to end on the fake today.

Swap it in via ``default_ports(pose="gvhmr")`` (wiring) — one fake replaced at a time,
satisfying the very same ``PoseEstimator`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.correction.anchor import blend_to_anchor
from ...core.ports.io import ClipRef
from ...core.ports.perception import Tracklet, Tracks
from ...core.ports.pose import PoseEstimator
from ...core.scene.field import MIN_SOLVED_CONFIDENCE, FieldCalibration
from ...core.scene.motion import PoseSequence, SmplxShape, SubjectMotion
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.units import FieldDimensions

#: Nominal SMPL-X pelvis height above the ground plane (m) — the mono Z anchor (R-4).
_DEFAULT_PELVIS_HEIGHT_M = 0.92

#: How far outside the pitch rectangle a grounded root may still be a real person (m). Generous
#: on purpose: a keeper behind his line, a thrown-in taker, a substitute on the touchline are all
#: real. 25 m past the paint is already in the stands, so anything beyond it is arithmetic, not a
#: player. This is the physical counterpart to `min_calib_confidence`: that one asks whether the
#: PLANE was solved, this one asks whether THIS POINT landed anywhere a footballer can stand.
_DEFAULT_OFF_PITCH_MARGIN_M = 25.0


@dataclass
class RawBodyMotion:
    """Backend output for one subject: camera-space SMPL-X articulation + shape, no world root.

    The world root is intentionally absent — placing it on the pitch is the field-homography
    job done in the pure half. This is the canned hand-off the grounding logic consumes.

    Attributes:
        track_id: The tracklet this articulation belongs to.
        frames: ``(T,)`` source frame indices.
        global_orient: ``(T, 3)`` root orientation, axis-angle.
        body_pose: ``(T, J, 3)`` body-joint rotations, axis-angle.
        betas: ``(n_betas,)`` per-subject shape coefficients (frame-invariant).
        pelvis_above_foot: ``(T,)`` per-frame vertical foot→pelvis offset (metres) from the
            backend's SMPL-X forward kinematics, or ``None``. When present it drives the world
            root Z in :meth:`GVHMRPoseEstimator._ground_root` (the T2 foot-plane anchor — pelvis
            height that varies with crouch/run/stride); when absent the estimator falls back to a
            fixed nominal pelvis height. Backends that do not compute FK leave it ``None``.
    """

    track_id: int
    frames: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray
    betas: np.ndarray
    pelvis_above_foot: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        t = self.frames.shape[0]
        self.global_orient = np.asarray(self.global_orient, dtype=float).reshape(-1, 3)
        body_pose = np.asarray(self.body_pose, dtype=float)
        self.body_pose = body_pose.reshape(body_pose.shape[0], -1, 3)  # (rows, J, 3)
        self.betas = np.asarray(self.betas, dtype=float).reshape(-1)
        if not (self.global_orient.shape[0] == self.body_pose.shape[0] == t):
            raise ValueError(
                f"ragged raw body motion {self.track_id}: {t} frames, "
                f"{self.global_orient.shape[0]} orient, {self.body_pose.shape[0]} body rows"
            )
        if self.pelvis_above_foot is not None:
            self.pelvis_above_foot = np.asarray(self.pelvis_above_foot, dtype=float).reshape(-1)
            if self.pelvis_above_foot.shape[0] != t:
                raise ValueError(
                    f"pelvis_above_foot for {self.track_id} has "
                    f"{self.pelvis_above_foot.shape[0]} rows, expected {t}"
                )


@runtime_checkable
class HMRBackend(Protocol):
    """The heavy half: run the HMR network per tracklet → camera-space articulation + shape.

    Kept behind this protocol so :class:`GVHMRPoseEstimator`'s grounding/assembly/refit logic
    can be tested with a stub returning canned :class:`RawBodyMotion` — no GPU required.
    """

    def estimate_bodies(self, clip: ClipRef, tracks: Tracks) -> dict[int, RawBodyMotion]:
        """Return SMPL-X articulation per subject, keyed by ``track_id``."""
        ...


@runtime_checkable
class OcclusionBackend(Protocol):
    """The cluster-occlusion heavy half (M3-2): amodally complete occluded segments.

    Behind this protocol so the re-fit's pure constraint/anchor logic stays testable. A real
    backend pairs amodal video completion (**Diffusion-VAS**) with pixel-level identity
    (**SAM-3** masklets) over the occluded ``frames`` and returns a re-fit :class:`SubjectMotion`.
    Whatever it produces must be validated against the homography anchor
    (:mod:`pitch3d.core.correction.anchor`) — a completion that drifts off the player's measured
    ground track is hallucinated, not measured (R-6). Injected, never on by default.
    """

    def complete_occlusions(
        self, clip: ClipRef, motion: SubjectMotion, frames: np.ndarray
    ) -> SubjectMotion:
        """Return ``motion`` with the occluded ``frames`` amodally completed."""
        ...


def _smooth_path(values: np.ndarray, window: int) -> np.ndarray:
    """Box-average a ``(T, D)`` path over a centred frame window (anti-jitter, edge-clamped)."""
    t = values.shape[0]
    if window <= 1 or t <= 2:
        return values
    half = (window if window % 2 else window + 1) // 2
    out = np.empty_like(values)
    for i in range(t):
        out[i] = values[max(0, i - half):min(t, i + half + 1)].mean(axis=0)
    return out


@dataclass
class GVHMRPoseEstimator(PoseEstimator):
    """SMPL-X HMR (FR-8) — pure root-grounding + assembly over an injected backend.

    Attributes:
        backend: The HMR network backend. If ``None``, a real :class:`GVHMRBackend` is built
            lazily on first use (needs the ``hmr`` extra + weights + GPU).
        occlusion_backend: Optional cluster-occlusion completer (M3-2). When injected and a re-fit
            requests ``complete_occlusions``, the selected frames are amodally completed by it
            (Diffusion-VAS + SAM-3, gated — R-8); when ``None`` that request is an actionable error
            pointing at the measured structural gap-fill instead. Never on by default.
        pelvis_height_m: World Z assigned to the grounded root (mono height anchor, R-4).
        smooth_window: Centred window (frames) for smoothing the grounded root path
            (anti-foot-sliding from box jitter); 1 disables it.
        min_calib_confidence: Calibration confidence a frame must carry before its foot is
            un-projected. Both calibrators write ``0.0`` on a frame they could not solve, so the
            default is a "was the plane measured at all" test, not a quality bar. ``0.0`` restores
            the pre-2026-08-07 behaviour of grounding through carried homographies — which is how
            a zooming phone clip produced roots 3 km apart. Auto default, manual override.
        n_betas: Shape-coefficient count expected from the backend (provenance only).
        device: Inference device for the default backend.
    """

    backend: HMRBackend | None = None
    occlusion_backend: OcclusionBackend | None = None
    pelvis_height_m: float = _DEFAULT_PELVIS_HEIGHT_M
    smooth_window: int = 1
    n_betas: int = 10
    device: str = "cuda"
    min_calib_confidence: float = MIN_SOLVED_CONFIDENCE
    off_pitch_margin_m: float = _DEFAULT_OFF_PITCH_MARGIN_M
    field_dimensions: FieldDimensions = field(default_factory=FieldDimensions)
    #: Filled by :meth:`estimate` so the caller can report the refusal instead of it being silent.
    dropped_frames: int = field(default=0, init=False)
    dropped_offpitch: int = field(default=0, init=False)
    dropped_subjects: list[int] = field(default_factory=list, init=False)

    def _on_pitch(self, world_xy: np.ndarray) -> np.ndarray:
        """Mask of world XY that could be a player: on the pitch, plus a generous margin."""
        half_x = self.field_dimensions.length / 2.0 + self.off_pitch_margin_m
        half_y = self.field_dimensions.width / 2.0 + self.off_pitch_margin_m
        xy = np.asarray(world_xy, dtype=float).reshape(-1, 2)
        return (np.abs(xy[:, 0]) <= half_x) & (np.abs(xy[:, 1]) <= half_y)

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="GVHMR",
            backend=Backend.LOCAL,
            license="see upstream (non-commercial SMPL-X)",
            params={"pelvis_height_m": self.pelvis_height_m, "device": self.device},
        )

    def estimate(
        self, clip: ClipRef, tracks: Tracks, calibration: FieldCalibration
    ) -> dict[int, SubjectMotion]:
        bodies = self._backend().estimate_bodies(clip, tracks)
        out: dict[int, SubjectMotion] = {}
        self.dropped_frames, self.dropped_offpitch, self.dropped_subjects = 0, 0, []
        for tl in tracks.tracklets:
            if tl.cls == "ball":
                continue  # the ball has its own BallTracker; HMR is for people only
            raw = bodies.get(tl.track_id)
            if raw is None:
                continue
            # Ground ONLY where the plane was measured. A carried homography is stale by however
            # many frames the calibrator failed for, and a foot un-projected through it lands near
            # the wrong horizon, where a pixel is metres — kilometres once a phone zooms in. Rows
            # without a solved plane are dropped here so the subject simply has no measurement
            # there; `add_temporal_coherence` then marks them `imputed` (R-6) and the #135 criteria
            # read them as what they are. Placing them anyway is the one thing that must not happen.
            keep = calibration.solved_mask(tl.frames, self.min_calib_confidence)
            # A solved plane is not the same as a sane un-projection. Measured on the fan clip
            # 2026-08-07: six frames whose calibration confidence was 0.546-0.575 — the TOP band —
            # put a foot 141-874 m from the pitch centre, while every frame below confidence 0.5
            # landed on the pitch. Confidence scores how well the homography fits the landmarks it
            # can see; it says nothing about a foot pixel that happens to sit near that
            # homography's vanishing line, where un-projection diverges. So the second test is on
            # the OUTPUT, and it is the physical one: a player is on a football pitch.
            if keep.any():
                world = self._ground_root(
                    replace(tl, frames=tl.frames[keep], bboxes_xyxy=tl.bboxes_xyxy[keep]),
                    calibration, None,
                )[:, :2]
                sane = self._on_pitch(world)
                self.dropped_offpitch += int((~sane).sum())
                keep[np.flatnonzero(keep)[~sane]] = False
            self.dropped_frames += int((~keep).sum())
            if not keep.any():
                # Not "lost by the tracker" — we never had a ground plane for any frame of his
                # life, so there is no position to infer from. Reported, not silently skipped.
                self.dropped_subjects.append(int(tl.track_id))
                continue
            kept_tl = tl if keep.all() else replace(
                tl, frames=tl.frames[keep], bboxes_xyxy=tl.bboxes_xyxy[keep]
            )
            rows = _align_rows(raw.frames, kept_tl.frames)
            height = raw.pelvis_above_foot[rows] if raw.pelvis_above_foot is not None else None
            transl = _smooth_path(
                self._ground_root(kept_tl, calibration, height), self.smooth_window
            )
            pose = PoseSequence(
                frames=kept_tl.frames,
                global_orient=raw.global_orient[rows],
                body_pose=raw.body_pose[rows],
                transl=transl,
            )
            out[tl.track_id] = SubjectMotion(shape=SmplxShape(betas=raw.betas), pose=pose)
        return out

    def refit(
        self, clip: ClipRef, motion: SubjectMotion, constraints: dict, frames: np.ndarray,
    ) -> SubjectMotion:
        """Apply operator constraints on ``frames`` and return a NEW motion (non-destructive).

        Geometric constraints are honoured purely here (no network): ``root_z_nudge`` (float, m),
        ``root_xy`` ((2,) world lock), ``foot_anchor`` (the measured homography ground anchor —
        ``(2,)`` or per-frame ``(M, 2)`` aligned to the selected frames — pulled in by
        ``anchor_blend`` ∈ ``[0, 1]``, default a hard lock), ``foot_floor`` (clamp root Z ≥ floor),
        ``relax_to_rest`` (scale body pose toward rest in ``[0, 1]``). ``complete_occlusions``
        (truthy) first routes the selected frames through an injected :class:`OcclusionBackend`
        (amodal cluster-occlusion completion, gated — R-8); validate its output against the
        homography anchor (:mod:`pitch3d.core.correction.anchor`). The correction engine wraps the
        result as a REFIT correction, so this stays a pure function of its inputs.
        """
        sel = np.asarray(frames, dtype=int).reshape(-1)
        refined = motion.copy()
        if constraints.get("complete_occlusions"):
            refined = self._complete_occlusions(clip, refined, sel)
        rows = np.isin(refined.pose.frames, sel)

        relax = constraints.get("relax_to_rest")
        if relax is not None:
            refined.pose.body_pose[rows] *= float(relax)
        nudge = float(constraints.get("root_z_nudge", 0.0))
        if nudge:
            refined.pose.transl[rows, 2] += nudge
        xy = constraints.get("root_xy")
        if xy is not None:
            refined.pose.transl[rows, :2] = np.asarray(xy, dtype=float).reshape(2)
        anchor = constraints.get("foot_anchor")
        if anchor is not None:
            refined.pose.transl[rows, :2] = blend_to_anchor(
                refined.pose.transl[rows, :2], anchor,
                float(constraints.get("anchor_blend", 1.0)),
            )
        floor = constraints.get("foot_floor")
        if floor is not None:
            refined.pose.transl[rows, 2] = np.maximum(refined.pose.transl[rows, 2], float(floor))
        return refined

    def _complete_occlusions(
        self, clip: ClipRef, motion: SubjectMotion, frames: np.ndarray
    ) -> SubjectMotion:
        """Amodally complete occluded ``frames`` via the injected backend (gated — R-8).

        Raises an actionable error when no :class:`OcclusionBackend` is wired, pointing at the
        measured structural alternative (``core.correction.coherence`` gap-fill) which needs no
        model. A real backend (Diffusion-VAS + SAM-3) returns a re-fit motion to validate against
        the homography anchor.
        """
        if self.occlusion_backend is None:
            raise NotImplementedError(
                "occlusion completion (`complete_occlusions`) needs an OcclusionBackend "
                "(Diffusion-VAS amodal masks + SAM-3 masklets) — install the `occlusion` extra or "
                "inject one via --occlusion-backend. The measured structural gap-fill/extension in "
                "pitch3d.core.correction.coherence (--coherence) bridges interior occlusions with "
                "no model and ships today."
            )
        return self.occlusion_backend.complete_occlusions(clip, motion, frames)

    def _ground_root(
        self,
        tl: Tracklet,
        calibration: FieldCalibration,
        pelvis_above_foot: np.ndarray | None = None,
    ) -> np.ndarray:
        """Place the root on the pitch: bbox foot point → world metres via the homography.

        The root Z is the pelvis height above the foot plane (z=0): the per-frame
        ``pelvis_above_foot`` the backend computed from SMPL-X FK (the T2 foot-plane anchor,
        varying with crouch/run/stride) when present, else the fixed nominal
        :attr:`pelvis_height_m`. Foot XY comes from the bbox bottom-centre through the homography.
        """
        foot_uv = np.column_stack(
            [(tl.bboxes_xyxy[:, 0] + tl.bboxes_xyxy[:, 2]) / 2.0, tl.bboxes_xyxy[:, 3]]
        )
        world_xy = np.stack(
            [calibration.image_to_world(int(f), foot_uv[i])[0] for i, f in enumerate(tl.frames)]
        )
        if pelvis_above_foot is None:
            z = np.full(tl.frames.shape[0], self.pelvis_height_m)
        else:
            z = np.asarray(pelvis_above_foot, dtype=float).reshape(-1)
        return np.column_stack([world_xy, z])

    def _backend(self) -> HMRBackend:
        return self.backend or GVHMRBackend(device=self.device)


def _align_rows(raw_frames: np.ndarray, want_frames: np.ndarray) -> np.ndarray:
    """Row indices selecting ``want_frames`` out of (sorted) ``raw_frames``; raises if uncovered."""
    idx = np.clip(np.searchsorted(raw_frames, want_frames), 0, raw_frames.shape[0] - 1)
    if not np.array_equal(raw_frames[idx], want_frames):
        raise ValueError("HMR backend frames do not cover the tracklet frames")
    return idx


@dataclass
class GVHMRBackend:
    """Real GVHMR inference: lazy torch/HMR imports, no import cost.

    Imports the heavy stack only when :meth:`estimate_bodies` is first called, so this module
    stays import-safe without the ``hmr`` extra installed.
    """

    weights: str | None = None
    device: str = "cuda"
    _model: object = None

    def estimate_bodies(  # pragma: no cover - heavy path
        self, clip: ClipRef, tracks: Tracks
    ) -> dict[int, RawBodyMotion]:
        self._load()
        raise NotImplementedError(
            "GVHMR inference is not wired yet (roadmap M1/P2.4) and is GPU-bound. Unlike the "
            "rfdetr/bytetrack reals, GVHMR is a research repo (not a pip package), so the `hmr` "
            "extra ships only its substrate (torch/smplx/chumpy, non-commercial SMPL-X) — not the "
            "network or its weights — and it is the one stage that may not be CPU-viable even for "
            "a single clip (ADR-0009). The pure half (root grounding + assembly + refit) is "
            "complete and tested; inject an HMRBackend that yields per-tracklet SMPL-X params, or "
            "keep the fake (`--pose fake`) until GVHMR is wired."
        )

    def _load(self) -> object:  # pragma: no cover - exercised only without the extra
        if self._model is None:
            try:
                import torch  # noqa: F401  (stand-in for the GVHMR/SMPL-X stack)
            except ImportError as exc:
                raise RuntimeError(
                    "GVHMR is not installed. Install the HMR extra: "
                    "`pip install 'pitch3d[hmr]'`, or inject an HMRBackend."
                ) from exc
            self._model = object()
        return self._model


@dataclass
class DiffusionVasOcclusionBackend:
    """Real cluster-occlusion completion — gated (R-8), not wired yet (roadmap M3-2).

    Amodal video completion (**Diffusion-VAS**) + pixel-level identity (**SAM-3** masklets) give a
    plausible body over the occluded segments, driving a constraint-guided re-fit of the hidden
    frames. Both are GPU-bound research repos (not pip packages), so the ``occlusion`` extra ships
    no weights/network and :meth:`complete_occlusions` raises an actionable error. The measured
    alternative — structural gap-fill/extension (``pitch3d.core.correction.coherence``) — needs no
    model and ships today; this seam is for *generative* completion of long cluster occlusions.
    """

    weights: str | None = None
    device: str = "cuda"
    _model: object = None

    def complete_occlusions(  # pragma: no cover - heavy path
        self, clip: ClipRef, motion: SubjectMotion, frames: np.ndarray
    ) -> SubjectMotion:
        raise NotImplementedError(
            "amodal occlusion completion is not wired yet (roadmap M3-2) and is GPU-bound: "
            "Diffusion-VAS amodal masks + SAM-3 masklets are research repos (not pip packages), so "
            "the `occlusion` extra ships no weights/network. Use the measured structural "
            "gap-fill/extension (pitch3d.core.correction.coherence, --coherence), which needs no "
            "model, or inject an OcclusionBackend via --occlusion-backend. Validate any completion "
            "against the homography anchor (pitch3d.core.correction.anchor) — off-anchor "
            "frames are hallucinated, not measured (R-6)."
        )
