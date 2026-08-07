"""Dependency injection — bundle the ports and build the default (fake-backed) app.

This is the composition root: the only place that knows about concrete adapters. The CLI and
the MCP server both call :func:`build_app`, so the LLM and the human drive the *same* wiring
(ADR-0008). ``default_ports`` returns the dependency-free fakes so the whole tool runs with no
GPU/Blender/models; swapping a real adapter is a one-line change here.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.ports.cache import Cache
from ..core.ports.export import Exporter
from ..core.ports.jobs import JobQueue
from ..core.ports.motion_prior import MotionPrior
from ..core.ports.observation import SceneObserver
from ..core.ports.perception import BallTracker, Detector, FieldCalibrator, Tracker
from ..core.ports.pose import PoseEstimator
from ..core.ports.reconstruction import AvatarBuilder, EnvReconstructor
from ..core.ports.render import RenderPass
from ..core.ports.view_synthesizer import ViewSynthesizer
from .controller import Application


@dataclass
class AppPorts:
    """Every adapter the application needs, behind its port type (ADR-0001)."""

    detector: Detector
    tracker: Tracker
    calibrator: FieldCalibrator
    pose: PoseEstimator
    ball: BallTracker
    env: EnvReconstructor
    avatar: AvatarBuilder
    viewsynth: ViewSynthesizer
    observer: SceneObserver
    render: RenderPass
    exporter: Exporter
    cache: Cache
    queue: JobQueue
    motion_prior: MotionPrior
    model_version: str = "fake-0"


def _resolve_backend(spec: str, protocol: Any) -> Any:
    """Import + instantiate a bring-your-own heavy backend from a dotted path (ADR-0006).

    ``spec`` is ``"package.module:Factory"`` (or ``"package.module.Factory"``) naming a
    zero-arg-constructible class or factory returning an object that satisfies ``protocol`` (one
    of the runtime-checkable backend protocols — ``HMRBackend``/``BallDetectionBackend``/
    ``KeypointBackend``). This is the seam that lets a workstation/GPU box inject a vendored
    GVHMR/TrackNet/keypoint network into the real adapter **without forking the wiring** — the
    research code stays out of the core tree (ADR-0001). Raises ``ValueError`` with an actionable
    message on a bad path or a backend that does not implement the protocol.
    """
    module_name, sep, attr = spec.partition(":")
    if not sep:
        module_name, _, attr = spec.rpartition(".")
    if not module_name or not attr:
        raise ValueError(
            f"backend spec {spec!r} must be 'package.module:Factory' (or 'package.module.Factory')"
        )
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"cannot import backend {spec!r}: {exc}") from exc
    backend = factory() if callable(factory) else factory
    if not isinstance(backend, protocol):
        raise ValueError(
            f"backend {spec!r} does not implement {protocol.__name__} (missing its method)"
        )
    return backend


def _bytetrack_backend(device: str):
    """The real association backend, with #133's mask cue attached when one is staged.

    `PITCH3D_MASK_CUE` points at a `track_masks.npz` from `scripts/build_track_masks.py`. Unset —
    the default — means plain ByteTrack, byte for byte, so this cannot change a run by accident;
    it exists so the cue can reach the *rendered* scene, not only the offline A/B harness.
    """
    import os

    from ..adapters.models.tracking import ByteTrackBackend, MaskCue

    path = os.environ.get("PITCH3D_MASK_CUE")
    cue = None
    if path:
        import numpy as np

        blob = np.load(path)
        cue = MaskCue(labels={
            int(f): lab for f, lab in zip(blob["frames"], blob["labels"], strict=True)
        })
        print(f"== tracking: mask cue ON from {path} ({len(cue.labels)} frames)")
    return ByteTrackBackend(device=device, mask_cue=cue)


def default_ports(
    *, out_dir: str | Path = "out", n_subjects: int = 4,
    detector: str = "fake", tracker: str = "fake", calibrator: str = "fake", pose: str = "fake",
    ball: str = "fake", env: str = "fake", avatar: str = "fake", render: str = "fake",
    export: str = "fake", observer: str = "fake", viewsynth: str = "fake",
    device: str = "cpu", detector_weights: str | None = None, detector_classes: str = "coco",
    pose_backend: str | None = None, ball_backend: str | None = None,
    calibrator_backend: str | None = None, tracker_backend: str | None = None,
    avatar_backend: str | None = None, occlusion_backend: str | None = None,
    motion_prior: str = "fake", camera_carry: int = 8, kit_split: bool = True,
    min_calib_confidence: float | None = None,
) -> AppPorts:
    """Default wiring: deterministic, dependency-free fakes, writing artifacts under ``out_dir``.

    ``detector`` / ``tracker`` / ``calibrator`` / ``pose`` / ``ball`` select real adapters so the
    fakes can be swapped one at a time (roadmap M1). ``detector``: ``"fake"`` (default) or
    ``"rfdetr"``. ``tracker``: ``"fake"`` or ``"bytetrack"`` (ByteTrack + team clustering).
    ``calibrator``: ``"fake"`` or ``"keypoints"`` (pitch-keypoint DLT homography). ``pose``:
    ``"fake"`` or ``"gvhmr"`` (SMPL-X HMR, ``hmr`` extra). ``ball``: ``"fake"`` or ``"tracknet"``
    (TrackNet 2D, ``ball`` extra). ``avatar``: ``"fake"``, ``"textured"`` (measured
    pixel-projection onto the tracked SMPL-X — the M2-0 *primary* realism path: the projection /
    visibility / per-vertex-colour-averaging half runs with no GPU, the SMPL-X meshing heavy half
    is gated behind the ``avatar`` extra), or ``"gaussian"`` (M3-1 strategy #3: one measured 3DGS
    splat anchored per SMPL-X vertex — same measured init runs GPU-free, the generative
    densify/inpaint refiner (IDOL/LHM/GART) is gated, R-8).
    ``env``: ``"fake"`` (placeholder marker) or ``"pitch"``
    (M2-1: the measured calibration-anchored pitch line markings as a vertex-coloured PLY — every
    vertex ``measured=1``, rendered by the splat pass; 3DGS/NeRF/generative env stay gated, R-8).
    ``render``: ``"fake"``, ``"overlay"`` (reproject the
    resolved 3D back onto per-frame PNGs — dependency-free, no extra), ``"splat"`` (M2-3: splat
    the measured M2-2 avatar meshes with a z-buffer, R-6-tinting unmeasured verts — the no-dep
    debug viz), ``"cycles"`` (M2-7: render those same measured meshes through Blender/Cycles for a
    real photoreal frame, R-6 tint intact, rigid-root placement only — needs ``$PITCH3D_BLENDER``)
    or ``"orbit"``
    (M2-5: ViewSynthesizer seam-A limited-orbit re-shoot of the source clip — a photoreal *video,
    not editable*; the authoritative cached path is ``Application.render_orbit``). ``export``:
    ``"fake"``
    or ``"gltf"`` (real SMPL-X ``.npz`` + canonical JSON now; glTF/GLB gated behind the
    ``export`` extra).
    ``observer``: ``"fake"`` (stdlib PNGs), ``"blender"`` (real proxy ``SCENE_3D`` via a
    ``blender --background`` subprocess) or ``"cycles"`` (M2-10/A-8: photoreal ``SCENE_3D`` — the
    resolved measured scene Cycles-rendered per viewpoint; needs ``$PITCH3D_BLENDER``).
    ``viewsynth`` (both ADR-0007 seams): ``"fake"`` (the deterministic seam-A/B stand-in),
    ``"cycles"`` (M2-10/A-9: the non-generative seam-A backend that re-renders the reconstructed 3D
    scene at the orbit cameras; ``Application.render_orbit`` then yields a photoreal *video, not
    editable*; needs ``$PITCH3D_BLENDER``), or ``"generative"`` (the real diffusion backend —
    importable but every method raises until the ``viewsynth`` extra is wired, R-8). The real model
    adapters need their extra (+ weights/GPU) at *call* time; importing them stays light.

    ``device`` (default ``"cpu"``, the local concept-validation profile) is forwarded to every
    real perception adapter; pass ``"cuda"`` where a GPU exists. Each adapter's own dataclass
    default stays ``"cuda"`` (production intent) — the composition root is the seam that picks
    the deployment profile. ``detector_weights`` is an optional RF-DETR weights path (the only
    adapter exposing a user-settable weights path today; others source weights in-backend).
    ``detector_classes`` selects the RF-DETR class map: ``"coco"`` (default — the validation
    profile, pairing the free COCO base weights' ids with the vocabulary, person→"player") or
    ``"sports"`` (the fine-tuned Roboflow checkpoint passed via ``detector_weights``, which splits
    players/goalkeepers/referees). Same adapter-vs-root split as ``device``: the adapter dataclass
    defaults to the sports map (production intent), the root to ``"coco"`` (what runs for free).

    ``camera_carry`` is the half-window (frames) over which ``calibrator="keypoints"`` re-estimates
    each frame's homography from its neighbours, carried on Lucas-Kanade inter-frame motion (R2,
    #104). CPU only. ``0`` disables it, which is what independent still images require.

    ``pose_backend`` / ``ball_backend`` / ``calibrator_backend`` / ``tracker_backend`` /
    ``avatar_backend`` inject a bring-your-own heavy backend by dotted path
    (``"package.module:Factory"``) into the matching real adapter — the on-box seam for wiring a
    vendored GVHMR/TrackNet/keypoint/tracking/SMPL-X-meshing network without forking this wiring
    (ADR-0006, see :func:`_resolve_backend`). Each requires its real adapter to be selected (e.g.
    ``pose_backend`` needs ``pose="gvhmr"``, ``avatar_backend`` needs ``avatar="textured"``);
    pairing one with ``"fake"`` raises. ``occlusion_backend`` injects a cluster-occlusion completer
    (Diffusion-VAS + SAM-3, M3-2) into the GVHMR adapter (needs ``pose="gvhmr"``); the re-fit only
    calls it when a correction sets ``complete_occlusions``.

    ``motion_prior`` (M3-8) selects the temporal denoiser a ``TEMPORAL_SMOOTHING`` correction with
    ``method="learned"`` calls: ``"fake"`` (default — the real GPU-free gaussian denoiser),
    ``"learned"`` (the gated HTD-Refine/StableMotion model — importable, raises until the ``motion``
    extra is wired, R-8), or a dotted-path ``"package.module:Factory"`` BYO ``MotionPrior``
    (ADR-0006). The pure ``moving_average``/``gaussian`` smoothing methods need no prior at all.
    """
    from ..adapters.fakes import (
        DiskCache,
        FakeAvatarBuilder,
        FakeBallTracker,
        FakeDetector,
        FakeEnvReconstructor,
        FakeExporter,
        FakeFieldCalibrator,
        FakeMotionPrior,
        FakePoseEstimator,
        FakeRenderPass,
        FakeSceneObserver,
        FakeTracker,
        FakeViewSynthesizer,
        InProcessJobQueue,
    )

    if detector == "fake":
        det: Detector = FakeDetector(n_subjects=n_subjects)
    elif detector == "rfdetr":
        from ..adapters.models import DETECTOR_CLASS_MAPS, RFDETRDetector

        if detector_classes not in DETECTOR_CLASS_MAPS:
            raise ValueError(
                f"unknown detector_classes {detector_classes!r}; "
                f"expected one of {sorted(DETECTOR_CLASS_MAPS)}"
            )
        det = RFDETRDetector(
            device=device, weights=detector_weights,
            class_map=dict(DETECTOR_CLASS_MAPS[detector_classes]),
        )
    else:
        raise ValueError(f"unknown detector {detector!r}; expected 'fake' or 'rfdetr'")

    if tracker == "fake":
        if tracker_backend:
            raise ValueError("tracker_backend requires --tracker bytetrack")
        trk: Tracker = FakeTracker()
    elif tracker == "bytetrack":
        from ..adapters.models import ByteTrackTracker
        from ..adapters.models.tracking import TrackingBackend

        trk = ByteTrackTracker(
            device=device,
            # Drop un-stitchable 1-frame singletons (no velocity -> noise, not a real track).
            # Kept low so genuine multi-frame fragments survive for the stitch pass to re-link
            # (raising this further would starve stitch, which runs AFTER this filter). See #202.
            min_track_frames=2,
            # #132: cut a track where its kit colour changes team. Measured on the target clip,
            # 9 of 38 tracks carried an avatar onto a different human; this takes that to 0.
            kit_split=kit_split,
            backend=_resolve_backend(tracker_backend, TrackingBackend)
            if tracker_backend else _bytetrack_backend(device),
        )
    else:
        raise ValueError(f"unknown tracker {tracker!r}; expected 'fake' or 'bytetrack'")

    if calibrator == "fake":
        if calibrator_backend:
            raise ValueError("calibrator_backend requires --calibrator keypoints")
        cal: FieldCalibrator = FakeFieldCalibrator()
    elif calibrator == "keypoints":
        from ..adapters.models import KeypointFieldCalibrator
        from ..adapters.models.calibration import KeypointBackend, LucasKanadeMotion

        cal = KeypointFieldCalibrator(
            device=device,
            backend=_resolve_backend(calibrator_backend, KeypointBackend)
            if calibrator_backend else None,
            # The per-frame solve swims by median 0.119 m between neighbouring frames while the
            # camera pans smoothly; carrying it on the measured inter-frame motion removes 92 %
            # (#104). CPU only. Set 0 to score each frame independently, as still-image evaluation
            # must — there is no motion to carry between unrelated frames.
            motion=LucasKanadeMotion() if camera_carry > 0 else None,
            carry_window=camera_carry,
        )
    else:
        raise ValueError(f"unknown calibrator {calibrator!r}; expected 'fake' or 'keypoints'")

    if pose == "fake":
        if pose_backend:
            raise ValueError("pose_backend requires --pose gvhmr")
        if occlusion_backend:
            raise ValueError("occlusion_backend requires --pose gvhmr")
        pse: PoseEstimator = FakePoseEstimator()
    elif pose == "gvhmr":
        from ..adapters.models import GVHMRPoseEstimator
        from ..adapters.models.pose import HMRBackend, OcclusionBackend

        pse = GVHMRPoseEstimator(
            device=device,
            backend=_resolve_backend(pose_backend, HMRBackend) if pose_backend else None,
            occlusion_backend=(
                _resolve_backend(occlusion_backend, OcclusionBackend)
                if occlusion_backend else None
            ),
            **({} if min_calib_confidence is None
               else {"min_calib_confidence": float(min_calib_confidence)}),
        )
    else:
        raise ValueError(f"unknown pose {pose!r}; expected 'fake' or 'gvhmr'")

    if ball == "fake":
        if ball_backend:
            raise ValueError("ball_backend requires --ball tracknet")
        blt: BallTracker = FakeBallTracker()
    elif ball == "tracknet":
        from ..adapters.models import TrackNetBallTracker
        from ..adapters.models.ball import BallDetectionBackend

        blt = TrackNetBallTracker(
            device=device,
            backend=_resolve_backend(ball_backend, BallDetectionBackend)
            if ball_backend else None,
        )
    else:
        raise ValueError(f"unknown ball {ball!r}; expected 'fake' or 'tracknet'")

    out = Path(out_dir)
    # One synthesizer instance backs both seams (seam-A render selector + render_orbit, and seam-B
    # amplify/inpaint). 'fake' is the deterministic stand-in (both seams, no deps); 'cycles' is the
    # non-generative seam-A backend that re-renders the reconstructed 3D scene at the orbit cameras
    # (A-9); 'generative' is the real diffusion backend — importable but every method raises until
    # the `viewsynth` extra is wired (R-8, ADR-0007), so it never silently runs.
    if viewsynth == "fake":
        vs: ViewSynthesizer = FakeViewSynthesizer(out_dir=out / "synth")
    elif viewsynth == "cycles":
        from ..adapters.render import CyclesViewSynthesizer

        vs = CyclesViewSynthesizer(out_dir=out / "synth")
    elif viewsynth == "generative":
        from ..adapters.viewsynth import GenerativeViewSynthesizer

        vs = GenerativeViewSynthesizer()
    else:
        raise ValueError(
            f"unknown viewsynth {viewsynth!r}; expected 'fake', 'cycles' or 'generative'"
        )
    if render == "fake":
        rnd: RenderPass = FakeRenderPass(out_dir=out / "render")
    elif render == "overlay":
        from ..adapters.render import ReprojectionOverlayRenderPass

        rnd = ReprojectionOverlayRenderPass(out_dir=out / "render")
    elif render == "splat":
        from ..adapters.render import SplatAvatarRenderPass

        rnd = SplatAvatarRenderPass(out_dir=out / "render")
    elif render == "cycles":
        from ..adapters.render import CyclesRenderPass

        rnd = CyclesRenderPass(out_dir=out / "render")
    elif render == "orbit":
        from ..adapters.render import ViewSynthOrbitRenderPass

        rnd = ViewSynthOrbitRenderPass(synthesizer=vs)
    else:
        raise ValueError(
            f"unknown render {render!r}; expected 'fake', 'overlay', 'splat', 'cycles' or 'orbit'"
        )

    if export == "fake":
        exp: Exporter = FakeExporter()
    elif export in ("gltf", "threejs"):
        from ..adapters.export import GltfExporter

        exp = GltfExporter()
    else:
        raise ValueError(f"unknown export {export!r}; expected 'fake', 'gltf' or 'threejs'")

    if observer == "fake":
        obs: SceneObserver = FakeSceneObserver(out_dir=out / "observations")
    elif observer == "blender":
        from ..adapters.blender import BlenderSceneObserver

        obs = BlenderSceneObserver(out_dir=out / "observations")
    elif observer == "cycles":
        from ..adapters.render import CyclesSceneObserver

        obs = CyclesSceneObserver(out_dir=out / "observations")
    else:
        raise ValueError(f"unknown observer {observer!r}; expected 'fake', 'blender' or 'cycles'")

    if env == "fake":
        env_rec: EnvReconstructor = FakeEnvReconstructor(out_dir=out / "assets")
    elif env == "pitch":
        from ..adapters.models import MeasuredPitchEnvReconstructor

        env_rec = MeasuredPitchEnvReconstructor(out_dir=out / "assets")
    else:
        raise ValueError(f"unknown env {env!r}; expected 'fake' or 'pitch'")

    if avatar == "fake":
        if avatar_backend:
            raise ValueError("avatar_backend requires --avatar textured or gaussian")
        avt: AvatarBuilder = FakeAvatarBuilder(out_dir=out / "assets")
    elif avatar == "textured":
        from ..adapters.models import TexturedSmplxAvatarBuilder
        from ..adapters.models.avatar import AvatarMeshBackend

        avt = TexturedSmplxAvatarBuilder(
            out_dir=out / "assets",
            device=device,
            backend=_resolve_backend(avatar_backend, AvatarMeshBackend)
            if avatar_backend else None,
        )
    elif avatar == "gaussian":
        from ..adapters.models import GaussianAvatarBuilder
        from ..adapters.models.avatar import AvatarMeshBackend

        avt = GaussianAvatarBuilder(
            out_dir=out / "assets",
            device=device,
            mesh_backend=_resolve_backend(avatar_backend, AvatarMeshBackend)
            if avatar_backend else None,
        )
    else:
        raise ValueError(f"unknown avatar {avatar!r}; expected 'fake', 'textured' or 'gaussian'")

    # Learned motion-prior denoiser (M3-8), engaged via a TEMPORAL_SMOOTHING correction with
    # method="learned": "fake" is the real GPU-free gaussian denoiser; "learned" is the gated
    # HTD-Refine/StableMotion model (raises until `motion` extra is wired, R-8); anything else is
    # a dotted-path BYO MotionPrior (ADR-0006). The pure moving_average/gaussian methods need none.
    if motion_prior == "fake":
        mp: MotionPrior = FakeMotionPrior()
    elif motion_prior == "learned":
        from ..adapters.models import LearnedMotionPrior

        mp = LearnedMotionPrior(device=device)
    else:
        mp = _resolve_backend(motion_prior, MotionPrior)

    return AppPorts(
        detector=det,
        tracker=trk,
        calibrator=cal,
        pose=pse,
        ball=blt,
        env=env_rec,
        avatar=avt,
        viewsynth=vs,
        observer=obs,
        render=rnd,
        exporter=exp,
        cache=DiskCache(root=out / "cache"),
        queue=InProcessJobQueue(),
        motion_prior=mp,
    )


def build_app(*, out_dir: str | Path = "out", ports: AppPorts | None = None) -> Application:
    """Build the application controller (fakes by default; pass ``ports`` to override)."""
    return Application(ports=ports or default_ports(out_dir=out_dir), out_dir=Path(out_dir))
