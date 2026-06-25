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


def default_ports(
    *, out_dir: str | Path = "out", n_subjects: int = 4,
    detector: str = "fake", tracker: str = "fake", calibrator: str = "fake", pose: str = "fake",
    ball: str = "fake", avatar: str = "fake", render: str = "fake", export: str = "fake",
    observer: str = "fake",
    device: str = "cpu", detector_weights: str | None = None, detector_classes: str = "coco",
    pose_backend: str | None = None, ball_backend: str | None = None,
    calibrator_backend: str | None = None, tracker_backend: str | None = None,
    avatar_backend: str | None = None,
) -> AppPorts:
    """Default wiring: deterministic, dependency-free fakes, writing artifacts under ``out_dir``.

    ``detector`` / ``tracker`` / ``calibrator`` / ``pose`` / ``ball`` select real adapters so the
    fakes can be swapped one at a time (roadmap M1). ``detector``: ``"fake"`` (default) or
    ``"rfdetr"``. ``tracker``: ``"fake"`` or ``"bytetrack"`` (ByteTrack + team clustering).
    ``calibrator``: ``"fake"`` or ``"keypoints"`` (pitch-keypoint DLT homography). ``pose``:
    ``"fake"`` or ``"gvhmr"`` (SMPL-X HMR, ``hmr`` extra). ``ball``: ``"fake"`` or ``"tracknet"``
    (TrackNet 2D, ``ball`` extra). ``avatar``: ``"fake"`` or ``"textured"`` (measured
    pixel-projection onto the tracked SMPL-X — the M2-0 *primary* realism path: the projection /
    visibility / per-vertex-colour-averaging half runs with no GPU, the SMPL-X meshing heavy half
    is gated behind the ``avatar`` extra). ``render``: ``"fake"`` or ``"overlay"`` (reproject the
    resolved 3D back onto per-frame PNGs — dependency-free, no extra). ``export``: ``"fake"``
    or ``"gltf"`` (real SMPL-X ``.npz`` + canonical JSON now; glTF/GLB gated behind the
    ``export`` extra).
    ``observer``: ``"fake"`` (stdlib PNGs) or ``"blender"`` (real proxy ``SCENE_3D`` via a
    ``blender --background`` subprocess; needs the binary on ``$PITCH3D_BLENDER``/``PATH``). The
    real model adapters need their extra (+ weights/GPU) at *call* time; importing them stays light.

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

    ``pose_backend`` / ``ball_backend`` / ``calibrator_backend`` / ``tracker_backend`` /
    ``avatar_backend`` inject a bring-your-own heavy backend by dotted path
    (``"package.module:Factory"``) into the matching real adapter — the on-box seam for wiring a
    vendored GVHMR/TrackNet/keypoint/tracking/SMPL-X-meshing network without forking this wiring
    (ADR-0006, see :func:`_resolve_backend`). Each requires its real adapter to be selected (e.g.
    ``pose_backend`` needs ``pose="gvhmr"``, ``avatar_backend`` needs ``avatar="textured"``);
    pairing one with ``"fake"`` raises.
    """
    from ..adapters.fakes import (
        DiskCache,
        FakeAvatarBuilder,
        FakeBallTracker,
        FakeDetector,
        FakeEnvReconstructor,
        FakeExporter,
        FakeFieldCalibrator,
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
            backend=_resolve_backend(tracker_backend, TrackingBackend)
            if tracker_backend else None,
        )
    else:
        raise ValueError(f"unknown tracker {tracker!r}; expected 'fake' or 'bytetrack'")

    if calibrator == "fake":
        if calibrator_backend:
            raise ValueError("calibrator_backend requires --calibrator keypoints")
        cal: FieldCalibrator = FakeFieldCalibrator()
    elif calibrator == "keypoints":
        from ..adapters.models import KeypointFieldCalibrator
        from ..adapters.models.calibration import KeypointBackend

        cal = KeypointFieldCalibrator(
            device=device,
            backend=_resolve_backend(calibrator_backend, KeypointBackend)
            if calibrator_backend else None,
        )
    else:
        raise ValueError(f"unknown calibrator {calibrator!r}; expected 'fake' or 'keypoints'")

    if pose == "fake":
        if pose_backend:
            raise ValueError("pose_backend requires --pose gvhmr")
        pse: PoseEstimator = FakePoseEstimator()
    elif pose == "gvhmr":
        from ..adapters.models import GVHMRPoseEstimator
        from ..adapters.models.pose import HMRBackend

        pse = GVHMRPoseEstimator(
            device=device,
            backend=_resolve_backend(pose_backend, HMRBackend) if pose_backend else None,
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
    if render == "fake":
        rnd: RenderPass = FakeRenderPass(out_dir=out / "render")
    elif render == "overlay":
        from ..adapters.render import ReprojectionOverlayRenderPass

        rnd = ReprojectionOverlayRenderPass(out_dir=out / "render")
    else:
        raise ValueError(f"unknown render {render!r}; expected 'fake' or 'overlay'")

    if export == "fake":
        exp: Exporter = FakeExporter()
    elif export == "gltf":
        from ..adapters.export import GltfExporter

        exp = GltfExporter()
    else:
        raise ValueError(f"unknown export {export!r}; expected 'fake' or 'gltf'")

    if observer == "fake":
        obs: SceneObserver = FakeSceneObserver(out_dir=out / "observations")
    elif observer == "blender":
        from ..adapters.blender import BlenderSceneObserver

        obs = BlenderSceneObserver(out_dir=out / "observations")
    else:
        raise ValueError(f"unknown observer {observer!r}; expected 'fake' or 'blender'")

    if avatar == "fake":
        if avatar_backend:
            raise ValueError("avatar_backend requires --avatar textured")
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
    else:
        raise ValueError(f"unknown avatar {avatar!r}; expected 'fake' or 'textured'")

    return AppPorts(
        detector=det,
        tracker=trk,
        calibrator=cal,
        pose=pse,
        ball=blt,
        env=FakeEnvReconstructor(out_dir=out / "assets"),
        avatar=avt,
        viewsynth=FakeViewSynthesizer(out_dir=out / "synth"),
        observer=obs,
        render=rnd,
        exporter=exp,
        cache=DiskCache(root=out / "cache"),
        queue=InProcessJobQueue(),
    )


def build_app(*, out_dir: str | Path = "out", ports: AppPorts | None = None) -> Application:
    """Build the application controller (fakes by default; pass ``ports`` to override)."""
    return Application(ports=ports or default_ports(out_dir=out_dir), out_dir=Path(out_dir))
