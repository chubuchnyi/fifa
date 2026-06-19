"""Dependency injection — bundle the ports and build the default (fake-backed) app.

This is the composition root: the only place that knows about concrete adapters. The CLI and
the MCP server both call :func:`build_app`, so the LLM and the human drive the *same* wiring
(ADR-0008). ``default_ports`` returns the dependency-free fakes so the whole tool runs with no
GPU/Blender/models; swapping a real adapter is a one-line change here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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


def default_ports(*, out_dir: str | Path = "out", n_subjects: int = 4) -> AppPorts:
    """All-fakes wiring: deterministic, dependency-free, writes artifacts under ``out_dir``."""
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

    out = Path(out_dir)
    return AppPorts(
        detector=FakeDetector(n_subjects=n_subjects),
        tracker=FakeTracker(),
        calibrator=FakeFieldCalibrator(),
        pose=FakePoseEstimator(),
        ball=FakeBallTracker(),
        env=FakeEnvReconstructor(out_dir=out / "assets"),
        avatar=FakeAvatarBuilder(out_dir=out / "assets"),
        viewsynth=FakeViewSynthesizer(out_dir=out / "synth"),
        observer=FakeSceneObserver(out_dir=out / "observations"),
        render=FakeRenderPass(out_dir=out / "render"),
        exporter=FakeExporter(),
        cache=DiskCache(root=out / "cache"),
        queue=InProcessJobQueue(),
    )


def build_app(*, out_dir: str | Path = "out", ports: AppPorts | None = None) -> Application:
    """Build the application controller (fakes by default; pass ``ports`` to override)."""
    return Application(ports=ports or default_ports(out_dir=out_dir), out_dir=Path(out_dir))
