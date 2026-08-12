"""Bring-your-own heavy backend over a dotted path (ADR-0006, roadmap M1 steps 5–7).

The GPU/workstation seam: a vendored GVHMR/TrackNet/keypoint network is injected into the real
adapter from the composition root (and the CLI) **without forking the wiring**, so the research
code stays out of the core tree (ADR-0001). The live networks can't run in CI, but the *seam* —
dotted-path resolution, protocol guarding, injection, and the override-needs-a-real-adapter rule
— is verified here headlessly, including a full CPU reconstruction driven by an in-repo stub
backend (proving an injected backend actually unblocks the otherwise-``NotImplementedError`` stub).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.ball import BallDetectionBackend
from pitch3d.adapters.models.pose import HMRBackend, RawBodyMotion
from pitch3d.app.wiring import _resolve_backend, build_app, default_ports
from pitch3d.core.ports.io import ClipRef


# --- in-repo stub backends (stand in for the vendored networks an on-box engineer would wire) ---
class StubHMRBackend:
    """Deterministic ``HMRBackend``: zero articulation for every non-ball tracklet."""

    def estimate_bodies(self, clip: ClipRef, tracks) -> dict[int, RawBodyMotion]:
        out: dict[int, RawBodyMotion] = {}
        for tl in tracks.tracklets:
            if tl.cls == "ball":
                continue
            t = tl.frames.shape[0]
            out[tl.track_id] = RawBodyMotion(
                track_id=tl.track_id, frames=tl.frames,
                global_orient=np.zeros((t, 3)), body_pose=np.zeros((t, 21, 3)),
                betas=np.zeros(10),
            )
        return out


class StubBallBackend:
    """Minimal ``BallDetectionBackend`` (presence of ``detect_ball`` is the protocol contract)."""

    def detect_ball(self, clip: ClipRef):  # pragma: no cover - not called in these tests
        raise NotImplementedError


class StubKeypointBackend:
    """Minimal ``KeypointBackend`` (presence of ``detect_keypoints`` is the protocol contract)."""

    def detect_keypoints(self, clip: ClipRef):  # pragma: no cover - not called in these tests
        raise NotImplementedError


class StubTrackingBackend:
    """Minimal ``TrackingBackend`` (presence of ``associate`` is the protocol contract)."""

    def associate(self, clip: ClipRef, detections):  # pragma: no cover - not called in these tests
        raise NotImplementedError


class StubAvatarMeshBackend:
    """Minimal ``AvatarMeshBackend`` (presence of ``observe`` is the protocol contract)."""

    def observe(self, subject, ref_crops):  # pragma: no cover - not called in these tests
        raise NotImplementedError


class NotABackend:
    """Implements none of the backend protocols — the guardrail must reject it."""


# --- _resolve_backend: dotted-path import + protocol guard --------------------------
def test_resolve_backend_imports_colon_form():
    backend = _resolve_backend(f"{__name__}:StubHMRBackend", HMRBackend)
    assert isinstance(backend, StubHMRBackend)


def test_resolve_backend_imports_dotted_form():
    backend = _resolve_backend(f"{__name__}.StubBallBackend", BallDetectionBackend)
    assert isinstance(backend, StubBallBackend)


def test_resolve_backend_rejects_a_malformed_spec():
    with pytest.raises(ValueError, match="must be 'package.module:Factory'"):
        _resolve_backend("justaname", HMRBackend)


def test_resolve_backend_reports_an_unimportable_path():
    with pytest.raises(ValueError, match="cannot import backend"):
        _resolve_backend("pitch3d.nope:Missing", HMRBackend)


def test_resolve_backend_enforces_the_protocol():
    with pytest.raises(ValueError, match="does not implement HMRBackend"):
        _resolve_backend(f"{__name__}:NotABackend", HMRBackend)


# --- default_ports injects the resolved backend into the real adapter ----------------
def test_pose_backend_is_injected(tmp_path):
    ports = default_ports(
        out_dir=tmp_path / "o", pose="gvhmr", pose_backend=f"{__name__}:StubHMRBackend"
    )
    assert isinstance(ports.pose.backend, StubHMRBackend)


def test_ball_backend_is_injected(tmp_path):
    ports = default_ports(
        out_dir=tmp_path / "o", ball="tracknet", ball_backend=f"{__name__}:StubBallBackend"
    )
    assert isinstance(ports.ball.backend, StubBallBackend)


def test_calibrator_backend_is_injected(tmp_path):
    ports = default_ports(
        out_dir=tmp_path / "o", calibrator="keypoints",
        calibrator_backend=f"{__name__}:StubKeypointBackend",
    )
    assert isinstance(ports.calibrator.backend, StubKeypointBackend)


def test_tracker_backend_is_injected(tmp_path):
    ports = default_ports(
        out_dir=tmp_path / "o", tracker="bytetrack",
        tracker_backend=f"{__name__}:StubTrackingBackend",
    )
    assert isinstance(ports.tracker.backend, StubTrackingBackend)


def test_avatar_backend_is_injected(tmp_path):
    ports = default_ports(
        out_dir=tmp_path / "o", avatar="textured",
        avatar_backend=f"{__name__}:StubAvatarMeshBackend",
    )
    assert isinstance(ports.avatar.backend, StubAvatarMeshBackend)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pose": "fake", "pose_backend": "x:Y"}, "pose_backend requires"),
        ({"ball": "fake", "ball_backend": "x:Y"}, "ball_backend requires"),
        ({"calibrator": "fake", "calibrator_backend": "x:Y"}, "calibrator_backend requires"),
        ({"tracker": "fake", "tracker_backend": "x:Y"}, "tracker_backend requires"),
        ({"avatar": "fake", "avatar_backend": "x:Y"}, "avatar_backend requires"),
    ],
)
def test_backend_override_requires_its_real_adapter(tmp_path, kwargs, message):
    with pytest.raises(ValueError, match=message):
        default_ports(out_dir=tmp_path / "o", **kwargs)


# --- end to end: an injected backend unblocks the otherwise-stubbed stage on CPU -----
def test_injected_pose_backend_drives_a_real_reconstruction(tmp_path):
    out = tmp_path / "out"
    ports = default_ports(
        out_dir=out, n_subjects=3, pose="gvhmr", pose_backend=f"{__name__}:StubHMRBackend"
    )
    app = build_app(out_dir=out, ports=ports)
    clip = ClipRef(
        source_id="demo", uri="memory://demo.mp4", frames=np.arange(6),
        width=1280, height=720, fps=25.0,
    )
    episode = app.register_clip(clip, name="demo episode")
    scene_id = app.run_reconstruction(episode.id)  # would raise NotImplementedError on the stub

    scene = app.get_scene(scene_id)
    assert scene.subjects
    transl = scene.subjects[0].proposal.pose.transl
    assert transl.shape[1] == 3
    assert np.isfinite(transl).all()  # roots grounded on the pitch, not NaN


# --- the cache must not hide a backend swap (found 2026-08-12) ------------------------
class _OtherStubHMRBackend(StubHMRBackend):
    """Same contract, different class — the only thing that differs between the two runs."""


def test_swapping_the_backend_invalidates_the_stage_cache(tmp_path):
    """A second run with a different backend must not be served the first one's result.

    Measured on the real pipeline before this was wired: BoT-SORT requested into an out-dir
    holding ByteTrack's cache reproduced ByteTrack's 38 subjects exactly, and no log line said
    the requested backend never ran. `ModelInfo.params` was always documented as feeding the
    cache key; nothing read it.
    """
    from pitch3d.core.orchestration.pipeline import ReconstructionPipeline
    from pitch3d.core.orchestration.stages import Stage

    out = tmp_path / "out"
    clip = ClipRef(
        source_id="demo", uri="memory://demo.mp4", frames=np.arange(6),
        width=1280, height=720, fps=25.0,
    )

    def pose_run(backend_path: str):
        p = default_ports(
            out_dir=out, n_subjects=3, pose="gvhmr", pose_backend=backend_path
        )
        pipeline = ReconstructionPipeline(
            detector=p.detector, tracker=p.tracker, calibrator=p.calibrator,
            pose=p.pose, ball=p.ball, cache=p.cache, queue=p.queue,
        )
        result = pipeline.run(clip)
        run = next(r for r in result.runs if r.stage is Stage.POSE)
        return run.key, run.cache_hit

    key_a, _ = pose_run(f"{__name__}:StubHMRBackend")
    key_b, hit_b = pose_run(f"{__name__}:_OtherStubHMRBackend")
    assert key_a != key_b, "the two backends share a POSE cache key — a swap would be silent"
    assert not hit_b, "the second backend was served the first one's cached result"

    key_a2, hit_a2 = pose_run(f"{__name__}:StubHMRBackend")
    assert (key_a2, hit_a2) == (key_a, True), "identical config must still hit the cache"
