"""Top-down tactical radar — pure 2D minimap maths + the observe ``include_radar`` wiring.

The radar drops the camera and draws the resolved world straight down onto the pitch plane
(FR-14, UX-3, M1 step 10). The pixel-mapping is pure numpy and checked directly; ``render_radar``
is exercised end-to-end (valid PNG, deterministic, dots only where subjects/ball are) and the
:class:`SceneObserver` ``include_radar`` opt-in is checked on the fake + the bare base contract.
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.fakes import FakeSceneObserver
from pitch3d.adapters.render.radar import radar_to_world, render_radar, world_to_radar
from pitch3d.core.ports.observation import (
    ObservationImage,
    ObservationKind,
    SceneObserver,
    Viewpoint,
)
from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.layers import ConfidenceMap
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.subject import Subject

_PITCH = {"length": 105.0, "width": 68.0, "px_w": 96, "px_h": 64, "margin": 4}


def _subject(make_motion, transl_z=0.0):
    return Subject(track_id=1, proposal=make_motion(range(4), transl_z=transl_z))


# --- pure pixel mapping --------------------------------------------------------
def test_world_centre_maps_to_image_centre():
    uv = world_to_radar(np.array([[0.0, 0.0]]), **_PITCH)
    np.testing.assert_allclose(uv[0], [96 / 2, 64 / 2])


def test_pitch_corners_map_to_the_inner_border():
    # top-left of the pitch (-L/2, +W/2) → (margin, margin); bottom-right → (W-margin, H-margin).
    corners = np.array([[-52.5, 34.0], [52.5, -34.0]])
    uv = world_to_radar(corners, **_PITCH)
    np.testing.assert_allclose(uv[0], [4, 4])
    np.testing.assert_allclose(uv[1], [96 - 4, 64 - 4])


def test_radar_to_world_inverts_world_to_radar():
    # a drag lands on a pixel; the inverse must recover the world XY the dot came from (ADR-0010).
    world = np.array([[0.0, 0.0], [-52.5, 34.0], [52.5, -34.0], [17.3, -9.1], [-4.0, 22.5]])
    uv = world_to_radar(world, **_PITCH)
    np.testing.assert_allclose(radar_to_world(uv, **_PITCH), world, atol=1e-9)


def test_radar_centre_pixel_maps_to_world_origin():
    np.testing.assert_allclose(radar_to_world(np.array([[96 / 2, 64 / 2]]), **_PITCH), [[0.0, 0.0]])


# --- render pass behaviour -----------------------------------------------------
def test_render_radar_is_valid_png_and_deterministic(make_scene, make_motion):
    scene = make_scene(subjects=[_subject(make_motion)])
    a = render_radar(scene, 0)
    b = render_radar(scene, 0)
    assert a[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG signature
    assert a == b                          # pure + stdlib zlib ⇒ byte-for-byte reproducible


def test_render_radar_draws_a_subject(make_scene, make_motion):
    populated = render_radar(make_scene(subjects=[_subject(make_motion)]), 0)
    empty = render_radar(make_scene(), 0)  # bare pitch, no subjects/ball
    assert populated != empty


def test_render_radar_includes_the_ball(make_scene):
    frames = np.arange(4)
    ball = BallTrack(
        frames=frames, positions_3d=np.zeros((4, 3)), height_confidence=np.ones(4)
    )
    assert render_radar(make_scene(ball=ball), 0) != render_radar(make_scene(), 0)


def test_render_radar_low_confidence_tints_the_subject(make_scene, make_motion):
    subj = _subject(make_motion)
    full = render_radar(make_scene(subjects=[subj]), 0)
    low = render_radar(
        make_scene(subjects=[subj], confidence=ConfidenceMap(subject_frame_conf={1: np.zeros(4)})),
        0,
    )
    assert full != low  # same dot, warning-tinted ⇒ different pixels (UX-3 parity with overlay)


# --- observe() include_radar wiring -------------------------------------------
def test_fake_observer_appends_radar_only_when_requested(tmp_path, make_scene, make_motion):
    obs = FakeSceneObserver(out_dir=tmp_path / "o")
    scene = make_scene(subjects=[_subject(make_motion)])
    assert not any(i.kind is ObservationKind.RADAR for i in obs.observe(scene).images)
    img = next(
        i for i in obs.observe(scene, include_radar=True).images if i.kind is ObservationKind.RADAR
    )
    assert img.viewpoint is Viewpoint.TOP
    assert img.uri.endswith("_radar_00000.png")


class _BareObserver(SceneObserver):
    """Implements only the abstract methods — does NOT override ``capture_radar``."""

    def capture_scene_views(self, scene, views, *, quality=RenderQuality.PREVIEW):
        return []

    def capture_frame_overlay(self, scene, frame):
        return ObservationImage(kind=ObservationKind.FRAME_OVERLAY, uri="x", frame=frame)

    def capture_ui(self, scene=None):
        return None


def test_base_capture_radar_defaults_to_none(make_scene):
    obs = _BareObserver().observe(make_scene(), include_radar=True)
    assert obs.images == []  # the headless default: opt-in, None unless an adapter overrides
