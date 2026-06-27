"""'Needs attention' UI panel + confidence map — the visual half of UX-4/FR-17 (M3-5).

The ranking (:func:`attention_list`) is already pure-core tested; here we pin the **picture**: the
dense per-subject×per-frame confidence heatmap (team colour pulled toward red as confidence drops,
honest background where nothing is measured), the ranked attention bars (most-urgent on top, bar
length ∝ severity, hue ∝ reason), and the :class:`FakeSceneObserver` wiring that returns it as the
``UI`` observation. All pure numpy + the stdlib PNG encoder — deterministic, pixel-testable, no
GPU/Blender/font engine (AC-7).
"""

from __future__ import annotations

import numpy as np

from pitch3d.adapters.fakes.observer import FakeSceneObserver
from pitch3d.adapters.render.attention import (
    _OK_COLOR,
    render_attention_panel,
    render_attention_ui,
    render_confidence_map,
)
from pitch3d.adapters.render.overlay import _BACKGROUND, _LOW_CONF_COLOR, _PLAYER_COLOR
from pitch3d.core.ports.observation import ObservationKind
from pitch3d.core.scene.layers import ConfidenceMap
from pitch3d.core.scene.motion import BallTrack
from pitch3d.core.scene.subject import Subject

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _subject(make_motion, track_id=1, n=2):
    return Subject(track_id=track_id, proposal=make_motion(range(n)))


def _ball(height_confidence):
    hc = np.asarray(height_confidence, dtype=float)
    return BallTrack(
        frames=np.arange(hc.shape[0]),
        positions_3d=np.tile([1.0, 0.0, 3.0], (hc.shape[0], 1)),
        height_confidence=hc,
    )


# --- confidence map ------------------------------------------------------------
def test_confidence_map_pulls_low_confidence_cells_toward_red(make_scene, make_motion):
    cmap = ConfidenceMap(subject_frame_conf={1: np.array([1.0, 0.0])})
    scene = make_scene(subjects=[_subject(make_motion)], confidence=cmap)
    fb = render_confidence_map(scene, width=2, height=1)  # 1 subject row, 2 frame cols
    assert tuple(fb[0, 0]) == _PLAYER_COLOR          # conf=1 → untouched base colour
    assert tuple(fb[0, 1]) == _LOW_CONF_COLOR        # conf=0 → full warning red


def test_confidence_map_is_flat_background_without_confidence(make_scene, make_motion):
    scene = make_scene(subjects=[_subject(make_motion)])  # no ConfidenceMap → nothing measured
    fb = render_confidence_map(scene, width=4, height=4)
    assert np.all(fb == np.array(_BACKGROUND, dtype=np.uint8))  # honest: don't fabricate (R-6)


def test_confidence_map_renders_a_ball_height_row(make_scene, make_motion):
    scene = make_scene(subjects=[_subject(make_motion)], ball=_ball([0.0]))
    fb = render_confidence_map(scene, width=1, height=2)  # row 0 = subject (no conf), row 1 = ball
    assert tuple(fb[0, 0]) == _BACKGROUND             # subject has no confidence → background
    assert tuple(fb[1, 0]) == _LOW_CONF_COLOR         # ball height conf=0 → red


# --- needs-attention panel -----------------------------------------------------
def test_attention_panel_ranks_worst_first_and_colours_by_reason(make_scene, make_motion):
    # reproj 50px → score 5.0 (orange, ranks first); conf 0.1 → score 0.8 (red, shorter bar)
    cmap = ConfidenceMap(
        subject_frame_conf={1: np.array([0.1])},
        reprojection_error_px={1: np.array([50.0])},
    )
    scene = make_scene(subjects=[_subject(make_motion, n=1)], confidence=cmap)
    panel = render_attention_panel(scene, width=100, height=2)
    assert tuple(panel[0, 0]) == (255, 140, 0)        # row 0: high_reprojection = orange
    assert tuple(panel[0, 99]) == (255, 140, 0)       # worst item → full-width bar
    assert tuple(panel[1, 0]) == _LOW_CONF_COLOR      # row 1: low_confidence = red
    assert tuple(panel[1, 99]) == _BACKGROUND         # lower score → shorter bar


def test_attention_panel_shows_an_all_clear_bar_when_nothing_flagged(make_scene, make_motion):
    scene = make_scene(subjects=[_subject(make_motion)])  # no confidence, no ball → empty list
    panel = render_attention_panel(scene, width=100, height=4)
    assert tuple(panel[0, 0]) == _OK_COLOR            # short green 'all clear' bar
    assert tuple(panel[0, 99]) == _BACKGROUND


# --- composite UI + observer wiring -------------------------------------------
def test_render_attention_ui_returns_a_valid_png(make_scene, make_motion):
    cmap = ConfidenceMap(subject_frame_conf={1: np.array([0.2, 0.9])})
    scene = make_scene(subjects=[_subject(make_motion)], confidence=cmap)
    png = render_attention_ui(scene, width=64, height=64)
    assert png.startswith(_PNG_MAGIC) and len(png) > len(_PNG_MAGIC)


def test_capture_ui_writes_the_real_panel_for_a_scene(tmp_path, make_scene, make_motion):
    cmap = ConfidenceMap(subject_frame_conf={1: np.array([0.1, 0.9])})
    scene = make_scene(subjects=[_subject(make_motion)], confidence=cmap)
    obs = FakeSceneObserver(out_dir=tmp_path)
    img = obs.capture_ui(scene)
    assert img is not None and img.kind is ObservationKind.UI
    assert (img.width, img.height) == (obs.ui_width, obs.ui_height)
    assert "attention" in (img.note or "")
    data = (tmp_path / f"{scene.id}_ui.png").read_bytes()
    assert data == render_attention_ui(scene, width=obs.ui_width, height=obs.ui_height)


def test_capture_ui_headless_is_the_flat_placeholder(tmp_path):
    obs = FakeSceneObserver(out_dir=tmp_path)
    img = obs.capture_ui(None)
    assert img is not None and img.kind is ObservationKind.UI
    assert (img.width, img.height) == (obs.width, obs.height)  # the flat placeholder, not the panel
