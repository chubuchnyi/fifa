"""Cycles photoreal observe + seam-A orbit (M2-10, A-8/A-9) — wiring + contract + a gated render.

A-8 (:class:`CyclesSceneObserver`) renders the resolved scene from each viewpoint through
Blender/Cycles, so the LLM loop sees the *same* photoreal pixels the final render produces, not a
proxy; the 2D radar/overlay/UI stay camera-free and delegate to the fake. A-9
(:class:`CyclesViewSynthesizer`) makes seam A honest — a limited orbit that *re-renders the measured
3D scene* at the orbit cameras (no generative; seam B stays gated, R-8). The error-prone wiring and
contracts are pinned with no Blender; two Blender-gated tests prove real photoreal pixels come out.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.blender import locate_blender
from pitch3d.adapters.models.avatar import write_vertex_colored_ply
from pitch3d.adapters.render import CyclesSceneObserver, CyclesViewSynthesizer
from pitch3d.adapters.render.cycles import CyclesRenderPass
from pitch3d.app.wiring import default_ports
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.observation import ObservationKind, Viewpoint, ViewpointCamera
from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.assets import RenderAssetKind, RenderAssetRef, SynthViewSeam
from pitch3d.core.scene.camera import CameraIntrinsics, CameraTrack
from pitch3d.core.scene.provenance import Backend, ModelInfo
from pitch3d.core.scene.subject import Subject


def _camera(n: int = 1, w: int = 160, h: int = 120) -> CameraTrack:
    intr = CameraIntrinsics(fx=120.0, fy=120.0, cx=w / 2.0, cy=h / 2.0, width=w, height=h)
    return CameraTrack.identity(intr, n)


def _triangle_avatar_scene(tmp_path, make_scene, make_motion) -> object:
    """A minimal scene with one rigid triangle avatar at z=8 (no SMPL-X model needed)."""
    verts = np.array([[-0.3, -0.3, 0.0], [0.3, -0.3, 0.0], [0.0, 0.4, 0.0]])
    faces = np.array([[0, 1, 2]])
    rgb = np.array([[210, 90, 40]] * 3, dtype=np.uint8)
    uri = write_vertex_colored_ply(
        tmp_path / "avatar_7.ply", verts, faces, rgb, np.array([True, True, True])
    )
    subj = Subject(track_id=7, proposal=make_motion([0], transl_z=8.0))
    scene = make_scene(subjects=[subj])
    scene.render_assets = [
        RenderAssetRef(
            id="a-7", kind=RenderAssetKind.AVATAR_TEXTURED_SMPLX, uri=uri,
            model=ModelInfo(name="t", backend=Backend.FAKE), subject_track_id=7,
        )
    ]
    return scene


# --- wiring: the CLI selectors resolve to the Cycles adapters (no Blender) ------
def test_default_ports_observer_cycles_selects_the_cycles_observer(tmp_path):
    ports = default_ports(out_dir=tmp_path / "out", observer="cycles")
    assert isinstance(ports.observer, CyclesSceneObserver)


def test_default_ports_viewsynth_cycles_selects_the_cycles_synth(tmp_path):
    ports = default_ports(out_dir=tmp_path / "out", viewsynth="cycles")
    assert isinstance(ports.viewsynth, CyclesViewSynthesizer)


def test_default_ports_rejects_unknown_viewsynth(tmp_path):
    with pytest.raises(ValueError):
        default_ports(out_dir=tmp_path / "out", viewsynth="nope")


# --- A-8 observer contract: empty/2D paths are Blender-free (delegate) ----------
def test_cycles_observer_empty_views_short_circuits(tmp_path, make_scene, make_motion):
    obs = CyclesSceneObserver(out_dir=tmp_path / "o")
    scene = make_scene(subjects=[Subject(7, make_motion([0]))])
    assert obs.capture_scene_views(scene, []) == []  # no views ⇒ no Blender


def test_cycles_observer_radar_delegates_to_the_fake(tmp_path, make_scene, make_motion):
    # The top-down radar is camera-free 2D, so it comes from the headless fallback (the fake),
    # never Blender — same split the proxy observer uses.
    obs = CyclesSceneObserver(out_dir=tmp_path / "o")
    scene = make_scene(subjects=[Subject(7, make_motion([0, 1]))])
    radar = obs.capture_radar(scene)
    assert radar is not None and radar.kind is ObservationKind.RADAR


# --- A-9 synth contract: scene-hint required, seam B gated, info LOCAL ----------
def test_cycles_viewsynth_info_is_local(tmp_path):
    vs = CyclesViewSynthesizer(out_dir=tmp_path / "s")
    assert vs.info().backend is Backend.LOCAL


def test_cycles_viewsynth_orbit_requires_the_resolved_scene_hint(tmp_path):
    # This backend re-renders the resolved 3D scene, so it needs scene_hints['scene'] — without it
    # there is nothing to re-shoot, and silently falling back to the clip would be a generative lie.
    vs = CyclesViewSynthesizer(out_dir=tmp_path / "s")
    clip = ClipRef(source_id="t", uri="", frames=np.arange(2), width=160, height=120, fps=0.0)
    cam = _camera(2)
    with pytest.raises(ValueError):
        vs.render_orbit(clip, cam, None)
    with pytest.raises(ValueError):
        vs.render_orbit(clip, cam, {})  # hints present but no 'scene'


def test_cycles_viewsynth_generative_seam_b_is_gated(tmp_path):
    vs = CyclesViewSynthesizer(out_dir=tmp_path / "s")
    clip = ClipRef(source_id="t", uri="", frames=np.arange(2), width=160, height=120, fps=0.0)
    with pytest.raises(NotImplementedError):
        vs.amplify(clip, n_views=3, deviation=10.0)
    with pytest.raises(NotImplementedError):
        vs.inpaint_occlusions([])


# --- A-8 Blender-gated: real photoreal SCENE_3D snapshots per viewpoint ---------
@pytest.mark.skipif(locate_blender() is None, reason="needs a Blender binary (env/PATH)")
def test_cycles_observer_captures_photoreal_scene_views(tmp_path, make_scene, make_motion):
    scene = _triangle_avatar_scene(tmp_path, make_scene, make_motion)
    obs = CyclesSceneObserver(
        out_dir=tmp_path / "o",
        render=CyclesRenderPass(out_dir=tmp_path / "o" / "_frames", samples=4),
    )
    views = [
        ViewpointCamera(Viewpoint.FRONT, _camera(1)),
        ViewpointCamera(Viewpoint.TOP, _camera(1)),
    ]
    images = obs.capture_scene_views(scene, views, quality=RenderQuality.PREVIEW)

    assert len(images) == 2
    assert all(i.kind is ObservationKind.SCENE_3D for i in images)
    assert all("cycles photoreal" in (i.note or "") for i in images)
    assert [i.viewpoint for i in images] == [Viewpoint.FRONT, Viewpoint.TOP]
    # each view is copied out to its own URI before the next overwrites the shared frame dir
    assert len({i.uri for i in images}) == 2
    for img in images:
        p = Path(img.uri)
        assert p.is_file() and p.stat().st_size > 0


# --- A-9 Blender-gated: the orbit re-renders the 3D scene (video, not editable) -
@pytest.mark.skipif(locate_blender() is None, reason="needs a Blender binary (env/PATH)")
def test_cycles_viewsynth_orbit_rerenders_the_resolved_scene(tmp_path, make_scene, make_motion):
    scene = _triangle_avatar_scene(tmp_path, make_scene, make_motion)
    vs = CyclesViewSynthesizer(
        out_dir=tmp_path / "s",
        render=CyclesRenderPass(out_dir=tmp_path / "s" / "orbit", samples=4),
    )
    clip = ClipRef(source_id="t", uri="", frames=np.arange(1), width=160, height=120, fps=0.0)
    ref = vs.render_orbit(clip, _camera(1), {"scene": scene, "max_deviation_deg": 20.0})

    assert ref.seam is SynthViewSeam.A_RENDER
    assert ref.editable is False
    assert "not editable" in (ref.note or "")
    # frustum_overlap falls with the requested re-aim — pin the exact formula (1 - dev/90)
    assert abs(ref.frustum_overlap - (1.0 - 20.0 / 90.0)) < 1e-6
    frame = Path(ref.uri) / "frame_00000.png"
    assert frame.is_file() and frame.stat().st_size > 0
