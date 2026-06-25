"""ViewSynthesizer seam A — the limited-orbit re-shoot (M2-5, ADR-0007, AC-4).

Seam A feeds the *eye*, never the reconstruction: it yields a photoreal **video, not editable**,
content-addressed and cached so the expensive generative pass never recomputes for the same
clip/orbit, and clearly flagged so it can't masquerade as an editable result (R-15). These pin
the controller use-case (``render_orbit``), the cache, the ``synth_views`` attachment, and the
``--render orbit`` selector wiring.
"""

from __future__ import annotations

from pathlib import Path

from pitch3d.adapters.fakes import FakeViewSynthesizer
from pitch3d.adapters.render import ViewSynthOrbitRenderPass, orbit_render_result
from pitch3d.app import build_app
from pitch3d.app.wiring import default_ports
from pitch3d.core.ports.render import RenderQuality
from pitch3d.core.scene.assets import SynthViewSeam


class _CountingSynth(FakeViewSynthesizer):
    """A fake that records how many times the backend actually re-shoots (cache-miss probe)."""

    calls = 0

    def render_orbit(self, clip, target_camera, scene_hints=None):
        self.calls += 1
        return super().render_orbit(clip, target_camera, scene_hints)


def test_render_orbit_yields_a_noneditable_video_ref(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id, max_deviation_deg=20.0)
    assert ref.seam is SynthViewSeam.A_RENDER
    assert ref.editable is False
    assert "not editable" in (ref.note or "")
    assert Path(ref.uri).exists()
    # the camera it re-shot along is a *prescribed* bounded orbit over the clip frames
    assert ref.camera.estimated is False
    assert ref.camera.frames.shape[0] == 8


def test_render_orbit_attaches_and_dedupes_on_synth_views(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id)
    app.render_orbit(scene_id)  # same clip/orbit ⇒ same id ⇒ no duplicate
    views = app.get_scene(scene_id).synth_views
    assert [v.id for v in views] == [ref.id]


def test_render_orbit_is_cached_no_recompute(app, clip):
    app.ports.viewsynth = _CountingSynth(out_dir=app.out_dir / "synth")
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    app.render_orbit(scene_id)
    app.render_orbit(scene_id)
    assert app.ports.viewsynth.calls == 1  # second call is a cache hit (ADR-0004)


def test_changing_the_orbit_deviation_is_a_new_cache_entry(app, clip):
    app.ports.viewsynth = _CountingSynth(out_dir=app.out_dir / "synth")
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    app.render_orbit(scene_id, max_deviation_deg=10.0)
    app.render_orbit(scene_id, max_deviation_deg=30.0)  # different params ⇒ recompute
    assert app.ports.viewsynth.calls == 2


def test_orbit_render_result_wraps_synthviewref_as_video(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id)
    result = orbit_render_result(ref, RenderQuality.PREVIEW)
    assert result.is_video is True
    assert result.n_frames == 8
    assert "not editable" in result.note


def test_render_selector_orbit_yields_a_video(tmp_path, clip):
    ports = default_ports(out_dir=tmp_path / "out", render="orbit")
    app = build_app(out_dir=tmp_path / "out", ports=ports)
    assert isinstance(app.ports.render, ViewSynthOrbitRenderPass)
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    result = app.render(scene_id)
    assert result.is_video is True
    assert result.n_frames == clip.n_frames
    assert "not editable" in result.note


# --- M2-6 fast low-q preview on the generative seam (UX-9) ---------------------------------------
def test_render_orbit_preview_downscales_the_reshot_camera(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id)  # default quality="preview"
    # the bounded orbit is composed at 1280x720; a preview re-shoots it at half-res 640x360
    assert (ref.camera.intrinsics.width, ref.camera.intrinsics.height) == (640, 360)
    assert ref.camera.frames.shape[0] == 8  # frame count preserved through the pixel rescale


def test_render_orbit_final_is_full_resolution(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id, quality="final")
    assert (ref.camera.intrinsics.width, ref.camera.intrinsics.height) == (1280, 720)


def test_preview_and_final_are_distinct_cache_entries(app, clip):
    app.ports.viewsynth = _CountingSynth(out_dir=app.out_dir / "synth")
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    app.render_orbit(scene_id, quality="preview")
    app.render_orbit(scene_id, quality="final")  # different quality ⇒ recompute, not a cache hit
    assert app.ports.viewsynth.calls == 2


def test_orbit_render_result_note_shows_size_and_quality(reconstructed):
    app, scene_id = reconstructed
    ref = app.render_orbit(scene_id)  # preview → 640x360
    result = orbit_render_result(ref, RenderQuality.PREVIEW)
    assert "640x360 preview" in result.note
    assert "not editable" in result.note
