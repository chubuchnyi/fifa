"""ViewSynthesizer seam B — the data amplifier (M3-3/M3-4, ADR-0007, AC-5b/AC-7).

Seam B feeds the *reconstruction*, not the eye: ``amplify`` turns the mono broadcast camera into N
pseudo-multi-views (FR-30) and ``inpaint_occlusions`` hallucinates a subject's unseen sides (FR-31),
both attached to the scene's ``synth_views`` and then *consumed* by env/avatar reconstruction. These
pin the two controller use-cases, the cache (ADR-0004), the per-subject routing of subject-only
inpaint vs. scene-shared amplify views, the **observable** "accepted by reconstruction" contract
(the fake records how many views it consumed — AC-5b), and the R-8 gate on the real generative
backend — all with no GPU/diffusion (AC-7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeViewSynthesizer
from pitch3d.adapters.viewsynth import GenerativeViewSynthesizer
from pitch3d.app.wiring import default_ports
from pitch3d.core.ports.io import CropRef
from pitch3d.core.scene.assets import SynthViewSeam


class _CountingAmplify(FakeViewSynthesizer):
    """Counts backend re-synthesis to prove ``amplify`` is content-addressed/cached (ADR-0004)."""

    calls = 0

    def amplify(self, clip, n_views, deviation):
        self.calls += 1
        return super().amplify(clip, n_views, deviation)


# --- amplify: mono → pseudo-multi-view (FR-30) ---------------------------------
def test_amplify_yields_n_bamplify_views_with_bounded_overlap(reconstructed):
    app, scene_id = reconstructed
    refs = app.amplify_views(scene_id, n_views=3, deviation=0.4)
    assert len(refs) == 3
    assert all(r.seam is SynthViewSeam.B_AMPLIFY for r in refs)
    assert all(Path(r.uri).exists() for r in refs)
    assert all(r.frustum_overlap == pytest.approx(1.0 - 0.4) for r in refs)  # R-14/R-16


def test_amplify_attaches_and_dedupes_on_synth_views(reconstructed):
    app, scene_id = reconstructed
    refs = app.amplify_views(scene_id, n_views=2, deviation=0.3)
    app.amplify_views(scene_id, n_views=2, deviation=0.3)  # same bounds ⇒ same ids ⇒ no dupes
    assert [v.id for v in app.get_scene(scene_id).synth_views] == [r.id for r in refs]


def test_amplify_is_cached_no_recompute(app, clip):
    app.ports.viewsynth = _CountingAmplify(out_dir=app.out_dir / "synth")
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    app.amplify_views(scene_id, n_views=4, deviation=0.3)
    app.amplify_views(scene_id, n_views=4, deviation=0.3)
    assert app.ports.viewsynth.calls == 1  # 2nd call is a cache hit (ADR-0004)


def test_amplify_params_are_distinct_cache_entries(app, clip):
    app.ports.viewsynth = _CountingAmplify(out_dir=app.out_dir / "synth")
    episode = app.register_clip(clip, name="t")
    scene_id = app.run_reconstruction(episode.id)
    app.amplify_views(scene_id, n_views=4, deviation=0.2)
    app.amplify_views(scene_id, n_views=4, deviation=0.5)  # different bound ⇒ recompute
    assert app.ports.viewsynth.calls == 2


# --- inpaint: a subject's unseen sides (FR-31) ---------------------------------
def test_inpaint_yields_a_binpaint_ref_tagged_for_the_subject(reconstructed):
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[0].track_id
    ref = app.inpaint_subject(scene_id, tid)
    assert ref.seam is SynthViewSeam.B_INPAINT
    assert ref.subject_track_id == tid
    assert ref.frustum_overlap == pytest.approx(0.6)  # plausible, not exact (R-16)
    assert ref.id in {v.id for v in app.get_scene(scene_id).synth_views}


def test_inpaint_defaults_to_a_placeholder_crop(reconstructed):
    # The pipeline does not source real crops yet; a zero placeholder still routes by track_id.
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[1].track_id
    assert app.inpaint_subject(scene_id, tid).subject_track_id == tid


def test_inpaint_accepts_explicit_ref_crops(reconstructed):
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[0].track_id
    crop = CropRef(subject_track_id=tid, uri="x", frame=2, bbox_xyxy=np.array([0.0, 0.0, 1.0, 1.0]))
    assert app.inpaint_subject(scene_id, tid, ref_crops=[crop]).subject_track_id == tid


# --- AC-5b: the views are ACCEPTED BY RECONSTRUCTION (observable) ---------------
def test_env_reconstruction_consumes_amplified_views(reconstructed):
    app, scene_id = reconstructed
    app.amplify_views(scene_id, n_views=3, deviation=0.3)
    assert app.build_env(scene_id).extra["synth_views"] == 3  # AC-5b: multi-view input accepted


def test_env_ignores_subject_inpaint_views(reconstructed):
    app, scene_id = reconstructed
    tid = app.get_scene(scene_id).subjects[0].track_id
    app.inpaint_subject(scene_id, tid)  # subject-only seam-B view, not environment input
    assert app.build_env(scene_id).extra["synth_views"] == 0


def test_avatars_consume_shared_amplify_plus_their_own_inpaint(reconstructed):
    app, scene_id = reconstructed
    subjects = app.get_scene(scene_id).subjects
    assert len(subjects) >= 2  # the routing claim is only meaningful with multiple subjects
    tid0 = subjects[0].track_id
    app.amplify_views(scene_id, n_views=2, deviation=0.3)  # shared by every subject
    app.inpaint_subject(scene_id, tid0)                    # only subject tid0
    refs = {r.subject_track_id: r for r in app.build_avatars(scene_id)}
    assert refs[tid0].extra["synth_views"] == 3            # 2 amplified + 1 own inpaint
    for other in subjects[1:]:
        assert refs[other.track_id].extra["synth_views"] == 2  # amplified only


def test_avatars_record_zero_synth_views_without_seam_b(reconstructed):
    app, scene_id = reconstructed
    refs = app.build_avatars(scene_id)
    assert all(r.extra["synth_views"] == 0 for r in refs)


# --- AC-7: real generative backend stays gated (R-8) ---------------------------
def test_generative_amplify_is_gated_actionably(clip):
    with pytest.raises(NotImplementedError, match=r"viewsynth"):
        GenerativeViewSynthesizer().amplify(clip, 4, 0.3)


def test_generative_inpaint_points_at_the_fake(reconstructed):
    with pytest.raises(NotImplementedError, match=r"FakeViewSynthesizer"):
        GenerativeViewSynthesizer().inpaint_occlusions([])


def test_wiring_generative_selector_is_the_gated_backend(tmp_path):
    ports = default_ports(out_dir=tmp_path / "out", viewsynth="generative")
    assert isinstance(ports.viewsynth, GenerativeViewSynthesizer)


def test_wiring_rejects_unknown_viewsynth(tmp_path):
    with pytest.raises(ValueError, match=r"generative"):
        default_ports(out_dir=tmp_path / "out", viewsynth="nope")
