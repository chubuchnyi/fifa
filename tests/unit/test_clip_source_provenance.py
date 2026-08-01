"""#125 — a scene must not be silently drawn over footage it was not reconstructed from.

A scene carries no pixels, so poseannot overlays it on whatever video the clip entry points at and
the result looks plausible either way. That is how the wrong-clip run reached the switcher: its
scene said `source_id: "clip"` while it was paired with the Colombia video, and it scored a healthy
1.0 px because `apply_rigid_camera.py` had replaced the calibration.
"""

from __future__ import annotations

import pytest

from poseannot.clips import Clip, _scene_source_id


def _clip(video: str | None, source_id: str | None) -> Clip:
    return Clip(
        id="c", label="c", source_video=video, scene_json="s.json",
        corrections_out="e.json", has_scene=True, scene_source_id=source_id,
    )


def test_the_pod_0801_pairing_is_flagged():
    # The real shape of the defect: a builtin borrowing the default clip's video, paired with a
    # scene reconstructed from `/runpod/clip.mp4`.
    c = _clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4", "clip")
    assert c.source_mismatch


def test_a_matching_pair_is_not_flagged():
    c = _clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4", "Colombia-1-0-Congo-DR1080p")
    assert not c.source_mismatch


@pytest.mark.parametrize("source_id", [None, ""])
def test_a_scene_that_does_not_say_is_unknown_not_mismatched(source_id):
    # Old scenes predate the field. Unknown must never be reported as wrong (R-6).
    assert not _clip("samples/video/Colombia-1-0-Congo-DR1080p.mp4", source_id).source_mismatch


def test_uploads_are_exempt_because_their_filename_is_normalised():
    # create_clip_from_upload stores every upload as `video.<ext>`, so the stem is not evidence.
    assert not _clip("poseannot/clips/x/video.mp4", "clip").source_mismatch


def test_source_id_is_read_without_parsing_the_whole_scene(tmp_path):
    p = tmp_path / "scene.json"
    p.write_text('{\n  "__type__": "Scene",\n  "fields": {\n    "source_id": "some-match",\n'
                 '    "subjects": [' + "0," * 100000 + "0]}}")
    assert _scene_source_id(p) == "some-match"


def test_an_unreadable_or_silent_scene_reads_as_unknown(tmp_path):
    assert _scene_source_id(tmp_path / "absent.json") is None
    assert _scene_source_id(None) is None
    silent = tmp_path / "silent.json"
    silent.write_text('{"__type__": "Scene", "fields": {"id": "scene-1"}}')
    assert _scene_source_id(silent) is None
