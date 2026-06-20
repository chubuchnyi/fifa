"""Live Blender edit bridge — a GUI drag becomes a canonical Correction (ADR-0010, ADR-0008).

The live GUI session itself needs a display + a Blender binary (not available in CI, the same
honest limitation as the GVHMR/TrackNet reals), so the *protocol* and the *host edit loop* are
verified here headlessly: drags are applied through the same ``apply_offset`` use-case the MCP
agent drives, a dragged subject lands exactly where it was dropped, and a real socket pair carries
the newline-JSON messages — acks, error-skips and EOF/bye termination all without touching Blender.
"""

from __future__ import annotations

import json
import socket

import numpy as np
import pytest

from pitch3d.adapters.blender.live import apply_drag, resolved_root_at, serve_edits
from pitch3d.adapters.blender.proxy import (
    build_proxy_plan,
    parse_subject_name,
    subject_object_name,
)
from pitch3d.app.wiring import build_app, default_ports
from pitch3d.core.ports.io import ClipRef


def _wired_scene(tmp_path, *, n_frames: int = 6, n_subjects: int = 3):
    """A fake-wired app with one reconstructed scene; returns (app, scene_id, first track_id)."""
    out = tmp_path / "out"
    app = build_app(out_dir=out, ports=default_ports(out_dir=out, n_subjects=n_subjects))
    clip = ClipRef(
        source_id="demo", uri="memory://demo.mp4", frames=np.arange(n_frames),
        width=1280, height=720, fps=25.0,
    )
    episode = app.register_clip(clip, name="demo episode")
    scene_id = app.run_reconstruction(episode.id)
    tid = app.get_scene(scene_id).subjects[0].track_id
    return app, scene_id, tid


def _drain(sock: socket.socket) -> list[dict]:
    """Read all newline-JSON the host wrote back, until EOF (peer must shut its write end)."""
    data = b""
    while True:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]


# --- the id<->name contract the live editor relies on -------------------------------
def test_subject_name_round_trips():
    assert subject_object_name(7) == "subject_7"
    assert parse_subject_name("subject_7") == 7
    assert parse_subject_name(subject_object_name(123)) == 123
    assert parse_subject_name("ball") is None  # the ball is not a draggable subject
    assert parse_subject_name("subject_x") is None


def test_proxy_object_names_parse_back_to_track_ids(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    plan = build_proxy_plan(app.resolved(scene_id))
    subject_names = [o.name for o in plan.objects if o.kind == "subject"]
    assert subject_names  # the builder produced subjects
    assert tid in {parse_subject_name(n) for n in subject_names}  # the builder/bridge agree
    assert all(parse_subject_name(o.name) is None for o in plan.objects if o.kind == "ball")


# --- apply_drag: a drop becomes a ROOT_TRANSLATION correction -----------------------
def test_resolved_root_at_reads_the_resolved_position(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    root = resolved_root_at(app.resolved(scene_id), tid, 2)
    assert root.shape == (3,)


def test_resolved_root_at_rejects_a_missing_frame(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    with pytest.raises(ValueError, match="no frame"):
        resolved_root_at(app.resolved(scene_id), tid, 999)


def test_apply_drag_lands_a_2d_drop_and_keeps_pelvis_height(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    base = resolved_root_at(app.resolved(scene_id), tid, 2)
    n0 = len(app.get_scene(scene_id).corrections)

    corr = apply_drag(app, scene_id, tid, 2, [10.0, -5.0])  # a radar drag: XY only

    assert corr.target.kind.value == "root_translation"
    assert corr.target.subject_track_id == tid
    assert corr.mode.value == "constant_offset"
    assert len(app.get_scene(scene_id).corrections) == n0 + 1  # committed, non-destructively
    landed = resolved_root_at(app.resolved(scene_id), tid, 2)
    np.testing.assert_allclose(landed, [10.0, -5.0, base[2]], atol=1e-6)  # exact drop, same Z


def test_apply_drag_3d_drop_moves_all_axes(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    apply_drag(app, scene_id, tid, 2, [3.0, 4.0, 1.5])  # a 3D-viewport drag
    np.testing.assert_allclose(
        resolved_root_at(app.resolved(scene_id), tid, 2), [3.0, 4.0, 1.5], atol=1e-6
    )


def test_apply_drag_defaults_to_the_whole_track(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path, n_frames=6)
    frames = app.get_scene(scene_id).subjects[0].proposal.pose.frames
    corr = apply_drag(app, scene_id, tid, 2, [1.0, 1.0])
    assert [corr.frame_range.start, corr.frame_range.end] == [int(frames[0]), int(frames[-1])]


def test_apply_drag_honours_an_explicit_range(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    corr = apply_drag(app, scene_id, tid, 2, [1.0, 1.0], frame_range=(1, 3))
    assert [corr.frame_range.start, corr.frame_range.end] == [1, 3]


def test_apply_drag_rejects_a_bad_location_shape(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    with pytest.raises(ValueError, match="x, y"):
        apply_drag(app, scene_id, tid, 2, [1.0, 2.0, 3.0, 4.0])


# --- serve_edits: the socket loop the live Blender talks to --------------------------
def test_serve_edits_commits_drags_over_a_socket(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    host_end, peer = socket.socketpair()
    messages = [
        {"type": "hello", "scene_id": scene_id},
        {"type": "edit", "track_id": tid, "frame": 2, "location": [10.0, -5.0]},
        {"type": "edit", "track_id": tid, "frame": 2, "location": [3.0, 4.0, 1.5]},
        {"type": "bye"},
    ]
    peer.sendall("".join(json.dumps(m) + "\n" for m in messages).encode())

    applied = serve_edits(app, scene_id, host_end)

    assert len(applied) == 2
    assert all(c.target.kind.value == "root_translation" for c in applied)
    np.testing.assert_allclose(resolved_root_at(app.resolved(scene_id), tid, 2), [3.0, 4.0, 1.5],
                               atol=1e-6)  # last drag wins at that frame
    host_end.close()
    replies = _drain(peer)
    assert [r["type"] for r in replies] == ["ready", "ack", "ack"]
    assert replies[1]["correction_id"] == applied[0].id
    peer.close()


def test_serve_edits_skips_bad_messages_without_corrupting_the_scene(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    host_end, peer = socket.socketpair()
    lines = [
        "definitely not json",
        json.dumps({"type": "frobnicate"}),                                  # unknown type
        json.dumps({"type": "edit", "track_id": tid, "frame": 999, "location": [0, 0]}),  # no frame
        json.dumps({"type": "edit", "track_id": tid, "frame": 2, "location": [1.0, 2.0]}),  # good
        json.dumps({"type": "bye"}),
    ]
    peer.sendall(("\n".join(lines) + "\n").encode())

    applied = serve_edits(app, scene_id, host_end)

    assert len(applied) == 1  # only the well-formed drag committed
    assert len(app.get_scene(scene_id).corrections) == 1
    host_end.close()
    assert [r["type"] for r in _drain(peer)] == ["error", "error", "error", "ack"]
    peer.close()


def test_serve_edits_stops_on_eof(tmp_path):
    app, scene_id, tid = _wired_scene(tmp_path)
    host_end, peer = socket.socketpair()
    peer.sendall((json.dumps({"type": "edit", "track_id": tid, "frame": 2,
                              "location": [1.0, 2.0]}) + "\n").encode())
    peer.shutdown(socket.SHUT_WR)  # EOF for the host; the editor closed Blender mid-session

    applied = serve_edits(app, scene_id, host_end)

    assert len(applied) == 1  # the buffered drag still landed before EOF returned the loop
    host_end.close()
    peer.close()
