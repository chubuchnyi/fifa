"""MCP driving adapter — pure dispatch over the fake-wired app + gated serve (ADR-0008).

The control surface (tool call → application use-case → text/image content blocks) is exercised
end to end on the dependency-free fakes with **no** ``mcp`` extra: every catalog tool dispatches,
``observe`` streams real PNG image blocks back to the agent, ``preview`` stays non-destructive,
and the live :func:`serve` fails *actionably* when the extra is absent — the same discipline the
model / export adapters follow. The SDK is never imported here.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from pitch3d.adapters.mcp import build_catalog, dispatch_tool, serve
from pitch3d.adapters.mcp.dispatch import ImageBlock, TextBlock
from pitch3d.app.wiring import build_app, default_ports
from pitch3d.core.ports.io import ClipRef

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _wired_app(tmp_path, *, n_frames: int = 6, n_subjects: int = 3):
    """A fully fake-wired Application with one registered, reconstructed episode."""
    out = tmp_path / "out"
    app = build_app(out_dir=out, ports=default_ports(out_dir=out, n_subjects=n_subjects))
    clip = ClipRef(
        source_id="demo", uri="memory://demo.mp4", frames=np.arange(n_frames),
        width=1280, height=720, fps=25.0,
    )
    episode = app.register_clip(clip, name="demo episode")
    return app, episode


def _one_text(blocks) -> dict:
    assert len(blocks) == 1 and isinstance(blocks[0], TextBlock)
    return json.loads(blocks[0].text)


def _reconstruct(app, episode_id: str) -> str:
    blocks = dispatch_tool(app, "run_reconstruction", {"episode_id": episode_id})
    return _one_text(blocks)["scene_id"]


def test_list_episodes_dispatches(tmp_path):
    app, episode = _wired_app(tmp_path)
    payload = _one_text(dispatch_tool(app, "list_episodes", {}))
    assert [e["id"] for e in payload] == [episode.id]
    assert payload[0]["n_frames"] == 6


def test_run_reconstruction_returns_scene_id(tmp_path):
    app, episode = _wired_app(tmp_path)
    scene_id = _reconstruct(app, episode.id)
    assert app.get_scene(scene_id).id == scene_id  # the controller really created it


def test_observe_streams_png_image_blocks(tmp_path):
    app, episode = _wired_app(tmp_path)
    scene_id = _reconstruct(app, episode.id)
    blocks = dispatch_tool(
        app, "observe",
        {"scene_id": scene_id, "frame": 2, "viewpoints": ["front", "top"], "include_ui": True},
    )
    assert isinstance(blocks[0], TextBlock) and scene_id in blocks[0].text  # manifest + summary
    images = [b for b in blocks if isinstance(b, ImageBlock)]
    assert len(images) >= 3  # two viewpoints + overlay + ui
    for img in images:
        assert img.mime_type == "image/png"
        assert img.data.startswith(_PNG_SIG)  # genuine pixels for the model to look at


def test_get_attention_dispatches_to_a_list(tmp_path):
    app, episode = _wired_app(tmp_path)
    scene_id = _reconstruct(app, episode.id)
    payload = _one_text(dispatch_tool(app, "get_attention", {"scene_id": scene_id}))
    assert isinstance(payload, list)


def test_apply_offset_then_preview_is_non_destructive(tmp_path):
    app, episode = _wired_app(tmp_path)
    scene_id = _reconstruct(app, episode.id)
    tid = app.get_scene(scene_id).subjects[0].track_id
    target = {"kind": "root_translation", "subject_track_id": tid}

    n0 = len(app.get_scene(scene_id).corrections)
    corr = _one_text(
        dispatch_tool(
            app, "apply_offset",
            {"scene_id": scene_id, "target": target, "frame_range": [0, 2], "delta": [0, 0, 0.1]},
        )
    )
    assert corr["mode"] == "constant_offset" and corr["target"]["subject_track_id"] == tid
    assert len(app.get_scene(scene_id).corrections) == n0 + 1  # the commit mutated

    candidate = {"target": target, "frame_range": [0, 2], "delta": [0, 0, 0.5]}
    preview = _one_text(
        dispatch_tool(app, "preview", {"scene_id": scene_id, "candidate": candidate})
    )
    assert preview["committed"] is False and preview["max_abs_change"] > 0
    assert len(app.get_scene(scene_id).corrections) == n0 + 1  # preview did NOT mutate (FR-23)


def test_render_and_export_dispatch(tmp_path):
    app, episode = _wired_app(tmp_path)
    scene_id = _reconstruct(app, episode.id)

    render = _one_text(dispatch_tool(app, "render", {"scene_id": scene_id}))
    assert render["n_frames"] == 6 and render["quality"] == "preview"

    out_path = str(tmp_path / "scene.json")
    export = _one_text(
        dispatch_tool(app, "export", {"scene_id": scene_id, "format": "json", "out_path": out_path})
    )
    assert export["format"] == "json" and export["paths"]


def test_full_catalog_drives_on_fakes(tmp_path):
    """Every advertised tool routes to a use-case (catalog ↔ dispatch parity), in dep order."""
    app, episode = _wired_app(tmp_path)
    listed = _one_text(dispatch_tool(app, "list_episodes", {}))
    assert [e["id"] for e in listed] == [episode.id]  # the registered episode is visible

    scene_id = _reconstruct(app, episode.id)
    tid = app.get_scene(scene_id).subjects[0].track_id
    target = {"kind": "root_translation", "subject_track_id": tid}
    corr = _one_text(
        dispatch_tool(
            app, "apply_offset",
            {"scene_id": scene_id, "target": target, "frame_range": [0, 2], "delta": [0, 0, 0.1]},
        )
    )

    args = {
        "list_episodes": {},
        "run_reconstruction": {"episode_id": episode.id},
        "observe": {"scene_id": scene_id},
        "get_attention": {"scene_id": scene_id},
        "apply_offset": {"scene_id": scene_id, "target": target, "frame_range": [0, 2],
                         "delta": [0, 0, 0.1]},
        "apply_keyframes": {"scene_id": scene_id, "target": target, "frame_range": [0, 2],
                            "key_frames": [0, 2], "key_values": [[0, 0, 0], [0, 0, 0.1]]},
        "apply_smoothing": {"scene_id": scene_id, "target": target, "frame_range": [0, 2]},
        "apply_refit": {"scene_id": scene_id, "target": target, "frame_range": [0, 2]},
        "set_correction_enabled": {"scene_id": scene_id, "correction_id": corr["id"],
                                   "enabled": False},
        "preview": {"scene_id": scene_id,
                    "candidate": {"target": target, "frame_range": [0, 2], "delta": [0, 0, 0.2]}},
        "render": {"scene_id": scene_id},
        "export": {"scene_id": scene_id, "format": "json", "out_path": str(tmp_path / "s.json")},
    }
    names = {t.name for t in build_catalog()}
    assert names == set(args), "catalog and dispatch test args drifted out of sync"
    for name in names:
        blocks = dispatch_tool(app, name, args[name])
        assert blocks and all(isinstance(b, (TextBlock, ImageBlock)) for b in blocks)


def test_unknown_tool_raises(tmp_path):
    app, _ = _wired_app(tmp_path)
    with pytest.raises(ValueError, match="unknown MCP tool"):
        dispatch_tool(app, "does_not_exist", {})


@pytest.mark.skipif(
    importlib.util.find_spec("mcp") is not None,
    reason="the 'mcp' extra is installed, so serve() starts the real server instead of erroring",
)
def test_serve_without_extra_is_actionable(tmp_path):
    app, _ = _wired_app(tmp_path)
    with pytest.raises(RuntimeError, match=r"pitch3d\[mcp\]"):
        serve(app)


def test_serve_rejects_unknown_transport(tmp_path):
    app, _ = _wired_app(tmp_path)
    with pytest.raises(ValueError, match="unsupported transport"):
        serve(app, transport="carrier-pigeon")
