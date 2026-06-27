"""Web viewer export — the dependency-free three.js bundle (M3-7, FR-27, AC-6).

The web view is the read-side, no-GPU complement to the radar: it reuses the glTF path's Z-up→Y-up
conversion so it opens *without scale/coord loss*, embeds its data so it runs straight off the
filesystem, and stays honest about scope (markers, not meshes). These pin the format support, the
bundle contents, the coord parity with the glTF export, the ball colour, and the selector wiring.
"""

from __future__ import annotations

import json

import numpy as np

from pitch3d.adapters.export import GltfExporter, build_viewer_payload
from pitch3d.adapters.export.gltf import build_gltf_scene
from pitch3d.app.wiring import default_ports
from pitch3d.core.ports.export import ExportFormat


def test_gltf_exporter_supports_threejs_and_not_usd():
    exp = GltfExporter()
    assert exp.supports(ExportFormat.THREEJS) is True
    assert exp.supports(ExportFormat.USD) is False  # honest scope unchanged


def test_threejs_export_writes_self_contained_bundle(reconstructed, tmp_path):
    app, scene_id = reconstructed
    app.ports.exporter = GltfExporter()
    out = tmp_path / "web"
    result = app.export(scene_id, "threejs", str(out))
    assert result.fmt is ExportFormat.THREEJS
    html, data = out / "index.html", out / "scene.json"
    assert html.exists() and data.exists()
    assert {str(html), str(data)} == set(result.paths)

    payload = json.loads(data.read_text())
    assert payload["up"] == "Y"
    assert payload["pitch"]["length"] > 0 and payload["pitch"]["width"] > 0
    assert payload["nodes"], "expected resolved subject/ball nodes"

    # self-contained: the placeholder is substituted and the payload is inlined (opens, no server)
    htext = html.read_text()
    assert "__SCENE_JSON__" not in htext
    assert "three" in htext
    assert json.dumps(payload["nodes"][0]["name"]) in htext  # data embedded in the page


def test_threejs_nodes_cover_subjects_and_ball(reconstructed):
    app, scene_id = reconstructed
    scene = app.get_scene(scene_id)
    payload = build_viewer_payload(scene)
    names = {n["name"] for n in payload["nodes"]}
    for subj in scene.subjects:
        assert f"subject_{subj.track_id}" in names
    if scene.ball is not None:
        assert "ball" in names
    for n in payload["nodes"]:
        assert len(n["color"]) == 3
        assert len(n["times"]) == len(n["positions"])
        assert all(len(p) == 3 for p in n["positions"])


def test_threejs_positions_match_gltf_yup_tracks(reconstructed):
    # AC-6: the web view reuses the glTF Z-up→Y-up conversion, so its positions/times are the glTF
    # tracks verbatim — it can never drift in scale or axis from the interchange export.
    app, scene_id = reconstructed
    scene = app.get_scene(scene_id)
    payload = build_viewer_payload(scene)
    gnodes = {n.name: n for n in build_gltf_scene(scene).nodes}
    assert set(gnodes) == {n["name"] for n in payload["nodes"]}
    for node in payload["nodes"]:
        g = gnodes[node["name"]]
        np.testing.assert_allclose(node["positions"], g.translations, atol=1e-3)
        np.testing.assert_allclose(node["times"], g.times, atol=1e-3)


def test_ball_node_is_gold(reconstructed):
    app, scene_id = reconstructed
    scene = app.get_scene(scene_id)
    if scene.ball is None:
        return
    ball = next(n for n in build_viewer_payload(scene)["nodes"] if n["name"] == "ball")
    assert ball["kind"] == "ball"
    assert ball["color"] == [1.0, 0.78, 0.16]


def test_export_selector_threejs_uses_gltf_exporter(tmp_path):
    ports = default_ports(out_dir=tmp_path / "out", export="threejs")
    assert isinstance(ports.exporter, GltfExporter)
