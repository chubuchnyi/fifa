"""In-Blender LIVE client — runs *inside* a GUI Blender's Python, never imported by us (ADR-0010).

:func:`pitch3d.adapters.blender.live.launch_live_session` invokes this via
``blender --python _live.py -- --plan plan.json --host H --port P --scene-id S`` (GUI, *not*
``--background``). It builds the same editable proxy as :mod:`._script` (reusing its builder), then
opens a socket back to the host and registers a ``depsgraph_update_post`` watcher: when the human
**grabs a subject's root Empty and moves it** (a drag, not a frame scrub), it reports the Empty's
new world location to the host, which turns it into a ``ROOT_TRANSLATION`` correction.

Self-contained (stdlib + ``bpy``/``mathutils``, both Blender-bundled) and ``bpy`` is imported only
when run as Blender's ``__main__`` — importing this file in a normal interpreter never pulls in
``bpy``. The whole file is exercised only inside Blender (no CI display), hence ``no cover``; the
host edit loop it talks to is unit-tested in :mod:`.live`.
"""

from __future__ import annotations

import json
import os
import socket
import sys

# Mirror of proxy.subject_object_name — this script cannot import the pitch3d package (it runs in
# Blender's interpreter), so the id↔name contract is duplicated here exactly, like _script.py.
_SUBJECT_PREFIX = "subject_"
_MOVE_EPS_M = 1e-4  # ignore sub-0.1mm jitter so only deliberate drags become corrections


def _parse_args(argv):  # pragma: no cover - exercised only inside Blender
    after = argv[argv.index("--") + 1:] if "--" in argv else []
    out = {"plan": None, "host": "127.0.0.1", "port": None, "scene_id": ""}
    flags = {"--plan": "plan", "--host": "host", "--port": "port", "--scene-id": "scene_id"}
    i = 0
    while i < len(after):
        key = flags.get(after[i])
        if key is not None and i + 1 < len(after):
            out[key] = after[i + 1]
            i += 2
        else:
            i += 1
    return out


def _track_id(name):  # pragma: no cover - exercised only inside Blender
    if not name.startswith(_SUBJECT_PREFIX):
        return None
    try:
        return int(name[len(_SUBJECT_PREFIX):])
    except ValueError:
        return None


def _send(sock, message):  # pragma: no cover - needs a live socket
    sock.sendall((json.dumps(message) + "\n").encode("utf-8"))


def main():  # pragma: no cover - runs only as Blender's __main__
    import bpy
    from mathutils import Vector

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import _script  # sibling builder; reused so the live proxy matches the batch one

    args = _parse_args(sys.argv)
    with open(args["plan"], encoding="utf-8") as fh:
        plan = json.load(fh)

    _script._clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = max(1, int(round(plan.get("fps", 25.0))))
    start, end = _script._frame_bounds(plan)
    scene.frame_start, scene.frame_end = start, end
    for obj_spec in plan["objects"]:
        _script._build_object(bpy, Vector, obj_spec)

    sock = socket.create_connection((args["host"], int(args["port"])))
    _send(sock, {"type": "hello", "scene_id": args["scene_id"]})

    def _subject_objects():
        return [(o, _track_id(o.name)) for o in bpy.data.objects if _track_id(o.name) is not None]

    def _snapshot():
        return {o.name: tuple(o.matrix_world.translation) for o, _ in _subject_objects()}

    # Cache last-known roots + frame. A frame scrub moves every animated Empty, which is NOT a
    # drag — so on a frame change we just refresh the cache and emit nothing; only a move while the
    # frame is unchanged is reported as a human edit.
    state = {"frame": scene.frame_current, "loc": _snapshot()}

    def _on_update(scene_arg, depsgraph=None):
        frame = scene_arg.frame_current
        if frame != state["frame"]:
            state["frame"] = frame
            state["loc"] = _snapshot()
            return
        for obj, tid in _subject_objects():
            now = tuple(obj.matrix_world.translation)
            was = state["loc"].get(obj.name)
            if was is None:
                state["loc"][obj.name] = now
                continue
            if max(abs(now[i] - was[i]) for i in range(3)) <= _MOVE_EPS_M:
                continue
            state["loc"][obj.name] = now
            try:
                _send(sock, {"type": "edit", "track_id": tid, "frame": int(frame),
                             "location": [float(now[0]), float(now[1]), float(now[2])]})
            except OSError:
                if _on_update in bpy.app.handlers.depsgraph_update_post:
                    bpy.app.handlers.depsgraph_update_post.remove(_on_update)
                return

    bpy.app.handlers.depsgraph_update_post.append(_on_update)
    print("PITCH3D_BLENDER_LIVE_OK")  # stdout marker, mirrors _script's success line


if __name__ == "__main__":
    main()
