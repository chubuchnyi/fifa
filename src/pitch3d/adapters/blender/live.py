"""Live interactive Blender edit bridge — a human drag becomes a Correction (ADR-0008, ADR-0010).

The batch :mod:`.runner` drives ``blender --background`` to *render* the proxy; this module is the
other direction: a **live**, GUI Blender session whose transform edits flow back into the scene as
canonical :class:`Correction`s. The host process owns the :class:`Scene` (single source of truth,
ADR-0002), so the wire protocol is deliberately thin — Blender reports a subject root's new *world
location* at a frame, and the host diffs it against the **resolved** root to mint the offset, then
commits it through the SAME ``apply_offset`` use-case the MCP agent calls. Human drags and LLM tool
calls are therefore one code path (ADR-0008).

Split like every other adapter (ADR-0001/0006/0009):

* the pure protocol + host edit loop (this module, **no** ``bpy``) — unit-tested over a socket pair
  with the fake-wired app, so the whole drag→Correction loop is verified headlessly;
* the in-Blender client (:mod:`._live`, ``bpy`` only) — a depsgraph watcher that emits edits;
* :func:`launch_live_session` — spawns GUI Blender and serves edits. It needs a display + a Blender
  binary, so it is *not* exercised in CI (the same honest limitation as the GVHMR/TrackNet reals,
  ADR-0009); everything it composes is tested in isolation.

Wire protocol (newline-delimited JSON, Blender → host unless noted):

* ``{"type": "hello", "scene_id": ...}``      → host replies ``{"type": "ready"}``
* ``{"type": "edit", "track_id": 7, "frame": 5, "location": [x, y] | [x, y, z],``
  ``"frame_range": [s, e]?, "note": str?}``    → host commits, replies ``{"type": "ack", ...}``
* ``{"type": "bye"}``                          → host stops serving (also stops on EOF)
"""

from __future__ import annotations

import json
import socket
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from ...core.scene.layers import Correction, TargetKind
from ...core.scene.scene import Scene
from .proxy import build_proxy_plan, write_plan
from .runner import _require_blender

_LIVE_SCRIPT = Path(__file__).with_name("_live.py")
_RECV_BYTES = 65536


def resolved_root_at(resolved: Scene, track_id: int, frame: int) -> np.ndarray:
    """The resolved world root ``(3,)`` of a subject at ``frame``.

    Expects the *resolved* scene (``Application.resolved``), whose subjects already carry
    ``proposal ⊕ corrections`` baked into ``proposal`` (so this reflects every prior edit, REFIT
    included). Raises if the subject does not exist at that frame.
    """
    motion = resolved.subject(int(track_id)).proposal
    frames = np.asarray(motion.pose.frames, dtype=int)
    rows = np.nonzero(frames == int(frame))[0]
    if rows.size == 0:
        raise ValueError(f"subject {track_id} has no frame {frame}")
    return np.asarray(motion.pose.transl[int(rows[0])], dtype=float)


def apply_drag(
    app,
    scene_id: str,
    track_id: int,
    frame: int,
    world_location,
    *,
    frame_range: tuple[int, int] | None = None,
    note: str | None = None,
) -> Correction:
    """Commit a human's drag of subject ``track_id`` to ``world_location`` at ``frame``.

    Diffs the drop against the resolved root to recover the offset, then commits a CONSTANT_OFFSET
    ``ROOT_TRANSLATION`` correction via ``app.apply_offset`` — so the subject lands *exactly* where
    it was dropped, regardless of prior corrections (the new offset stacks on the resolved base).
    A 2-vector ``location`` (a radar drag) moves only XY and keeps the resolved pelvis height; a
    3-vector (a 3D-viewport drag) moves all axes. Defaults to a constant offset over the subject's
    whole track (the drag nudges the entire trajectory); pass ``frame_range`` to scope it.
    """
    resolved = app.resolved(scene_id)
    base = resolved_root_at(resolved, track_id, frame)
    loc = np.asarray(world_location, dtype=float).reshape(-1)
    if loc.shape[0] == 2:
        target_xyz = np.array([loc[0], loc[1], base[2]])
    elif loc.shape[0] == 3:
        target_xyz = loc
    else:
        raise ValueError("world_location must be (x, y) or (x, y, z)")
    delta = target_xyz - base
    if frame_range is None:
        frames = np.asarray(resolved.subject(int(track_id)).proposal.pose.frames, dtype=int)
        frame_range = (int(frames[0]), int(frames[-1]))
    target = {"kind": TargetKind.ROOT_TRANSLATION.value, "subject_track_id": int(track_id)}
    return app.apply_offset(
        scene_id, target, [int(frame_range[0]), int(frame_range[1])], delta,
        note=note or f"live drag @f{int(frame)}",
    )


def _send(conn: socket.socket, message: dict) -> None:
    conn.sendall((json.dumps(message) + "\n").encode("utf-8"))


def _iter_lines(conn: socket.socket):
    """Yield decoded text lines from a stream socket until EOF (handles split/joined frames)."""
    buf = b""
    while True:
        chunk = conn.recv(_RECV_BYTES)
        if not chunk:
            if buf.strip():
                yield buf.decode("utf-8", "replace")
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            yield line.decode("utf-8", "replace")


def serve_edits(
    app,
    scene_id: str,
    conn: socket.socket,
    *,
    default_frame_range: tuple[int, int] | None = None,
    on_edit: Callable[[Correction], None] | None = None,
) -> list[Correction]:
    """Read edit messages off ``conn`` and commit each as a Correction; return them in order.

    Loops until the peer sends ``{"type": "bye"}`` or closes the socket. Malformed or unknown
    messages get a ``{"type": "error"}`` reply and are skipped (a wobbly editor never corrupts the
    scene). ``on_edit`` is called with each committed Correction (e.g. to push fresh feedback back).
    """
    applied: list[Correction] = []
    for line in _iter_lines(conn):
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _send(conn, {"type": "error", "message": "invalid json"})
            continue
        mtype = msg.get("type")
        if mtype == "bye":
            break
        if mtype == "hello":
            _send(conn, {"type": "ready", "scene_id": scene_id})
            continue
        if mtype != "edit":
            _send(conn, {"type": "error", "message": f"unknown message type {mtype!r}"})
            continue
        try:
            corr = apply_drag(
                app, scene_id, msg["track_id"], msg["frame"], msg["location"],
                frame_range=tuple(msg["frame_range"]) if msg.get("frame_range") else
                default_frame_range,
                note=msg.get("note"),
            )
        except (KeyError, ValueError) as exc:
            _send(conn, {"type": "error", "message": str(exc)})
            continue
        applied.append(corr)
        _send(conn, {
            "type": "ack",
            "correction_id": corr.id,
            "track_id": corr.target.subject_track_id,
            "frame_range": [corr.frame_range.start, corr.frame_range.end],
            "delta": corr.payload.delta.tolist(),
        })
        if on_edit is not None:
            on_edit(corr)
    return applied


def launch_live_session(  # pragma: no cover - needs a display + a Blender binary
    app,
    scene_id: str,
    *,
    blender: str | None = None,
    host: str = "127.0.0.1",
    fps: float = 25.0,
    frame_range: tuple[int, int] | None = None,
    accept_timeout: float = 120.0,
) -> dict:
    """Open a live GUI Blender on the resolved scene and stream its drags back as Corrections.

    Binds a localhost socket, launches Blender (GUI — *no* ``--background``) running
    :mod:`._live` with the proxy plan + the socket address, then serves edits until the human
    closes Blender. Returns ``{"scene_id", "applied": [correction ids]}``. Not run in CI (a
    headless box has no GUI Blender); the protocol and edit loop it composes are unit-tested.
    """
    binary = _require_blender(blender)
    plan = build_proxy_plan(app.resolved(scene_id), fps=fps, include_pose=False)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((host, 0))
    srv.listen(1)
    bound_host, bound_port = srv.getsockname()
    applied: list[Correction] = []
    with tempfile.TemporaryDirectory(prefix="pitch3d-live-") as tmp:
        plan_path = write_plan(plan, Path(tmp) / "plan.json")
        cmd = [
            binary, "--python", str(_LIVE_SCRIPT), "--",
            "--plan", str(plan_path), "--host", str(bound_host),
            "--port", str(bound_port), "--scene-id", scene_id,
        ]
        proc = subprocess.Popen(cmd)
        srv.settimeout(accept_timeout)
        try:
            conn, _ = srv.accept()
            srv.settimeout(None)
            with conn:
                applied = serve_edits(app, scene_id, conn, default_frame_range=frame_range)
        finally:
            srv.close()
            if proc.poll() is None:
                proc.terminate()
            proc.wait(timeout=30)
    return {"scene_id": scene_id, "applied": [c.id for c in applied]}


__all__ = [
    "apply_drag",
    "launch_live_session",
    "resolved_root_at",
    "serve_edits",
]
