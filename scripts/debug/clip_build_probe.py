#!/usr/bin/env python
"""Find which clip wedges the poseannot server, and in which phase.

Symptom (2026-07-09): after a clip switch the whole server hangs — EVERY route
(even /static) times out with 0 bytes and the worker spins at 100% one core.
That is the signature of an INFINITE LOOP on the asyncio event-loop thread: the
render endpoints (`/api/pitch`, `/api/frame/{n}/skeletons`) are ``async def`` and
call synchronous CPU code (frame_projector / project / FK) directly on the loop,
so one non-terminating call starves the entire process, static files included.

`clips._ACTIVE_ID` is in-memory only, so restarting the server resets to the
default clip and the hang vanishes — until the user re-selects the bad clip. This
probe reproduces the exact production path OFFLINE, one fresh process per clip,
each phase flush-printed BEFORE it runs. Wrap the call in an OS-level timeout
(Python SIGALRM can't interrupt a GIL-holding C loop):

    for c in default A_smplestx B_sam3dbody clip; do
        echo "===== $c ====="
        timeout 90 python scripts/debug/clip_build_probe.py "$c"
        echo "  (exit $?  — 124 = TIMED OUT in the phase named on the last line)"
    done

The phase whose "-> ..." line printed with no matching "   ok" after it is the
culprit; combine with the clip name to localise the bug (bad camera, degenerate
homography, etc.). Read-only: selects the clip override + reads, never writes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def _say(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    clip_id = sys.argv[1] if len(sys.argv) > 1 else "default"
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    from poseannot import clips, scene_state
    from poseannot.config import load as load_config

    _say(f"-> select({clip_id!r})")
    clips.select(clip_id)
    cfg = load_config()
    _say(f"   ok  scene_json={cfg.scene_json.name}  video={cfg.source_video.name}")

    _say("-> build_scene_state / get_state(force_reload=True)")
    t = time.perf_counter()
    st = scene_state.get_state(force_reload=True)
    _say(f"   ok  {time.perf_counter()-t:.2f}s  subjects={len(st.subjects)}  n_frames={st.n_frames}")

    from poseannot.app import _joints2d_for
    from poseannot.camera import frame_projector, project_points
    from poseannot.video import frame_size

    from pitch3d.core.scene.pitch import pitch_line_world_points

    _say("-> frame_size(source_video)")
    t = time.perf_counter()
    vsize = frame_size(str(cfg.source_video))
    _say(f"   ok  {time.perf_counter()-t:.2f}s  vsize={vsize}")

    _say(f"-> frame_projector(camera, frame={frame})")
    t = time.perf_counter()
    proj = frame_projector(st.scene.camera, frame, video_size=vsize)
    _say(f"   ok  {time.perf_counter()-t:.2f}s  flipped={proj.frame_flipped}")

    _say("-> pitch project (/api/pitch)")
    t = time.perf_counter()
    field = st.scene.field
    world = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5)
    uv = project_points(world, proj)
    _say(f"   ok  {time.perf_counter()-t:.2f}s  pts={len(uv)}")

    _say("-> skeletons project (/api/frame/{n}/skeletons)")
    t = time.perf_counter()
    n = 0
    for tid, sub in sorted(st.subjects.items()):
        if 0 <= frame < sub.frames.shape[0]:
            _joints2d_for(st, sub, frame, vsize, None)
            n += 1
    _say(f"   ok  {time.perf_counter()-t:.2f}s  subjects_projected={n}")

    _say("ALL PHASES OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
