# ADR-0011 — Deliverable video path: package CLI, versioned export contract, virtual-operator cameras

- **Status:** Accepted
- **Date:** 2026-07-03
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0003 (external Blender render), ADR-0005 (canonical
  scene JSON); R-6 (mark, don't invent); STATUS §6 2026-07-03 (eye-verdict of the first full
  deliverable)

## Context

The deliverable video path is two processes that can only talk through files:
`anim_export` (pipeline venv: torch/smplx) writes a directory of `.npz` artifacts and
`scripts/blender_animate.py` (Blender's own Python, `--factory-startup`) renders them. Eye-judging
the first full deliverable (48 f × 4 cameras, 2026-06-28) found it **unwatchable as broadcast**
despite every subsystem being individually validated:

1. **Framing.** The renderer derived four STATIC cameras from the bbox of everything it loaded;
   with the 105×68 m pitch folded into that bbox, every camera framed the whole stadium bowl from
   OUTSIDE and the players became 5–10 px specks. The close-ups that had passed eye-validation came
   from an "action" camera that was never in the deliverable camera set.
2. **Implicit contract.** The npz schema lived as string keys agreed across two files — no version,
   no validation. Drift (a renamed key, a stale artifact from a previous run, a half-written dir)
   was only ever caught by eye in a finished render, the most expensive failure point we have.
3. **Env spray + knob divergence.** The exporter was configured through six env variables with no
   CLI; the two shell wrappers each restated default knobs and had already diverged
   (`COHERENCE=1` in `demo_video.sh` vs unset → `0` in `pod_make_video.sh`, so the pod deliverable
   rendered raw unsmoothed poses; `PITCH3D_STADIUM_VIDEO` was never wired in the official path, so
   stadium/texture/lighting were silently skipped).

## Decision

1. **Camera planning is a first-class core competence — a "virtual operator", not a bbox.**
   `core/scene/cameras.py` (pure numpy, unit-tested) plans what a real TV rig does: FIXED mounts
   placed inside the stadium-bowl envelope (main stand over the halfway line, low pitchside,
   behind the action-side goal, overhead) that never dolly, but PAN (aim at the smoothed action
   centroid) and ZOOM (fov fits the smoothed action radius) per frame. Aim and zoom are
   straggler-robust (median centroid, bulk-quantile radius: an idle goalkeeper must not zoom the
   shot out to the stadium) and zero-phase-smoothed with separate windows (aim ~1/3 s, zoom ~1 s).
   The exporter writes the planned tracks to `cameras.npz`; the renderer only aims `bpy` cameras
   from it (`sensor_fit=HORIZONTAL`, per-frame look-at + `angle`). The old bbox cameras remain
   solely as a fallback for pre-operator exports.
2. **The export↔render boundary is a versioned, validated contract.**
   `adapters/blender/anim_contract.py` (stdlib+numpy, imported by file on the Blender side like
   `scene_builders.py`) defines `SCHEMA_VERSION`, per-artifact required npz keys, and
   `manifest.json`. The exporter validates and records everything it writes (`write_manifest`);
   the renderer calls `load_manifest` FIRST and refuses loudly — missing manifest, schema
   mismatch, listed file missing, npz key missing — before building any scene. Which artifacts
   exist stays optional by design (no source clip → no `stadium.npz`); *what the renderer may
   expect* is not. A crashed export leaves no manifest, so half-written directories are refused.
3. **The exporter lives in the package with a real CLI.** `pitch3d.app.anim_export` exposes
   `main(argv)` with flags (`--scene/--out/--smplx-models/--source-video/--fade-frames/
   --canonical-up`); env variables remain as defaults (flags > env > `.env`), and
   `scripts/anim_export.py` stays as a thin shim so every wrapper keeps working.
4. **Shell knob defaults are single-sourced.** `scripts/video_defaults.sh` is the one place the
   deliverable-path defaults (frames, coherence, stitch, cameras, res, samples…) are written;
   both wrappers source it. The `COHERENCE=1/0` divergence class cannot recur.

## Consequences

- The framing logic is unit-testable numpy (`tests/unit/test_virtual_cameras.py`: mounts inside
  the bowl, action fits in frame for every tracking camera, zoom never pumps, stragglers don't
  blow up the fov) instead of untested Blender-side code.
- Contract drift now fails in milliseconds with a named cause instead of after a GPU render
  (`tests/unit/test_anim_contract.py`; `tests/e2e/test_video_path_smoke.py` drives the real
  boundary end-to-end and asserts the renderer refuses an unmanifested directory).
- Old export directories (pre-manifest) are deliberately refused: re-run the exporter. This is
  the loud-failure trade we chose over silently rendering stale artifacts.
- The deliverable and formal render paths still share Blender-side code only via pitch3d-free
  modules imported by file (`scene_builders.py`, now `anim_contract.py`) — the ADR-0003 boundary
  (no `pitch3d` import inside Blender) is preserved.
