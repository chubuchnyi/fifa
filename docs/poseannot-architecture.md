# poseannot — architecture

Browser-based annotator and pipeline inspector for pitch3d. It sits on top of a `scene.json`, lets
a human step through a clip, edit SMPL-X poses, nudge the pitch calibration, re-run the physics
gates and read the pipeline's own stage manifest. Every edit persists as **`Correction` records**
layered over the immutable pipeline output, so the pipeline stays reproducible.

**Re-measured 2026-08-08.** The previous version of this document described six modules and framed
editing, clip switching and gate re-runs as "v1+/v2+" future work. All three ship. What follows is
measured against the tree; where a docstring inside the code still says otherwise, that is called
out rather than quietly reconciled.

Its place in the wider system — and why it is architecturally interesting — is
[`architecture.md` §4a](architecture.md#4a-poseannot--the-human-driving-adapter). One line of that
matters enough to repeat here:

```bash
grep -rho "from pitch3d[.a-z_]*" poseannot/*.py | sort -u   # 25 imports, ALL core.*, zero adapters
```

**poseannot imports `pitch3d.core.*` and nothing else.** It is a driving adapter that reaches the
domain directly, never through `adapters/` or `app.wiring`. That constraint is deliberate and it is
what keeps the app torch-free and startable anywhere a `scene.json` exists.

## Modules

12 files, ~3 630 lines.

| module | lines | owns |
|---|---|---|
| `app.py` | 1225 | the FastAPI app: **37 routes**, Pydantic request models, startup FK prewarm, `/static` mount |
| `scene_state.py` | 445 | the in-memory `SceneState` singleton — loads `scene.json`, runs SMPL-X FK per subject, caches joints + verts, applies and persists edits, rebuilds affected caches |
| `camera.py` | 417 | world (Z-up) → pixel projection, homography lift/decompose, the 180°-roll fix, plane-similarity math for pitch nudges |
| `rerun.py` | 342 | Studio correction re-run — runs `core.correction` gates as an **ephemeral, revertable** Correction layer |
| `clips.py` | 279 | runtime clip registry over `poseannot/clips/` — discovery, select, upload |
| `edits.py` | 251 | `Correction` persistence to `edits.json`, builders for body-pose / root / calibration edits, undo |
| `studio.py` | 244 | Pipeline Studio: the per-stage manifest — DAG, params, per-clip availability. Read-only |
| `pitch_evidence.py` | 175 | does the video actually have paint under the pitch line we are drawing? |
| `auth.py` | 107 | JWT issue/decode, bcrypt verify, `users.yaml`, `python -m poseannot.auth hash` |
| `config.py` | 91 | `PoseAnnotConfig` — YAML + `POSEANNOT_*` env + the runtime clip override slot |
| `video.py` | 48 | OpenCV frame read → JPEG |

## Topology

```
┌──────────────────────────────── browser ───────────────────────────────┐
│  /app   annotator + Studio tabs   —  Alpine.js 3.14 state, HTML shell  │
│  /world 3D scene view             —  Three.js 0.170 WebGL              │
│  both via <script type="importmap"> → jsdelivr CDN. No build step.     │
└───┬────────────────────────────────────────────────────────────────────┘
    │  JWT in HttpOnly cookie (or Authorization: Bearer)
    ▼
┌───────────────────────── FastAPI (uvicorn) — poseannot/ ───────────────┐
│  auth.py · config.py · video.py · camera.py                            │
│  clips.py         runtime clip registry (switch WITHOUT a restart)      │
│  scene_state.py   scene.json → resolve(proposal ⊕ corrections) → FK    │
│                   cache; prewarmed on a startup thread, 5-22 s         │
│  edits.py         Correction rows → edits.json (atomic whole-file)      │
│  studio.py        stage manifest (read-only)                           │
│  rerun.py         12 gates as an ephemeral layer, in memory only        │
│  pitch_evidence.py  per-point paint check against the frame            │
│  app.py           the 37 routes                                        │
└───┬────────────────────────────────────────────────────────────────────┘
    │  reads: scene.json, source video, SMPL-X model files
    │  writes: edits.json only — scene.json is never mutated
    ▼
┌──────────── pitch3d.core — the ONLY pitch3d dependency ────────────────┐
│  core/scene/serialization.py   load_scene / to_json                    │
│  core/correction/engine.py     resolve_subject_motion                  │
│  core/correction/*             the gate implementations rerun.py calls │
│  core/config/physics.py        the same defaults the pipeline uses      │
│  core/orchestration/continuity.py                                       │
└─────────────────────────────────────────────────────────────────────────┘

  SMPL-X forward kinematics comes from the upstream `smplx` package, imported
  directly by scene_state.py — NOT from adapters/models/smplx_lbs.py.
```

## Route surface (37)

```
/, /app, /world           pages: login · annotator+Studio · 3D world view
/login, /logout           JWT cookie issue / clear
/api/scene                clip metadata + track list
/api/frame/{n}/…          JPEG frame, skeletons, ground, poses3d
/api/subject/{tid}/…      mesh, joints, joints2d, mesh2d — per frame
/api/subject/place        POST — reposition a subject
/api/world/…              geometry + per-frame skeletons for the 3D view
/api/pitch/…              overlay, calibrated overlay, adjust / layout / undo (POST)
/api/camera/{frame}       camera pose per frame
/api/overlay3d/{frame}    combined 3D overlay payload
/api/studio/…             stage manifest, ball, rerun catalog / run / clear
/api/edit, /api/edits     write an edit, undo, list edited frames, download edits.json
/api/clips                list, select, upload
/api/health               liveness
```

**There is no `/studio` page route.** Studio is a tab inside `static/index.html`, served at `/app`.

## Editing — live, and the docstring that says otherwise is stale

`app.py`'s module docstring still opens `"""poseannot FastAPI app — v0 (read-only)."""`. **That
line is wrong.** Ten routes write: `/api/edit`, `/api/edit/undo`, `/api/pitch/adjust`,
`/api/pitch/layout`, `/api/pitch/adjust/undo`, `/api/subject/place`, `/api/clips/select`,
`/api/clips/upload`, `/api/studio/rerun`, `/api/studio/rerun/clear`.

Edits land in `cfg.corrections_out`, or in a per-clip `poseannot/clips/<id>/edits.json`. The file
is `{"corrections": [Correction, …]}` — the pipeline's own `Correction` class — written by
`edits._atomic_write` as a **whole-file atomic rewrite**, not an append. One row looks like:

```json
{
  "target": { "kind": "POSE_BODY_JOINT", "subject_track_id": 15, "joint_index": 4 },
  "frame_range": [30, 30],
  "kind": "KEYFRAME_INTERP",
  "key_frames": [30],
  "key_values": [[0.1, -0.3, 0.05]],
  "note": "manual-admin-2026-07-07T10:42",
  "confidence": 1.0
}
```

- **Provenance** — `note` carries user + ISO timestamp, so the audit trail is free.
- **Undo** — pop the last row; the resolver returns to the previous state.
- **Scene immutability** — `scene.json` is never mutated by the UI.
- **Three editable families** — body-pose joints, subject root placement, and pitch calibration
  (`camera.py`'s plane-similarity nudge), each with its own builder in `edits.py`.

## Correction re-run (`rerun.py`) — the Studio's mutating half

The old doc listed "re-run the corrections stack" as a v2+ maybe. It ships, and its shape is the
part worth knowing:

- **12 gates run**: `coherence`, `kinematic`, `foot_floor`, `joint_kinematic`, `orientation`,
  `collision`, `orient_verticality`, `pose_motion_sync`, `facing_align`, `inertia_smooth`,
  `jerk_clamp`, `joint_smooth`. Entry points `gate_catalog()`, `run_corrections()`,
  `clear_corrections()`; `FLAGSHIP_GATE = "orient_verticality"`.
- **4 are declared unavailable**: `foot_plant`, `momentum_smooth`, `contact_lock`,
  `gravity_project`. They need pelvis-target / foot-position **provider callables** that only exist
  inside a live reconstruction — a `scene.json` cannot supply them. The split is a **hardcoded
  list** (`_PROVIDER_GATES`), not runtime detection, so it can drift.
- **It cannot drift silently.** `tests/unit/test_gate_chain_parity.py` parses
  `app/controller.py`'s gate chain and fails if the two lists disagree. That test is the only thing
  keeping a hand-maintained mirror honest, which is why it exists.
- **Re-run output is in memory only** and is *never* written to `edits.json`. It is a preview
  layer: run it, look, clear it. A user's own edit is the durable thing.

## Pipeline Studio (`studio.py`) — read-only introspection

`build_manifest()` emits the pipeline's stage DAG (decode → detect → track → calibrate → pose →
ball2d → ball3d → …), and per stage: temporal shape, dependencies, cost, family, an in/out
description, and the **real default params** pulled from the pure `core/` configs and
`config/physics.yaml` — not a hand-written copy. `_availability()` then tags each stage `live`,
`partial` or `unmaterialized` against the scene actually loaded.

It is **torch-free by construction**: it imports no adapters, which is why the Studio opens on a
laptop with no GPU and no model weights. The module is read-only; the *Studio tab* is not, because
`rerun.py` hangs off the same stage ids.

## Pitch evidence (`pitch_evidence.py`)

Answers one question per projected pitch-line point: **is there real paint under it?**
`classify()` returns `ok` (paint within tolerance), `unknown` (the point falls off the playing
surface — crowd, boards) or `off` (clear turf, no paint: the overlay is claiming a marking that is
not there). HSV turf/surface masks, `lru_cache`d per `(video, frame)`. This is what turns "the
calibration looks about right" into a per-marking verdict.

## Frontend

`poseannot/static/` — `index.html` (127 K, the annotator + Studio), `world.html` + `world_view.js`
(the 3D view), `style.css`, `login.html`. **No build step, no bundler**: pure ES modules with an
`importmap` pointing at jsdelivr for Three.js 0.170 and Alpine.js 3.14.1.

`poseannot/vendor/` exists but is **empty and untracked** — nothing is vendored today. If the CDN
becomes a problem, that directory is where the answer goes; do not assume it already is one.

> ⚠ **Editing `static/style.css`? Bump the `?v=` token in `index.html`.** It is currently
> `?v=layout8`. Without the bump the browser re-verifies the *old* stylesheet, and a "fix" gets
> judged against the file it replaced.

## Auth

Minimal, sized for a pod and a handful of operators:

- `poseannot/users.yaml` — bcrypt hashes. `python -m poseannot.auth hash` mints one.
- `POST /login` → JWT signed with `POSEANNOT_JWT_SECRET`, returned as an `HttpOnly SameSite=strict`
  cookie.
- `GET /api/*` requires the cookie or `Authorization: Bearer <token>`.
- No reset flow, no self-signup, no OAuth. Add them when the team passes ~5 people.

## Deployment

```
export POSEANNOT_JWT_SECRET=$(openssl rand -hex 32)
export POSEANNOT_SCENE_JSON=/workspace/out/anim_full_realism/scene.json
export POSEANNOT_SOURCE_VIDEO=/workspace/samples/video/…mp4
.venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000
```

One `uvicorn` process, **one *active* clip** — but `clips.py` switches it at runtime:
`clips.select(id)` installs a `config.set_override()` and `/api/clips/select` calls
`get_state(force_reload=True)` to rebuild the FK cache. `config.py`'s docstring still says
"Everything is one-clip-at-a-time (per user request)", which is now only half true: one at a time,
yes; one per process lifetime, no.

For A/B work the env vars take two scenes — `POSEANNOT_SCENE_JSON` and `POSEANNOT_SCENE_JSON_B` —
which is what `scripts/view_cue_ab.sh` and `scripts/view_handover_ab.sh` drive.

RunPod's HTTPS proxy terminates TLS; we serve plain HTTP on 8000.

## Non-obvious decisions

- **The FK cache is warmed on a startup thread, not per request.** `app.py`'s
  `@app.on_event("startup")` spawns a daemon calling `get_state()`; its own docstring puts the
  SMPL-X build at **5–22 s** (an earlier draft of this document said 30–55 s — that number was
  never re-measured after the caching changed). Afterwards every joint/mesh request is a numpy
  slice.
- **`joints2d` is computed server-side.** Doing it in the browser would mean shipping the
  calibration *and* the 180°-roll detection to the client. One implementation, in Python; the
  client stays a thin renderer.
- **The projector handles two calibration facts.** `scene.camera.intrinsics` is calibrated at
  1280×720 while the source video is 1920×1080, so `camera.frame_projector` scales fx/fy/cx/cy by
  the ratio; and the solved camera track is sometimes 180° rolled, auto-detected on `R[1,2] < 0`
  and composed out, so 2D pixels are always right-side-up.
- **Zero build tooling on the front end** — vanilla + CDN over React + Vite, because the UI grows
  slowly and every extra moving part is one more thing that can fail on deploy. Migrating later is
  a change local to `static/`.
- **Body-pose editing is the primary use case**, per the user: per-joint axis-angle rotation with a
  Three.js gizmo. Root placement and pitch calibration are editable too; `global_orient` is not.
- **The app never imports `adapters/`.** `studio.py` says so in a comment where it reaches for
  defaults — "*pulled from the pure-half configs (never adapters)*". Keep it that way: it is what
  makes the annotator startable without torch, without weights and without a GPU.
