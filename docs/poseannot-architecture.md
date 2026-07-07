# poseannot — architecture

Browser-based pose annotator for the pitch3d pipeline.  Sits on top of
`scene.json`, lets a human step through a clip, and (from v1) edit the
SMPL-X poses per subject per frame.  Edits persist as a **new layer of
`Correction` records** on top of the immutable pipeline output, so the
whole pipeline stays reproducible.

## High-level topology

```
┌──────────────────────────── browser ────────────────────────────┐
│                                                                  │
│   Alpine.js state ⟷ vanilla HTML shell ⟷ Three.js WebGL scene    │
│                                                                  │
│   ▲ JPEG frames · joint 2D pts · joint/mesh 3D · JWT cookie     │
└───┬──────────────────────────────────────────────────────────────┘
    │  HTTPS (RunPod proxy)  ·  JWT in cookie or Authorization: Bearer
    ▼
┌───────────────────────── FastAPI (uvicorn) ─────────────────────┐
│                                                                  │
│   auth.py    ─  bcrypt hashes in users.yaml + JWT (jose)         │
│   config.py  ─  YAML + POSEANNOT_* env override                  │
│   video.py   ─  OpenCV VideoCapture, cached, thread-locked read  │
│   camera.py  ─  scene.camera → intrinsic scale + 180° flip fix   │
│   scene_state.py                                                 │
│               ├ loads scene.json once (start)                    │
│               ├ resolves subject motion (proposal ⊕ corrections) │
│               ├ SMPL-X FK per (subject, frame) → cached verts    │
│               └ per-subject 22 body joints in world Z-up         │
│   app.py     ─  FastAPI routes                                   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
         │
         │  reads (read-only): scene.json, source video, SMPL-X models
         │  writes (v1+): edits.json — new corrections layered on top
         │
         ▼
┌───────────────────── pitch3d python package ───────────────────┐
│                                                                 │
│   core/scene/serialization.py        load_scene / to_json      │
│   core/correction/engine.py          resolve_subject_motion    │
│   core/correction/*                  20+ physics gates          │
│   adapters/models/smplx_lbs.py       SMPL-X mesh + FK           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## What each layer owns

**pitch3d (unchanged)** — the *canonical* scene model, physics gates, and
SMPL-X FK.  The annotator never bypasses this layer; every edit ends up
as a `Correction` record consumed by `resolve_subject_motion`.

**Backend (`poseannot/`)** — a thin FastAPI service that:
- authenticates the caller (JWT, cookie-primary + Bearer-secondary);
- keeps one scene in memory with a fully populated per-frame FK cache
  (~30 s cold start, then per-frame requests are dict lookups);
- serves per-frame JPEG (via OpenCV) + joint/mesh JSON on demand;
- persists edits (v1+) to `edits.json` — a plain `Correction` list — so
  the pipeline reproducibility invariant (scene.json immutable) holds.

**Frontend (`poseannot/static/`)** — a single-page app with **no build
step**: vanilla HTML, Alpine.js 3.x from CDN for reactive state, Three.js
0.170 from CDN for the 3D view.  Everything is pure ES modules, so
"deploy" is `git clone && uvicorn ...`.

## Data model

Scene → subjects → per-subject motion is exactly the pipeline's
`SubjectMotion(shape=SmplxShape(betas), pose=PoseSequence(frames, global_orient, body_pose, transl))`.

For the annotator, we serialize a per-frame slice:

- `/api/subject/{tid}/joints/{frame}` → 22 world-Z-up (x, y, z) joint positions
- `/api/subject/{tid}/joints2d/{frame}` → same joints projected to source pixels
- `/api/subject/{tid}/mesh/{frame}` → 10 475 vertices + 20 908 faces (float32 JSON)
- `/api/camera/{frame}` → per-frame fx/fy/cx/cy + world→camera R+t

The `joints2d` route wraps two facts about our calibration path:
1. `scene.camera.intrinsics` is calibrated at 1280×720; the source video
   is 1920×1080.  `camera.frame_projector` scales fx/fy/cx/cy by the
   ratio so projected pixels land on the browser's un-resized video.
2. The camera track occasionally comes out 180° rolled (per the
   `reference_camera_180_roll` memory).  The projector auto-detects
   (`R[1,2] < 0`) and composes a roll so 2D pixels are always right-side-up.

## Edit persistence (v1)

Every edit is one row appended to `edits.json`:

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

Semantics:
- **Provenance**: `note` includes user + ISO timestamp — audit trail is free.
- **Undo**: pop the last row; the resolver returns to the previous state.
- **Composition with physics**: on save, the backend can optionally re-run
  the corrections stack (v2+) — user's edit is respected as the highest-
  authority signal, physics gates only propagate/smooth around it.
- **Scene immutability**: `scene.json` is never mutated by the UI.

## Auth

Minimal to fit RunPod:
- `poseannot/users.yaml` — bcrypt-hashed passwords for a small operator team.
- `POST /login username/password` → JWT signed with `POSEANNOT_JWT_SECRET`,
  returned as an `HttpOnly SameSite=strict` cookie.
- `GET /api/*` requires the cookie or `Authorization: Bearer <token>`.
- No password reset flow, no self-signup, no OAuth — add later if the
  team grows past ~5 people.

## Deployment

The RunPod pod is the target.  One `uvicorn` process per pod, one clip
loaded per process (per user request — no multi-clip switching in v0/v1).

```
export POSEANNOT_JWT_SECRET=$(openssl rand -hex 32)
export POSEANNOT_SCENE_JSON=/workspace/out/anim_full_realism/scene.json
export POSEANNOT_SOURCE_VIDEO=/workspace/samples/video/…mp4
.venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000
```

RunPod's HTTPS proxy handles TLS; we serve plain HTTP on 8000.

## Non-obvious decisions

- **Backend caches SMPL-X FK on startup** rather than streaming per
  request.  A single FK call takes ~40 ms; a 23-subject 60-frame clip is
  55 s upfront but every subsequent joint/mesh request is a numpy slice.
- **Frontend has zero build tooling**.  We deliberately picked vanilla +
  CDN over React + Vite because the UI grows slowly and every extra
  moving part is one more failure mode on deploy.  Migrating to React
  later is a local change to `static/`.
- **`joints2d` is server-side, not client-side**.  Doing the projection in
  the browser would require shipping the calibration + the 180° roll
  logic to the client; keeping it in Python means one implementation and
  the client stays a thin renderer.
- **Body-pose editing is the primary use case** (per user).  The 3D view
  is designed around per-joint axis-angle rotation with a Three.js
  gizmo; global_orient / transl are read-only for now.
