# poseannot — v0 (read-only pose annotator)

Browser UI for stepping through a single clip, per-frame:
- video frame with SMPL-X joints projected as a 2D overlay
- Three.js 3D view of the selected subject's SMPL-X mesh + skeleton
- tracks sidebar (click a track to focus)
- frame timeline (click a cell to jump)
- keyboard: `J` / `→` next frame · `K` / `←` prev

Editing arrives in v1.

## Local run

```bash
# 1. Deps
.venv/bin/pip install fastapi 'uvicorn[standard]' python-jose[cryptography] python-multipart bcrypt

# 2. Config (edit poseannot/config.yaml if the clip is different)
cat poseannot/config.yaml

# 3. Add / rotate the admin user
.venv/bin/python -m poseannot.auth hash <newpassword>
# → paste into poseannot/users.yaml under password_hash

# 4. Start
.venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000

# 5. Browse
open http://127.0.0.1:8000/
# default creds: admin / physics    (rotate before deploying!)
```

First `/api/scene` call warms the SMPL-X FK cache — expect ~30 s on a
23-subject 60-frame clip.  Frames after that are instant.

## RunPod deploy

`.venv` + `pitch3d` package + `SMPL-X/models/` + video file + `scene.json`
must all be on the pod's persistent volume.  Then:

```bash
export POSEANNOT_JWT_SECRET=$(openssl rand -hex 32)
export POSEANNOT_SCENE_JSON=/workspace/out/anim_full_realism/scene.json
export POSEANNOT_SOURCE_VIDEO=/workspace/samples/video/…mp4
.venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000
```

RunPod exposes port 8000 as `https://<pod>.proxy.runpod.net/`.  Point
browsers there.

## API (JWT-guarded, cookie or Bearer)

| method | path                                        | body / response                                   |
|--------|---------------------------------------------|---------------------------------------------------|
| POST   | `/login`                                    | form `username`, `password` → sets cookie        |
| GET    | `/logout`                                   | clears cookie                                     |
| GET    | `/api/scene`                                | tracks list + n_frames + joint names             |
| GET    | `/api/frame/{n}`                            | JPEG of source frame n                            |
| GET    | `/api/subject/{tid}/mesh/{frame}`           | verts (10475) + faces (20908) in world           |
| GET    | `/api/subject/{tid}/joints/{frame}`         | 22 body joints in world                          |
| GET    | `/api/subject/{tid}/joints2d/{frame}`       | same 22 joints projected to source pixels        |
| GET    | `/api/camera/{frame}`                       | per-frame intrinsics + extrinsics                 |
| GET    | `/api/health`                               | `{"ok": true}`                                    |

## Data layer

The app reads `scene.json` at start; user edits (v1+) will land in a
separate `edits.json` layered on top via `pitch3d`'s existing ADR-0002
corrections seam.  Original `scene.json` is never mutated by the UI.
