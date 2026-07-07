"""poseannot FastAPI app — v0 (read-only).

Serves:
    /                                    → login page (redirects to /app if authed)
    /login (POST)                        → issue JWT cookie
    /logout                              → clear cookie
    /app                                 → main annotation UI (static)
    /api/scene                           → clip metadata + tracks list
    /api/frame/{n}                       → JPEG of source frame n
    /api/subject/{tid}/mesh/{frame}      → SMPL-X vertices + faces (JSON)
    /api/subject/{tid}/joints/{frame}    → 22 body joints (JSON)
    /api/subject/{tid}/joints2d/{frame}  → same 22 joints projected to pixels
    /api/camera/{frame}                  → per-frame intrinsics + extrinsics
    /static/*                            → assets

Run:
    .venv/bin/uvicorn poseannot.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel, Field

from . import clips as clips_mod
from .auth import authenticate, current_user, issue_token
from .camera import frame_projector, project_points
from .config import load as load_config
from .scene_state import (
    BODY_JOINT_NAMES,
    apply_and_persist_edit,
    edited_frames,
    get_state,
    undo_last_edit,
)
from .video import encode_jpeg, frame_size, read_frame

app = FastAPI(title="poseannot v0", docs_url=None, redoc_url=None)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ─── auth pages ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/app", response_class=HTMLResponse)
async def main_app(user: str = Depends(current_user)) -> FileResponse:
    del user
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/login")
async def login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
) -> JSONResponse:
    if not authenticate(username, password):
        raise HTTPException(status_code=401, detail="invalid credentials")
    cfg = load_config()
    token = issue_token(username, cfg)
    resp = JSONResponse({"ok": True, "redirect": "/app"})
    resp.set_cookie(
        key="poseannot_token", value=token, httponly=True, samesite="strict",
        max_age=cfg.jwt_expire_hours * 3600,
    )
    return resp


@app.get("/logout")
async def logout() -> RedirectResponse:
    resp = RedirectResponse("/")
    resp.delete_cookie("poseannot_token")
    return resp


# ─── read-only API ───────────────────────────────────────────────────────────
@app.get("/api/scene")
async def api_scene(user: str = Depends(current_user)) -> dict:
    del user
    st = get_state()
    tracks = []
    for tid, sub in sorted(st.subjects.items()):
        tracks.append({
            "track_id": tid,
            "n_frames": int(sub.frames.shape[0]),
            "frame_range": [int(sub.frames.min()), int(sub.frames.max())],
            "transl_bbox_xy": [
                float(sub.transl[:, 0].min()), float(sub.transl[:, 1].min()),
                float(sub.transl[:, 0].max()), float(sub.transl[:, 1].max()),
            ],
        })
    cfg = load_config()
    return {
        "n_frames": st.n_frames,
        "n_subjects": len(st.subjects),
        "joint_names": BODY_JOINT_NAMES,
        "tracks": tracks,
        "clip": {
            "video_name": Path(str(cfg.source_video)).name,
            "scene_json": Path(str(cfg.scene_json)).name,
            "fps": cfg.fps,
        },
    }


@app.get("/api/frame/{n}")
async def api_frame(n: int, user: str = Depends(current_user)) -> Response:
    del user
    cfg = load_config()
    bgr = read_frame(str(cfg.source_video), n)
    return Response(content=encode_jpeg(bgr), media_type="image/jpeg")


@app.get("/api/subject/{tid}/mesh/{frame}")
async def api_mesh(
    tid: int, frame: int, user: str = Depends(current_user),
) -> dict:
    del user
    st = get_state()
    if tid not in st.subjects:
        raise HTTPException(404, f"no subject {tid}")
    sub = st.subjects[tid]
    if frame < 0 or frame >= sub.frames.shape[0]:
        raise HTTPException(404, f"frame {frame} out of range for subject {tid}")
    return {
        "verts": sub.verts[frame].tolist(),
        "faces": sub.faces.tolist(),
        "transl": sub.transl[frame].tolist(),
    }


@app.get("/api/subject/{tid}/joints/{frame}")
async def api_joints(
    tid: int, frame: int, user: str = Depends(current_user),
) -> dict:
    del user
    st = get_state()
    if tid not in st.subjects:
        raise HTTPException(404, f"no subject {tid}")
    sub = st.subjects[tid]
    if frame < 0 or frame >= sub.frames.shape[0]:
        raise HTTPException(404, f"frame {frame} out of range for subject {tid}")
    return {
        "joints": sub.joints[frame].tolist(),
        "names": BODY_JOINT_NAMES,
    }


@app.get("/api/subject/{tid}/joints2d/{frame}")
async def api_joints2d(
    tid: int, frame: int, user: str = Depends(current_user),
) -> dict:
    """SMPL-X joints projected to pixel coordinates for the 2D overlay."""
    del user
    st = get_state()
    if tid not in st.subjects:
        raise HTTPException(404, f"no subject {tid}")
    sub = st.subjects[tid]
    if frame < 0 or frame >= sub.frames.shape[0]:
        raise HTTPException(404, f"frame {frame} out of range for subject {tid}")
    if st.scene.camera is None:
        raise HTTPException(500, "scene has no camera track")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(st.scene.camera, frame, video_size=vsize)
    joints2d = project_points(sub.joints[frame], proj)
    # replace NaN (behind camera) with None so JSON serializes cleanly
    pts = []
    for uv in joints2d:
        if np.isnan(uv).any():
            pts.append(None)
        else:
            pts.append([float(uv[0]), float(uv[1])])
    return {
        "pts": pts,
        "names": BODY_JOINT_NAMES,
        "frame_flipped": bool(proj.frame_flipped),
    }


@app.get("/api/frame/{n}/skeletons")
async def api_frame_skeletons(n: int, user: str = Depends(current_user)) -> dict:
    """All subjects' 2D body joints at position ``n`` — one round-trip overlay."""
    del user
    st = get_state()
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if n < 0 or n >= sub.frames.shape[0]:
            continue
        subjects.append({"track_id": tid, "pts": _joints2d_for(st, sub, n, vsize)})
    return {"subjects": subjects, "names": BODY_JOINT_NAMES}


@app.get("/api/frame/{n}/poses3d")
async def api_frame_poses3d(n: int, user: str = Depends(current_user)) -> dict:
    """All subjects' 3D body joints at position ``n`` — for the 3D show-all view."""
    del user
    st = get_state()
    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if n < 0 or n >= sub.frames.shape[0]:
            continue
        subjects.append({"track_id": tid, "joints": sub.joints[n].tolist()})
    return {"subjects": subjects}


@app.get("/api/camera/{frame}")
async def api_camera(frame: int, user: str = Depends(current_user)) -> dict:
    del user
    st = get_state()
    if st.scene.camera is None:
        raise HTTPException(500, "scene has no camera track")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(st.scene.camera, frame, video_size=vsize)
    return {
        "fx": proj.fx, "fy": proj.fy, "cx": proj.cx, "cy": proj.cy,
        "R": proj.R.tolist(),
        "t": proj.t.tolist(),
        "frame_flipped": proj.frame_flipped,
    }


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


# ─── write API (v1 — joint edits) ───────────────────────────────────────────
class EditRequest(BaseModel):
    track_id: int
    frame: int
    joint_index: int = Field(..., ge=0, lt=21, description="body_pose joint index 0..20")
    axis_angle: list[float] = Field(..., min_length=3, max_length=3)


class UndoRequest(BaseModel):
    track_id: int
    frame: int
    joint_index: int | None = None


def _serialize_subject_frame(cache, frame: int, joints2d_pts: list) -> dict:
    return {
        "verts": cache.verts[frame].tolist(),
        "faces": cache.faces.tolist(),
        "joints": cache.joints[frame].tolist(),
        "transl": cache.transl[frame].tolist(),
        "body_pose": cache.body_pose[frame].tolist(),
        "joints2d": joints2d_pts,
    }


def _joints2d_for(state, cache, frame: int, video_size) -> list:
    if state.scene.camera is None:
        return []
    proj = frame_projector(state.scene.camera, frame, video_size=video_size)
    j2d = project_points(cache.joints[frame], proj)
    out = []
    for uv in j2d:
        if np.isnan(uv).any():
            out.append(None)
        else:
            out.append([float(uv[0]), float(uv[1])])
    return out


@app.post("/api/edit")
async def api_edit(req: EditRequest, user: str = Depends(current_user)) -> dict:
    """Persist a single-frame body_pose axis-angle edit; return refreshed frame."""
    st = get_state()
    if req.track_id not in st.subjects:
        raise HTTPException(404, f"no subject {req.track_id}")
    if req.frame < 0 or req.frame >= st.subjects[req.track_id].frames.shape[0]:
        raise HTTPException(404, f"frame {req.frame} out of range")

    cache, _ = apply_and_persist_edit(
        st,
        track_id=req.track_id,
        frame=req.frame,
        joint_index=req.joint_index,
        axis_angle=req.axis_angle,
        user=user,
    )
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    j2d = _joints2d_for(st, cache, req.frame, vsize)
    return {"ok": True, **_serialize_subject_frame(cache, req.frame, j2d)}


@app.post("/api/edit/undo")
async def api_edit_undo(req: UndoRequest, user: str = Depends(current_user)) -> dict:
    del user
    st = get_state()
    if req.track_id not in st.subjects:
        raise HTTPException(404, f"no subject {req.track_id}")
    cache = undo_last_edit(
        st, track_id=req.track_id, frame=req.frame,
        joint_index=req.joint_index,
    )
    if cache is None:
        return {"ok": False, "reason": "no matching edit"}
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    j2d = _joints2d_for(st, cache, req.frame, vsize)
    return {"ok": True, **_serialize_subject_frame(cache, req.frame, j2d)}


@app.get("/api/edits")
async def api_edits(user: str = Depends(current_user)) -> dict:
    """Return {track_id: [edited_frames]} for timeline colouring."""
    del user
    m = edited_frames()
    return {"edits": {str(tid): sorted(f) for tid, f in m.items()}}


# ─── clips API (runtime clip switch + upload) ───────────────────────────────
class ClipSelectRequest(BaseModel):
    clip_id: str


@app.get("/api/clips")
async def api_clips(user: str = Depends(current_user)) -> dict:
    del user
    return {
        "active": clips_mod.active_id(),
        "clips": [clips_mod.clip_to_dict(c) for c in clips_mod.list_clips()],
    }


@app.post("/api/clips/select")
async def api_clips_select(
    req: ClipSelectRequest, user: str = Depends(current_user),
) -> dict:
    del user
    try:
        clip = clips_mod.select(req.clip_id)
    except KeyError:
        raise HTTPException(404, f"no clip '{req.clip_id}'")
    except ValueError as e:
        raise HTTPException(400, str(e))
    get_state(force_reload=True)   # rebuild FK cache for the new clip
    return {"ok": True, "active": clip.id}


@app.post("/api/clips/upload")
async def api_clips_upload(
    user: str = Depends(current_user),
    clip_id: str = Form(...),
    video: UploadFile = File(...),
    scene: UploadFile = File(...),
    edits: UploadFile | None = File(None),
) -> dict:
    """Persist an uploaded (video + scene.json[+edits]) bundle as a new clip."""
    del user
    video_bytes = await video.read()
    scene_bytes = await scene.read()
    edits_bytes = await edits.read() if edits is not None else None
    try:
        clip = clips_mod.create_clip_from_upload(
            clip_id,
            video_bytes=video_bytes,
            video_filename=video.filename or "",
            scene_bytes=scene_bytes,
            edits_bytes=edits_bytes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "clip": clips_mod.clip_to_dict(clip)}
