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
from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    RedirectResponse,
)
from fastapi.staticfiles import StaticFiles

from .auth import authenticate, current_user, issue_token
from .camera import frame_projector, project_points
from .config import load as load_config
from .scene_state import BODY_JOINT_NAMES, get_state
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
    return {
        "n_frames": st.n_frames,
        "n_subjects": len(st.subjects),
        "joint_names": BODY_JOINT_NAMES,
        "tracks": tracks,
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
