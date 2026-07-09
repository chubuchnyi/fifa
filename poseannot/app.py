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
    /api/subject/{tid}/mesh2d/{frame}    → subsampled mesh verts projected to pixels (+depth)
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
from . import studio as studio_mod
from .auth import authenticate, current_user, issue_token
from .camera import CameraAdjust, frame_projector, project_points
from .config import load as load_config
from pitch3d.core.scene.pitch import pitch_line_world_points
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


@app.on_event("startup")
def _prewarm_scene_state() -> None:
    """Build the FK cache off the request path so the first /api/scene is a cache
    hit, not a 5-22s SMPL-X build. Daemon thread → startup never blocks/fails on
    it; any real error resurfaces on the actual request."""
    import threading

    def _warm() -> None:
        try:
            get_state()
        except Exception:  # noqa: BLE001 — best-effort warm-up
            pass

    threading.Thread(target=_warm, name="poseannot-prewarm", daemon=True).start()


# ─── auth pages ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "login.html")


@app.get("/app", response_class=HTMLResponse)
async def main_app(user: str = Depends(current_user)) -> FileResponse:
    del user
    # no-store: the annotation UI iterates fast — never let a browser serve a
    # stale index.html (a cached build silently masks shipped fixes).
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


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
def api_scene(user: str = Depends(current_user)) -> dict:
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
def api_frame(n: int, user: str = Depends(current_user)) -> Response:
    del user
    cfg = load_config()
    bgr = read_frame(str(cfg.source_video), n)
    return Response(content=encode_jpeg(bgr), media_type="image/jpeg")


@app.get("/api/subject/{tid}/mesh/{frame}")
def api_mesh(
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
def api_joints(
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


def camera_adjust_params(
    zoom: float = 1.0, panx: float = 0.0, pany: float = 0.0,
    yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0, dolly: float = 0.0,
) -> CameraAdjust:
    """Parse the overlay camera-nudge query params (all default to identity)."""
    return CameraAdjust(
        zoom=zoom, panx=panx, pany=pany,
        yaw=yaw, pitch=pitch, roll=roll, dolly=dolly,
    )


@app.get("/api/subject/{tid}/joints2d/{frame}")
def api_joints2d(
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


@app.get("/api/subject/{tid}/mesh2d/{frame}")
def api_mesh2d(
    tid: int, frame: int, stride: int = 0,
    adj: CameraAdjust = Depends(camera_adjust_params),
    user: str = Depends(current_user),
) -> dict:
    """SMPL-X mesh vertices projected to pixels (subsampled) for the 3D overlay.

    Returns ``pts = [[u, v, d] | None, ...]`` where ``d`` is the vertex's
    camera-space depth normalised to ``[0, 1]`` per subject (0 = nearest the
    camera) so the client can depth-shade the projected surface. ``stride``
    keeps every Nth vertex; 0 = auto (~1200 points). Vertices behind the camera
    serialize as ``None``.
    """
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
    proj = frame_projector(st.scene.camera, frame, video_size=vsize, adjust=adj)
    verts = sub.verts[frame]
    if stride <= 0:
        stride = max(1, verts.shape[0] // 1200)
    vsub = verts[::stride]
    uv = project_points(vsub, proj)
    depth = (vsub @ proj.R.T + proj.t)[:, 2]   # camera-space z (same R,t as project)
    ok = np.isfinite(uv).all(axis=1) & np.isfinite(depth) & (depth > 1e-6)
    if ok.any():
        dmin, dmax = float(depth[ok].min()), float(depth[ok].max())
    else:
        dmin, dmax = 0.0, 1.0
    span = (dmax - dmin) or 1.0
    pts = []
    for (u, v), dz, good in zip(uv, depth, ok):
        if not good:
            pts.append(None)
        else:
            pts.append([float(u), float(v), float((dz - dmin) / span)])
    return {
        "track_id": tid,
        "pts": pts,
        "stride": int(stride),
        "frame_flipped": bool(proj.frame_flipped),
    }


@app.get("/api/frame/{n}/skeletons")
def api_frame_skeletons(
    n: int, adj: CameraAdjust = Depends(camera_adjust_params),
    user: str = Depends(current_user),
) -> dict:
    """All subjects' 2D body joints at position ``n`` — one round-trip overlay."""
    del user
    st = get_state()
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if n < 0 or n >= sub.frames.shape[0]:
            continue
        subjects.append({"track_id": tid, "pts": _joints2d_for(st, sub, n, vsize, adj)})
    return {"subjects": subjects, "names": BODY_JOINT_NAMES}


@app.get("/api/pitch/{frame}")
def api_pitch(
    frame: int, adj: CameraAdjust = Depends(camera_adjust_params),
    user: str = Depends(current_user),
) -> dict:
    """Measured pitch markings projected to pixels — the overlay's alignment
    reference. Pose-independent (uses the field geometry + camera only), so it
    shows where the calibration *thinks* the pitch is; drag the camera controls
    until it lands on the painted lines and the players follow."""
    del user
    st = get_state()
    if st.scene.camera is None or getattr(st.scene, "field", None) is None:
        raise HTTPException(500, "scene has no camera/field")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(st.scene.camera, frame, video_size=vsize, adjust=adj)
    field = st.scene.field
    world = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5)
    uv = project_points(world, proj)
    pts = [[float(u), float(v)] for u, v in uv if np.isfinite(u) and np.isfinite(v)]
    return {"pts": pts, "frame_flipped": bool(proj.frame_flipped)}


@app.get("/api/frame/{n}/poses3d")
def api_frame_poses3d(n: int, user: str = Depends(current_user)) -> dict:
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
def api_camera(frame: int, user: str = Depends(current_user)) -> dict:
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


@app.get("/api/overlay3d/{frame}")
def api_overlay3d(frame: int, user: str = Depends(current_user)) -> dict:
    """Everything the browser needs to project + hand-orient the overlay ITSELF,
    in one round-trip: the flip-corrected camera (K, R, t), the 3D pitch-marking
    world points with their bbox centre, and every subject's 3D body joints — all
    in the SAME world frame. The client applies the user's rigid transform
    (rotate/translate/flip about the pitch centre) and projects with a plain
    pinhole on each slider tick, so orientation edits never touch the server.
    Replaces the old per-tick ``?zoom=&pan=&yaw=`` re-projection round-trips."""
    del user
    st = get_state()
    if st.scene.camera is None or getattr(st.scene, "field", None) is None:
        raise HTTPException(500, "scene has no camera/field")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(st.scene.camera, frame, video_size=vsize)
    field = st.scene.field
    world = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5)
    world = np.asarray(world, dtype=float)
    if len(world):
        center = ((world.min(axis=0) + world.max(axis=0)) / 2.0).tolist()
    else:
        center = [0.0, 0.0, float(field.plane_z)]
    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if 0 <= frame < sub.frames.shape[0]:
            subjects.append({"track_id": tid, "joints": sub.joints[frame].tolist()})
    return {
        "camera": {
            "fx": proj.fx, "fy": proj.fy, "cx": proj.cx, "cy": proj.cy,
            "R": proj.R.tolist(), "t": proj.t.tolist(),
            "frame_flipped": bool(proj.frame_flipped),
        },
        "pitch": {"points": world.tolist(), "center": center},
        "subjects": subjects,
        "video_size": [int(vsize[0]), int(vsize[1])],
    }


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True}


# ─── studio API (Pipeline Studio — per-stage manifest, read-only) ────────────
@app.get("/api/studio/stages")
def api_studio_stages(user: str = Depends(current_user)) -> dict:
    """The pipeline stage DAG as an inspectable manifest (docs/pipeline-studio.md).

    Static structure + params from the pure configs, plus a per-clip ``status``
    per stage telling the UI which stages it can visualise from this scene.json
    now (``live``) vs. which need a Phase-0 run bundle (``unmaterialized``)."""
    del user
    st = get_state()
    return studio_mod.build_manifest(
        st.scene, n_frames=st.n_frames, active_clip=clips_mod.active_id(),
    )


@app.get("/api/studio/ball/{frame}")
def api_studio_ball(frame: int, user: str = Depends(current_user)) -> dict:
    """Ball world position at ``frame`` projected to pixels (Ball 2D/3D stage layer).

    Returns ``{pt: [u,v] | None, xyz, on_ground, conf}``; 404-free when the scene
    carries no ball track (the manifest already flags that stage unmaterialized)."""
    del user
    st = get_state()
    ball = getattr(st.scene, "ball", None)
    if ball is None:
        raise HTTPException(404, "scene has no ball track")
    pos = np.asarray(getattr(ball, "positions_3d", []), dtype=float)
    if frame < 0 or frame >= pos.shape[0]:
        raise HTTPException(404, f"frame {frame} out of range for ball track")
    on_ground = getattr(ball, "on_ground", None)
    conf = getattr(ball, "height_confidence", None)
    out: dict = {
        "xyz": pos[frame].tolist(),
        "on_ground": bool(on_ground[frame]) if on_ground is not None else None,
        "conf": float(conf[frame]) if conf is not None else None,
        "pt": None,
    }
    if st.scene.camera is not None:
        cfg = load_config()
        vsize = frame_size(str(cfg.source_video))
        proj = frame_projector(st.scene.camera, frame, video_size=vsize)
        uv = project_points(pos[frame][None, :], proj)[0]
        if np.isfinite(uv).all():
            out["pt"] = [float(uv[0]), float(uv[1])]
        out["frame_flipped"] = bool(proj.frame_flipped)
    return out


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


def _joints2d_for(state, cache, frame: int, video_size, adjust=None) -> list:
    if state.scene.camera is None:
        return []
    proj = frame_projector(state.scene.camera, frame, video_size=video_size, adjust=adjust)
    j2d = project_points(cache.joints[frame], proj)
    out = []
    for uv in j2d:
        if np.isnan(uv).any():
            out.append(None)
        else:
            out.append([float(uv[0]), float(uv[1])])
    return out


@app.post("/api/edit")
def api_edit(req: EditRequest, user: str = Depends(current_user)) -> dict:
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
def api_edit_undo(req: UndoRequest, user: str = Depends(current_user)) -> dict:
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
def api_edits(user: str = Depends(current_user)) -> dict:
    """Return {track_id: [edited_frames]} for timeline colouring."""
    del user
    m = edited_frames()
    return {"edits": {str(tid): sorted(f) for tid, f in m.items()}}


@app.get("/api/edits/download")
def api_edits_download(user: str = Depends(current_user)) -> FileResponse:
    """Download the persisted corrections file (edits.json) for the active clip."""
    del user
    path = load_config().corrections_out
    if not path.exists():
        raise HTTPException(404, "no edits saved yet")
    return FileResponse(str(path), media_type="application/json", filename="edits.json")


# ─── clips API (runtime clip switch + upload) ───────────────────────────────
class ClipSelectRequest(BaseModel):
    clip_id: str


@app.get("/api/clips")
def api_clips(user: str = Depends(current_user)) -> dict:
    del user
    return {
        "active": clips_mod.active_id(),
        "clips": [clips_mod.clip_to_dict(c) for c in clips_mod.list_clips()],
    }


@app.post("/api/clips/select")
def api_clips_select(
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
