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
from typing import Any

import numpy as np
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
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

from pitch3d.core.scene.layers import TargetKind
from pitch3d.core.scene.pitch import (
    pitch_line_world_points,
    pitch_polylines,
    pitch_upright_polylines,
)
from pitch3d.core.scene.projection import quat_to_rotation_matrix
from pitch3d.core.scene.units import FieldDimensions

from . import clips as clips_mod
from . import rerun as rerun_mod
from . import studio as studio_mod
from .auth import authenticate, current_user, issue_token
from .camera import (
    CameraAdjust,
    camera_centre,
    decompose_similarity,
    focal_from_homography,
    frame_projector,
    image_to_ground,
    plane_orientation,
    plane_similarity,
    plane_similarity_params,
    project_ground,
    project_points,
    project_world,
    world_to_image,
)
from .config import load as load_config
from .pitch_evidence import DEFAULT_TOLERANCE_PX, classify
from .scene_state import (
    BODY_JOINT_NAMES,
    apply_and_persist_calibration_edit,
    apply_and_persist_edit,
    apply_and_persist_root_edit,
    calibration,
    camera,
    edited_frames,
    get_state,
    get_state_b,
    layout_panel_context,
    layout_panel_matrix,
    set_layout_panel_edit,
    undo_last_calibration_edit,
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


@app.get("/world", response_class=HTMLResponse)
async def world_view(user: str = Depends(current_user)) -> FileResponse:
    """Whole-pitch 3D scrubber: every player's skeleton on the measured pitch, free to orbit.

    Exists so pose and physics work can be judged by scrubbing frames instead of paying ~30 min
    of GPU for a full render just to see whether the bodies move like bodies. Separate from /app
    on purpose — that view is a single-subject joint editor, pelvis-centred on one player.
    """
    del user
    return FileResponse(
        STATIC_DIR / "world.html",
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
    if camera(st) is None:
        raise HTTPException(500, "scene has no camera track")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(camera(st), frame, video_size=vsize)
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
    if camera(st) is None:
        raise HTTPException(500, "scene has no camera track")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(camera(st), frame, video_size=vsize, adjust=adj)
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


@app.get("/api/world/geometry")
def api_world_geometry(user: str = Depends(current_user)) -> dict:
    """The pitch itself, in world metres — markings on Z=0 plus goals and flagposts.

    Static for a clip, so the viewer fetches it once and then only asks for skeletons per frame.
    Both lists come from `core.scene.pitch`, the same geometry the overlay and the render use, so
    the 3D view cannot drift from the 2D one.
    """
    del user
    return {
        "markings": [p.tolist() for p in pitch_polylines()],
        "uprights": [p.tolist() for p in pitch_upright_polylines()],
        "length": FieldDimensions().length,
        "width": FieldDimensions().width,
    }


@app.get("/api/world/{n}/skeletons")
def api_world_skeletons(
    n: int, arm: str = "a", user: str = Depends(current_user),
) -> dict:
    """Every subject's 3D body joints at position ``n``, in world metres — one round trip.

    The per-subject `/api/subject/{tid}/joints/{frame}` endpoint is for the joint editor, which
    works on one player at a time. Scrubbing a whole pitch needs all of them at once or the view
    spends its time in HTTP; this is the 3D twin of `/api/frame/{n}/skeletons`.
    """
    del user
    # arm="b" serves POSEANNOT_SCENE_JSON_B, so the viewer can lay one reconstruction over another
    # and see where they disagree. Absent second scene -> empty, not an error: the overlay is a
    # toggle, and a viewer with nothing to compare should just draw one run.
    st = get_state() if arm == "a" else get_state_b()
    if st is None:
        return {"frame": n, "subjects": [], "names": BODY_JOINT_NAMES, "arm": arm}
    teams = {s.track_id: s.team_id for s in st.scene.subjects}
    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if n < 0 or n >= sub.frames.shape[0]:
            continue
        subjects.append({
            "track_id": tid,
            "team": teams.get(tid),
            # Already world z-up: scene_state adds the root translation when it bakes the cache.
            "joints": np.round(sub.joints[n], 4).tolist(),
        })
    # The MEASURED kit colour, not a palette: team ids are k-means cluster labels, so "A" is not
    # reliably one team or the other and hard-coding a colour per id gets the two sides swapped —
    # which is exactly what happened, and the user caught it before I did.
    teams_rgb = {
        t.id: [round(float(c), 4) for c in (t.color_rgb or (0.7, 0.7, 0.7))]
        for t in st.scene.teams
    }
    return {"frame": n, "subjects": subjects, "names": BODY_JOINT_NAMES, "arm": arm,
            "teams": teams_rgb}


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
    if camera(st) is None or getattr(st.scene, "field", None) is None:
        raise HTTPException(500, "scene has no camera/field")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(camera(st), frame, video_size=vsize, adjust=adj)
    field = st.scene.field
    world = pitch_line_world_points(field.dimensions, plane_z=field.plane_z, spacing=0.5)
    uv = project_points(world, proj)
    pts = [[float(u), float(v)] for u, v in uv if np.isfinite(u) and np.isfinite(v)]
    return {"pts": pts, "frame_flipped": bool(proj.frame_flipped)}


def _clip_focal(cal, width: int, height: int) -> float | None:
    """One focal for the whole clip — the median of what each frame's homography implies.

    Per-frame it swings 28% on the target clip, which would make the goal frame breathe as you
    scrub. The camera's zoom barely moves here, so the median is both steadier and closer.
    """
    est = [focal_from_homography(world_to_image(cal, int(f)), width, height) for f in cal.frames]
    good = [f for f in est if f is not None]
    return float(np.median(good)) if good else None


#: Pitch landmarks pushed to the corners. Two cameras that disagree in focal still agree near the
#: principal point, so a centre-huddled probe reads ~0 px on a scene that is 12686 px out.
_AGREE_PROBE = np.array(
    [[0.0, 0.0], [52.5, 34.0], [-52.5, -34.0], [52.5, -34.0], [-52.5, 34.0], [0.0, 34.0]]
)


def _camera_provenance(cam, w2i: np.ndarray, frame: int, width: int) -> dict:
    """How far ``scene.camera`` is from the calibration it is supposed to *be*, in pixels.

    ``real`` here means "the same camera the pitch is drawn through", which is the only sense the
    overlay can check. A scene whose camera was measured from this clip reads ~0; one carrying the
    synthetic broadcast pose (#107) reads thousands, and the players sit off their own feet.

    Both sides are read at *this* frame: a real camera pans, so comparing its frame-0 pose against
    this frame's homography would report the pan itself as disagreement.
    """
    if cam is None:
        return {"source": "none", "agree_px": None}
    row = int(np.argmin(np.abs(np.asarray(cam.frames, dtype=int) - int(frame))))
    rot = quat_to_rotation_matrix(cam.rotation_quat[row])
    ground = np.column_stack([_AGREE_PROBE, np.zeros(len(_AGREE_PROBE))])
    p = (ground @ rot.T + cam.translation[row]) @ cam.intrinsics.matrix().T
    with np.errstate(invalid="ignore", divide="ignore"):
        through_cam = p[:, :2] / p[:, 2, None]
    # The camera may be stored at a render size; compare in ITS pixel space, not the video's.
    scale = float(cam.intrinsics.width) / float(width)
    through_hom = project_ground(_AGREE_PROBE, w2i) * scale
    d = np.linalg.norm(through_cam - through_hom, axis=1)
    agree = float(np.nanmax(d)) if np.isfinite(d).any() else None
    return {
        "source": "measured" if (agree is not None and agree <= 1.0) else "synthetic",
        "agree_px": None if agree is None else round(agree, 2),
        "focal_px": round(float(cam.intrinsics.fx), 1),
        "size": [int(cam.intrinsics.width), int(cam.intrinsics.height)],
        "static": bool(np.ptp(np.asarray(cam.translation, dtype=float), axis=0).max() == 0.0),
    }


#: The pitch-layout gizmo, as a fraction of the frame. Fixed in SCREEN space rather than pinned
#: to a landmark: a broadcast shot contains whatever part of the pitch it happens to contain, and
#: a handle on the corner flag is unreachable in every frame where that corner is out of shot.
#: These two are always on screen and always on the lawn, and the world points beneath them are
#: read back through the current calibration — so the gesture is a pitch-plane gesture regardless.
_GIZMO_UV = ((0.50, 0.60), (0.80, 0.60))


def _gizmo(cal, frame: int, width: int, height: int) -> list[dict]:
    """The two draggable layout handles at ``frame``, in pixels and in metres."""
    px = np.array([[fu * width, fv * height] for fu, fv in _GIZMO_UV])
    xy = image_to_ground(px, cal, frame)
    return [
        {"id": name, "uv": [float(u), float(v)], "world": [float(x), float(y)]}
        for name, (u, v), (x, y) in zip(("move", "turn"), px, xy, strict=True)
    ]


@app.get("/api/pitch/calibrated/{frame}")
def api_pitch_calibrated(
    frame: int,
    tolerance: float | None = None,
    focal: float | None = None,
    user: str = Depends(current_user),
) -> dict:
    """The pitch markings drawn through the SOLVED per-frame calibration.

    The sibling ``/api/pitch`` projects through ``scene.camera``, which #107 showed is a
    synthetic frozen pose rather than this clip's camera — the two disagree by a median
    ~1300 px on a 1920x1080 frame, which is why hand-aligning the overlay never converged.
    This one uses ``field.calibration.homographies``, an exact map of the pitch plane that
    tracks the pan frame by frame and needs no focal (#61's unsolved part).

    Returned as POLYLINES, not a point cloud: connectivity is what makes a misalignment
    legible — a line that should be straight and isn't tells you more than scattered dots.
    Runs break wherever the marking passes behind the camera, so segments never wrap around
    the horizon.
    """
    del user
    st = get_state()
    field = getattr(st.scene, "field", None)
    cal = calibration(st)
    if cal is None:
        raise HTTPException(400, "scene has no field calibration — nothing to draw")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    try:
        w2i = world_to_image(cal, frame)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    cfg = load_config()
    vw, vh = frame_size(str(cfg.source_video))
    tol = DEFAULT_TOLERANCE_PX if tolerance is None else float(tolerance)

    polylines: list[dict] = []
    errors: list[np.ndarray] = []
    for poly in pitch_polylines(field.dimensions, spacing=0.5):
        uv = project_ground(poly, w2i)
        # A marking is only drawable where it is in front of the camera AND not so far
        # outside the frame that it would blow up the SVG path.
        good = np.isfinite(uv).all(axis=1) & (np.abs(uv) < 20000).all(axis=1)
        if not good.any():
            continue
        cut = np.nonzero(np.diff(good.astype(int)) != 0)[0] + 1
        for run in np.split(np.arange(len(uv)), cut):
            if not len(run) or not good[run[0]]:
                continue
            seg = uv[run]
            labels, dist = classify(seg, str(cfg.source_video), int(frame), tol)
            errors.append(dist)
            # split again wherever the verdict changes, so a single marking can be drawn
            # part confirmed and part extrapolated
            bnd = np.nonzero(labels[1:] != labels[:-1])[0] + 1
            for piece in np.split(np.arange(len(seg)), bnd):
                if len(piece) < 2:
                    continue
                polylines.append({
                    "pts": [[float(u), float(v)] for u, v in seg[piece]],
                    "status": str(labels[piece[0]]),
                })

    frames = np.asarray(cal.frames, dtype=int)
    idx = int(np.nonzero(frames == int(frame))[0][0])
    conf = float(np.asarray(cal.confidence, dtype=float)[idx])
    # Count only what the user can actually see. Most sampled points land outside the
    # 1920x1080 frame (the pitch is much bigger than the shot) and would otherwise swamp
    # the summary with "unknown".
    counts = {k: 0 for k in ("ok", "off", "unknown")}
    for p in polylines:
        counts[p["status"]] += sum(
            1 for u, v in p["pts"] if 0 <= u < vw and 0 <= v < vh
        )
    # The goal frames and corner flags stand off the plane, so unlike everything above they are
    # not measured — they are drawn through a focal the pitch itself cannot supply, and that is
    # the point: a wrong focal is invisible on the lawn and obvious on a goalpost.
    auto_focal = _clip_focal(cal, vw, vh)
    used_focal = float(focal) if focal else auto_focal
    uprights: list[dict] = []
    cam = None
    if used_focal and used_focal > 1.0:
        for poly in pitch_upright_polylines(field.dimensions, plane_z=field.plane_z):
            uv = project_world(poly, w2i, used_focal, vw, vh)
            if not np.isfinite(uv).all() or (np.abs(uv) > 20000).any():
                continue
            uprights.append({"pts": [[float(u), float(v)] for u, v in uv]})
        cam = [round(float(x), 1) for x in camera_centre(w2i, used_focal, vw, vh)]

    all_err = np.concatenate(errors) if errors else np.array([np.nan])
    fit = float(np.nanmedian(all_err)) if np.isfinite(all_err).any() else None
    _lp_panel = layout_panel_matrix(st)
    _lp_pre, _lp_post = layout_panel_context(st, frame)
    return {
        "polylines": polylines,
        "uprights": uprights,
        # Whether the players and these markings are drawn through the SAME camera (#61/#107).
        # The pitch above comes from the homography; the players come from ``scene.camera``. For
        # months those were two different cameras 12686 px apart and every consumer read only one
        # of them, so nothing could see it. This is where it becomes visible without a script.
        "camera": _camera_provenance(camera(st), w2i, frame, vw),
        "focal_px": used_focal,
        "focal_auto_px": auto_focal,
        # Where the chosen focal puts the camera, in metres. A broadcast rig is ~15-25 m up and
        # tens of metres past the touchline, and it does not move — so this is the readout that
        # says whether a hand-set focal is physical before the overlay is even looked at.
        "camera_m": cam,
        # Which world this calibration was solved in (#118). Calibrations solved before the fix
        # sit in PnLCalib's top-down template frame, where "Z up" is a left-handed label — every
        # pixel on the lawn is identical, so this readout is the only place it can ever show.
        "frame_handed": "right" if plane_orientation(w2i, vw, vh) > 0 else "mirrored",
        "confidence": conf,
        # Measured against the painted pixels in THIS frame. Unlike ``confidence`` — which
        # #105/#106 showed is anti-predictive — this one is checkable by eye.
        "fit_px": fit,
        "counts": counts,
        "tolerance_px": tol,
        "frame": int(frame),
        "video_size": [int(vw), int(vh)],
        # Where to grab the layout to re-register it by hand (#112).
        "handles": _gizmo(cal, frame, vw, vh),
        # The plane map, both ways. #127: the drag used to have no feedback until pointerup,
        # because only the server could redraw the outline — so the gesture could not be
        # modulated while it was still open. With these the browser previews it live, using
        # H'_w2i = H_w2i @ B ⇒ the drawn pixels move by H @ B @ H⁻¹.
        "w2i": [[float(x) for x in row] for row in w2i],
        "i2w": [[float(x) for x in row] for row in np.linalg.inv(w2i)],
        "adjusted": any(
            c.target.kind is TargetKind.FIELD_CALIBRATION and c.enabled
            for c in st.scene.corrections
        ),
        # What the typed panel's sliders must read. Sent every refresh so they survive a reload,
        # an undo, or a second browser — the panel shows scene state, not local state.
        "layout": decompose_similarity(_lp_panel),
        # The panel's preview basis: the plane map with the panel's own slot taken OUT, plus what
        # composes before and after it. ``w2i`` already has the panel folded in, so it cannot
        # preview a *replacement* of it — only an addition on top.
        "layout_basis": {
            "w2i_bare": [
                [float(x) for x in row]
                for row in w2i @ np.linalg.inv(_lp_pre @ _lp_panel @ _lp_post)
            ],
            "pre": [[float(x) for x in row] for row in _lp_pre],
            "post": [[float(x) for x in row] for row in _lp_post],
        },
    }


class PitchAdjustRequest(BaseModel):
    frame: int
    #: which gizmo handle was dragged — ``move`` translates, ``turn`` rotates and scales.
    handle: str
    #: where it was dropped, in video pixels.
    uv: list[float]


@app.post("/api/pitch/adjust")
def api_pitch_adjust(req: PitchAdjustRequest, user: str = Depends(current_user)) -> dict:
    """Drag the pitch layout onto the painted lines (#112).

    The solve can be a perfectly good *camera* and still sit a metre off along its own plane;
    no residual we compute can see that, because the residual is scored against the very lines
    that placed it. So the operator's eye is the instrument, and what their drag produces is a
    world-plane similarity stored as an ordinary correction — non-destructive, undoable, and
    provably unable to break the single-camera property (see ``PlaneTransformPayload``).
    """
    st = get_state()
    cal = calibration(st)
    if cal is None:
        raise HTTPException(400, "scene has no field calibration — nothing to adjust")
    if req.handle not in ("move", "turn"):
        raise HTTPException(400, f"unknown handle {req.handle!r}")
    if req.frame < 0 or req.frame >= st.n_frames:
        raise HTTPException(404, f"frame {req.frame} out of range")

    cfg = load_config()
    vw, vh = frame_size(str(cfg.source_video))
    try:
        handles = _gizmo(cal, req.frame, vw, vh)
        dst = image_to_ground(np.asarray([req.uv], dtype=float), cal, req.frame)[0]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    if not np.isfinite(dst).all():
        raise HTTPException(400, "that pixel is not on the pitch plane")

    anchor = np.asarray(handles[0]["world"], dtype=float)
    src = np.asarray(handles[1 if req.handle == "turn" else 0]["world"], dtype=float)
    b = plane_similarity(anchor=anchor, src=src, dst=dst, turn=req.handle == "turn")
    corr = apply_and_persist_calibration_edit(
        st, frame=0, frame_end=st.n_frames - 1, matrix=b, user=user,
    )
    return {
        "ok": True,
        "correction_id": corr.id,
        "frame_range": [corr.frame_range.start, corr.frame_range.end],
        # What the drag actually did, in units the user can sanity-check against a pitch.
        "moved_m": round(float(np.linalg.norm(dst - src)), 2),
        "scale": round(float(np.hypot(b[0, 0], b[1, 0])), 4),
        "turn_deg": round(float(np.degrees(np.arctan2(b[1, 0], b[0, 0]))), 2),
    }


class PitchLayoutRequest(BaseModel):
    """Where the layout should END UP — not how far to nudge it."""

    dx: float = 0.0
    dy: float = 0.0
    deg: float = 0.0
    scale: float = 1.0


@app.post("/api/pitch/layout")
def api_pitch_layout(req: PitchLayoutRequest, user: str = Depends(current_user)) -> dict:
    """Set the typed panel's layout transform absolutely (#127).

    Separate from ``/api/pitch/adjust`` because it means something different. A drag is a
    *gesture*: it was aimed at the layout as the last gesture left it, so it appends and one
    gesture is one undo. The panel is a *state*: its sliders say where the pitch should sit, so
    it rewrites one correction and its sliders keep their value. Sliders that sprang back to
    zero after every commit was the defect this endpoint exists to remove — the layout moved and
    the control that moved it showed nothing.

    The four scalars are the whole request, so the server is still the only thing that ever
    builds ``B`` and it is a similarity by construction. A raw 3x3 from the browser would have
    been the shortcut, and it would have handed the client the ability to post a transform that
    breaks the single-camera guarantee.
    """
    st = get_state()
    if calibration(st) is None:
        raise HTTPException(400, "scene has no field calibration — nothing to adjust")
    if not 0.5 <= req.scale <= 2.0:
        raise HTTPException(400, f"scale {req.scale} is outside 0.5–2.0")
    b = plane_similarity_params(req.dx, req.dy, req.deg, req.scale)
    corr = set_layout_panel_edit(st, matrix=b, user=user)
    return {
        "ok": True,
        "correction_id": None if corr is None else corr.id,
        "layout": decompose_similarity(b),
    }


@app.post("/api/pitch/adjust/undo")
def api_pitch_adjust_undo(user: str = Depends(current_user)) -> dict:
    """Undo the most recent layout drag."""
    del user
    st = get_state()
    popped = undo_last_calibration_edit(st)
    return {"ok": popped is not None, "correction_id": None if popped is None else popped.id}


#: SMPL-X body joints whose midpoint is "where this player is standing".
_FOOT_JOINTS = (10, 11)  # left_foot, right_foot


def _stance_xy(cache, frame: int) -> np.ndarray:
    """World XY of the subject's stance at ``frame`` — the midpoint between the feet."""
    return np.asarray(cache.joints[frame][list(_FOOT_JOINTS), :2], dtype=float).mean(axis=0)


@app.get("/api/frame/{n}/ground")
def api_frame_ground(n: int, user: str = Depends(current_user)) -> dict:
    """Every subject's stance point, projected through the SOLVED calibration.

    This is the handle the user grabs. It is deliberately the feet and not the pelvis:
    the homography is only exact ON the pitch plane, and the feet are the one part of a
    player that is actually on it. Pixels here and metres in the reply are two views of
    the same point, so the client can drag in pixels and never do geometry itself.
    """
    del user
    st = get_state()
    cal = calibration(st)
    if cal is None:
        raise HTTPException(400, "scene has no field calibration — nothing to place against")
    if n < 0 or n >= st.n_frames:
        raise HTTPException(404, f"frame {n} out of range")
    try:
        w2i = world_to_image(cal, n)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    subjects = []
    for tid, sub in sorted(st.subjects.items()):
        if n >= sub.frames.shape[0]:
            continue
        xy = _stance_xy(sub, n)
        uv = project_ground(xy.reshape(1, 2), w2i)[0]
        if not np.isfinite(uv).all():
            continue
        subjects.append({
            "track_id": int(tid),
            "uv": [float(uv[0]), float(uv[1])],
            "world": [float(xy[0]), float(xy[1])],
        })
    return {"subjects": subjects, "frame": int(n)}


class PlaceRequest(BaseModel):
    track_id: int
    frame: int
    uv: list[float]
    #: last frame the move applies to; omitted means "the rest of the track".
    frame_end: int | None = None


@app.post("/api/subject/place")
def api_subject_place(req: PlaceRequest, user: str = Depends(current_user)) -> dict:
    """Drop a player at a pixel: the homography says which point of the pitch that is.

    The move is applied from ``frame`` to the end of the track, not to the single frame
    the user was looking at. Placement error inherited from the calibration is systematic
    along a track, so a one-frame fix would swap a steady offset for a visible pop.
    """
    st = get_state()
    if req.track_id not in st.subjects:
        raise HTTPException(404, f"no subject {req.track_id}")
    sub = st.subjects[req.track_id]
    if req.frame < 0 or req.frame >= sub.frames.shape[0]:
        raise HTTPException(404, f"frame {req.frame} out of range")
    cal = calibration(st)
    if cal is None:
        raise HTTPException(400, "scene has no field calibration — cannot place")

    try:
        target = image_to_ground(np.asarray([req.uv], dtype=float), cal, req.frame)[0]
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    delta = target - _stance_xy(sub, req.frame)
    end = sub.frames.shape[0] - 1 if req.frame_end is None else int(req.frame_end)

    cache, corr = apply_and_persist_root_edit(
        st,
        track_id=req.track_id,
        frame=req.frame,
        kind="root_translation",
        delta=[float(delta[0]), float(delta[1]), 0.0],
        user=user,
        frame_end=end,
    )
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    j2d = _joints2d_for(st, cache, req.frame, vsize)
    return {
        "ok": True,
        "delta_m": [float(delta[0]), float(delta[1])],
        "frame_range": [int(req.frame), int(end)],
        "correction_id": corr.id,
        **_serialize_subject_frame(cache, req.frame, j2d),
    }


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
    if camera(st) is None:
        raise HTTPException(500, "scene has no camera track")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(camera(st), frame, video_size=vsize)
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
    if camera(st) is None or getattr(st.scene, "field", None) is None:
        raise HTTPException(500, "scene has no camera/field")
    if frame < 0 or frame >= st.n_frames:
        raise HTTPException(404, f"frame {frame} out of range")
    cfg = load_config()
    vsize = frame_size(str(cfg.source_video))
    proj = frame_projector(camera(st), frame, video_size=vsize)
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
    if camera(st) is not None:
        cfg = load_config()
        vsize = frame_size(str(cfg.source_video))
        proj = frame_projector(camera(st), frame, video_size=vsize)
        uv = project_points(pos[frame][None, :], proj)[0]
        if np.isfinite(uv).all():
            out["pt"] = [float(uv[0]), float(uv[1])]
        out["frame_flipped"] = bool(proj.frame_flipped)
    return out


# ─── studio re-run API (correction gates as an ephemeral, revertable layer) ──
class RerunRequest(BaseModel):
    profile: str = "default"
    #: per-gate on/off override; absent gates fall back to the profile default.
    overrides: dict[str, bool] = Field(default_factory=dict)
    #: per-gate field overrides ({gate_id: {field: value}}) applied on top of the
    #: profile defaults; unknown/uncastable fields are dropped server-side.
    params: dict[str, dict[str, Any]] = Field(default_factory=dict)


@app.get("/api/studio/rerun/catalog")
def api_studio_rerun_catalog(
    profile: str = "default", user: str = Depends(current_user),
) -> dict:
    """Ordered correction-gate list + profile default-enabled flags, for the UI.

    Gates available from scene.json alone are ``available:true``; the four that
    need a live-pipeline provider are surfaced ``available:false`` with a reason."""
    del user
    return rerun_mod.gate_catalog(profile)


@app.post("/api/studio/rerun")
def api_studio_rerun(req: RerunRequest, user: str = Depends(current_user)) -> dict:
    """Run the enabled correction gates on the frozen baseline; rebuild affected FK.

    The gates append revertable ``Correction`` layers (never mutate poses), so the
    existing joint/mesh endpoints then serve the corrected poses. Ephemeral: nothing
    is written to edits.json. Synchronous — a full-scene re-run is FK-bound (~seconds
    to tens of seconds); the UI shows a spinner."""
    del user
    st = get_state()
    return rerun_mod.run_corrections(
        st, profile=req.profile, overrides=req.overrides, params=req.params,
    )


@app.post("/api/studio/rerun/clear")
def api_studio_rerun_clear(user: str = Depends(current_user)) -> dict:
    """Drop the ephemeral studio corrections; restore + rebuild the baseline poses."""
    del user
    st = get_state()
    return rerun_mod.clear_corrections(st)


# ─── write API (v1 — joint edits) ───────────────────────────────────────────
from pitch3d.core.scene.layers import TargetKind as _TargetKind

#: wire edit-kind → scene TargetKind (undo filter). Root kinds edit the SMPL-X
#: root; pose_body_joint edits one body joint.
_EDIT_KIND_TO_TARGET: dict[str, _TargetKind] = {
    "pose_body_joint": _TargetKind.POSE_BODY_JOINT,
    "root_orientation": _TargetKind.ROOT_ORIENTATION,
    "root_translation": _TargetKind.ROOT_TRANSLATION,
}
_ROOT_EDIT_KINDS = {"root_orientation", "root_translation"}


class EditRequest(BaseModel):
    track_id: int
    frame: int
    kind: str = "pose_body_joint"
    joint_index: int | None = Field(
        None, ge=0, lt=21, description="body_pose joint index 0..20 (pose_body_joint only)",
    )
    axis_angle: list[float] = Field(
        ..., min_length=3, max_length=3,
        description="pose_body_joint: absolute joint axis-angle; root_*: delta nudge",
    )


class UndoRequest(BaseModel):
    track_id: int
    frame: int
    kind: str | None = None
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
    cam = camera(state)
    if cam is None:
        return []
    proj = frame_projector(cam, frame, video_size=video_size, adjust=adjust)
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
    """Persist a single-frame edit (body_pose joint, or root orient/transl offset)."""
    st = get_state()
    if req.track_id not in st.subjects:
        raise HTTPException(404, f"no subject {req.track_id}")
    if req.frame < 0 or req.frame >= st.subjects[req.track_id].frames.shape[0]:
        raise HTTPException(404, f"frame {req.frame} out of range")

    if req.kind == "pose_body_joint":
        if req.joint_index is None:
            raise HTTPException(400, "pose_body_joint edit requires joint_index")
        cache, _ = apply_and_persist_edit(
            st,
            track_id=req.track_id,
            frame=req.frame,
            joint_index=req.joint_index,
            axis_angle=req.axis_angle,
            user=user,
        )
    elif req.kind in _ROOT_EDIT_KINDS:
        cache, _ = apply_and_persist_root_edit(
            st,
            track_id=req.track_id,
            frame=req.frame,
            kind=req.kind,
            delta=req.axis_angle,
            user=user,
        )
    else:
        raise HTTPException(400, f"unknown edit kind {req.kind!r}")

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
    tk = _EDIT_KIND_TO_TARGET.get(req.kind) if req.kind else None
    cache = undo_last_edit(
        st, track_id=req.track_id, frame=req.frame,
        joint_index=req.joint_index, kind=tk,
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
