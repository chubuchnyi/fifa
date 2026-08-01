"""Runtime clip registry — switch the annotated clip without a restart.

A *clip* is a (video + scene.json) pair. ``scene.json`` is the pipeline
output (SMPL-X poses, tracks, calibrated camera); a bare video without it
has nothing to overlay or edit. So "load your own clip" means uploading a
reconstructed bundle, not a raw video.

Layout — each subdir under ``poseannot/clips/`` is one clip::

    poseannot/clips/<id>/video.mp4
    poseannot/clips/<id>/scene.json
    poseannot/clips/<id>/edits.json   (optional; per-clip user edits)

The built-in ``default`` clip comes from ``config.yaml`` and is always
listed first. Selecting a clip installs a config override (see
``config.set_override``) so the next ``get_state(force_reload=True)`` runs
FK against the new pair.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from . import config as _config
from .config import REPO_ROOT

CLIPS_DIR = REPO_ROOT / "poseannot" / "clips"
_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


@dataclass(frozen=True)
class Clip:
    id: str
    label: str
    source_video: str | None
    scene_json: str | None
    corrections_out: str
    has_scene: bool
    builtin: bool = False


def _compute_default() -> Clip:
    cfg = _config.load()
    return Clip(
        id="default",
        label=Path(str(cfg.source_video)).name,
        source_video=str(cfg.source_video),
        scene_json=str(cfg.scene_json),
        corrections_out=str(cfg.corrections_out),
        has_scene=Path(str(cfg.scene_json)).exists(),
        builtin=True,
    )


# Snapshot at import time — override is guaranteed unset before any request.
_DEFAULT: Clip = _compute_default()
_ACTIVE_ID: str = "default"


def _discover() -> list[Clip]:
    if not CLIPS_DIR.exists():
        return []
    out: list[Clip] = []
    for d in sorted(CLIPS_DIR.iterdir()):
        if not d.is_dir():
            continue
        vids = [p for p in sorted(d.iterdir()) if p.suffix.lower() in _VIDEO_EXTS]
        # ``scene_rigid.json`` wins when it is there: same poses, but a camera that exists (see
        # _CURATED). Written by scripts/apply_rigid_camera.py, so a clip is upgraded by running
        # that against it — no re-upload, and the original stays for comparison.
        scene = next((d / n for n in ("scene_rigid.json", "scene.json") if (d / n).exists()), None)
        out.append(Clip(
            id=d.name,
            label=d.name,
            source_video=str(vids[0]) if vids else None,
            scene_json=str(scene) if scene else None,
            corrections_out=str(d / (f"{scene.stem}_edits.json" if scene else "edits.json")),
            has_scene=scene is not None,
            builtin=False,
        ))
    return out


# Curated raw/debug scenes under ``out/`` — surfaced as builtins so the Studio
# correction re-run has an inverted-body scene to fix (the flagship demo) and the
# operator can compare a raw pose against the finished, physics-corrected default.
# (id, label, scene.json relative to repo root); listed only if the file exists.
#
# Every entry is a ``_rigid`` scene: the producer's own camera cannot be used. PnLCalib solves
# each frame as a free 8-DOF homography with nothing tying it to a pinhole, and on this clip the
# result is not a bad camera but *no* camera — the nearest realizable one is 525 px away at every
# focal from 200 to 12000 px. ``camera_from_calibration`` therefore refuses, ``app/controller.py``
# invents a ``Viewpoint.BROADCAST`` camera instead, and the players come out 3.9x too small while
# the pitch (drawn through the homography) stays right. That is #61, and it is what "the ground
# marks land but the players do not" looks like. ``scripts/apply_rigid_camera.py`` swaps in the
# one camera #119 measured off the video; the pose content is untouched.
_CURATED: list[tuple[str, str, str]] = [
    ("raw-pose", "Colombia · uncorrected pose (inverted bodies) · #119 cam",
     "out/anim_full_realism/scene_rigid.json"),
    ("carry-off", "Colombia · carry_off export · #119 cam",
     "out/carry_off/export/scene_rigid.json"),
    # The same 60 frames through today's producer (pod, 2026-07-31, every real backend:
    # RF-DETR → ByteTrack → PnLCalib → SMPLest-X → WASB). The two scenes above were written
    # by older code and carry its fixed-in-tree defects; this one is the honest baseline to
    # judge current output against, and the only one whose poses match the source.
    ("fresh-60", "Colombia · fresh pod run 2026-07-31 · #119 cam",
     "out/fresh60/export/scene_rigid.json"),
]


def _extra_builtins() -> list[Clip]:
    out: list[Clip] = []
    for cid, label, rel in _CURATED:
        scene = REPO_ROOT / rel
        if not scene.exists() or _DEFAULT.source_video is None:
            continue
        out.append(Clip(
            id=cid, label=label,
            source_video=_DEFAULT.source_video,
            scene_json=str(scene),
            # Per-scene, not per-directory: a correction is a delta in the scene's own world, so
            # edits made against a mirrored scene mean something else in a right-handed one.
            corrections_out=str(scene.with_name(scene.stem + "_edits.json")),
            has_scene=True, builtin=True,
        ))
    return out


def list_clips() -> list[Clip]:
    return [_DEFAULT, *_extra_builtins(), *_discover()]


def get_clip(clip_id: str) -> Clip | None:
    for c in list_clips():
        if c.id == clip_id:
            return c
    return None


def active_id() -> str:
    return _ACTIVE_ID


def select(clip_id: str) -> Clip:
    """Install the clip's paths as the active config override."""
    global _ACTIVE_ID
    c = get_clip(clip_id)
    if c is None:
        raise KeyError(clip_id)
    if not c.has_scene or not c.source_video:
        raise ValueError(f"clip '{clip_id}' has no scene.json/video — nothing to annotate")
    if c.id == "default":
        _config.set_override(None)
    else:
        _config.set_override({
            "source_video": c.source_video,
            "scene_json": c.scene_json,
            "corrections_out": c.corrections_out,
        })
    _ACTIVE_ID = clip_id
    return c


def _safe_id(clip_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", clip_id).strip("_")
    return safe or "clip"


def _unique_dir(clip_id: str) -> Path:
    base = _safe_id(clip_id)
    d = CLIPS_DIR / base
    n = 2
    while d.exists():
        d = CLIPS_DIR / f"{base}-{n}"
        n += 1
    return d


def create_clip_from_upload(
    clip_id: str,
    *,
    video_bytes: bytes,
    video_filename: str,
    scene_bytes: bytes,
    edits_bytes: bytes | None = None,
) -> Clip:
    """Persist an uploaded (video + scene.json[+edits]) bundle as a new clip."""
    if not video_bytes:
        raise ValueError("empty video upload")
    try:
        json.loads(scene_bytes.decode("utf-8"))
    except Exception as e:  # noqa: BLE001 — surface any parse failure to the caller
        raise ValueError(f"scene.json is not valid JSON: {e}") from e

    ext = Path(video_filename or "").suffix.lower()
    if ext not in _VIDEO_EXTS:
        ext = ".mp4"
    d = _unique_dir(clip_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"video{ext}").write_bytes(video_bytes)
    (d / "scene.json").write_bytes(scene_bytes)
    if edits_bytes:
        (d / "edits.json").write_bytes(edits_bytes)
    clip = get_clip(d.name)
    if clip is None:
        raise RuntimeError(f"clip '{d.name}' vanished after write")
    return clip


def clip_to_dict(c: Clip) -> dict:
    return {
        "id": c.id,
        "label": c.label,
        "video": Path(c.source_video).name if c.source_video else None,
        "has_scene": c.has_scene,
        "builtin": c.builtin,
    }
