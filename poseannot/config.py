"""poseannot config — read once at process start.

`config.yaml` lives beside this module and controls WHICH clip is loaded
(scene.json + source video) plus WHERE the auth db lives. Everything is
one-clip-at-a-time (per user request).

Example ``poseannot/config.yaml``::

    scene_json: out/anim_full_realism/scene.json
    source_video: samples/video/Colombia-1-0-Congo-DR1080p.mp4
    smplx_models: models/smplx
    fps: 29.97
    users_yaml: poseannot/users.yaml
    jwt_secret: change-me-before-deploy
    jwt_expire_hours: 24
    corrections_out: out/physics_debug/edits.json

Override any field via env var ``POSEANNOT_<UPPER_KEY>`` (e.g.
``POSEANNOT_JWT_SECRET=…``) so RunPod ``.env`` can supply secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = Path(__file__).resolve().parent / "config.yaml"

# Runtime clip override (see clips.py). Only the clip-specific fields
# (source_video / scene_json / corrections_out) may be swapped at runtime;
# deploy-wide fields (smplx_models, users, jwt) always come from yaml/env.
_ACTIVE_OVERRIDE: dict[str, str] | None = None


def set_override(override: dict[str, str] | None) -> None:
    global _ACTIVE_OVERRIDE
    _ACTIVE_OVERRIDE = dict(override) if override else None


def get_override() -> dict[str, str] | None:
    return dict(_ACTIVE_OVERRIDE) if _ACTIVE_OVERRIDE else None


@dataclass(frozen=True)
class PoseAnnotConfig:
    scene_json: Path
    source_video: Path
    smplx_models: Path
    fps: float
    users_yaml: Path
    jwt_secret: str
    jwt_expire_hours: int
    corrections_out: Path


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def load(path: Path | None = None) -> PoseAnnotConfig:
    path = path or DEFAULT_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    # env vars override YAML
    def _get(key: str, cast=str):
        env_key = f"POSEANNOT_{key.upper()}"
        return cast(os.environ.get(env_key, raw.get(key)))

    ov = _ACTIVE_OVERRIDE or {}
    return PoseAnnotConfig(
        scene_json=_resolve(
            ov.get("scene_json") or _get("scene_json") or "out/anim_full_realism/scene.json"
        ),
        source_video=_resolve(
            ov.get("source_video") or _get("source_video")
            or "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
        ),
        smplx_models=_resolve(_get("smplx_models") or "models/smplx"),
        fps=float(_get("fps") or 29.97),
        users_yaml=_resolve(_get("users_yaml") or "poseannot/users.yaml"),
        jwt_secret=str(_get("jwt_secret") or "dev-secret-change-me"),
        jwt_expire_hours=int(_get("jwt_expire_hours") or 24),
        corrections_out=_resolve(
            ov.get("corrections_out") or _get("corrections_out")
            or "out/physics_debug/edits.json"
        ),
    )
