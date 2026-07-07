"""poseannot config — read once at process start.

`config.yaml` lives beside this module and controls WHICH clip is loaded
(scene.json + source video) plus WHERE the auth db lives. Everything is
one-clip-at-a-time (per user request).

Example ``poseannot/config.yaml``::

    scene_json: out/anim_full_realism/scene.json
    source_video: samples/video/Colombia-1-0-Congo-DR1080p.mp4
    smplx_models: SMPL-X/models
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

    return PoseAnnotConfig(
        scene_json=_resolve(_get("scene_json") or "out/anim_full_realism/scene.json"),
        source_video=_resolve(
            _get("source_video") or "samples/video/Colombia-1-0-Congo-DR1080p.mp4"
        ),
        smplx_models=_resolve(_get("smplx_models") or "SMPL-X/models"),
        fps=float(_get("fps") or 29.97),
        users_yaml=_resolve(_get("users_yaml") or "poseannot/users.yaml"),
        jwt_secret=str(_get("jwt_secret") or "dev-secret-change-me"),
        jwt_expire_hours=int(_get("jwt_expire_hours") or 24),
        corrections_out=_resolve(
            _get("corrections_out") or "out/physics_debug/edits.json"
        ),
    )
