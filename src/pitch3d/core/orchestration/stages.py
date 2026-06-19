"""Pipeline stages + cache-keyed, job-wrapped execution (ADR-0004).

Every heavy stage runs as a :class:`~pitch3d.core.ports.jobs.JobQueue` job whose output is
stored in a content-addressable :class:`~pitch3d.core.ports.cache.Cache`, keyed by
``hash(stage, input_hash, params, model_version)``. Re-running with unchanged inputs is a
cache hit, so generative passes (avatars, ViewSynthesizer) are never recomputed needlessly
(NFR-4). The fake queue runs jobs synchronously in-process; a real worker is an adapter swap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..ports.cache import Cache
from ..ports.io import ClipRef
from ..ports.jobs import JobQueue


class Stage(str, Enum):
    """The reconstruction → render → export stage DAG (architecture §7, §9)."""

    AMPLIFY = "amplify"      # optional ViewSynthesizer seam B (mono → pseudo-multi-view)
    DETECT = "detect"
    TRACK = "track"          # tracking + team assignment
    CALIBRATE = "calibrate"  # field homography
    POSE = "pose"            # HMR → SMPL-X, root anchored by homography
    BALL = "ball"            # 2D ball track + core 3D lift
    ASSEMBLE = "assemble"    # proposal Scene + confidence map
    ENV = "env"              # 3DGS/NeRF environment
    AVATAR = "avatar"        # per-subject render asset
    RENDER = "render"
    EXPORT = "export"
    OBSERVE = "observe"      # multi-view snapshots for the LLM feedback loop (ADR-0008)


#: Core reconstruction order that produces the proposal :class:`~...scene.scene.Scene`.
RECON_ORDER: tuple[Stage, ...] = (
    Stage.DETECT,
    Stage.TRACK,
    Stage.CALIBRATE,
    Stage.POSE,
    Stage.BALL,
    Stage.ASSEMBLE,
)


def clip_hash(clip: ClipRef) -> str:
    """Stable content hash of a clip's identity (source + uri + frame span)."""
    blob = json.dumps(
        {
            "source_id": clip.source_id,
            "uri": clip.uri,
            "frames": [int(clip.frames[0]), int(clip.frames[-1]), int(clip.n_frames)]
            if clip.n_frames
            else [],
            "wh": [int(clip.width), int(clip.height)],
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class StageRun:
    """Outcome of running one stage: its result, the cache key, and whether it was a hit."""

    stage: Stage
    key: str
    result: Any
    cache_hit: bool


def run_cached(
    queue: JobQueue,
    cache: Cache,
    stage: Stage,
    thunk: Callable[[], Any],
    *,
    input_hash: str,
    params: dict | None = None,
    model_version: str,
    meta: dict | None = None,
) -> StageRun:
    """Run ``stage`` through the cache then the queue.

    On a cache hit the thunk is never invoked. On a miss the thunk is submitted as a job and
    its result stored. Assumes a queue whose result is available once the job finishes (the
    in-process fake completes on ``submit``); an async orchestrator that polls
    :meth:`JobQueue.state` is a future variant, not core.
    """
    params = params or {}
    key = cache.key_for(stage.value, input_hash, params, model_version)
    if cache.has(key):
        return StageRun(stage=stage, key=key, result=cache.get(key), cache_hit=True)
    handle = queue.submit(stage.value, thunk, meta={**(meta or {}), "cache_key": key})
    result = queue.result(handle)
    cache.put(key, result)
    return StageRun(stage=stage, key=key, result=result, cache_hit=False)
