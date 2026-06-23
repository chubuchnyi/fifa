"""Real-frames bake-off scene — reuse the synthetic harness against a *real* dataset.

:class:`SyntheticScene` already carries everything the bake-off harness needs (GT camera, GT
world joints, GT articulation, boxes). The only thing it lacks for a **real** dataset (3DPW /
EMDB / WorldPose) is a pointer to the actual RGB frames so the heavy keypoint/HMR backend can
run on them instead of on the in-memory synthetic stand-in. :class:`PoseEvalScene` adds exactly
that — ``clip_uri`` / ``source_id`` / ``fps`` — and nothing else, so a real scene scores through
the *same* :mod:`pitch3d.eval.harness` code path as synthetic (Condition A / B, Global/Local
MPJPE in metres). The harness reads these three attrs by ``getattr`` with synthetic defaults, so
this module is purely additive: synthetic eval is unchanged.

The concrete dataset loaders (e.g. :func:`pitch3d.eval.datasets_3dpw.load_3dpw_sequence`) build a
:class:`PoseEvalScene` from on-disk GT; this module is the dataset-agnostic seam they target.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .harness import run_conditions
from .synthetic import SyntheticScene

if TYPE_CHECKING:
    from ..adapters.models.pose import HMRBackend
    from ..core.scene.field import FieldCalibration


@dataclass
class PoseEvalScene(SyntheticScene):
    """A :class:`SyntheticScene` whose frames are *real* RGB on disk, not the synthetic stand-in.

    Adds the clip pointer the harness hands to the backend (:func:`harness._clip_and_tracks`):

    * ``clip_uri`` — where the real frames live (e.g. ``file:///data/3dpw/imageFiles/seqA``); the
      backend decodes these instead of the ``memory://synthetic`` URI the fake backends ignore.
    * ``source_id`` — stable scene id used in the assembled ``ClipRef`` (and any export path).
    * ``fps`` — frame rate carried on the ``ClipRef``.

    Every GT field (camera, world/image joints, boxes, articulation) is inherited unchanged, so
    scoring is identical to the synthetic path. Build one from a plain scene with
    :meth:`from_scene`, or directly from a dataset loader.
    """

    clip_uri: str = "memory://synthetic"
    source_id: str = "eval"
    fps: float = 25.0

    @classmethod
    def from_scene(
        cls,
        scene: SyntheticScene,
        *,
        clip_uri: str,
        source_id: str = "eval",
        fps: float = 25.0,
    ) -> PoseEvalScene:
        """Wrap an existing :class:`SyntheticScene` as a real-frames scene pointing at ``clip_uri``.

        Copies every base field by value (via :func:`dataclasses.fields`), so the result scores
        identically to ``scene`` but routes the backend at real RGB. Handy in tests (dress a
        synthetic scene as a dataset scene) and for any loader that first builds the GT geometry
        as a :class:`SyntheticScene` and then attaches its frame source.
        """
        base = {f.name: getattr(scene, f.name) for f in dataclasses.fields(SyntheticScene)}
        return cls(**base, clip_uri=clip_uri, source_id=source_id, fps=fps)


def evaluate_dataset(
    scene: SyntheticScene,
    backend: HMRBackend,
    calibration: FieldCalibration | None = None,
    root_joint: int = 0,
    visible_only: bool = True,
) -> dict[str, dict[str, float] | None]:
    """Score one backend on a (real or synthetic) eval scene → ``{'A': grid, 'B': grid | None}``.

    A thin wrapper over :func:`pitch3d.eval.harness.run_conditions` that flips the default to
    ``visible_only=True`` — real datasets mask occluded joints like the official evaluators do, so
    that is the right default here (the synthetic ``run_*`` helpers keep the permissive default for
    sanity floors). ``A`` is the GT-camera number (pose-net only); ``B`` is the grounded product
    number when a ``calibration`` is supplied, else ``None``.
    """
    return run_conditions(
        scene, backend, calibration, root_joint, visible_only=visible_only
    )
