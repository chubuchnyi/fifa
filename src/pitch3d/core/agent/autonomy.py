"""A-10: bounded, attention-driven autonomous correction — the agent's hands, kept honest.

The MCP agent (ADR-0008) must *act*, not only observe: read the "needs attention" list, target the
worst problem, and apply a correction — without the freedom to wander. This module is that policy as
pure core logic (no LLM, no port, no GPU), so the loop is deterministic and unit-tested:

* **Attention-driven targeting** — the next edit is always the top :func:`attention_list` item,
  never a blind scan (UX-4).
* **Bounded edits** — an :class:`EditBudget` caps how many corrections and how far (metres) one edit
  may move a root; a candidate beyond the cap is *clipped*, never applied wholesale, so a large
  error converges over several small honest steps instead of one teleport.
* **Measured verification** — "fixed" means the resolved root is back on its **measured homography
  anchor** (``core.correction.anchor``); the loop re-scores from the anchor residual after every
  edit, so the attention list clearing is a measured fact, not the agent's say-so (R-6).

:func:`auto_correct` runs the closed loop and returns a new scene + an :class:`AutonomyReport`. The
input scene is never mutated (ADR-0002): every fix is a normal, inspectable, disableable correction
layered on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..correction.anchor import DEFAULT_MAX_RESIDUAL_M, anchor_residuals
from ..correction.engine import make_offset
from ..orchestration.assemble import resolve_scene
from ..scene.layers import ConfidenceMap, Correction, CorrectionTarget, TargetKind
from ..scene.review import AttentionItem, attention_list
from ..scene.scene import Scene

#: ``track_id -> (T, 2)`` per-frame, or ``(2,)`` locked, measured ground-plane anchor for a subject.
Anchors = dict[int, np.ndarray]


@dataclass(frozen=True)
class EditBudget:
    """The leash: bounds on what the autonomous loop may do."""

    max_edits: int = 8                              # hard cap on corrections applied across the run
    max_abs_change_m: float = 1.0                   # per-edit XY move cap on a root (metres)
    blend: float = 1.0                              # how hard each edit pulls to the anchor, [0, 1]
    max_residual_m: float = DEFAULT_MAX_RESIDUAL_M  # on-anchor tolerance; sets conf crossover
    min_score: float = 0.0                          # ignore attention items at/below this urgency


@dataclass
class AutonomyStep:
    """One bounded edit the loop applied (for inspection / the report)."""

    track_id: int
    frame_range: tuple[int, int]
    delta_m: tuple[float, float]    # the XY ground offset applied
    residual_before_m: float        # worst off-anchor residual on targeted frames pre-edit
    clipped: bool                   # True if the pull was capped by max_abs_change_m


@dataclass
class AutonomyReport:
    """What the run did — attention before/after is the measured proof it helped (R-6)."""

    attention_before: int
    attention_after: int
    edits_applied: int
    cleared: bool
    steps: list[AutonomyStep] = field(default_factory=list)


def residual_to_confidence(residuals: np.ndarray, max_residual_m: float) -> np.ndarray:
    """Map per-frame anchor residual (m) to confidence in ``(0, 1]``.

    ``m / (m + r)``: 1.0 at zero residual, exactly 0.5 at ``r == max_residual_m`` (the on-anchor
    tolerance), decaying toward 0 as the root drifts. The 0.5 crossover lines up with the default
    attention threshold, so an off-anchor frame trips the "needs attention" list and an on-anchor
    one does not.
    """
    r = np.asarray(residuals, dtype=float)
    m = max(float(max_residual_m), 1e-9)
    return m / (m + r)


def rescore_from_anchors(scene: Scene, anchors: Anchors, *, max_residual_m: float) -> Scene:
    """Return a copy of ``scene`` whose ``confidence`` reflects each root's *resolved* anchor fit.

    Resolves the correction stack (pure — offsets only) and recomputes ``subject_frame_conf`` from
    the anchor residual, so the attention list scores the current edited state, not the raw
    proposal. Subjects without an anchor keep their existing confidence. The input scene is not
    mutated.
    """
    resolved = resolve_scene(scene)
    base = scene.confidence or ConfidenceMap()
    frame_conf = dict(base.subject_frame_conf)
    for s in resolved.subjects:
        anchor = anchors.get(s.track_id)
        if anchor is None:
            continue
        res = anchor_residuals(s.proposal.pose.transl, anchor)
        frame_conf[s.track_id] = residual_to_confidence(res, max_residual_m)
    return replace(scene, confidence=replace(base, subject_frame_conf=frame_conf))


def _root_target(track_id: int) -> CorrectionTarget:
    return CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=track_id)


def _anchor_rows(anchor: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """The anchor XY for the selected ``rows``; a ``(2,)`` locked anchor broadcasts to all."""
    a = np.asarray(anchor, dtype=float)
    if a.ndim == 1:
        return np.broadcast_to(a.reshape(1, 2), (int(np.count_nonzero(rows)), 2))
    return a.reshape(-1, 2)[rows]


def propose_anchor_offset(
    scene: Scene, track_id: int, anchor: np.ndarray, *, budget: EditBudget, edit_id: str,
) -> tuple[Correction, AutonomyStep] | None:
    """Propose one bounded CONSTANT_OFFSET pulling a subject's off-anchor root toward the anchor.

    Returns the correction + a step record, or ``None`` if the subject is already on-anchor. The
    offset is the mean ground displacement over the off-anchor frames, scaled by ``blend`` and
    clipped to ``max_abs_change_m`` — a large error is reduced one bounded step at a time.
    """
    s = resolve_scene(scene).subject(track_id)
    transl = s.proposal.pose.transl
    frames = np.asarray(s.proposal.pose.frames, dtype=int)
    res = anchor_residuals(transl, anchor)
    off = res > budget.max_residual_m
    if not np.any(off):
        return None

    fr = (int(frames[off].min()), int(frames[off].max()))
    rows = (frames >= fr[0]) & (frames <= fr[1])
    delta_xy = budget.blend * np.mean(_anchor_rows(anchor, rows) - transl[rows, :2], axis=0)
    mag = float(np.linalg.norm(delta_xy))
    clipped = mag > budget.max_abs_change_m
    if clipped and mag > 0.0:
        delta_xy = delta_xy * (budget.max_abs_change_m / mag)

    delta = np.array([float(delta_xy[0]), float(delta_xy[1]), 0.0])
    corr = make_offset(edit_id, _root_target(track_id), fr, delta, note="auto: pull root to anchor")
    step = AutonomyStep(
        track_id=track_id,
        frame_range=fr,
        delta_m=(float(delta[0]), float(delta[1])),
        residual_before_m=float(res[off].max()),
        clipped=clipped,
    )
    return corr, step


def attention_targets(scene: Scene, anchors: Anchors, *, budget: EditBudget) -> list[AttentionItem]:
    """The attention items this loop owns: off-anchor frames of anchored subjects, worst first.

    Re-scores from the anchors, then keeps only the ``low_confidence`` items for subjects we hold an
    anchor for and whose urgency clears ``min_score`` — the ball-height / reprojection signals are a
    different concern the anchor loop deliberately does not touch (R-4).
    """
    scored = rescore_from_anchors(scene, anchors, max_residual_m=budget.max_residual_m)
    return [
        it
        for it in attention_list(scored)
        if it.reason == "low_confidence"
        and it.track_id is not None
        and it.track_id in anchors
        and it.score > budget.min_score
    ]


def auto_correct(
    scene: Scene, anchors: Anchors, *, budget: EditBudget | None = None,
) -> tuple[Scene, AutonomyReport]:
    """Iteratively fix the worst off-anchor subject until attention clears or the budget is spent.

    Each round: re-score from the anchors, take the top :func:`attention_targets` item, and apply
    one bounded anchor-pull offset to that subject. Returns the edited scene (corrections layered,
    the input never mutated) + an :class:`AutonomyReport` scored over the off-anchor attention it
    owns. Deterministic.
    """
    budget = budget or EditBudget()
    work = scene
    steps: list[AutonomyStep] = []

    before = attention_targets(work, anchors, budget=budget)
    for i in range(budget.max_edits):
        items = attention_targets(work, anchors, budget=budget)
        if not items:
            break
        tid = items[0].track_id
        assert tid is not None  # guaranteed by attention_targets
        proposed = propose_anchor_offset(
            work, tid, anchors[tid], budget=budget, edit_id=f"auto-fix-{i}-{tid}"
        )
        if proposed is None:
            break
        corr, step = proposed
        work = replace(work, corrections=[*work.corrections, corr])
        steps.append(step)

    after = attention_targets(work, anchors, budget=budget)
    report = AutonomyReport(
        attention_before=len(before),
        attention_after=len(after),
        edits_applied=len(steps),
        cleared=len(after) == 0,
        steps=steps,
    )
    return work, report
