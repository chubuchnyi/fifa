"""Two ids, one human: merge a handover pair and drop the mannequin (П3 + П2, #135).

The user judged every track of a reference reconstruction by eye on 2026-08-07 and named three
pairs that are **one player each**. The measured criterion behind that judgement — a subject whose
measurements *stop* mid-clip, and another whose measurements *start* where the first one stopped,
within a few metres — is implemented in ``scripts/track_quality.py`` and scores **20 of 21** judged
tracks against the eye. This is that criterion made to act.

**Why the pass lives here and not in the stitcher.** The obvious place is
:mod:`~pitch3d.core.orchestration.continuity`, which re-links fragments *before* pose. Measured
2026-08-07 (`scripts/bench_handover_stitch.py`, and the W3 entry in `docs/work-plan-2026-08.md`):
run pre-pose, the same rule accepts six merges on the broadcast clip and the **video pixels say
four of them join two different shirts**. The reason is the population, not the rule — pre-pose the
tracker hands over 32 tracklets including eight 4-13-frame kit-split children, and
nearest-endpoint in a crowd of fragments picks the man the id jumped *to*. After pose there are 24
long subjects and the same rule is right 20 times in 21.

**What "drop the mannequin" means.** A pose frame marked ``imputed`` carries **exactly 0.00 rad**
of limb articulation while the root keeps coasting — mechanically a sliding mannequin, and what
the eye reads as a phantom. Both halves of a handover pair are full of them: the head is imputed
after it dies, the tail before it is born. The merged subject is built from the **measured rows of
both halves only**; the seam between them is bridged by the same interpolator the coherence pass
uses, so it articulates. Nothing is invented that was not already there.

**R-6 is the reason this is off by default.** Deleting a subject is the one operation that can
*hide* a real player, so ``HandoverConfig.enabled`` starts ``False`` and the pass is opt-in
(`--handover` on the CLI). A wrong merge does not mark, it erases.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from ..correction.coherence import extend_pose_to_span, fill_pose_gaps
from ..scene.layers import ConfidenceMap
from ..scene.motion import PoseSequence, Provenance, SubjectMotion
from ..scene.scene import Scene
from ..scene.subject import Subject


@dataclass(frozen=True)
class HandoverConfig:
    """Gates for deciding that two subjects are one human, and how to rebuild him.

    The defaults are the ones ``scripts/track_quality.py`` was scored with — changing them
    invalidates the 20/21 against the eye, so change them deliberately.
    """

    enabled: bool = False
    #: Frames between the head's last measured frame and the tail's first. Signed: a negative
    #: gap means their measured runs overlap, which happens when an id drifts for a few frames
    #: before dying.
    max_gap: int = 14
    #: Metres between the two measured endpoints on the pitch. Deliberately generous — this is
    #: an endpoint distance, not an extrapolation, and a player crosses 6 m in half a second.
    max_dist: float = 6.0
    #: Frames on which **both** are measured. Above this they are two humans standing close, not
    #: one human with two ids, however near their endpoints are. This is the load-bearing gate.
    max_both: int = 4
    #: Never merge across resolved teams. ``None`` on either side is *not* treated as a wildcard
    #: here (unlike ``StitchConfig``): an unlabelled subject may be merged, a *differently*
    #: labelled one may not.
    require_same_team: bool = True
    #: Frames at either end of the clip within which a subject counts as "measured to the edge",
    #: and therefore neither dies early nor is born late.
    edge_tolerance: int = 3
    #: Metres inside which a merge is *reported* as suspect — never rejected. If the rebuilt
    #: subject ends up this close to a third subject **on frames both of them measured**, the
    #: merge may have taken measurements that belong to that third man. It is a flag, not a gate,
    #: because WorldPose says real players genuinely do it: 39 pairs in 20 clips come within
    #: 0.5 m, for up to 151 frames (3.0 s) at a stretch. Only the eye can settle those.
    twin_radius: float = 0.5
    #: Both-measured frames inside ``twin_radius`` needed before a merge is flagged.
    twin_min_frames: int = 3


@dataclass
class HandoverReport:
    """What the pass did, for logging and for the A/B (R-6 transparency)."""

    #: ``(survivor_track_id, absorbed_track_id)`` per merge, in the order applied.
    merges: list[tuple[int, int]] = field(default_factory=list)
    #: Candidates the assignment saw but did not take, as ``(head, tail, metres)``.
    runners_up: list[tuple[int, int, float]] = field(default_factory=list)
    n_in: int = 0
    n_out: int = 0
    #: Imputed ("mannequin") frames that no longer reach the renderer.
    mannequin_frames_dropped: int = 0
    #: Frames the seam interpolator had to bridge across all merges.
    seam_frames_filled: int = 0
    #: ``(survivor, third_party, min_metres, close_frames, of_which_both_measured)`` — merges
    #: that left the rebuilt subject inside ``twin_radius`` of someone else. Flagged, never
    #: rejected. The last field separates the strong case (two *measurements* in one body) from
    #: the weak one (an interpolated seam passing through a third man).
    suspect: list[tuple[int, int, float, int, int]] = field(default_factory=list)


def _measured_positions(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Indices of measured rows, and the frame numbers they sit on."""
    prov = pose.provenance
    frames = np.asarray(pose.frames, dtype=int).reshape(-1)
    if prov is None:
        return np.arange(frames.shape[0]), frames
    idx = np.flatnonzero(np.asarray(prov) == Provenance.MEASURED.value)
    return idx, frames[idx]


def _candidates(scene: Scene, cfg: HandoverConfig) -> list[dict]:
    """Every legal ``head -> tail`` handover, with its endpoint distance in metres."""
    info: dict[int, dict] = {}
    for s in scene.subjects:
        idx, mframes = _measured_positions(s.proposal.pose)
        if idx.size == 0:
            continue                      # never measured at all: nothing to hand over
        span = np.asarray(s.proposal.pose.frames, dtype=int).reshape(-1)
        info[s.track_id] = {
            "s": s, "idx": idx, "mframes": mframes,
            "first_m": int(mframes[0]), "last_m": int(mframes[-1]),
            "span0": int(span[0]), "span1": int(span[-1]),
            "transl": np.asarray(s.proposal.pose.transl, dtype=float),
        }

    out: list[dict] = []
    for h, hi in info.items():
        if hi["last_m"] >= hi["span1"] - cfg.edge_tolerance:
            continue                      # measured to the end: he never went missing
        for t, ti in info.items():
            if h == t or ti["first_m"] <= ti["span0"] + cfg.edge_tolerance:
                continue                  # measured from the start: he was never born mid-clip
            if cfg.require_same_team:
                ha, ta = hi["s"].team_id, ti["s"].team_id
                if ha is not None and ta is not None and ha != ta:
                    continue
            gap = ti["first_m"] - hi["last_m"]
            if abs(gap) > cfg.max_gap:
                continue
            both = np.intersect1d(hi["mframes"], ti["mframes"]).size
            if both > cfg.max_both:
                continue                  # two humans measured at once, however close
            here = hi["transl"][hi["idx"][-1], :2]
            there = ti["transl"][ti["idx"][0], :2]
            d = float(np.linalg.norm(here - there))
            if d > cfg.max_dist:
                continue
            out.append({"head": h, "tail": t, "gap": int(gap), "dist": d, "both": int(both)})
    return sorted(out, key=lambda c: c["dist"])


def _assign(cands: list[dict]) -> tuple[list[dict], list[dict]]:
    """One partner each, nearest handover first — an assignment, not a candidate list.

    This is the half that keeps a real player whole. On the reference scene t20 is a candidate
    head for t25 at 2.09 m, but t15 claims t25 at 0.85 m; taking every candidate would have
    convicted t20 — whom the eye called correct — of being a phantom.
    """
    taken: set[int] = set()
    kept, dropped = [], []
    for c in cands:
        if c["head"] in taken or c["tail"] in taken:
            dropped.append(c)
            continue
        taken.add(c["head"])
        taken.add(c["tail"])
        kept.append(c)
    return kept, dropped


def _rows(pose: PoseSequence, idx: np.ndarray) -> dict:
    """The selected rows of a pose, as plain arrays ready to concatenate."""
    return {
        "frames": np.asarray(pose.frames, dtype=int).reshape(-1)[idx],
        "global_orient": np.asarray(pose.global_orient, dtype=float)[idx],
        "body_pose": np.asarray(pose.body_pose, dtype=float)[idx],
        "transl": np.asarray(pose.transl, dtype=float)[idx],
    }


def _merge_pair(head: Subject, tail: Subject, cfg: HandoverConfig,
                span: tuple[int, int]) -> tuple[Subject, int, int]:
    """Rebuild one human from the measured rows of both halves. Returns (subject, filled, dropped).

    The survivor keeps the **lower** track id — the same rule the 2D stitcher uses, so ids stay
    comparable across a run with and without this pass — and the shape of whichever half was
    measured on more frames, because betas fitted on 46 frames beat betas fitted on 3.
    """
    hi, _ = _measured_positions(head.proposal.pose)
    ti, _ = _measured_positions(tail.proposal.pose)
    hr, tr = _rows(head.proposal.pose, hi), _rows(tail.proposal.pose, ti)

    primary = head if hi.size >= ti.size else tail
    # On a frame both halves measured, the longer-measured half wins; there are at most
    # `max_both` such frames by construction.
    keep_t = ~np.isin(tr["frames"], hr["frames"]) if primary is head else np.ones(
        tr["frames"].shape[0], dtype=bool)
    keep_h = np.ones(hr["frames"].shape[0], dtype=bool) if primary is head else ~np.isin(
        hr["frames"], tr["frames"])

    frames = np.concatenate([hr["frames"][keep_h], tr["frames"][keep_t]])
    order = np.argsort(frames, kind="stable")

    def cat(key: str) -> np.ndarray:
        return np.concatenate([hr[key][keep_h], tr[key][keep_t]], axis=0)[order]

    merged = PoseSequence(
        frames=frames[order],
        global_orient=cat("global_orient"),
        body_pose=cat("body_pose"),
        transl=cat("transl"),
        provenance=np.full(frames.shape[0], Provenance.MEASURED.value),
    )
    # The seam is a genuine gap between two measurements — exactly what the coherence
    # interpolator is for, and it articulates (unlike an imputed hold).
    merged, filled = fill_pose_gaps(merged, max(cfg.max_gap, 1))
    merged, _added = extend_pose_to_span(merged, span[0], span[1])

    before = sum(
        int(np.sum(np.asarray(s.proposal.pose.provenance) == Provenance.IMPUTED.value))
        for s in (head, tail)
        if s.proposal.pose.provenance is not None
    )
    after = int(np.sum(np.asarray(merged.provenance) == Provenance.IMPUTED.value))

    survivor = head if head.track_id <= tail.track_id else tail
    out = replace(
        survivor,
        track_id=min(head.track_id, tail.track_id),
        proposal=SubjectMotion(shape=primary.proposal.shape, pose=merged),
        team_id=survivor.team_id if survivor.team_id is not None else primary.team_id,
    )
    return out, int(filled.size), max(before - after, 0)


def _flag_twins(
    subjects: list[Subject], merged_ids: set[int], cfg: HandoverConfig
) -> list[tuple[int, int, float, int, int]]:
    """Did a merge leave its survivor standing inside someone else?

    A merged subject that now shares a body with a third man has plausibly taken that man's
    measurements. This does not reject the merge — WorldPose says real players do get that close
    (39 pairs in 20 clips inside 0.5 m, one for 3.0 s straight) and only the eye can separate the
    two cases — it says so out loud (R-6: mark, never hide).

    Frames where *either* subject is ``imputed`` are excluded, matching what
    ``scripts/track_quality.py --twins`` counts, so the two tools cannot disagree. The returned
    tuple separates the strong case from the weak one: an interpolated seam that passes through a
    third man is a visible interpenetration but only inference, whereas two **measurements** in
    one body means one of them is on the wrong human.
    """
    def usable(s: Subject) -> tuple[dict[int, np.ndarray], set[int]]:
        pose = s.proposal.pose
        frames = np.asarray(pose.frames, dtype=int).reshape(-1)
        transl = np.asarray(pose.transl, dtype=float)
        prov = pose.provenance
        prov = (np.full(frames.shape[0], Provenance.MEASURED.value) if prov is None
                else np.asarray(prov))
        keep = prov != Provenance.IMPUTED.value
        pts = {int(f): transl[i, :2] for i, f in enumerate(frames) if keep[i]}
        meas = {int(f) for i, f in enumerate(frames) if prov[i] == Provenance.MEASURED.value}
        return pts, meas

    info = {s.track_id: usable(s) for s in subjects}
    out: list[tuple[int, int, float, int, int]] = []
    for tid in sorted(merged_ids):
        mine, mine_m = info.get(tid, ({}, set()))
        for s in subjects:
            if s.track_id == tid:
                continue
            theirs, their_m = info[s.track_id]
            shared = sorted(set(mine) & set(theirs))
            if not shared:
                continue
            d = np.array([np.linalg.norm(mine[f] - theirs[f]) for f in shared])
            near_f = [f for f, dist in zip(shared, d, strict=True) if dist < cfg.twin_radius]
            if len(near_f) < cfg.twin_min_frames:
                continue
            both = sum(1 for f in near_f if f in mine_m and f in their_m)
            out.append((tid, s.track_id, round(float(d.min()), 3), len(near_f), both))
    return out


def merge_handovers(
    scene: Scene, cfg: HandoverConfig | None = None
) -> tuple[Scene, HandoverReport]:
    """Merge every accepted handover pair into one subject. The input scene is not mutated."""
    cfg = cfg or HandoverConfig()
    report = HandoverReport(n_in=len(scene.subjects), n_out=len(scene.subjects))
    if not cfg.enabled or len(scene.subjects) < 2:
        return scene, report

    cands = _candidates(scene, cfg)
    pairs, losers = _assign(cands)
    report.runners_up = [(c["head"], c["tail"], round(c["dist"], 2)) for c in losers]
    if not pairs:
        return scene, report

    by_id = {s.track_id: s for s in scene.subjects}
    spans = [np.asarray(s.proposal.pose.frames, dtype=int).reshape(-1) for s in scene.subjects]
    span = (int(min(f[0] for f in spans)), int(max(f[-1] for f in spans)))

    absorbed: set[int] = set()
    merged_subjects: dict[int, Subject] = {}
    for c in pairs:
        head, tail = by_id[c["head"]], by_id[c["tail"]]
        sub, filled, dropped = _merge_pair(head, tail, cfg, span)
        merged_subjects[sub.track_id] = sub
        absorbed.update({head.track_id, tail.track_id} - {sub.track_id})
        report.merges.append((sub.track_id, max(head.track_id, tail.track_id)))
        report.seam_frames_filled += filled
        report.mannequin_frames_dropped += dropped

    out = [merged_subjects.get(s.track_id, s)
           for s in scene.subjects if s.track_id not in absorbed]
    out.sort(key=lambda s: s.track_id)
    report.suspect = _flag_twins(out, set(merged_subjects), cfg)

    base = scene.confidence or ConfidenceMap()
    frame_conf = {k: v for k, v in base.subject_frame_conf.items() if k not in absorbed}
    report.n_out = len(out)
    return replace(scene, subjects=out,
                   confidence=replace(base, subject_frame_conf=frame_conf)), report
