"""Track continuity: stitch fragmented tracklets back into stable identities (FR-6).

ByteTrack is forward-causal and has no re-identification, so a player who is briefly
occluded re-enters with a *new* ``track_id``. Downstream that reads as a subject who
"appears from nowhere". This pure-core pass re-links such fragments **conservatively**:
it joins ``A -> B`` only when B starts *strictly after* A ends (never overlapping frames —
that would be two people at once), within a short gap, with compatible class and team,
similar box size, and a predicted-position match (constant-velocity extrapolation from
A's tail).

It is deliberately **structural**: it changes *which identities exist* before pose runs,
so each real player is posed once. It fabricates **nothing** — no synthetic frames are
added; the gap between two fragments is left empty for the separate, non-destructive
coherence corrections to fill. Honesty over coverage (R-6): a missed merge leaves two
fragments, but a *wrong* merge teleports a body — so the gates are strict and unmatched
fragments survive untouched.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from ..ports.perception import Tracklet, Tracks


@dataclass(frozen=True)
class StitchConfig:
    """Gates for linking one fragment to its continuation.

    Pixel distances unless noted; ``max_center_dist`` is in units of the pair's mean
    bbox width, so it scales with apparent player size (near vs. far from camera).
    """

    max_gap: int = 12               # max missing frames between A's end and B's start
    max_center_dist: float = 1.5    # predicted-vs-actual centre gap, in mean-bbox-widths
    max_size_ratio: float = 1.6     # larger/smaller box (w and h each) must stay under this
    min_track_frames: int = 3       # final tracklets shorter than this are dropped as blips
    require_same_team: bool = True  # never link across resolved teams (None is a wildcard)
    require_same_cls: bool = True   # never link player <-> referee <-> ball
    velocity_window: int = 5        # samples at A's tail used to extrapolate its velocity


@dataclass
class StitchReport:
    """What the pass did, for logging / inspection (R-6 transparency)."""

    merges: list[list[int]] = field(default_factory=list)  # each: sorted original ids joined
    dropped: list[int] = field(default_factory=list)       # original ids removed as blips
    n_in: int = 0
    n_out: int = 0


@dataclass
class _Summary:
    """The few geometry/label signals the stitch reasons over (built once per tracklet)."""

    tid: int
    start_frame: int
    end_frame: int
    start_center: np.ndarray
    end_center: np.ndarray
    velocity: np.ndarray   # px/frame at the tail
    mean_w: float
    mean_h: float
    cls: str
    team_id: str | None
    n_frames: int


def _summarize(t: Tracklet, velocity_window: int) -> _Summary:
    frames = np.asarray(t.frames, dtype=int).reshape(-1)
    boxes = np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4)
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    boxes = boxes[order]
    centers = np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0)
    )
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    k = min(velocity_window, len(frames))
    if k >= 2 and (df := float(frames[-1] - frames[-k])) > 0:
        vel = (centers[-1] - centers[-k]) / df
    else:
        vel = np.zeros(2)
    return _Summary(
        tid=int(t.track_id),
        start_frame=int(frames[0]),
        end_frame=int(frames[-1]),
        start_center=centers[0],
        end_center=centers[-1],
        velocity=vel,
        mean_w=float(np.mean(widths)),
        mean_h=float(np.mean(heights)),
        cls=t.cls,
        team_id=t.team_id,
        n_frames=len(frames),
    )


def _link_score(a: _Summary, b: _Summary, cfg: StitchConfig) -> float | None:
    """Normalized spatial mismatch in [0, 1] if ``a -> b`` is a legal link, else ``None``."""
    gap = b.start_frame - a.end_frame - 1
    if gap < 0 or gap > cfg.max_gap:          # strictly no frame overlap; short gap only
        return None
    if cfg.require_same_cls and a.cls != b.cls:
        return None
    if (
        cfg.require_same_team
        and a.team_id is not None
        and b.team_id is not None
        and a.team_id != b.team_id
    ):
        return None
    for sa, sb in ((a.mean_w, b.mean_w), (a.mean_h, b.mean_h)):
        lo, hi = min(sa, sb), max(sa, sb)
        if lo <= 1e-6 or hi / lo > cfg.max_size_ratio:
            return None
    dt = b.start_frame - a.end_frame          # >= 1
    predicted = a.end_center + a.velocity * dt
    dist = float(np.linalg.norm(predicted - b.start_center))
    scale = cfg.max_center_dist * (a.mean_w + b.mean_w) / 2.0
    if scale <= 1e-6 or dist > scale:
        return None
    return dist / scale


def _merge(members: list[Tracklet]) -> Tracklet:
    """Concatenate a chain of fragments into one tracklet — no frames invented."""
    new_id = min(int(t.track_id) for t in members)
    frames = np.concatenate([np.asarray(t.frames, dtype=int).reshape(-1) for t in members])
    boxes = np.concatenate(
        [np.asarray(t.bboxes_xyxy, dtype=float).reshape(-1, 4) for t in members]
    )
    order = np.argsort(frames, kind="stable")
    frames, boxes = frames[order], boxes[order]
    keep = np.concatenate(([True], np.diff(frames) > 0))  # dedupe by frame (safety net)
    frames, boxes = frames[keep], boxes[keep]

    counts: Counter[str] = Counter()
    for t in members:
        counts[t.cls] += int(np.asarray(t.frames).reshape(-1).shape[0])
    top = max(counts.values())
    cls = sorted(c for c, v in counts.items() if v == top)[0]

    team: str | None = None
    for t in sorted(members, key=lambda x: (int(np.min(x.frames)), int(x.track_id))):
        if t.team_id is not None:
            team = t.team_id
            break
    return Tracklet(track_id=new_id, frames=frames, bboxes_xyxy=boxes, cls=cls, team_id=team)


def stitch_tracks_with_report(
    tracks: Tracks, cfg: StitchConfig | None = None
) -> tuple[Tracks, StitchReport]:
    """Re-link fragments and drop blips, returning the new :class:`Tracks` and a report."""
    cfg = cfg or StitchConfig()
    tls = list(tracks.tracklets)
    n = len(tls)
    if n <= 1:
        # a lone tracklet is still subject to blip rejection
        kept = [t for t in tls if np.asarray(t.frames).reshape(-1).shape[0] >= cfg.min_track_frames]
        dropped = [int(t.track_id) for t in tls if t not in kept]
        return (
            Tracks(tracklets=list(kept), teams=list(tracks.teams)),
            StitchReport(dropped=dropped, n_in=n, n_out=len(kept)),
        )

    summaries = [_summarize(t, cfg.velocity_window) for t in tls]

    candidates: list[tuple[float, int, int, int, int, int]] = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            score = _link_score(summaries[i], summaries[j], cfg)
            if score is None:
                continue
            gap = summaries[j].start_frame - summaries[i].end_frame - 1
            candidates.append((score, gap, summaries[i].tid, summaries[j].tid, i, j))
    candidates.sort()

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    used_out = [False] * n
    used_in = [False] * n
    for _score, _gap, _ida, _idb, i, j in candidates:
        if used_out[i] or used_in[j] or find(i) == find(j):
            continue
        used_out[i] = used_in[j] = True
        parent[find(i)] = find(j)

    comps: dict[int, list[int]] = {}
    for idx in range(n):
        comps.setdefault(find(idx), []).append(idx)

    out: list[Tracklet] = []
    report = StitchReport(n_in=n)
    for members in comps.values():
        member_ids = sorted(int(summaries[m].tid) for m in members)
        merged = _merge([tls[m] for m in members])
        if len(members) > 1:
            report.merges.append(member_ids)
        if merged.frames.shape[0] < cfg.min_track_frames:
            report.dropped.extend(member_ids)
            continue
        out.append(merged)

    out.sort(key=lambda t: int(t.track_id))
    report.merges.sort()
    report.dropped.sort()
    report.n_out = len(out)
    return Tracks(tracklets=out, teams=list(tracks.teams)), report


def stitch_tracks(tracks: Tracks, cfg: StitchConfig | None = None) -> Tracks:
    """Conservatively re-link fragmented tracklets and drop blips (see module docstring)."""
    return stitch_tracks_with_report(tracks, cfg)[0]
