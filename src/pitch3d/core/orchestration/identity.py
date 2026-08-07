"""Identity gate — GTA-style split of tracks whose appearance changed mid-clip.

Symptom (user report 2026-07-06): a specific player's shirt colour flickers
across the final video, even though the aggregate team hue is stable.
Root cause: ByteTrack sometimes ID-swaps two physical players who cross
paths; the fused tracklet then carries features from BOTH people. Our
``_assign_teams`` k-means over the mean appearance lands the wrong team
label for half the frames, and every downstream stage (SMPL-X, kit-inject,
v2v) inherits the mismatch.

**Fix (GTA — Global Tracklet Association, Sun et al. ACCV 2024):** DBSCAN
over per-frame appearance features INSIDE each track. Two clusters
separated by ≥ ``min_split_gap_frames`` contiguous frames → the track
fused two people → split at the boundary. New tracklets get fresh
``track_id``s and downstream ``_assign_teams`` sees clean unimodal
appearance distributions.

Design:
* Pure core function; no torch dependency.
* Feature source is INJECTED as a callable ``appearance_provider(tracklet)
  → np.ndarray | None`` of shape ``(T, D)``. Callers wire a real Re-ID
  backbone (OSNet / CLIP-ReIdent) or a cheap HSV extractor.
* When the provider returns ``None`` for a tracklet, we skip that track
  and record it in the report.
* Cross-track merge is a natural next step, deferred to a follow-up
  iteration.

Config lives in ``config/physics.yaml → identity:`` (parametric, YAML-only).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..config.gates import IdentityConfig
from ..ports.perception import Tracklet, Tracks

#: Callable that returns ``(T, D)`` per-frame appearance features for a
#: tracklet, or ``None`` if unavailable. ``T`` MUST equal ``len(tracklet.frames)``.
AppearanceProvider = Callable[[Tracklet], "np.ndarray | None"]


@dataclass
class IdentitySplit:
    """One split event: an original tracklet became two."""

    original_track_id: int
    child_track_ids: tuple[int, int]
    split_frame: int         # first frame of the SECOND child
    intra_cluster_dist: float


@dataclass
class IdentityMerge:
    """One cross-track merge event: two disjoint tracklets fused into one."""

    child_track_ids: tuple[int, int]
    merged_track_id: int
    gap_frames: int
    cosine_distance: float


@dataclass
class IdentityReport:
    n_input_tracks: int = 0
    n_output_tracks: int = 0
    tracks_split: int = 0
    tracks_merged: int = 0
    tracks_no_features: int = 0
    #: Tracklets whose team this gate blanked and :func:`_restore_team_labels` gave back (#137).
    #: Before that pass every split half and every merge reached the scene unlabelled, and a null
    #: team is a **wildcard** to ``StitchConfig.require_same_team``.
    teams_restored: int = 0
    splits: list[IdentitySplit] = field(default_factory=list)
    merges: list[IdentityMerge] = field(default_factory=list)


# ─── DBSCAN (numpy-only mini implementation to avoid sklearn dep) ────────────

def _cosine_distance_matrix(x: np.ndarray) -> np.ndarray:
    """Return pairwise cosine-distance matrix for rows of ``x``."""
    n = x.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    sim = xn @ xn.T
    return 1.0 - np.clip(sim, -1.0, 1.0)


def _dbscan_labels(
    features: np.ndarray, eps: float, min_samples: int,
) -> np.ndarray:
    """Minimal DBSCAN over precomputed cosine distances; -1 = noise."""
    n = features.shape[0]
    labels = np.full(n, -1, dtype=int)
    if n == 0:
        return labels
    dist = _cosine_distance_matrix(features)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbours = np.where(dist[i] <= eps)[0]
        if neighbours.shape[0] < min_samples:
            continue  # noise for now, may get absorbed later
        labels[i] = cluster_id
        stack = list(neighbours)
        while stack:
            j = stack.pop()
            if not visited[j]:
                visited[j] = True
                nn = np.where(dist[j] <= eps)[0]
                if nn.shape[0] >= min_samples:
                    stack.extend(int(k) for k in nn)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


# ─── split logic ─────────────────────────────────────────────────────────────

def _find_split(
    labels: np.ndarray, frames: np.ndarray, cfg: IdentityConfig,
) -> int | None:
    """Given per-frame DBSCAN cluster labels, return the row index that starts
    the second half of a valid split — or None if no clean split exists.

    We only accept a clean two-cluster split: the WHOLE track partitions into
    two contiguous stretches, each with ``>= dbscan_min_samples`` frames, and
    the transition between them is a single boundary (not per-frame flicker).
    """
    n = labels.shape[0]
    if n < 2 * cfg.dbscan_min_samples:
        return None
    # Drop leading/trailing noise
    valid = labels != -1
    if not valid.any():
        return None
    first = int(np.argmax(valid))
    last = n - 1 - int(np.argmax(valid[::-1]))
    core = labels[first:last + 1]
    unique = np.unique(core[core >= 0])
    if unique.shape[0] < 2:
        return None
    # find the change point: earliest index inside [first, last] where the
    # dominant cluster switches from unique[0] to unique[1]
    change: int | None = None
    prev_seen = None
    for i, lbl in enumerate(core):
        if lbl == -1:
            continue
        if prev_seen is None:
            prev_seen = lbl
            continue
        if lbl != prev_seen:
            change = first + i
            break
    if change is None:
        return None
    # protect against per-frame flicker
    left = labels[:change]
    right = labels[change:]
    if (left >= 0).sum() < cfg.dbscan_min_samples:
        return None
    if (right >= 0).sum() < cfg.dbscan_min_samples:
        return None
    # confirm the boundary — at least min_split_gap_frames of one label after
    # the switch before we see the other again
    right_gap = 0
    right_leader = None
    for lbl in right:
        if lbl == -1:
            continue
        if right_leader is None:
            right_leader = lbl
            right_gap = 1
        elif lbl == right_leader:
            right_gap += 1
            if right_gap >= cfg.min_split_gap_frames:
                break
        else:
            return None
    if right_gap < cfg.min_split_gap_frames:
        return None
    return change


def _split_tracklet(t: Tracklet, split_row: int, new_id: int) -> Tracklet:
    """Build the SECOND half of a split, starting at ``split_row``."""
    return Tracklet(
        track_id=new_id,
        frames=np.asarray(t.frames, dtype=int)[split_row:].copy(),
        bboxes_xyxy=np.asarray(t.bboxes_xyxy, dtype=float)[split_row:].copy(),
        cls=t.cls,
        team_id=None,   # let downstream re-assign on the clean identity
    )


def _truncate_tracklet(t: Tracklet, split_row: int) -> Tracklet:
    """Build the FIRST half of a split, ending BEFORE ``split_row``."""
    return Tracklet(
        track_id=int(t.track_id),
        frames=np.asarray(t.frames, dtype=int)[:split_row].copy(),
        bboxes_xyxy=np.asarray(t.bboxes_xyxy, dtype=float)[:split_row].copy(),
        cls=t.cls,
        team_id=None,
    )


def _merge_tracklets(a: Tracklet, b: Tracklet) -> Tracklet:
    """Concatenate two disjoint tracklets; keep the smaller track_id."""
    frames = np.concatenate([np.asarray(a.frames), np.asarray(b.frames)])
    boxes = np.concatenate(
        [np.asarray(a.bboxes_xyxy).reshape(-1, 4),
         np.asarray(b.bboxes_xyxy).reshape(-1, 4)]
    )
    order = np.argsort(frames, kind="stable")
    frames = frames[order]
    boxes = boxes[order]
    new_id = min(int(a.track_id), int(b.track_id))
    return Tracklet(
        track_id=new_id,
        frames=frames,
        bboxes_xyxy=boxes,
        cls=a.cls,
        team_id=None,
    )


def _restore_team_labels(
    tracks: list[Tracklet], feats_by_id: dict[int, np.ndarray],
) -> int:
    """Re-label every tracklet this gate blanked, against the ones that kept their team (#137).

    ``_split_tracklet``, ``_truncate_tracklet`` and ``_merge_tracklets`` each emit
    ``team_id=None`` under the note *"let downstream re-assign on the clean identity"* — and for a
    whole pod run nothing did. The team assignment lives in ``ByteTrackTracker._assign_teams``,
    which runs **before** this gate, so there is no downstream. Measured on `out/vert137`
    (355 frames, `--identity` on): **23 of 27 subjects reached `scene.json` with no team**, while
    the scene's own ``teams`` block still held both teams with member-averaged colours.

    That is not cosmetic. ``StitchConfig.require_same_team`` treats ``None`` as a **wildcard**, so
    every unlabelled subject becomes stitchable to anyone — the gate meant to *clean* identities
    was quietly removing the one appearance constraint that survives at our subject size.

    Re-assign against the tracklets that **kept** a label rather than re-clustering from scratch,
    so the result stays consistent with the ``teams`` block and with every id the tracker already
    stamped. Same cosine metric the split stage uses. A split's two halves are re-derived
    independently and may land on different teams — which is the point of splitting there.

    Returns the number of tracklets relabelled. Leaves ``None`` in place when there is nothing to
    anchor against (every track touched, or no features) — R-6: an unlabelled subject is better
    than an invented team, and the caller now says so out loud.
    """
    anchors: list[tuple[str, np.ndarray]] = []
    for t in tracks:
        feats = feats_by_id.get(int(t.track_id))
        if t.team_id is not None and feats is not None and np.size(feats):
            anchors.append((t.team_id, np.asarray(feats, dtype=float).mean(axis=0)))
    if not anchors:
        return 0

    centres = np.stack([c for _t, c in anchors])
    restored = 0
    for t in tracks:
        if t.team_id is not None:
            continue
        feats = feats_by_id.get(int(t.track_id))
        if feats is None or not np.size(feats):
            continue
        centre = np.asarray(feats, dtype=float).mean(axis=0)
        d = _cosine_distance_matrix(np.vstack([centre[None, :], centres]))[0, 1:]
        t.team_id = anchors[int(np.argmin(d))][0]
        restored += 1
    return restored


def _cross_track_merge(
    tracks: list[Tracklet],
    feats_by_id: dict[int, np.ndarray],
    cfg: IdentityConfig,
) -> tuple[list[Tracklet], list[IdentityMerge]]:
    """Iterative greedy merge across tracklets with disjoint frame ranges.

    A pair (a, b) is merged iff:
      * ``a.frames.max() < b.frames.min()`` (temporally disjoint), AND
      * ``b.frames.min() - a.frames.max() <= merge_max_gap_frames`` (short gap), AND
      * ``cosine(mean_feat(a), mean_feat(b)) <= merge_cosine_threshold``.

    The candidate list is sorted by (distance, gap) and processed greedily so
    the closest identities are matched first. After each merge the working
    set is rebuilt.
    """
    if not tracks:
        return tracks, []
    merges: list[IdentityMerge] = []
    current = list(tracks)

    # per-track mean feature (or None when not measurable)
    def _centroid(t: Tracklet) -> np.ndarray | None:
        f = feats_by_id.get(int(t.track_id))
        if f is None or f.shape[0] == 0:
            return None
        return f.mean(axis=0)

    changed = True
    while changed:
        changed = False
        centroids = {int(t.track_id): _centroid(t) for t in current}
        candidates: list[tuple[float, int, int, int]] = []
        for i, a in enumerate(current):
            ca = centroids[int(a.track_id)]
            if ca is None:
                continue
            a_end = int(np.asarray(a.frames).max())
            for j, b in enumerate(current):
                if j <= i:
                    continue
                cb = centroids[int(b.track_id)]
                if cb is None:
                    continue
                b_start = int(np.asarray(b.frames).min())
                # ensure a is earlier; if not, swap the working pointers
                if b_start <= a_end:
                    a2, b2 = b, a
                    a2_end = int(np.asarray(a2.frames).max())
                    b2_start = int(np.asarray(b2.frames).min())
                    if b2_start <= a2_end:
                        continue  # overlapping — never merge
                    idx_a, idx_b = j, i
                    a_end_use, b_start_use = a2_end, b2_start
                    ca_use, cb_use = cb, ca
                else:
                    idx_a, idx_b = i, j
                    a_end_use, b_start_use = a_end, b_start
                    ca_use, cb_use = ca, cb
                gap = b_start_use - a_end_use - 1
                if gap > cfg.merge_max_gap_frames:
                    continue
                # cosine distance between centroids
                na = np.linalg.norm(ca_use) + 1e-9
                nb = np.linalg.norm(cb_use) + 1e-9
                cos = float(np.dot(ca_use, cb_use) / (na * nb))
                dist = 1.0 - cos
                if dist > cfg.merge_cosine_threshold:
                    continue
                candidates.append((dist, gap, idx_a, idx_b))
        if not candidates:
            break
        candidates.sort(key=lambda x: (x[0], x[1]))
        dist, gap, idx_a, idx_b = candidates[0]
        a, b = current[idx_a], current[idx_b]
        merged = _merge_tracklets(a, b)
        # rebuild feature map: mean-of-means weighted by n frames
        fa = feats_by_id.get(int(a.track_id))
        fb = feats_by_id.get(int(b.track_id))
        if fa is not None and fb is not None:
            feats_by_id[int(merged.track_id)] = np.vstack([fa, fb])
        # drop the two used tracks, add the merged one
        current = [t for k, t in enumerate(current) if k not in (idx_a, idx_b)]
        current.append(merged)
        merges.append(IdentityMerge(
            child_track_ids=(int(a.track_id), int(b.track_id)),
            merged_track_id=int(merged.track_id),
            gap_frames=int(gap),
            cosine_distance=float(dist),
        ))
        changed = True
    # keep a stable output order: sort by first frame
    current.sort(key=lambda t: (int(np.asarray(t.frames).min()), int(t.track_id)))
    return current, merges


def identity_gate(
    tracks: Tracks,
    cfg: IdentityConfig | None = None,
    appearance_provider: AppearanceProvider | None = None,
) -> tuple[Tracks, IdentityReport]:
    """GTA-style intra-track split. Returns new tracks + report.

    * ``cfg is None`` or ``cfg.enabled is False`` → passthrough; report says
      how many tracks would have been split, but the tracks are unchanged.
    * ``appearance_provider is None`` → also passthrough (nothing to cluster
      on); every track counted as ``tracks_no_features``.
    * ``cfg.dry_run is True`` → detect splits, populate report, but do NOT
      emit new tracklets. Useful for measuring in isolation.
    """
    cfg = cfg or IdentityConfig()
    report = IdentityReport(n_input_tracks=len(tracks.tracklets))
    if not cfg.enabled or appearance_provider is None:
        # measure-only fallthrough
        out_tracks = list(tracks.tracklets)
        if appearance_provider is None:
            report.tracks_no_features = report.n_input_tracks
        report.n_output_tracks = len(out_tracks)
        return Tracks(tracklets=out_tracks, teams=list(tracks.teams)), report

    out_tracks: list[Tracklet] = []
    #: Track the CURRENT per-tracklet feature slice by the tracklet's output
    #: track_id so the merge stage doesn't have to re-query the provider on
    #: truncated post-split tracklets (which would trip the shape check).
    output_feats: dict[int, np.ndarray] = {}
    next_id = 1 + max((int(t.track_id) for t in tracks.tracklets), default=0)

    for t in tracks.tracklets:
        feats = appearance_provider(t)
        if feats is None:
            report.tracks_no_features += 1
            out_tracks.append(t)
            continue
        feats = np.asarray(feats, dtype=float)
        if feats.shape[0] != np.asarray(t.frames).shape[0]:
            raise ValueError(
                f"appearance provider returned {feats.shape[0]} rows for track "
                f"{t.track_id} with {np.asarray(t.frames).shape[0]} frames"
            )
        labels = _dbscan_labels(feats, cfg.dbscan_eps, cfg.dbscan_min_samples)
        split_row = _find_split(labels, np.asarray(t.frames), cfg)
        if split_row is None:
            out_tracks.append(t)
            output_feats[int(t.track_id)] = feats
            continue
        left_rows = labels[:split_row]
        right_rows = labels[split_row:]
        left_lbl = int(np.median(left_rows[left_rows >= 0])) if (left_rows >= 0).any() else -1
        right_lbl = int(np.median(right_rows[right_rows >= 0])) if (right_rows >= 0).any() else -1
        # intra-cluster distance: cosine distance between the two centroids
        left_ctr = feats[:split_row][left_rows == left_lbl].mean(axis=0)
        right_ctr = feats[split_row:][right_rows == right_lbl].mean(axis=0)
        intra = float(_cosine_distance_matrix(
            np.stack([left_ctr, right_ctr])
        )[0, 1])
        first_frame_of_right = int(np.asarray(t.frames)[split_row])
        new_id = next_id
        next_id += 1
        report.splits.append(IdentitySplit(
            original_track_id=int(t.track_id),
            child_track_ids=(int(t.track_id), new_id),
            split_frame=first_frame_of_right,
            intra_cluster_dist=intra,
        ))
        report.tracks_split += 1
        if cfg.dry_run:
            out_tracks.append(t)   # keep the original untouched in dry-run
            output_feats[int(t.track_id)] = feats
        else:
            out_tracks.append(_truncate_tracklet(t, split_row))
            out_tracks.append(_split_tracklet(t, split_row, new_id))
            output_feats[int(t.track_id)] = feats[:split_row]
            output_feats[new_id] = feats[split_row:]

    # ─── stage 2: cross-track merge ─────────────────────────────────────
    if cfg.merge_enabled and not cfg.dry_run:
        # reuse the features already sliced for the split stage
        out_tracks, merges = _cross_track_merge(out_tracks, output_feats, cfg)
        report.merges.extend(merges)
        report.tracks_merged = len(merges)
    elif cfg.merge_enabled and cfg.dry_run:
        # measure-only pass on the pre-split tracks so the report is honest
        feats_by_id_dry: dict[int, np.ndarray] = {}
        for t in tracks.tracklets:
            f = appearance_provider(t)
            if f is not None:
                feats_by_id_dry[int(t.track_id)] = np.asarray(f, dtype=float)
        _, merges = _cross_track_merge(
            list(tracks.tracklets), feats_by_id_dry, cfg,
        )
        report.merges.extend(merges)
        report.tracks_merged = len(merges)

    if not cfg.dry_run:
        report.teams_restored = _restore_team_labels(out_tracks, output_feats)

    report.n_output_tracks = len(out_tracks)
    return Tracks(tracklets=out_tracks, teams=list(tracks.teams)), report


__all__ = [
    "AppearanceProvider",
    "IdentityMerge",
    "IdentityReport",
    "IdentitySplit",
    "identity_gate",
]
