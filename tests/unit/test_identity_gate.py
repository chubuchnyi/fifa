"""Identity gate (roadmap step 1): GTA-style intra-track appearance split."""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.config.gates import IdentityConfig
from pitch3d.core.orchestration.identity import (
    _dbscan_labels,
    _find_split,
    identity_gate,
)
from pitch3d.core.ports.perception import Tracklet, Tracks


def _tracklet(track_id: int, frames: np.ndarray, cls: str = "player") -> Tracklet:
    n = frames.shape[0]
    return Tracklet(
        track_id=track_id, frames=frames.astype(int),
        bboxes_xyxy=np.zeros((n, 4)), cls=cls,
    )


def _tracks(*tls: Tracklet) -> Tracks:
    return Tracks(tracklets=list(tls), teams=[])


# ─── DBSCAN mini-impl ─────────────────────────────────────────────────────

def test_dbscan_finds_two_clusters_on_disjoint_features():
    """10 rows: 5 yellow + 5 azure → DBSCAN returns 2 labels."""
    yellow = np.tile([1.0, 0.7, 0.0], (5, 1))
    azure = np.tile([0.0, 0.5, 1.0], (5, 1))
    feats = np.vstack([yellow, azure])
    labels = _dbscan_labels(feats, eps=0.05, min_samples=3)
    unique = set(labels.tolist())
    unique.discard(-1)
    assert len(unique) == 2


def test_dbscan_single_cluster_on_similar_features():
    yellow = np.tile([1.0, 0.7, 0.0], (10, 1)) + 0.01 * np.random.default_rng(0).standard_normal((10, 3))
    labels = _dbscan_labels(yellow, eps=0.1, min_samples=3)
    unique = set(labels.tolist())
    unique.discard(-1)
    assert len(unique) == 1


# ─── split logic ──────────────────────────────────────────────────────────

def test_find_split_returns_boundary_index():
    """Labels 0,0,0,0,1,1,1,1 → split at index 4."""
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    frames = np.arange(8)
    cfg = IdentityConfig(dbscan_min_samples=3, min_split_gap_frames=3)
    assert _find_split(labels, frames, cfg) == 4


def test_find_split_rejects_per_frame_flicker():
    """Alternating labels are NOT a valid split — flicker guard."""
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    frames = np.arange(8)
    cfg = IdentityConfig(dbscan_min_samples=3, min_split_gap_frames=3)
    assert _find_split(labels, frames, cfg) is None


def test_find_split_none_when_single_cluster():
    labels = np.zeros(10, dtype=int)
    cfg = IdentityConfig(dbscan_min_samples=3, min_split_gap_frames=3)
    assert _find_split(labels, np.arange(10), cfg) is None


def test_find_split_respects_min_samples_on_each_side():
    """Left side < min_samples → no split."""
    labels = np.array([0, 1, 1, 1, 1, 1, 1, 1])   # only one leading 0
    cfg = IdentityConfig(dbscan_min_samples=3, min_split_gap_frames=3)
    assert _find_split(labels, np.arange(8), cfg) is None


# ─── gate integration ────────────────────────────────────────────────────

def _appearance_provider(track_features: dict[int, np.ndarray]):
    def provider(tracklet: Tracklet):
        return track_features.get(int(tracklet.track_id))
    return provider


def test_disabled_gate_is_passthrough():
    t = _tracklet(1, np.arange(10))
    tracks, report = identity_gate(_tracks(t), IdentityConfig(enabled=False))
    assert len(tracks.tracklets) == 1
    assert tracks.tracklets[0].track_id == 1
    assert report.tracks_split == 0


def test_no_provider_is_measure_only():
    t = _tracklet(1, np.arange(10))
    tracks, report = identity_gate(_tracks(t), IdentityConfig(enabled=True))
    assert len(tracks.tracklets) == 1
    assert report.tracks_no_features == 1


def test_gate_splits_id_swapped_track():
    """One track carries yellow (frames 0-4) then azure (5-9) — split at 5."""
    frames = np.arange(10)
    t = _tracklet(track_id=1, frames=frames)
    feats = np.vstack([
        np.tile([1.0, 0.7, 0.0], (5, 1)),
        np.tile([0.0, 0.5, 1.0], (5, 1)),
    ])
    cfg = IdentityConfig(enabled=True, dbscan_eps=0.10,
                        dbscan_min_samples=3, min_split_gap_frames=3)
    tracks, report = identity_gate(
        _tracks(t), cfg,
        appearance_provider=_appearance_provider({1: feats}),
    )
    assert report.tracks_split == 1
    # child tracklets: original id keeps first half, new id gets second half
    assert len(tracks.tracklets) == 2
    a, b = tracks.tracklets
    assert a.track_id == 1
    assert b.track_id == 2       # next_id above the existing max
    assert a.frames.shape[0] == 5
    assert b.frames.shape[0] == 5
    # neither child carries the old team_id — force reassignment
    assert a.team_id is None
    assert b.team_id is None
    # audit trail
    s = report.splits[0]
    assert s.original_track_id == 1
    assert s.child_track_ids == (1, 2)
    assert s.split_frame == 5
    assert 0.5 < s.intra_cluster_dist <= 1.0


def test_gate_dry_run_reports_but_leaves_tracks_intact():
    frames = np.arange(10)
    t = _tracklet(track_id=1, frames=frames)
    feats = np.vstack([
        np.tile([1.0, 0.7, 0.0], (5, 1)),
        np.tile([0.0, 0.5, 1.0], (5, 1)),
    ])
    cfg = IdentityConfig(
        enabled=True, dbscan_eps=0.10, dbscan_min_samples=3,
        min_split_gap_frames=3, dry_run=True,
    )
    tracks, report = identity_gate(
        _tracks(t), cfg,
        appearance_provider=_appearance_provider({1: feats}),
    )
    assert report.tracks_split == 1
    assert len(tracks.tracklets) == 1        # unchanged
    assert tracks.tracklets[0].track_id == 1


def test_gate_leaves_clean_track_alone():
    frames = np.arange(20)
    t = _tracklet(track_id=1, frames=frames)
    feats = np.tile([1.0, 0.7, 0.0], (20, 1))
    cfg = IdentityConfig(enabled=True)
    tracks, report = identity_gate(
        _tracks(t), cfg,
        appearance_provider=_appearance_provider({1: feats}),
    )
    assert report.tracks_split == 0
    assert len(tracks.tracklets) == 1
    assert tracks.tracklets[0].track_id == 1


def test_gate_provider_shape_mismatch_raises():
    """Provider must return one row per frame; anything else is a bug."""
    t = _tracklet(1, np.arange(10))
    cfg = IdentityConfig(enabled=True)
    with pytest.raises(ValueError, match="rows"):
        identity_gate(
            _tracks(t), cfg,
            appearance_provider=lambda _: np.zeros((7, 3)),
        )


def test_multi_track_split_independent_ids():
    """Two swapped tracks → four output tracks with unique ids."""
    frames = np.arange(10)
    t1 = _tracklet(1, frames)
    t2 = _tracklet(2, frames)
    swap = np.vstack([
        np.tile([1.0, 0.7, 0.0], (5, 1)),
        np.tile([0.0, 0.5, 1.0], (5, 1)),
    ])
    cfg = IdentityConfig(enabled=True, dbscan_min_samples=3,
                        min_split_gap_frames=3)
    tracks, report = identity_gate(
        _tracks(t1, t2), cfg,
        appearance_provider=_appearance_provider({1: swap, 2: swap}),
    )
    assert report.tracks_split == 2
    ids = [tl.track_id for tl in tracks.tracklets]
    assert len(set(ids)) == len(ids)         # unique
    assert set(ids) >= {1, 2}                # original ids kept for LEFT halves


def test_track_with_none_features_skipped():
    frames = np.arange(10)
    t = _tracklet(1, frames)
    tracks, report = identity_gate(
        _tracks(t), IdentityConfig(enabled=True),
        appearance_provider=lambda _: None,
    )
    assert report.tracks_no_features == 1
    assert report.tracks_split == 0
    assert tracks.tracklets == [t]


def test_empty_tracks_returns_empty():
    tracks, report = identity_gate(
        Tracks(tracklets=[], teams=[]),
        IdentityConfig(enabled=True),
    )
    assert tracks.tracklets == []
    assert report.n_input_tracks == 0
