"""The identity gate must not silently un-team the tracks it cleans (#137).

Measured on the pod, `out/vert137` (355 frames, `--identity` on): **23 of 27 subjects** reached
`scene.json` with `team_id=None`, while the scene's own `teams` block still held both teams with
member-averaged colours — so the labels existed and the gate dropped them. Three constructors in
`identity.py` emit `team_id=None` under the note *"let downstream re-assign on the clean
identity"*, and there is no downstream: `ByteTrackTracker._assign_teams` runs **before** this gate.

It is not cosmetic. `StitchConfig.require_same_team` treats `None` as a wildcard, so each of those
23 became stitchable to anyone — the gate meant to clean identities was removing the one appearance
constraint that survives at our subject size (a 28 x 72 px player is ~573 px of shirt, and eleven
teammates share it, so team is the *only* appearance signal left).
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.orchestration.identity import IdentityConfig, identity_gate
from pitch3d.core.ports.perception import Tracklet, Tracks

_N = 40
_SPLIT = 20


def _boxes(n: int) -> np.ndarray:
    return np.stack([np.array([10.0 + i, 10.0, 40.0 + i, 100.0]) for i in range(n)])


def _track(tid: int, team: str | None, n: int = _N) -> Tracklet:
    return Tracklet(track_id=tid, frames=np.arange(n), bboxes_xyxy=_boxes(n),
                    cls="player", team_id=team)


#: Two well-separated kit features, the shape `make_hsv_appearance_provider` returns.
_KIT_A = np.array([1.0, 0.0, 0.0, 0.5])
_KIT_B = np.array([0.0, 1.0, 0.0, 0.5])


def _provider(swap_ids: set[int]):
    """Appearance per tracklet: `swap_ids` wear kit A then kit B, everyone else one kit."""
    def provider(t: Tracklet) -> np.ndarray:
        n = np.asarray(t.frames).shape[0]
        if int(t.track_id) in swap_ids:
            rows = [(_KIT_A if i < _SPLIT else _KIT_B) for i in range(n)]
        else:
            rows = [(_KIT_A if int(t.track_id) % 2 == 0 else _KIT_B)] * n
        return np.stack(rows) + 1e-6
    return provider


def _cfg() -> IdentityConfig:
    return IdentityConfig(enabled=True, dbscan_eps=0.2, dbscan_min_samples=3,
                          min_split_gap_frames=4, merge_enabled=False)


def test_a_split_leaves_no_subject_without_a_team():
    """The regression itself: both halves of a split must come back labelled."""
    tracks = Tracks(
        tracklets=[_track(1, "A"), _track(2, "B"), _track(3, "A")],
        teams=[],
    )
    out, report = identity_gate(tracks, _cfg(), _provider({3}))

    assert report.tracks_split == 1, "the fixture must actually split, or this tests nothing"
    assert len(out.tracklets) == 4
    unlabelled = [t.track_id for t in out.tracklets if t.team_id is None]
    assert not unlabelled, f"gate returned {len(unlabelled)} subject(s) with no team: {unlabelled}"
    assert report.teams_restored >= 1


def test_the_two_halves_may_land_on_different_teams():
    """A split cuts where the human changed, so inheriting the parent's label would be wrong."""
    tracks = Tracks(tracklets=[_track(1, "A"), _track(2, "B"), _track(3, "A")], teams=[])
    out, _ = identity_gate(tracks, _cfg(), _provider({3}))

    halves = [t for t in out.tracklets if t.track_id not in (1, 2)]
    assert len(halves) == 2
    assert {t.team_id for t in halves} == {"A", "B"}, (
        "the half wearing kit B must be re-derived as B, not inherited as A"
    )


def test_untouched_tracks_keep_the_label_the_tracker_gave_them():
    """Re-assignment anchors on the survivors; it must never move them."""
    tracks = Tracks(tracklets=[_track(1, "A"), _track(2, "B"), _track(3, "A")], teams=[])
    out, _ = identity_gate(tracks, _cfg(), _provider({3}))
    kept = {t.track_id: t.team_id for t in out.tracklets if t.track_id in (1, 2)}
    assert kept == {1: "A", 2: "B"}


def test_no_anchor_leaves_none_rather_than_inventing_a_team():
    """R-6: with nothing labelled to anchor against, an honest `None` beats a guess."""
    tracks = Tracks(tracklets=[_track(1, None), _track(2, None), _track(3, None)], teams=[])
    out, report = identity_gate(tracks, _cfg(), _provider({3}))
    assert report.teams_restored == 0
    assert all(t.team_id is None for t in out.tracklets)


def test_dry_run_does_not_relabel():
    """`dry_run` measures; it must not mutate what the caller passed in."""
    tracks = Tracks(tracklets=[_track(1, "A"), _track(2, "B"), _track(3, "A")], teams=[])
    cfg = IdentityConfig(enabled=True, dbscan_eps=0.2, dbscan_min_samples=3,
                         min_split_gap_frames=4, merge_enabled=False, dry_run=True)
    out, report = identity_gate(tracks, cfg, _provider({3}))
    assert report.teams_restored == 0
    assert [t.team_id for t in out.tracklets] == ["A", "B", "A"]


@pytest.mark.parametrize("merge", [False, True])
def test_output_is_fully_labelled_with_and_without_the_merge_stage(merge: bool):
    """The merge constructor blanks the team too, so both stages have to be covered."""
    tracks = Tracks(
        tracklets=[_track(1, "A"), _track(2, "B"), _track(3, "A"), _track(4, "B")],
        teams=[],
    )
    cfg = IdentityConfig(enabled=True, dbscan_eps=0.2, dbscan_min_samples=3,
                         min_split_gap_frames=4, merge_enabled=merge)
    out, _ = identity_gate(tracks, cfg, _provider({3}))
    assert all(t.team_id is not None for t in out.tracklets)
