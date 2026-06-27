"""Tracker adapters — shared port contract + ByteTrack vote/team-clustering/provenance (FR-6, M1).

The real ByteTrack adapter is exercised with an *injected* stub backend, so its pure half
(majority-vote class resolution + deterministic 2-means team assignment) is verified with
**no supervision, no cv2, no GPU** — the same AC-7 discipline the fakes follow.
"""

from __future__ import annotations

import importlib.util
from collections import Counter

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeDetector, FakeTracker
from pitch3d.adapters.models.tracking import (
    ByteTrackBackend,
    ByteTrackTracker,
    RawTracklet,
    TrackingBackend,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Detections, Tracker, Tracks

_ROLES = {"player", "goalkeeper", "referee"}


def _clip(frames=(0, 1, 2), width=640, height=360) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


#: Real detections so the fake tracker has something to associate; the stub backend ignores it.
_DETS: Detections = FakeDetector(n_subjects=4).detect(_clip())


class _StubTrackingBackend:
    """Returns canned associated tracklets — stands in for ByteTrack + appearance sampling."""

    def __init__(self, raw: list[RawTracklet]):
        self.raw = raw

    def associate(self, clip: ClipRef, detections: Detections) -> list[RawTracklet]:
        return list(self.raw)


def _raw(track_id, cls, appearance, *, frames=(0, 1, 2), classes=None) -> RawTracklet:
    frames = np.asarray(frames, dtype=int)
    return RawTracklet(
        track_id=track_id,
        frames=frames,
        bboxes_xyxy=np.tile([0.0, 0.0, 10.0, 20.0], (frames.shape[0], 1)),
        classes=classes or [cls] * frames.shape[0],
        appearance=appearance,
    )


# Two colour clusters (HSV-ish): reds near hue 10, blues near hue 120; a no-appearance referee.
_CANNED = [
    _raw(0, "player", [10.0, 120.0, 120.0]),
    _raw(1, "player", [125.0, 120.0, 120.0]),
    _raw(2, "goalkeeper", [14.0, 118.0, 121.0]),
    _raw(3, "referee", None, classes=["referee"] * 3),
    _raw(4, "player", [121.0, 119.0, 119.0]),
]


def _fake_case():
    return FakeTracker(), _clip(), _DETS


def _bytetrack_case():
    return ByteTrackTracker(backend=_StubTrackingBackend(_CANNED)), _clip(), _DETS


@pytest.mark.parametrize("make", [_fake_case, _bytetrack_case], ids=["fake", "bytetrack"])
def test_tracker_port_contract(make):
    tracker, clip, dets = make()
    assert isinstance(tracker, Tracker)
    tracks = tracker.track(clip, dets)
    assert isinstance(tracks, Tracks)
    defined = {t.id for t in tracks.teams}
    for tl in tracks.tracklets:
        assert tl.cls in _ROLES
        assert tl.frames.ndim == 1
        assert tl.bboxes_xyxy.shape == (tl.frames.shape[0], 4)
        if tl.team_id is not None:
            assert tl.team_id in defined  # every referenced team is defined


def test_bytetrack_clusters_two_teams_and_excludes_referee():
    tracks = ByteTrackTracker(backend=_StubTrackingBackend(_CANNED)).track(_clip(), _DETS)
    by_id = {t.track_id: t for t in tracks.tracklets}
    # Cluster holding the smallest track id (0) is always team "A".
    assert by_id[0].team_id == "A"
    assert by_id[2].team_id == "A"  # goalkeeper clusters with track 0 by colour
    assert by_id[1].team_id == "B"
    assert by_id[4].team_id == "B"
    assert by_id[3].team_id is None  # referee → no team
    assert {t.id for t in tracks.teams} == {"A", "B"}


def test_team_color_rgb_is_measured_from_the_cluster():
    # color_rgb was never set (render fell back to an arbitrary tab10 index); it must now carry the
    # cluster's *measured* kit colour so the render paints the real shirts: A red-ish, B blue-ish.
    tracks = ByteTrackTracker(backend=_StubTrackingBackend(_CANNED)).track(_clip(), _DETS)
    by_id = {t.id: t for t in tracks.teams}
    assert by_id["A"].color_rgb is not None and by_id["B"].color_rgb is not None
    ar, ag, ab = by_id["A"].color_rgb
    br, bg, bb = by_id["B"].color_rgb
    assert ar > ag and ar > ab  # team A reads red
    assert bb > br and bb > bg  # team B reads blue


def test_brightness_outlier_does_not_collapse_the_split():
    # The 19/1 collapse: raw-HSV euclidean k-means seeds its 2nd centroid on a light/shadow (high-V)
    # torso, so that one player splits off and everyone else lands in a single team. A hue-aware
    # feature must instead split on kit colour — all reds one team, all blues the other, 5/5.
    reds = [_raw(i, "player", [10.0, 180.0, 120.0]) for i in range(4)]
    reds.append(_raw(4, "player", [10.0, 180.0, 250.0]))  # same hue, much brighter — the outlier
    blues = [_raw(i, "player", [120.0, 180.0, 120.0]) for i in range(5, 10)]
    tracks = ByteTrackTracker(backend=_StubTrackingBackend(reds + blues)).track(_clip(), _DETS)
    team_of = {t.track_id: t.team_id for t in tracks.tracklets}
    assert len({team_of[i] for i in range(5)}) == 1  # every red (incl. the bright one) → one team
    assert len({team_of[i] for i in range(5, 10)}) == 1  # every blue → one team
    assert team_of[0] != team_of[5]  # and the two kits are different teams
    assert sorted(Counter(team_of.values()).values()) == [5, 5]  # balanced, not the old 1/9


def test_majority_vote_resolves_flickering_class():
    raw = [_raw(0, "player", [10.0, 10.0, 10.0], classes=["player", "player", "goalkeeper"])]
    tracks = ByteTrackTracker(backend=_StubTrackingBackend(raw)).track(_clip(), _DETS)
    assert tracks.tracklets[0].cls == "player"


def test_team_ids_are_stable_under_input_order():
    raw = [_raw(0, "player", [10.0, 110.0, 110.0]), _raw(1, "player", [120.0, 110.0, 110.0])]
    a = ByteTrackTracker(backend=_StubTrackingBackend(raw)).track(_clip(), _DETS)
    b = ByteTrackTracker(backend=_StubTrackingBackend(list(reversed(raw)))).track(_clip(), _DETS)
    a_teams = {t.track_id: t.team_id for t in a.tracklets}
    b_teams = {t.track_id: t.team_id for t in b.tracklets}
    assert a_teams == b_teams
    assert next(t for t in a.tracklets if t.track_id == 0).team_id == "A"  # smallest id → A


def test_no_appearance_yields_no_teams():
    raw = [_raw(0, "player", None), _raw(1, "player", None)]
    tracks = ByteTrackTracker(backend=_StubTrackingBackend(raw)).track(_clip(), _DETS)
    assert tracks.teams == []
    assert all(t.team_id is None for t in tracks.tracklets)


def test_min_track_frames_drops_blips():
    raw = [
        _raw(0, "player", [10.0, 1.0, 1.0], frames=(0, 1, 2)),
        _raw(1, "player", [12.0, 1.0, 1.0], frames=(5,)),
    ]
    tracks = ByteTrackTracker(
        backend=_StubTrackingBackend(raw), min_track_frames=2
    ).track(_clip(), _DETS)
    assert [t.track_id for t in tracks.tracklets] == [0]
    assert tracks.tracklets[0].team_id == "A"


def test_bytetrack_provenance():
    info = ByteTrackTracker(n_teams=2).info()
    assert info.name == "ByteTrack+BoT-SORT"
    assert info.backend.value == "local"
    assert info.license == "MIT"
    assert info.params["n_teams"] == 2


def test_backends_satisfy_protocol():
    assert isinstance(_StubTrackingBackend([]), TrackingBackend)
    assert isinstance(ByteTrackBackend(), TrackingBackend)  # structural: has associate


def test_raw_tracklet_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        RawTracklet(
            track_id=0, frames=np.array([0, 1]), bboxes_xyxy=np.zeros((2, 4)), classes=["player"]
        )


@pytest.mark.skipif(
    importlib.util.find_spec("supervision") is not None, reason="cv extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `cv` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[cv\]"):
        ByteTrackTracker().track(_clip(), _DETS)
