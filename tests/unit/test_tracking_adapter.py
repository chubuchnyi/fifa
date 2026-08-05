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


# --- #132: a track that changes player mid-way must not keep one identity ------------------

#: Two kits as mean-HSV rows (OpenCV H∈[0,180]): Colombia light blue, Congo DR yellow. Measured
#: off the target clip, not invented -- these are the hues `_kit_scan` separates the teams on.
_BLU_HSV, _YEL_HSV = (105.0, 180.0, 190.0), (25.0, 200.0, 195.0)


def _swap_track(n_blue: int, n_yellow: int, track_id: int = 97) -> RawTracklet:
    series = np.array([_BLU_HSV] * n_blue + [_YEL_HSV] * n_yellow, dtype=float)
    t = n_blue + n_yellow
    return RawTracklet(
        track_id=track_id,
        frames=np.arange(t),
        bboxes_xyxy=np.tile(np.array([0.0, 0.0, 10.0, 20.0]), (t, 1)),
        classes=["player"] * t,
        appearance=np.median(series, axis=0),
        appearance_series=series,
    )


def _centroids():
    from pitch3d.adapters.models.tracking import _hsv_to_feature

    return _hsv_to_feature(np.array([_BLU_HSV, _YEL_HSV], dtype=float))


def test_kit_change_splits_the_track_and_each_piece_wears_one_kit():
    from pitch3d.adapters.models.tracking import _hsv_to_feature, split_on_kit_change

    pieces = split_on_kit_change(_swap_track(8, 8), _centroids(), min_run=4, next_id=500)

    assert len(pieces) == 2, "a track wearing two kits must not stay one identity"
    assert [p.track_id for p in pieces] == [97, 500], "the first piece keeps the original id"
    # Every frame of a piece must sit nearer its own kit centre than the other one.
    for piece, kit in zip(pieces, (_BLU_HSV, _YEL_HSV), strict=True):
        d = np.sum((_hsv_to_feature(piece.appearance_series)
                    - _hsv_to_feature(np.array([kit]))) ** 2, axis=1)
        assert float(d.max()) < 1e-9, "a piece still contains the other team's frames"
    assert sum(p.frames.shape[0] for p in pieces) == 16, "splitting must not drop frames"


def test_a_track_that_never_changes_kit_is_returned_untouched():
    from pitch3d.adapters.models.tracking import split_on_kit_change

    one = _swap_track(16, 0)
    assert split_on_kit_change(one, _centroids(), min_run=4, next_id=500) == [one]


def test_a_brief_flicker_is_not_a_swap():
    from pitch3d.adapters.models.tracking import split_on_kit_change

    # The crossing itself puts the other player's shirt in the box for a frame or two. That is
    # the occlusion, not a handover, and splitting on it would shred healthy tracks.
    flicker = _swap_track(8, 0)
    flicker.appearance_series[4:6] = _YEL_HSV
    assert len(split_on_kit_change(flicker, _centroids(), min_run=4, next_id=500)) == 1
