"""Track-continuity stitching (FR-6) — pure-core fragment re-linking + blip rejection.

ByteTrack re-IDs occluded players, so one identity arrives as several fragments. These
tests drive ``stitch_tracks`` with synthetic fragments and assert the conservative gates:
merge only across a real, non-overlapping gap with matching team/class/size and a
velocity-predicted position; never merge overlaps, wrong teams, far jumps, or size
mismatches; drop blips *after* stitching; fabricate no frames; stay deterministic and pure.
"""

from __future__ import annotations

import numpy as np

from pitch3d.core.orchestration.continuity import (
    StitchConfig,
    stitch_tracks,
    stitch_tracks_with_report,
)
from pitch3d.core.ports.perception import Tracklet, Tracks


def _track(tid, f0, f1, x0, y0, *, vx=0.0, vy=0.0, w=20.0, h=40.0, cls="player", team=None):
    """A synthetic tracklet whose centre starts at (x0,y0) and moves (vx,vy) px/frame."""
    frames = np.arange(f0, f1 + 1)
    cx = x0 + vx * (frames - f0)
    cy = y0 + vy * (frames - f0)
    boxes = np.column_stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return Tracklet(track_id=tid, frames=frames, bboxes_xyxy=boxes, cls=cls, team_id=team)


def _ids(tracks):
    return sorted(int(t.track_id) for t in tracks.tracklets)


def test_merge_across_gap_keeps_original_frames():
    # A ends f9 @ x=118 moving +2/f; B resumes f13 right where A is predicted to be.
    a = _track(7, 0, 9, 100.0, 50.0, vx=2.0, team="A")
    b = _track(4, 13, 20, 126.0, 50.0, vx=2.0, team="A")
    out, rep = stitch_tracks_with_report(Tracks(tracklets=[a, b]))
    assert _ids(out) == [4]                      # merged onto the min original id
    merged = out.tracklets[0]
    # frames are exactly the union of both fragments — the gap (10,11,12) is NOT filled
    assert merged.frames.tolist() == list(range(0, 10)) + list(range(13, 21))
    assert 10 not in merged.frames and 11 not in merged.frames and 12 not in merged.frames
    assert rep.merges == [[4, 7]] and rep.dropped == [] and rep.n_out == 1


def test_no_merge_on_overlapping_frames():
    # frames 8,9,10 shared -> two bodies at once -> must stay separate.
    a = _track(0, 0, 10, 100.0, 50.0, team="A")
    b = _track(1, 8, 18, 100.0, 50.0, team="A")
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_no_merge_different_teams():
    a = _track(0, 0, 9, 100.0, 50.0, team="A")
    b = _track(1, 12, 20, 100.0, 50.0, team="B")
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_no_merge_gap_too_large():
    a = _track(0, 0, 9, 100.0, 50.0, team="A")
    b = _track(1, 30, 40, 100.0, 50.0, team="A")   # 20-frame gap > max_gap=12
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_no_merge_spatial_mismatch():
    a = _track(0, 0, 9, 100.0, 50.0, vx=0.0, team="A")
    b = _track(1, 13, 22, 400.0, 50.0, vx=0.0, team="A")  # 300px away, velocity says ~0
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_no_merge_size_mismatch():
    a = _track(0, 0, 9, 100.0, 50.0, h=40.0, team="A")
    b = _track(1, 13, 22, 100.0, 50.0, h=120.0, team="A")  # 3x taller box
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_no_merge_different_class():
    a = _track(0, 0, 9, 100.0, 50.0, cls="player", team="A")
    b = _track(1, 13, 22, 100.0, 50.0, cls="referee", team="A")
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0, 1]


def test_blip_dropped_when_isolated():
    blip = _track(5, 0, 1, 100.0, 50.0)            # 2 frames < min_track_frames=3
    out, rep = stitch_tracks_with_report(Tracks(tracklets=[blip]))
    assert out.tracklets == [] and rep.dropped == [5] and rep.n_out == 0


def test_two_blips_stitched_survive():
    # neither fragment alone clears min_track_frames, but joined (4 frames) they do.
    a = _track(2, 0, 1, 100.0, 50.0, vx=2.0, team="A")
    b = _track(9, 3, 4, 106.0, 50.0, vx=2.0, team="A")
    out, rep = stitch_tracks_with_report(Tracks(tracklets=[a, b]))
    assert _ids(out) == [2]
    assert out.tracklets[0].frames.tolist() == [0, 1, 3, 4]
    assert rep.merges == [[2, 9]] and rep.dropped == []


def test_chain_of_three_merges_to_one():
    a = _track(8, 0, 5, 100.0, 50.0, vx=2.0, team="A")
    b = _track(3, 8, 13, 116.0, 50.0, vx=2.0, team="A")
    c = _track(6, 16, 21, 132.0, 50.0, vx=2.0, team="A")
    out, rep = stitch_tracks_with_report(Tracks(tracklets=[a, b, c]))
    assert _ids(out) == [3]                        # min of {8,3,6}
    merged = out.tracklets[0]
    expected = list(range(0, 6)) + list(range(8, 14)) + list(range(16, 22))
    assert merged.frames.tolist() == expected
    assert rep.merges == [[3, 6, 8]]


def test_merged_team_takes_first_non_none():
    a = _track(0, 0, 9, 100.0, 50.0, vx=2.0, team=None)
    b = _track(1, 12, 20, 124.0, 50.0, vx=2.0, team="B")
    out = stitch_tracks(Tracks(tracklets=[a, b]))
    assert _ids(out) == [0]
    assert out.tracklets[0].team_id == "B"         # wildcard A inherits B's resolved team


def test_deterministic_and_input_not_mutated():
    a = _track(7, 0, 9, 100.0, 50.0, vx=2.0, team="A")
    b = _track(4, 13, 20, 126.0, 50.0, vx=2.0, team="A")
    tracks = Tracks(tracklets=[a, b])
    frames_before = [t.frames.copy() for t in tracks.tracklets]
    boxes_before = [t.bboxes_xyxy.copy() for t in tracks.tracklets]

    out1 = stitch_tracks(tracks)
    out2 = stitch_tracks(tracks)

    assert _ids(out1) == _ids(out2) == [4]
    assert out1.tracklets[0].frames.tolist() == out2.tracklets[0].frames.tolist()
    # inputs untouched
    assert len(tracks.tracklets) == 2
    for t, f0, b0 in zip(tracks.tracklets, frames_before, boxes_before, strict=True):
        assert t.frames.tolist() == f0.tolist()
        assert np.array_equal(t.bboxes_xyxy, b0)


def test_empty_passthrough():
    out, rep = stitch_tracks_with_report(Tracks(tracklets=[]))
    assert out.tracklets == [] and rep.n_in == 0 and rep.n_out == 0


def test_single_long_track_passthrough():
    a = _track(3, 0, 9, 100.0, 50.0)
    out = stitch_tracks(Tracks(tracklets=[a]))
    assert _ids(out) == [3] and out.tracklets[0].frames.tolist() == list(range(10))


def test_teams_list_preserved():
    from pitch3d.core.scene.subject import Team

    teams = [Team(id="A", name="A", color_rgb=(1.0, 0.0, 0.0))]
    a = _track(0, 0, 9, 100.0, 50.0, team="A")
    out = stitch_tracks(Tracks(tracklets=[a], teams=teams))
    assert [t.id for t in out.teams] == ["A"]


def test_config_can_disable_team_gate():
    a = _track(0, 0, 9, 100.0, 50.0, vx=2.0, team="A")
    b = _track(1, 12, 20, 124.0, 50.0, vx=2.0, team="B")
    cfg = StitchConfig(require_same_team=False)
    out = stitch_tracks(Tracks(tracklets=[a, b]), cfg)
    assert _ids(out) == [0]                          # now A and B fragments join
