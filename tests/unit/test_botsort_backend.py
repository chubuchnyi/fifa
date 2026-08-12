"""BoT-SORT backend: the parts that decide whether the A/B is honest.

The association itself needs ultralytics + decoded media, so it is measured by
``scripts/bench_association.py`` (real detections, real clip), not here. What IS worth pinning
without media is everything that could make the two backends stop being comparable, or make a
configured capability silently not reach the run:

* the tracker args are *derived from the inherited fields*, so ByteTrack and BoT-SORT cannot
  drift onto different thresholds while still being reported as an A/B;
* the mask cue is refused loudly instead of ignored (#141);
* the shared tracklet assembly really is shared — the same input builds the same output through
  either class.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.models.botsort_backend import BotSortBackend, _DetView, make
from pitch3d.adapters.models.tracking import ByteTrackBackend, MaskCue, TrackingBackend


def test_satisfies_the_backend_protocol():
    assert isinstance(BotSortBackend(device="cpu"), TrackingBackend)


def test_detview_converts_xyxy_to_the_xywh_bytetracker_reads():
    view = _DetView(
        np.array([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 4.0, 8.0]]),
        np.array([0.9, 0.4]),
        np.array([2.0, 2.0]),
    )
    # centre-x, centre-y, w, h — ultralytics' `results.xywh`, not the corner form we store.
    assert view.xywh.tolist() == [[20.0, 40.0, 20.0, 40.0], [2.0, 4.0, 4.0, 8.0]]
    assert view.conf.tolist() == [0.9, 0.4]
    assert len(view) == 2


def test_tracker_args_follow_the_inherited_thresholds():
    """The A/B is only an A/B if both arms run the same knobs — so derive, never restate."""
    backend = BotSortBackend(
        device="cpu", match_threshold=0.55, activation_threshold=0.4, lost_buffer=12
    )
    args = backend._tracker_args()
    assert args.match_thresh == 0.55
    assert args.track_high_thresh == args.new_track_thresh == 0.4
    assert args.track_buffer == 12
    assert args.with_reid is False

    shared = {"match_threshold", "activation_threshold", "lost_buffer"}
    defaults = ByteTrackBackend(device="cpu")
    same = BotSortBackend(device="cpu")
    for name in shared:
        assert getattr(same, name) == getattr(defaults, name), (
            f"{name} differs between the two backends by default — the A/B would confound "
            "the association algorithm with a threshold change"
        )


def test_gmc_method_is_validated_at_construction():
    with pytest.raises(ValueError, match="unknown gmc_method"):
        BotSortBackend(device="cpu", gmc_method="opticalflow")
    assert BotSortBackend(device="cpu", gmc_method="none").gmc_method == "none"


def test_mask_cue_is_refused_not_ignored():
    """#141: a capability that is configured and silently never reaches the run."""
    backend = BotSortBackend(device="cpu", mask_cue=MaskCue(labels={}))
    with pytest.raises(ValueError, match="mask cue"):
        backend.associate(clip=None, detections=None)


def test_tracklet_assembly_is_shared_with_bytetrack(monkeypatch):
    """Both classes must build identical RawTracklets from identical association output."""
    boxes = {
        7: [(1, np.array([0.0, 0.0, 10.0, 20.0]), "player"),
            (0, np.array([1.0, 1.0, 11.0, 21.0]), "player")],
        3: [(4, np.array([5.0, 5.0, 15.0, 25.0]), "referee")],
    }
    monkeypatch.setattr(ByteTrackBackend, "_sample_appearance", lambda self, clip, b: {})

    a = ByteTrackBackend(device="cpu")._build_tracklets(None, boxes)
    b = BotSortBackend(device="cpu")._build_tracklets(None, boxes)

    assert [t.track_id for t in a] == [t.track_id for t in b] == [3, 7]
    for left, right in zip(a, b, strict=True):
        assert left.frames.tolist() == right.frames.tolist()
        assert left.classes == right.classes
        np.testing.assert_array_equal(left.bboxes_xyxy, right.bboxes_xyxy)
    # frames come back sorted even though track 7 was appended out of order
    assert a[1].frames.tolist() == [0, 1]
    assert a[1].bboxes_xyxy[0].tolist() == [1.0, 1.0, 11.0, 21.0]


def test_make_reads_the_gmc_override(monkeypatch):
    monkeypatch.setenv("PITCH3D_GMC_METHOD", "none")
    monkeypatch.setenv("PITCH3D_DEVICE", "cpu")
    backend = make()
    assert backend.gmc_method == "none"
    assert backend.device == "cpu"
