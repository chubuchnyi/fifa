"""Detector adapters — shared port contract + RF-DETR mapping/threshold/provenance (FR-5, M1).

The real RF-DETR adapter is exercised with an *injected* stub backend, so its pure half
(class-map → vocabulary, score threshold, per-frame assembly) is verified with **no torch,
no cv2, no GPU** — the same AC-7 discipline the fakes follow.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from pitch3d.adapters.fakes import FakeDetector
from pitch3d.adapters.models.detection import (
    DetectionBackend,
    RawFrameDetections,
    RFDETRBackend,
    RFDETRDetector,
)
from pitch3d.core.ports.io import ClipRef
from pitch3d.core.ports.perception import Detections, Detector

_VOCAB = {"player", "goalkeeper", "referee", "ball"}


def _clip(frames=(0, 1, 2), width=640, height=360) -> ClipRef:
    return ClipRef(
        source_id="s", uri="x", frames=np.array(frames), width=width, height=height, fps=25.0
    )


class _StubBackend:
    """Returns canned raw detections per frame — stands in for decode + inference."""

    def __init__(self, per_frame: dict[int, tuple]):
        self.per_frame = per_frame

    def detect_raw(self, clip: ClipRef) -> list[RawFrameDetections]:
        out = []
        for f in clip.frames.tolist():
            boxes, ids, scores = self.per_frame[int(f)]
            out.append(
                RawFrameDetections(frame=int(f), boxes_xyxy=boxes, class_ids=ids, scores=scores)
            )
        return out


# A frame mixing every case: player, goalkeeper, referee, ball (all kept),
# a below-threshold player, and an unknown class id (both dropped).
_BOXES = np.array(
    [[10, 10, 20, 40], [30, 10, 40, 40], [50, 10, 60, 40],
     [70, 10, 80, 40], [90, 10, 100, 40], [110, 10, 120, 40]], dtype=float,
)
_IDS = np.array([2, 1, 3, 0, 2, 7])              # player, gk, ref, ball, player(low), unknown
_SCORES = np.array([0.9, 0.8, 0.7, 0.85, 0.1, 0.99])


def _rfdetr_with_stub(frames=(0, 1, 2), threshold=0.3) -> RFDETRDetector:
    per = {int(f): (_BOXES, _IDS, _SCORES) for f in frames}
    return RFDETRDetector(backend=_StubBackend(per), score_threshold=threshold)


@pytest.mark.parametrize(
    "make_detector",
    [lambda: FakeDetector(n_subjects=3), _rfdetr_with_stub],
    ids=["fake", "rfdetr"],
)
def test_detector_port_contract(make_detector):
    det = make_detector()
    assert isinstance(det, Detector)
    clip = _clip()
    result = det.detect(clip)
    assert isinstance(result, Detections)
    assert [fd.frame for fd in result.frames] == clip.frames.tolist()  # one entry per frame, in order
    for fd in result.frames:
        for d in fd.items:
            assert d.cls in _VOCAB
            assert d.bbox_xyxy.shape == (4,)
            assert 0.0 <= d.score <= 1.0


def test_rfdetr_maps_thresholds_and_drops_unknown_ids():
    items = _rfdetr_with_stub(frames=(0,)).detect(_clip(frames=(0,))).frames[0].items
    # kept: ball(0.85), gk(0.8), player(0.9), ref(0.7); dropped: low-score player(0.1), unknown id 7
    assert sorted(d.cls for d in items) == ["ball", "goalkeeper", "player", "referee"]
    assert all(d.score >= 0.3 for d in items)


def test_rfdetr_preserves_box_and_score():
    items = _rfdetr_with_stub(frames=(0,)).detect(_clip(frames=(0,))).frames[0].items
    gk = next(d for d in items if d.cls == "goalkeeper")
    np.testing.assert_array_equal(gk.bbox_xyxy, [30, 10, 40, 40])
    assert gk.score == pytest.approx(0.8)


def test_rfdetr_threshold_is_configurable():
    # floor 0.75 is above the referee's 0.7 (drops) but below the keeper's 0.8 (kept)
    items = _rfdetr_with_stub(frames=(0,), threshold=0.75).detect(_clip(frames=(0,))).frames[0].items
    assert sorted(d.cls for d in items) == ["ball", "goalkeeper", "player"]  # 0.85, 0.8, 0.9


def test_rfdetr_handles_empty_frame():
    per = {0: (np.zeros((0, 4)), np.zeros((0,), int), np.zeros((0,)))}
    result = RFDETRDetector(backend=_StubBackend(per)).detect(_clip(frames=(0,)))
    assert result.frames[0].items == []


def test_rfdetr_provenance():
    info = RFDETRDetector(score_threshold=0.5).info()
    assert info.name == "RF-DETR" and info.version == "sports"
    assert info.backend.value == "local"
    assert info.license == "Apache-2.0"
    assert info.params["score_threshold"] == 0.5


def test_backends_satisfy_protocol():
    assert isinstance(_StubBackend({}), DetectionBackend)
    assert isinstance(RFDETRBackend(), DetectionBackend)  # structural: has detect_raw


def test_raw_frame_detections_rejects_ragged():
    with pytest.raises(ValueError, match="ragged"):
        RawFrameDetections(
            frame=0, boxes_xyxy=np.zeros((2, 4)), class_ids=np.array([1]), scores=np.array([0.5, 0.5])
        )


@pytest.mark.skipif(
    importlib.util.find_spec("rfdetr") is not None, reason="cv extra installed"
)
def test_default_backend_without_extra_is_actionable():
    # No backend injected and the `cv` extra absent → a clear, install-pointing error.
    with pytest.raises(RuntimeError, match=r"pitch3d\[cv\]"):
        RFDETRDetector().detect(_clip(frames=(0,)))
