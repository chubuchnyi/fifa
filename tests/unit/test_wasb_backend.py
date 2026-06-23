"""WASB ball backend: the torch-free seam (windowing, assembly, factory, dotted-path resolution).

The GPU inference path (``_load``/``_preprocess_window``/``detect_ball``) needs a WASB checkout +
weight + CUDA and is exercised on the pod, not here (``# pragma: no cover``). What *is* pinned
headlessly is everything the wiring depends on: the pure window tiling and per-frame assembly, that
importing the module is torch-free, that ``make`` reads its env config and satisfies the
``BallDetectionBackend`` protocol, and that the dotted-path seam resolves it (ADR-0006).
"""

from __future__ import annotations

import pytest

from pitch3d.adapters.models.ball import BallDetectionBackend, RawBallDetections
from pitch3d.adapters.models.wasb_backend import (
    WASBBallBackend,
    _assemble_detections,
    _window_starts,
    make,
)
from pitch3d.app.wiring import _resolve_backend


# --- pure windowing ------------------------------------------------------------------
def test_window_starts_non_overlapping_exact():
    # n a multiple of frames_in: clean non-overlapping tiling, no clamped tail.
    assert _window_starts(6, frames_in=3, step=3) == [0, 3]
    assert _window_starts(9, frames_in=3, step=3) == [0, 3, 6]


def test_window_starts_clamps_a_final_window_to_cover_the_tail():
    # n not a multiple: a final window is clamped to end=n-frames_in so the last frame is covered.
    assert _window_starts(8, frames_in=3, step=3) == [0, 3, 5]
    assert _window_starts(7, frames_in=3, step=3) == [0, 3, 4]


def test_window_starts_short_and_empty_clips():
    assert _window_starts(0, frames_in=3, step=3) == []
    assert _window_starts(2, frames_in=3, step=3) == [0]  # shorter than a window -> one padded
    assert _window_starts(3, frames_in=3, step=3) == [0]  # exactly one window


def test_window_starts_sliding_step_one():
    assert _window_starts(5, frames_in=3, step=1) == [0, 1, 2]


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 20])
def test_every_frame_is_covered_by_some_window(n):
    frames_in, step = 3, 3
    covered: set[int] = set()
    for start in _window_starts(n, frames_in, step):
        for j in range(frames_in):
            if start + j < n:  # the same clamp-pad guard detect_ball applies
                covered.add(start + j)
    assert covered == set(range(n))


# --- pure assembly -------------------------------------------------------------------
def test_assemble_marks_visible_detections_and_zeros_occluded():
    frames = [10, 11, 12]
    xy = {0: (100.0, 50.0), 1: (0.0, 0.0), 2: (120.0, 60.0)}
    visi = {0: True, 1: False, 2: True}
    raw = _assemble_detections(frames, xy, visi)

    assert isinstance(raw, RawBallDetections)
    assert raw.frames.tolist() == [10, 11, 12]
    assert raw.scores.tolist() == [1.0, 0.0, 1.0]  # visibility surfaced as confidence
    assert raw.points_xy[0].tolist() == [100.0, 50.0]
    assert raw.points_xy[1].tolist() == [0.0, 0.0]  # occluded -> placeholder for the gap-filler


def test_assemble_empty_clip_is_well_formed():
    raw = _assemble_detections([], {}, {})
    assert raw.frames.shape == (0,)
    assert raw.points_xy.shape == (0, 2)
    assert raw.scores.shape == (0,)


# --- factory + seam ------------------------------------------------------------------
def test_make_reads_env(monkeypatch):
    monkeypatch.setenv("PITCH3D_WASB_REPO", "/x/WASB")
    monkeypatch.setenv("PITCH3D_WASB_CKPT", "/x/wasb_soccer.pth.tar")
    monkeypatch.setenv("PITCH3D_WASB_DATASET", "soccer")
    monkeypatch.setenv("PITCH3D_DEVICE", "cuda")
    backend = make()
    assert isinstance(backend, WASBBallBackend)
    assert backend.repo_dir == "/x/WASB"
    assert backend.weights == "/x/wasb_soccer.pth.tar"
    assert backend.device == "cuda"


def test_make_satisfies_ball_detection_backend_protocol():
    # runtime-checkable protocol: presence of detect_ball is the contract the adapter injects.
    assert isinstance(make(), BallDetectionBackend)


def test_resolves_over_the_dotted_path_seam():
    backend = _resolve_backend(
        "pitch3d.adapters.models.wasb_backend:make", BallDetectionBackend
    )
    assert isinstance(backend, WASBBallBackend)
