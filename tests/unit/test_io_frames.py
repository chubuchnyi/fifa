"""Shared frame decoder (``adapters/io/frames``) — directory branch + URI-scheme stripping.

The video-seek branch needs real media + cv2 and is exercised on the pod; here we cover the
directory branch (deterministic, lossless PNGs) and the pure ``file://`` strip.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.adapters.io.frames import iter_clip_frames, resolve_source_path


def test_resolve_source_path_strips_file_scheme():
    assert resolve_source_path("file:///a/b.mp4") == "/a/b.mp4"
    assert resolve_source_path("/a/b.mp4") == "/a/b.mp4"
    assert resolve_source_path("memory://demo.mp4") == "memory://demo.mp4"  # other schemes kept


def test_iter_clip_frames_reads_requested_directory_frames(tmp_path):
    cv2 = pytest.importorskip("cv2")
    # Three solid frames; the decoder returns them unchanged in OpenCV-native BGR order, indexed by
    # position in the sorted file list (so the caller's frame numbers select the right images).
    colors = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]
    for i, c in enumerate(colors):
        img = np.zeros((4, 5, 3), np.uint8)
        img[:] = c
        cv2.imwrite(str(tmp_path / f"{i:03d}.png"), img)
    got = dict(iter_clip_frames(str(tmp_path), [0, 2]))
    assert set(got) == {0, 2}
    assert got[0].shape == (4, 5, 3)
    np.testing.assert_array_equal(got[0][0, 0], colors[0])  # BGR preserved (no flip in the decoder)
    np.testing.assert_array_equal(got[2][0, 0], colors[2])
