"""PnLCalib backend: import-safety + seam contract, the parts verifiable with no GPU/weights.

The detection path needs PnLCalib + CUDA + weights (box only), so here we pin only what must hold
on any machine: the module imports without the heavy stack, ``make()`` builds an env-configured
backend that satisfies **both** the :class:`KeypointBackend` (DLT path) and
:class:`HomographyBackend` (camera-module path) protocols the dotted-path seam checks, and the
``PNLCALIB_*`` environment knobs actually land on the backend.
"""

from __future__ import annotations

import importlib

from pitch3d.adapters.models.calibration import HomographyBackend, KeypointBackend


def test_module_is_import_safe() -> None:
    # Must import with no torch/cv2/PnLCalib present (all heavy imports are lazy in _load).
    mod = importlib.import_module("pitch3d.adapters.models.pnlcalib_backend")
    assert hasattr(mod, "make")


def test_make_satisfies_both_backend_protocols() -> None:
    from pitch3d.adapters.models.pnlcalib_backend import make

    backend = make()
    # One backend, two solver paths the eval A/Bs via --solver dlt|camera.
    assert isinstance(backend, KeypointBackend)  # runtime_checkable: has detect_keypoints
    assert isinstance(backend, HomographyBackend)  # runtime_checkable: has calibrate_frames
    assert callable(backend.detect_keypoints)
    assert callable(backend.calibrate_frames)


def test_make_reads_env(monkeypatch) -> None:
    from pitch3d.adapters.models.pnlcalib_backend import make

    monkeypatch.setenv("PNLCALIB_REPO", "/tmp/pnl")
    monkeypatch.setenv("PNLCALIB_WEIGHTS_KP", "/tmp/kp.pt")
    monkeypatch.setenv("PNLCALIB_WEIGHTS_LINES", "/tmp/lines.pt")
    monkeypatch.setenv("PNLCALIB_DEVICE", "cpu")
    monkeypatch.setenv("PNLCALIB_KP_THRESHOLD", "0.2")
    monkeypatch.setenv("PNLCALIB_LINE_THRESHOLD", "0.5")

    backend = make()
    assert backend.repo == "/tmp/pnl"
    assert backend.weights_kp == "/tmp/kp.pt"
    assert backend.weights_lines == "/tmp/lines.pt"
    assert backend.device == "cpu"
    assert backend.kp_threshold == 0.2
    assert backend.line_threshold == 0.5
    assert backend.pnl_refine is True  # camera-path PnL line refinement defaults on


def test_make_reads_pnl_refine_flag(monkeypatch) -> None:
    from pitch3d.adapters.models.pnlcalib_backend import make

    assert make().pnl_refine is True  # unset → default on
    for raw, expected in [("0", False), ("false", False), ("no", False),
                          ("1", True), ("true", True), ("on", True)]:
        monkeypatch.setenv("PNLCALIB_PNL_REFINE", raw)
        assert make().pnl_refine is expected


def test_make_kwargs_override_env(monkeypatch) -> None:
    # The threshold sweep passes explicit thresholds; an explicit kwarg must win over the env
    # default (precedence: kwarg > env > PnLCalib default), without touching other env-set fields.
    from pitch3d.adapters.models.pnlcalib_backend import make

    monkeypatch.setenv("PNLCALIB_KP_THRESHOLD", "0.3434")
    monkeypatch.setenv("PNLCALIB_LINE_THRESHOLD", "0.7867")

    backend = make(kp_threshold=0.15, line_threshold=0.6)
    assert backend.kp_threshold == 0.15  # kwarg wins
    assert backend.line_threshold == 0.6
    # An omitted kwarg still falls back to the env value.
    assert make(kp_threshold=0.15).line_threshold == 0.7867
