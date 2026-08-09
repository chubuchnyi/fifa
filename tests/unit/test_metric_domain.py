"""A number must carry the window it is valid in (#141, A-bis).

`fit_rigid_camera.py` printed a bare `jitter` computed by
`bench_frame_preprocessing.smooth_residual`, which fits a **cubic in time across the whole span**.
Its docstring says the assumption — *"a broadcast pan is smooth over 2 s"* — and the number was
read over 236 frames, 7.9 s.

    60 frames = 2.0 s ->  6.42 px
    236 frames = 7.9 s -> 60.42 px

The second was quoted as jitter. At 120x the measured swim (#104: 0.011 m) it sent a whole
investigation after temporal instability that does not exist. Neither number was wrong; the
second was quoted where the first applies.

The fix is not a framework. It is the pattern `apply_rigid_camera.py` already uses — refuse, and
say which range you cover — applied to a value that escapes as a string rather than as a call.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "bfp", _ROOT / "scripts" / "bench_frame_preprocessing.py"
)
_BFP = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BFP)

smooth_residual = _BFP.smooth_residual
smooth_residual_domain = _BFP.smooth_residual_domain
MAX_FRAMES = _BFP.SMOOTH_RESIDUAL_MAX_FRAMES


def test_inside_the_window_the_number_is_unlabelled():
    assert smooth_residual_domain(60) == ""
    assert smooth_residual_domain(MAX_FRAMES) == ""


def test_outside_it_the_number_carries_a_warning_naming_the_span():
    msg = smooth_residual_domain(236)
    assert "OUT OF DOMAIN" in msg
    assert "236" in msg, "the label must name the span it was computed over"
    assert str(MAX_FRAMES) in msg, "and the span it is valid over"


def test_the_warning_says_what_the_number_actually_measures_out_there():
    """'Too long' is not actionable; 'this is camera motion, not noise' is."""
    assert "camera motion" in smooth_residual_domain(236)


def test_the_effect_the_domain_exists_to_describe_is_real():
    """A cubic follows a short pan and not a long one — the residual grows with span alone.

    Same synthetic pan sampled over two spans. Nothing about the motion changes; only how much
    of it a degree-3 polynomial can absorb.
    """
    def pan(n: int) -> np.ndarray:
        t = np.linspace(0.0, n / 30.0, n)            # seconds at 30 fps
        # a smooth but not-cubic sweep: a real operator accelerates, holds, decelerates
        u = 900.0 * np.sin(0.9 * t) + 40.0 * np.sin(5.0 * t)
        return np.stack([np.column_stack([u, np.zeros_like(u)])], axis=1).transpose(0, 1, 2)

    short = float(np.median(smooth_residual(pan(60))))
    long_ = float(np.median(smooth_residual(pan(236))))
    assert long_ > short * 2.0, (
        f"the span effect this domain guard exists for should dominate: {short:.2f} -> {long_:.2f}"
    )
