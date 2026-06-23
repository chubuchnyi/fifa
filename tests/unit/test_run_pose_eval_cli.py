"""Smoke-test the pose-eval CLI end-to-end (argparse → harness → JSON) on synthetic GT.

Runs ``scripts/run_pose_eval.py`` as a subprocess so the whole front door is exercised: the
oracle backend must drive Condition A to ~0 and emit a Condition-B grid, and the JSON contract
the runbook documents must hold. This is the cheap guard that the "first number is one command
away" claim stays true.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_CLI = _REPO / "scripts" / "run_pose_eval.py"


def _run(*args):
    env = {**os.environ, "PYTHONPATH": str(_REPO / "src")}
    out = subprocess.run(
        [sys.executable, str(_CLI), *args],
        capture_output=True, text=True, env=env, check=True,
    )
    return json.loads(out.stdout)


def test_cli_synthetic_oracle_scores_zero_with_condition_b():
    res = _run("--dataset", "synthetic", "--seed", "1", "--condition-b")
    assert res["dataset"] == "synthetic"
    assert res["grid"]["A"]["global_mpjpe_m"] < 1e-9
    assert res["grid"]["A"]["local_mpjpe_m"] < 1e-9
    assert res["grid"]["B"] is not None
    assert res["grid"]["B"]["local_mpjpe_m"] < 1e-9  # root-relative == Condition A


def test_cli_synthetic_zero_backend_is_positive_floor():
    res = _run("--dataset", "synthetic", "--backend", "zero")
    assert res["grid"]["A"]["local_mpjpe_m"] > 0.0
    assert res["grid"]["B"] is None  # no calibration without --condition-b
