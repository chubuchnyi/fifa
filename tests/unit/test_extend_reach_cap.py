"""R-6 bounds presence in distance and speed, and until now not in time.

`extend_to_span` reconstructs a tracker-lost subject instead of blinking it out — the rule the
user set, and it is right. What nothing bounded was **how long the claim of presence survives**.
`extrapolate_decay` caps the coasted *distance*, `coast_max_speed` the *velocity*; the frame count
was the whole clip regardless of how little evidence stood behind it.

Measured 2026-08-09 on the two scenes on disk, counting every non-measured subject-frame by its
distance to the nearest measured one:

* `out/res_ab236/f236_res896.json` — 38 subjects, all 236 frames, median 37 % measured, worst 2 %
  (5 real frames of 236). **47.9 %** of all subject-frames sit further than 12 frames from ANY
  measurement, **11.9 %** further than 120. Worst: 228 frames held from one edge measurement.
* `out/fan_auto/scene_fan_auto.json` — 32 subjects, all 120 frames. **41.4 %** beyond 12 frames.

12 frames is `max_fill_gap` — the distance this same pass refuses to bridge *between two* real
observations. An edge has no observation on the far side at all, so it was the weaker inference
running the longer distance.

`max_extend_frames` is the bound. Default `None` keeps the old behaviour, because changing what
every scene contains without asking is the failure mode this repo keeps hitting.
"""

from __future__ import annotations

import numpy as np
import pytest

from pitch3d.core.correction.coherence import CoherenceConfig, add_temporal_coherence
from pitch3d.core.scene.motion import (
    PoseSequence,
    Provenance,
    SmplxShape,
    SubjectMotion,
)
from pitch3d.core.scene.scene import Scene, Subject

_SPAN = 100


def _subject(track_id: int, first: int, last: int) -> Subject:
    """A subject measured only on ``[first, last]`` — the tracker saw it there and nowhere else."""
    n = last - first + 1
    frames = np.arange(first, last + 1)
    return Subject(
        track_id=track_id,
        proposal=SubjectMotion(
            shape=SmplxShape(betas=np.zeros(10)),
            pose=PoseSequence(
                frames=frames,
                global_orient=np.zeros((n, 3)),
                body_pose=np.zeros((n, 21, 3)),
                # a real walk, so the coast has a velocity to inherit
                transl=np.stack([np.linspace(0, n * 0.1, n), np.zeros(n), np.full(n, 0.92)], 1),
                provenance=np.full(n, Provenance.MEASURED.value),
            ),
        ),
    )


def _scene(*subjects: Subject) -> Scene:
    return Scene(id="s", episode_id="e", source_id="c", subjects=list(subjects))


def _run(cfg: CoherenceConfig, *subjects: Subject):
    out, report = add_temporal_coherence(_scene(*subjects), cfg, fps=25.0)
    return {s.track_id: s.proposal.pose for s in out.subjects}, report


def _cfg(**kw) -> CoherenceConfig:
    return CoherenceConfig(**kw)


# --- the defect, pinned so it cannot come back silently --------------------------------------

def test_without_a_cap_five_measured_frames_still_fill_the_whole_clip():
    """The measured behaviour. If this ever changes by accident, the default changed."""
    brief = _subject(1, 0, 4)          # 5 measured frames
    anchor = _subject(2, 0, _SPAN)     # someone has to establish the span
    poses, _ = _run(_cfg(), brief, anchor)

    prov = np.asarray(poses[1].provenance)
    assert poses[1].frames.shape[0] == _SPAN + 1
    assert int((prov == Provenance.MEASURED.value).sum()) == 5
    assert int((prov == Provenance.IMPUTED.value).sum()) == _SPAN - 4


def test_a_cap_ends_the_subject_where_the_evidence_does():
    brief = _subject(1, 0, 4)
    anchor = _subject(2, 0, _SPAN)
    poses, _ = _run(_cfg(max_extend_frames=12), brief, anchor)

    f = poses[1].frames
    assert int(f[0]) == 0 and int(f[-1]) == 4 + 12, "12 frames past the last measurement, no more"
    assert f.shape[0] == 17


def test_the_cap_reaches_backwards_too():
    """A subject acquired late is extrapolated *before* its first frame by the same rule."""
    late = _subject(1, 60, 64)
    anchor = _subject(2, 0, _SPAN)
    poses, _ = _run(_cfg(max_extend_frames=10), late, anchor)

    f = poses[1].frames
    assert int(f[0]) == 50 and int(f[-1]) == 74


def test_a_subject_that_spans_the_clip_is_untouched_by_the_cap():
    """A guard that shortened a fully measured subject would be worse than the defect."""
    full = _subject(1, 0, _SPAN)
    poses, _ = _run(_cfg(max_extend_frames=5), full)

    assert poses[1].frames.shape[0] == _SPAN + 1
    assert set(np.asarray(poses[1].provenance).tolist()) == {Provenance.MEASURED.value}


def test_the_cap_never_extends_past_the_clip_span():
    """min/max against the span, not a bare +-N: frame 140 of a 100-frame clip renders nothing."""
    brief = _subject(1, 95, _SPAN)
    anchor = _subject(2, 0, _SPAN)
    poses, _ = _run(_cfg(max_extend_frames=40), brief, anchor)

    assert int(poses[1].frames[-1]) == _SPAN
    assert int(poses[1].frames[0]) == 55


def test_the_cap_is_per_subject_not_per_scene():
    """Different evidence must earn different reach from one config — the point of clamping here."""
    brief = _subject(1, 10, 12)
    long = _subject(2, 0, 80)
    anchor = _subject(3, 0, _SPAN)   # holds the span open past subject 2's last frame
    poses, _ = _run(_cfg(max_extend_frames=6), brief, long, anchor)

    assert (int(poses[1].frames[0]), int(poses[1].frames[-1])) == (4, 18)
    assert (int(poses[2].frames[0]), int(poses[2].frames[-1])) == (0, 86)
    assert (int(poses[3].frames[0]), int(poses[3].frames[-1])) == (0, _SPAN)


def test_capped_edge_frames_are_still_imputed_and_low_confidence():
    """The cap shortens the claim; it must not upgrade what is left into a measurement."""
    brief = _subject(1, 0, 4)
    anchor = _subject(2, 0, _SPAN)
    out, _ = add_temporal_coherence(
        _scene(brief, anchor), _cfg(max_extend_frames=8), fps=25.0
    )
    pose = {s.track_id: s.proposal.pose for s in out.subjects}[1]
    prov = np.asarray(pose.provenance)

    assert int((prov == Provenance.IMPUTED.value).sum()) == 8
    conf = out.confidence.subject_frame_conf[1]
    assert float(np.max(conf[prov == Provenance.IMPUTED.value])) == pytest.approx(0.2)


def test_the_report_counts_the_frames_that_were_actually_added():
    """`extended_frames` is what the run log prints; a stale count hides the change."""
    brief = _subject(1, 0, 4)
    anchor = _subject(2, 0, _SPAN)
    _, uncapped = _run(_cfg(), brief, anchor)
    _, capped = _run(_cfg(max_extend_frames=12), brief, anchor)

    assert uncapped.extended_frames == _SPAN - 4
    assert capped.extended_frames == 12
    assert capped.extended_frames < uncapped.extended_frames


def test_zero_means_no_extrapolation_at_all_rather_than_being_read_as_off():
    """`0` and `None` must not collapse — one is 'do not extend', the other 'no limit'."""
    brief = _subject(1, 0, 4)
    anchor = _subject(2, 0, _SPAN)
    poses, report = _run(_cfg(max_extend_frames=0), brief, anchor)

    assert poses[1].frames.shape[0] == 5
    assert report.extended_frames == 0


# --- the override chain: yaml -> env -> CLI, the rule in CLAUDE.md ----------------------------

def test_the_shipped_default_is_unbounded_so_no_scene_changes_without_being_asked():
    from pitch3d.core.config import load_physics_config

    assert load_physics_config(env={}).coherence.max_extend_frames is None


def test_the_env_var_sets_it():
    from pitch3d.core.config import load_physics_config

    cfg = load_physics_config(env={"PITCH3D_COH_MAX_EXTEND": "20"})
    assert cfg.coherence.max_extend_frames == 20
    assert cfg.lineage["coherence.max_extend_frames"].startswith("env")


def test_the_cli_flag_reaches_the_config_and_beats_the_yaml():
    """The chain has to work end to end. A flag argparse accepts and nobody reads is #141."""
    import inspect

    from pitch3d.app import cli

    src = inspect.getsource(cli)
    assert '"--max-extend-frames"' in src, "the flag must be declared"
    assert "max_extend_frames=args.max_extend_frames" in src, "and forwarded to run()"
    assert "max_extend_frames" in inspect.signature(cli.run_dry_run).parameters
    assert "replace(coherence_cfg, max_extend_frames=" in src, \
        "and it must override the loaded physics config, not sit unused next to it"
