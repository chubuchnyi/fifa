# pitch3d — how to work in this repo

Read this first, then `docs/STATUS.md` for what is true right now. Those two are the whole
cold start; everything else is opened on demand.

**Goal:** one broadcast clip → a realistic novel-view video of the same episode. Same players
(kit + numbers), same stadium. **Judged by eye, not by metrics.**

---

## Where to look

| You need | Read |
|----------|------|
| What is open / what to do next | `docs/STATUS.md` (~120 lines, always read in full) |
| Why an open item is open, what was measured | `docs/findings/` |
| Which file owns a subsystem | `docs/code-map.md` |
| What happened on some past date | `docs/archive/status-log-2026-07.md` (verbatim log, 3.3k lines — grep it, don't read it) |
| What was tried and rejected, with numbers | `docs/adr/0012-rejected-approaches-log.md` |
| Pipeline shape end to end | `docs/pipeline.md` |

Do not read `docs/roadmap.md` or `docs/m1-status-and-plan.md` for current state — they are
historical build logs about platform plumbing, not result quality.

## Working rules

These come from repeated user feedback. They override generic good practice.

- **Results over process.** Only work that makes the real-clip output visibly better. No milestone
  flips, no seam-wiring for its own sake, no tests against fake adapters as evidence of progress.
- **The user's eye is ground truth.** On any visual or pixel-alignment question, make ONE small
  change and let the user judge. Believe their verdict over a headless check of mine.
- **Verify end to end every iteration.** Run the whole decode → … → export (→ Blender) path, not
  just the unit that changed.
- **Auto-detect plus manual override.** Every measured estimator ships an override chain:
  auto (npz) → CLI flag → default. Never force the single automated answer.
- **R-6 honesty: mark, never hide.** A subject the tracker lost but that is certainly present gets
  interpolated or extrapolated at low confidence — it never blinks out. The same rule applies to my
  own status claims: measure against the goal, don't narrow the exit criteria to fit what exists.
- **Track state in `docs/STATUS.md` and commit at every checkpoint.** Sessions break and context
  compresses; the CC task list is at most a transient mirror. Commit is standing-authorized,
  **push is not** — ask first.
- **Be decisive when scope is known.** Chain the steps and execute; don't poll for obvious
  confirmations.

## Commands

Run everything from the repo root. Local env is `.venv` — CPU/core work needs no GPU.

```bash
# tests / lint / types
.venv/bin/python -m pytest                 # full suite: 1114 passed / 14 skipped, >5 min (2026-08-01)
.venv/bin/python -m pytest tests/<path>    # focused — prefer this, the full run is slow
.venv/bin/ruff check <files>
.venv/bin/mypy

# end-to-end CLI (the real pipeline entrypoint; see app/cli.py for args)
.venv/bin/python -m pitch3d.app.cli ...          # e.g. --stitch, --real-calib
PYTHONPATH=src python3 -m pitch3d --out-dir out/dryrun   # fakes-only dry run

# R2 camera propagation (#94): default 8, `0` = per-frame (the pre-R2 control side of the A/B).
# On the pod chain the same knob is the CAMERA_CARRY env var, read by demo_video.sh.
python -m pitch3d.app.cli --calibrator keypoints --camera-carry 8 ...

# deliverable video, one command on the pod
bash scripts/pod_finish_batch.sh           # TAIL_ONLY=1 iterates just the generative tail (~15 min, ~$0.2)
```

`just` targets wrap the same things: `setup`, `setup-local`, `cloud-setup`, `test`, `dryrun`,
`lint`, `clean`.

**Runnable evidence.** ADR-0012 cites these — re-run them instead of trusting the write-up:

```bash
.venv/bin/python scripts/mutate_projection_sign.py                  # do the R6 sign guards still catch anything?
PYTHONPATH=src .venv/bin/python scripts/bench_ransac_usac.py        # R10 rejection
PYTHONPATH=src .venv/bin/python scripts/bench_line_constraints.py   # R3 point-on-line gain
PYTHONPATH=src .venv/bin/python scripts/bench_novel_view_metric.py  # R7: why 0.35–0.45 m is not a bar
PYTHONPATH=src .venv/bin/python scripts/bench_joint_limits.py       # R5: is hyperextension ever reached?
PYTHONPATH=src .venv/bin/python scripts/bench_camera_swim.py        # R2: does the camera swim?
PYTHONPATH=src .venv/bin/python scripts/bench_calib_confidence.py   # #105: is calib confidence predictive?
python scripts/motion_stats.py --scene <export/scene.json>          # #207: speed/accel plausibility
```

## Architecture

Hexagonal (ADR-0001): pure `core/` (numpy, CPU, unit-testable) behind ports; `adapters/` hold the
heavy ML, live behind extras and are lazy-imported by dotted path (ADR-0006). Corrections are the
sole edit path (ADR-0002). Human ≡ LLM as editors, over MCP (ADR-0008/0010).

Subsystem → files: `docs/code-map.md`.

**Known structural debt — do not add to it** (measured 2026-08-01, remediation not started):

- **6 pipeline entry points** re-implement orchestration: `app/cli.py`, `app/controller.py`,
  `app/anim_export.py`, `poseannot/`, `scripts/*`, pod shell scripts. New work goes through
  `controller.Application`; do not start a seventh path.
- **4 calibration routes, 1 wired.** Only `KeypointFieldCalibrator` is in `wiring.py`.
  `CameraModuleFieldCalibrator`, `PnLCalibBackend` and `scripts/{fit,apply,bench}_rigid_camera.py`
  are parallel. Check `wiring.py` before assuming which one runs.
- **Stubs that raise `NotImplementedError`** are publicly exported from `adapters/models/__init__.py`
  and can be instantiated on the fly (`adapters/models/pose.py`). A surprise `NotImplementedError`
  at runtime is this, not a missing dependency.
- **Config has no single source.** `config/*.yaml` + `.env` + CLI flags + hardcoded constants
  (`core/scene/cameras.py`, `core/correction/kinematics.py`) with no documented precedence.

## Testing

The suite is green and that does **not** mean the pipeline works. `tests/conftest.py` is
fakes-backed by design ("no GPU/Blender/models"), there is no golden test over a real clip, and
~6000 lines of the user-facing path (`controller.py`, `cli.py`, `anim_export.py`, `poseannot/*`,
`blender_animate.py`) have no direct coverage.

So: a green run is a smoke signal, not evidence. Evidence is the rendered output plus the numbers
from the probe scripts above.

When adding tests, prefer one assertion against real captured data over ten against fakes. Do not
add tests that re-implement the function under test in the assertion, or that only check a call
does not raise.

## Lint

311 ruff errors exist repo-wide (87 E501, 71 F401, 45 E702, 34 I001) and there is no CI or
pre-commit. Lint **only the lines you changed** — baseline-diff rather than "fixing" unrelated
violations:

```bash
git show HEAD:<file> | .venv/bin/ruff check --stdin-filename <file> -
```

## GPU pod (RunPod)

~$0.74/hr — **stop it whenever you are not actively rendering.** The network volume persists and
restart is free.

The pod repo `/workspace/fifa` is a **stale mirror**: it reports already-pushed work as
"uncommitted". Reconcile by byte-comparing against pushed HEAD, then checkout + rm + pull. Never
blind-commit on the pod. Pod findings and validation numbers land in committed scripts/docs, not
just in chat.

Watcher scripts: `pgrep -f batch.sh` over ssh matches the watcher **itself** — use `[b]atch.sh` or
grep the log for `BATCH_FINISH_OK`.

## Gotchas that cost a session before

- **The solved camera is 180° rolled** relative to raw video. Any raw-pixel consumer must rotate the
  frame first (auto-detect: `-R[1][2] < 0`).
- **The clip is a floodlit night match** — no sky, neutral-cool light, soft multi-shadows. Never
  light it as daytime.
- **Editing `poseannot/static/style.css`?** Bump the `?v=` token in `index.html`, or the browser
  verifies the old stylesheet.
- **Wan amplifies stated colour adjectives.** State every large surface's colour in the v2v prompt
  at its *measured* intensity, or the prior repaints it.
- **Zoom before a visual verdict.** Thumbnail-scale eye errors have produced two false defects.
