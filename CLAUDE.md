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
| What we validated / refuted / never judged, across every thread | `docs/findings/research-ledger-2026-08-07.md` |
| What was tried and rejected, with numbers | `docs/adr/0012-rejected-approaches-log.md` |
| Pipeline shape end to end | `docs/pipeline.md` |
| Exact data contract: types, shapes, units, npz keys | `docs/pipeline-io.md` |
| How to run on the local RTX 4080 instead of the pod | `docs/local-gpu-box.md` |

Do not read `docs/roadmap.md` or `docs/archive/m1-status-and-plan.md` for current state — they are
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
.venv/bin/python -m pytest                 # full suite: 1168 passed / 19 skipped in 21 s (2026-08-07)
.venv/bin/python -m pytest tests/<path>    # focused, for a tight edit loop — but 21 s is cheap,
                                           # so run the full suite before you call anything done
# NB: pyproject already sets `-q`. Adding another `-q` makes it `-qq`, which silently swallows
# the "N passed" summary line — that is how the ">5 min" claim here survived unchallenged.
.venv/bin/ruff check <files>
.venv/bin/mypy

# end-to-end CLI (the real pipeline entrypoint; see app/cli.py for args)
.venv/bin/python -m pitch3d.app.cli --clip <mp4> --frames 48 --out-dir out/run \
  --detector rfdetr --tracker bytetrack --device cpu --render overlay --export gltf
# `--export scene` is NOT a value — the choices are fake|gltf|threejs (`gltf` writes scene.json).
# `--real-calib` is NOT a CLI flag — it is scripts/demo_video.sh's (on by default there) and
# needs the pod-only PnLCalib weights. The stitch flag is `--no-stitch`; `--stitch` is not real.
# Adding `--calibrator keypoints` locally raises NotImplementedError: it needs a backend injected
# via `--calibrator-backend` (ADR-0006). Verified end-to-end on the target clip 2026-08-02.
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
PYTHONPATH=src python scripts/bench_camera_model_gap.py             # what one focal per clip costs, in metres
python scripts/motion_stats.py --scene <export/scene.json>          # #207: speed/accel plausibility
.venv/bin/python scripts/check_layout_preview.py                    # #127: does the browser's drag preview match the commit? (needs the server + node)
.venv/bin/python scripts/track_quality.py --camera calib/Colombia-1-0-Congo-DR1080p.npz \
  --labels docs/findings/track-labels-2026-08-07.json               # #135: is each player correct? 19/20 vs the user's eye
```

**#135's criteria are the fastest way to judge a scene without rendering it.** `track_quality.py`
prints a provenance timeline per track; `imputed` means *frozen limbs, coasting root* — a
mannequin, and the thing the eye calls a phantom. Always pass `--camera`: the stored scenes carry
an invented 772 px fallback under which every subject is trivially inside the image, which turns
the in-frame/off-frame test (the user's own display rule) into a constant.

## Architecture

Hexagonal (ADR-0001): pure `core/` (numpy, CPU, unit-testable) behind ports; `adapters/` hold the
heavy ML, live behind extras and are lazy-imported by dotted path (ADR-0006). Corrections are the
sole edit path (ADR-0002). Human ≡ LLM as editors, over MCP (ADR-0008/0010).

Subsystem → files: `docs/code-map.md`.

**Known structural debt — do not add to it.** Re-measured 2026-08-02, and two of the four
entries that used to sit here were wrong. They had sent at least one agent (me) hunting
duplication that does not exist, so what follows is only what was verified in the code:

- **One orchestration path, and it is `controller.Application`.** `app/cli.py` and
  `adapters/mcp/server.py` both go through it. New pipeline work goes there too.
  `app/anim_export.py` and `scripts/blender_animate.py` are *not* rival pipelines — they consume
  an already-exported `scene.json` (JSON→mesh, then mesh→render) and never reconstruct, so
  "collapsing" them onto `Application` would be wiring for its own sake.
- **The gate chain is mirrored by hand in Studio.** `Application.run_reconstruction` runs 16
  physics gates inline; `poseannot/rerun.py` re-runs 12 of them against a loaded `scene.json` and
  declares the other 4 `available: false` (they need live-pipeline providers). That is a
  legitimate split, not a bug — but the mirror used to be maintained by eye.
  `tests/unit/test_gate_chain_parity.py` now parses the controller and fails if the two drift.
- **2 calibration routes reachable, more present.** `wiring.py` accepts `calibrator="fake"` or
  `"keypoints"` only. `CameraModuleFieldCalibrator`, `_PnLCalibBackend` and
  `scripts/{fit,apply,bench}_rigid_camera.py` exist alongside. Check `wiring.py` before assuming
  which one ran.
- **9 exported stubs raise `NotImplementedError`.** Of the 24 names in
  `adapters/models/__init__.py.__all__`: `ApiAvatarBuilder`, `SplatEnvReconstructor`, `GVHMRBackend`,
  `GVHMRPoseEstimator`, `LearnedMotionPrior`, `PitchKeypointBackend`, `TrackNetBackend`,
  `DiffusionVasOcclusionBackend`, `FeedForwardGaussianRefiner`. All import and construct fine and
  fail only when called. A surprise `NotImplementedError` is this, not a missing dependency.

**Config is not the free-for-all this file used to claim.** `core/config/physics.py`
`load_physics_config()` resolves in a documented order — shipped `config/physics.yaml` → named
profile → `PITCH3D_*` env vars → Python overrides — and records every scalar's provenance in
`PhysicsConfig.lineage`. The constants in `core/correction/kinematics.py` are dataclass defaults
mirrored by `config/physics.yaml`, not a second source of truth. `.env` is read by the pod shell
scripts; no Python in this repo loads it. The real gap is narrower: the `BOWL_*` stadium geometry
in `core/scene/cameras.py` has no override path at all.

## Testing

The suite is green and that does **not** mean the pipeline works. `tests/conftest.py` is
fakes-backed by design ("no GPU/Blender/models"), and ~6000 lines of the user-facing path
(`controller.py`, `cli.py`, `anim_export.py`, `poseannot/*`, `blender_animate.py`) have no
direct coverage.

So: a green run is a smoke signal, not evidence. Evidence is the rendered output plus the numbers
from the probe scripts above.

**The one exception is `tests/e2e/test_golden_real_camera.py`** — the only test backed by a real
measurement rather than a fake. It runs the camera solve over
`calib/Colombia-1-0-Congo-DR1080p.npz` (7 kB, committed, so it runs in CI too) and pins what the
code *derives*: focal 4169.32 px, one optical centre for all 60 frames, camera at
(−2.29, −70.13, 17.22) m, and the framing the operator actually shot. It is mutation-checked —
the table in its docstring lists which injected regressions it catches and the one it does not.

If these numbers fail, do not nudge them until they pass. Either the camera code regressed, or
someone refit the calibration and the whole file needs re-measuring against the new clip.

When adding tests, prefer one assertion against real captured data over ten against fakes. Do not
add tests that re-implement the function under test in the assertion, or that only check a call
does not raise.

## Lint, hooks and CI

The rule is **"a file may not gain violations"**, not "a file must be clean". A 152-error
backlog remains (87 E501, 45 E702, plus a tail) and is not being fixed in one go — so leave
unrelated violations alone and just don't add any. `scripts/lint_changed.py` diffs each changed
file against its previous version and fails only on an increase; `pre-commit` and
`.github/workflows/ci.yml` both call that same script, so they cannot disagree.

```bash
pre-commit install                       # once per clone, before your first commit
.venv/bin/python scripts/lint_changed.py # the exact verdict the hook and CI will give
```

ruff is pinned to **0.16.1** in FOUR places — `[dev]`, `.pre-commit-config.yaml`,
`.github/workflows/ci.yml` and `requirements-dev.txt`. Bump them together or the hook and CI
disagree. The fourth was missed once: `requirements-dev.txt` sat on 0.15.18 while the other three
said 0.16.1, so `.venv/bin/ruff` — installed from it — checked a different rule set than CI and
under-reported the backlog by one (UP034). Fixed 2026-08-07; the comment in
`.pre-commit-config.yaml:10` still says "three", treat this list as the count.

Two deliberate gaps, so you do not mistake them for working:

- **`ruff format` is not wired up.** It would rewrite 302 of 388 files (~13 600 lines) in one
  commit and destroy `git blame` on code whose history is the main record of why it looks the
  way it does.
- **`UP042` is switched off.** It wants `(str, Enum)` → `StrEnum` on 23 core domain types, which
  changes what `f"{role}"` renders (`Role.PLAYER` → `player`) — and those enums land in
  `scene.json` and in agent summaries. A real migration with tests, not a lint autofix.
- **`mypy` currently checks nothing.** Modern numpy stubs use 3.12 `type` statements, which mypy
  rejects under our `python_version = "3.11"` and then stops at the first error. It is in `[dev]`
  and out of CI. Treat a clean `mypy` run as meaningless until this is fixed.

## Where to run GPU work

**Reconstruction now runs locally — try `demorig-pc` before starting a pod.** An RTX 4080 (16 GB)
box reachable as `ssh demorig`, running the chain in Docker: all five real backends, 48 frames in
75 s, peak 24% of VRAM. Free, and the suite runs ~3× faster there. Full runbook,
including the WSL traps that will otherwise cost you an afternoon: **`docs/local-gpu-box.md`**.

It does **not** yet do the generative tail (Wan/SeedVR2 unstaged, 16 GB unproven for them) or
Blender rendering. Those are still the pod.

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
