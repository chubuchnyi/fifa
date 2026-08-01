# pitch3d — STATUS (current state only)

<!--
  This file is VOLATILE STATE: what is true right now and what to do next.
  Keep it under ~150 lines. It is read in full at the start of every session.

  It is NOT the place for evidence, measurements or reasoning:
    - how to work here, commands, rules  -> CLAUDE.md
    - per-item detail / root causes      -> docs/findings/
    - what happened and when             -> docs/archive/status-log-2026-07.md
    - where code lives                   -> docs/code-map.md

  Update + commit at every meaningful step. Chat history and the CC task list do
  not survive a session; this file does.
-->

**Last updated:** 2026-08-01 · **Repo:** /home/chubuchnyi/AVATAR · **Target clip:**
`samples/video/Colombia-1-0-Congo-DR1080p.mp4`

---

## 1. Goal

From one source broadcast clip → a **realistic novel-view video of the SAME episode** (different
camera angle), as faithful as possible. Players look like the originals (same kit + shirt numbers);
the stadium is realistic and the same as the source. **Judged by eye.**

Approximations are OK where one clip cannot recover the truth — backstopped by manual Blender
editing and generative prompt-editing (ADR-0008, LLM-over-MCP).

## 2. Staged bar (in order; each gated on eye-judgement)

- [x] **v0 — correct geometry** (2026-06-27). 20 players, root spread 34×40 m, pitch lines + goals,
  cameras that frame the action. Validated end-to-end on the pod.
- [x] **v1 — recognizability** (2026-06-28). Kit colours (10/10 split), shirt numbers (honest blanks
  where illegible), hybrid stadium backdrop.
- [ ] **v2 — photoreal** (started 2026-06-28). The agreed «A через B, 1→2→3, свет из клипа» plan is
  **complete**: lever 1 measured per-vertex body texture, lever 2 grass PBR via shared
  `scene_builders.py`, lever 3 light-from-clip (floodlit night, auto + manual override).
  Work since then is the generative finishing tail — see §3.

## 3. Current focus

**v2 finishing tail**, iterated on the pod as numbered batches (t1…t23). The chain is one command:
`bash scripts/pod_finish_batch.sh` — recon → quilt export → render → night-grade → Wan-VACE →
SeedVR2 → mask pass → screen-space pins (stages 9–14).

Best finals: `out/kitzones_pod/sideline_t21_pinned8.mp4` (sideline) ·
`goal3_pinned4_xflat.mp4` (goal).

**Next candidate (t24):** panel-row lime-dash periodicity (ours reads as repeating bright green LED
segments, the clip is calm dark-green panels with gold text), and player-silhouette recovery — the
root cause that the stage-14 shadow pin can only mitigate.

Lever-by-lever history: [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md).

## 4. Open board

One line per item. Reasoning, measurements and root causes live in
[`findings/open-items-2026-08-01.md`](findings/open-items-2026-08-01.md).

| ID | Item | Status |
|----|------|--------|
| #61 | Camera-calibration accuracy (offset + ~3× scale) | root cause exact + fixed for every clip 2026-08-01 — **awaiting eye** |
| #119 | Re-solve the calibration as ONE camera, not 60 free homographies | done 2026-07-31 — **awaiting eye** |
| #112 | Drag the pitch layout to correct the homography | done 2026-07-31 — **awaiting eye** |
| #60 | Re-run overlays + verify acceptable alignment | pending — the eye-check that closes the calibration thread |
| #125 | A run that solved no calibration frame still exported a finished scene | gate fixed 2026-08-01; **why PnLCalib solved nothing on that pod is still open** (needs the pod) |
| #120 | Stored scenes declare a world frame they are not in | body mirror 2026-07-31; **corrections mirror 2026-08-01** (user saw it, measured 0.114→0.323); stale `handedness` labels remain |
| #109 | `jersey_numbers.py` must crop from the real camera at native resolution | pending, unblocked by #107 |
| #108 | R3's line-constraint path is a no-op on this clip | pending — needs a log line at the decision first, then a run |
| #45 | F2: raw video → frame range → auto `scene.json` behind the GUI | **BLOCKED on a user decision** (where it runs). Do NOT stub a fake generate button |
| #107 | Render the measured camera, not the synthetic one | done 2026-07-31 |
| #117 | Frame preprocessing to feed auto-calibration | research done 2026-07-31; its focal reading superseded by #119 |
| #122 | An expired session silently degraded the UI instead of asking for a re-login | done 2026-07-31 |
| #124 | The pitch-layout drag was unverifiable | done 2026-08-01 |

**Bottleneck:** four items (#61, #119, #112, #60) are built and waiting on a visual verdict. Nothing
in the calibration thread closes until they are judged — worth batching into one A/B pass.

## 5. Health (measured 2026-08-01)

Honest baseline, so the next session does not mistake green for safe.

| Signal | Measured | Note |
|--------|----------|------|
| Test suite | **1114 passed / 14 skipped / 0 failed**, >5 min | fakes-backed (`conftest.py` says so) |
| Real-measurement coverage | **1 file, 8 assertions** | `tests/e2e/test_golden_real_camera.py` over the committed 7 kB camera fit — the only non-fake evidence in the suite, and mutation-checked. Everything downstream of the camera (detection, pose, export) is still fakes-only |
| Untested user-facing paths | ~6000 lines | `app/controller.py`, `app/cli.py`, `app/anim_export.py`, `poseannot/app.py`, `poseannot/camera.py`, `scripts/blender_animate.py` |
| Lint | **152 ruff errors** (was 311) | 87 E501 · 45 E702 · tail. 148 auto-fixed 2026-08-01; UP042 switched off (its fix changes enum serialisation) |
| CI | **pre-commit + GH Actions** | gate = `scripts/lint_changed.py`: a changed file may not *gain* violations. The 152 are reported, not gated — the backlog can shrink, not grow |
| Type checking | **mypy checks nothing** | numpy's stubs use 3.12 `type` syntax, rejected under `python_version = "3.11"`; mypy stops at error 1. Declared but dead |
| Declared deps | **fixed 2026-08-01** | `pyyaml`/`scipy` were imported by `core/` but undeclared — a clean `pip install -e ".[dev]"` could not collect a single test. CI now guards this |
| Pipeline entry points | **6** | cli · controller · anim_export · poseannot · scripts/* · pod shell — orchestration re-implemented in each |
| Calibration backends | **4 paths, 1 wired** | only `KeypointFieldCalibrator` is in `wiring.py`; `CameraModuleFieldCalibrator`, `PnLCalibBackend` and the `*_rigid_camera.py` scripts are parallel routes |

Remediation plan agreed 2026-08-01: (1) agent entry point — this split; (2) pre-commit + CI;
(3) one golden test on real measured data; (4) collapse the 6 entry points onto
`controller.Application`. **Steps 1–3 done; 4 not started.**

Step 3 landed against the camera solve, not the 30-frame clip originally sketched: the clip is
not committed (too large) so a test over it cannot run in CI, whereas the 7 kB fit derived from
it can. The real-video path was measured — 58.63 s for 30 frames, 19 subjects, CPU — and is a
viable opt-in local test, but it is not written. That is the honest gap in step 3: the golden
test proves the camera is real, not that the export downstream of it is.

Step 2 is a *mechanical* fence only. It stops new lint debt and proves the package installs and
imports from a clean checkout. It does **not** make the suite meaningful — that is step 3, and
until then a green CI still means "the fakes agree with each other".

## 6. Key references

- **How to work here (commands, rules, architecture):** [`../CLAUDE.md`](../CLAUDE.md)
- **Where code lives:** [`code-map.md`](code-map.md)
- **Open-item detail:** [`findings/open-items-2026-08-01.md`](findings/open-items-2026-08-01.md)
- **History (verbatim log, … 2026-08-01):** [`archive/status-log-2026-07.md`](archive/status-log-2026-07.md)
- **v0 defects + code root-causes:** [`v0-geometry-defects.md`](v0-geometry-defects.md)
- **Pipeline overview:** [`pipeline.md`](pipeline.md) · **Rejected approaches:** [`adr/0012-rejected-approaches-log.md`](adr/0012-rejected-approaches-log.md)
- **Historical build log (M0–M4 = plumbing, not result quality):** [`roadmap.md`](roadmap.md) ·
  **M1 live state:** [`m1-status-and-plan.md`](m1-status-and-plan.md)
