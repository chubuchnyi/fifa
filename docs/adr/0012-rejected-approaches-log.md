# ADR-0012 — Rejected approaches log: what we tried, what we declined, and what would re-open it

- **Status:** Accepted
- **Date:** 2026-07-29
- **Deciders:** architecture
- **Related:** R-6 (mark, don't erase); ADR-0002 (correction stack); `docs/roadmap.md` "Research intake"; `docs/research/football-3d-pipeline-v2.md` §9; roadmap R8 (#100)

## Context

Two long research briefs arrived (`docs/research/football-3d-*.md`) carrying a large table of ideas
already considered and rejected, and we have separately accumulated our own rejections — some of them
*measured*, at real cost. Without one place to put these, the same questions get re-opened every few
sessions by whoever reads the briefs next, and the expensive part (the measurement) gets repeated or,
worse, skipped in favour of the plausible-sounding recommendation.

A rejection is only durable if it records **what would reverse it**. A bare "we rejected X" invites
re-litigation; "we rejected X, here is the number, here is the condition under which the number would
change" does not. This ADR is therefore organised by *strength of evidence*, and every entry is tagged
with what it is conditional on — because many of these rejections are conditional on **our** goal (one
broadcast clip → a novel-view video judged by eye) and would flip for an analytics product.

## Decision

Three tiers, strongest evidence first. **Do not re-open a Tier 1 entry without new measurement.**

### Tier 1 — Rejected on our own measurement

The evidence is in this repo and re-runnable. These are the most expensive to re-derive.

| Idea | Verdict | Measured reason | What would re-open it |
|---|---|---|---|
| **USAC / MAGSAC++ replacing our RANSAC** for image→world calibration (v2 §3.3, its headline calibration item) | **Rejected** | Ours **0.07–0.12 m** world RMS; USAC **28–180 m**. USAC keeps a degenerate 4-point consensus (inlier recall 0.05–0.42) and with *zero* outliers fails outright **0/20**. Threshold-independent (1→50 px), both directions, and Hartley pre-conditioning makes it worse. Root cause: MAGSAC++ marginalises over **one global** noise scale, but a pitch homography is heteroscedastic — uniform 2 px image error spans 0.027 m near to 0.227 m far, **8.4× p95/p5**. Repro: `scripts/bench_ransac_usac.py` | A robust estimator that accepts a **per-correspondence** noise scale, or one that ingests per-landmark confidence. Both properties of `solve_homography_ransac` (world-metre threshold, confidence-weighted DLT refit) are load-bearing |
| **"Unify the four SMPL-X→world constants"** (brief §2.1) | **Rejected — premise false** | The matrices are **both correct**: they map two *different* source frames (canonical `+y` up vs SMPLest-X camera `+y` down). Unifying them inverts half the pipeline. Encoded as a failing test: `tests/unit/test_frames.py::test_the_two_remaps_are_not_interchangeable` | Nothing — this is a fact about SMPL-X output, not a preference. (The brief still earned its keep: chasing it found a real 206 mm → 18 mm ground-plane bug next door in `smplx_foot_pos.py`) |
| **Fitting the two *edges* of a painted pitch line** (v2 §9's own reformulation of the line-thickness idea, adopted as R3) | **Rejected for this footage** | The edges are not resolvable. On the target clip the paint is **2.00 px** median thick (distance transform, p90 2.80; 4.0 px in the nearest band) and **1–2 px** by raw intensity FWHM with no mask involved — profile `114 118 152 196 145 116`, a single-lobe **PSF, not a band**. Two independent edge positions cannot come out of that. The thickness-as-range-cue variant dies too: 2.0 → 2.8 px across the *whole* visible pitch is ~1 px of dynamic range. Repro: `scripts/measure_pitch_line_width.py` | Materially higher effective resolution on the pitch — a true close-up, a 4K source, or a broadcast with less compression. This rejection is **conditional on the target clip**, not on the idea. **The salvage shipped 2026-07-29** — PnLCalib's discarded line head (`pnlcalib_backend.py:115`) now enters the DLT as one point-on-line row each. It is the **first brief item to survive measurement**, and it buys *robustness*, not precision: −9 % median at the clip's 10 kp/frame, but p95 **95.8 → 1.26 m** at 4 kp, and 3-kp frames become solvable at all. Repro: `scripts/bench_line_constraints.py`. Caveat worth keeping: the first version of that benchmark said **−43 %**, because it made all 17 lines visible and modelled line error as iid per point. A stripe detector mislocalises the *whole* line — a bias no amount of sampling along it averages away. Fixing both assumptions took it to −9 %; the script now prints both tables so the gap stays visible |
| **Iterative moving-average low-pass on HMR yaw** | **Rejected** | Removes 90 % of angular acceleration but flattens **100°+ real turns** — it attenuates exactly the signal we measure. Replaced by `facing_align` (structural yaw discipline from velocity direction, not statistical smoothing) | A filter whose stopband is defined by *implausibility* rather than frequency. Same objection as v2 §9's own "MPC/iLQR for smoothing" entry — we arrived at it independently |
| **Sparse SMPL-X FK sampling for foot position** (30-frame cap) | **Rejected** | Held-between-samples interpolation *synthesised fake stances* — `contact_probe` read constant XY across held rows as a planted foot. Raising the cap to 240 cut aggregate foot slide **15.4 m → 0.3 m (98 %)** | Nothing at current clip lengths; revisit only if FK cost dominates on full-match ingest |
| **GREEN source-kit chroma keying** for grass/player separation | **Rejected** | Grass collision — the pitch *is* green. Built the AOV-mask design instead | Nothing |

### Tier 2 — v2 §9's own rejections, ported with our verdict

We agree with these, but several are agreed for **different reasons** than the brief gives, and the
conditionality matters. Where our verdict differs from theirs, that is called out.

| Idea | v2 §9 verdict | Ours | Note |
|---|---|---|---|
| **NeRF / PixelNeRF for calibration** | Reject | **Agree** | We have a metric CAD model of the scene (105×68 m, regulated markings). NeRF supplies appearance, not the missing geometry. *Unconditional for calibration* — but note NeRF-class methods are **not** rejected for the v2 appearance tier, which is a different question |
| **ALIKED / LightGlue as the primary pitch matcher** | Reject | **Agree, conditionally** | Uniform periodic grass, near-mirror-symmetric pitch (half-aliasing), a third of the frame moving. Semantic line detection against a known model wins. Ships in OpenCV 5's `Features` module, so it is cheap to try — **they remain fair game as a *secondary* cue**, and as a matcher for frame-to-frame propagation (R2), which is a px→px problem without the pitch's symmetry |
| **Ball shadow for height** | Reject | **Agree, and more strongly for us** | Their reason is detection (3–4 px shadow ellipse). Ours is stronger and measured: our target clip is a **floodlit night match** with multiple soft shadows and no sun — the geometry they call "free from geolocation + timestamp" does not exist here at all |
| **Line thickness as an independent constraint** | Reject, but reformulate | **Agree with the rejection; their reformulation also fails here** | Perspective changes thickness without zoom (chicken-and-egg with range) — agreed. But their salvage, two line **edges** as parallels at a regulated separation, needs the edges to be resolvable, and on our clip the paint is 2 px wide. **See Tier 1** — measured and rejected, with the surviving version (use PnLCalib's discarded line detections as point-on-line residuals) adopted as the re-scoped R3 |
| **Audio TDOA on broadcast** | Reject | **Agree** | Mixed stereo program audio, not a multichannel field feed, so multilateration is impossible. *Conditional on the broadcast source*: inverts for own-camera training capture (their §5.1) |
| **Game engine as a post-filter** | Reject | **Agree** | It cannot see pixels: it resolves collisions by shifting people arbitrarily, destroying the observations. Constraints belong in the same objective (ADR-0002's correction stack is our version of this) |
| **Ragdoll for SMPL** | Reject | **Agree** | A passive chain does not run, jump, or strike. Needs an actuated controller |
| **MPC / iLQR for smoothing** | Reject | **Agree** | Receding-horizon *control*, not *estimation*; offline we want batch MAP. "Minimise energy" biases exactly the quantity being measured — the same trap as our own Tier 1 yaw low-pass |
| **Elastic ball bounce** | Reject | **Agree** | Turf restitution 0.6–0.8 and humidity-dependent; the error moves switching times, which is what the ballistic segmentation keys on |
| **YOLO-World / CLIP for kit changes** | Reject | **Agree** | Solves a problem we do not have. Bibs are a **team-assignment** problem, solved by within-match torso-colour clustering |
| **G-API (OpenCV 5)** | Not applicable | **Agree, verified** | Moved to `opencv_contrib` — and we confirmed by inventory that **G-API is used nowhere** in our 23 cv2-importing files |
| **Isaac Gym** | Not applicable | **Agree** | Deprecated; replacement is Isaac Lab |
| **UE5 synthetic data for *appearance*** | Defer (P3) | **Agree, defer** | Domain gap on spectral artefacts, motion blur, H.264 — all real for ReID and OCR |
| **UE5 synthetic data for *geometry*** | **Accept (P1)** | **Defer for now** | Their evidence is good (SynLoc, SoccerNet MDE 2025, NVS-2026). But our binding constraint is appearance fidelity on **one** clip, not a trained estimator's accuracy. Re-open when we need a *learned* component to generalise across matches |

### Tier 3 — The briefs' own prescriptions we declined

Distinct from Tier 2: these are things the briefs *recommend*, not things they reject.

| Item | Verdict | Reason | What would re-open it |
|---|---|---|---|
| **Factor graph** (§4 / §7 — 136k variables, Theseus, staged A/B/C) | **Defer, not adopt** | Entirely `[est.]` by the doc's own admission, and the largest, most expensive section. It buys *accuracy*; our binding constraint is appearance fidelity, not the last 10 cm. ADR-0002's correction stack + the human/LLM edit loop reach the same visible result more cheaply | R7's metric (#99) showing that **inter-player residual** is what breaks the render. That is the specific number that would justify it |
| **Their accuracy envelope** (Global 0.35–0.45 m; "do not spec below") | **Reject as our bar** | Correct for a pitch-control analytics product, wrong for a video judged by eye. A large *common-mode* camera error is nearly free at a novel viewpoint — it is a rigid re-render. Superseded by R7 | Pivoting the product toward analytics |
| **Shot segmentation** (§3.1) | **Defer** | We reconstruct one continuous clip | Becomes P0 the moment we ingest a full match |
| **Off-screen imputation B2/B4** (§3.6) | **Defer, and note the conflict** | For analytics an imputed ghost is a useful estimate. In a photoreal video, rendering an imputed player is **fabrication**, which R-6 forbids | Only behind R4's `Provenance` type (#96), gated **at the renderer** — not at the analytics boundary. Note this sits in deliberate tension with our own rule that near-certain tracker-lost subjects must be interpolated rather than blinked out; the dividing line is confidence, and `Provenance` is what makes it checkable |
| **Frozen per-frame `PlayerState` / `BallState` dataclasses** (brief §3) | **Adopt the *types*, decline the *shape*** | R4 (#96) shipped `Provenance` and `BallMode` — that part was right, and better-motivated than the brief argued (see below). But the brief hangs them off a per-frame state object, and our motion is already `(T, …)` arrays in `PoseSequence`/`BallTrack`. A parallel per-frame representation would have to be kept in sync with the arrays that every gate actually writes, so the labels ride the arrays instead | A move to a genuinely per-frame scene graph (e.g. USD time-samples as the in-memory model, not just the export target) |
| **Brief §0 / §10 repo layout** (split into `CLAUDE.md` + `docs/spec/`; `src/core/`, `src/adapters/`) | **Ignore** | Written greenfield. We already have `src/pitch3d/{core,adapters}`, 12 ADRs and `STATUS.md` as SSOT; following it literally forks both the tree and the tracking | Nothing |

### One inherited caveat, already ours

v2 §12.6 warns that TrackNet/TOTNet-class ball trackers are validated on racket sports and table
tennis, and that football (deforming ball, ~20× scene scale, occlusion by 22 legs) is unproven
transfer. Our `adapters/models/wasb_backend.py` is from that lineage — so this warning applies to a
choice **we have already made**, not to a recommendation we are weighing. Recorded here so it is not
mistaken for a rejection.

## Consequences

**Positive**
- Settled questions stay settled, and the cost of re-opening one is explicit: produce the number.
- The conditionality tags mean a future pivot (analytics, full-match ingest, multi-clip generalisation)
  can find exactly which rejections it invalidates instead of re-arguing all of them.
- Tier 1 entries are backed by runnable artifacts (`scripts/bench_ransac_usac.py`,
  `tests/unit/test_frames.py`), so the evidence does not decay into folklore.

**Negative / costs**
- A rejection log ages. Tier 2 entries inherited from the briefs are *argued*, not measured by us —
  they carry the briefs' evidence, not ours, and should be treated as weaker.
- There is a real risk of over-trusting this file. The briefs' **diagnoses** have been far better than
  their **prescriptions**: **five** of their items have now been measured (R1, R10, R3-edges,
  R3-salvage, R4) and **three premises were false** — while every one of those investigations turned
  up something real next door (a 206 → 18 mm foot-placement bug; why our world-metre RANSAC threshold
  is load-bearing; an entire discarded line-detection head, which then shipped; a ball `on_ground`
  bool that was lossy for 46 of 48 frames). Read them, measure them, do not implement them. R5 and R7
  remain unmeasured hypotheses, not work orders.
- R4 sharpens that pattern rather than breaking it. It is the first item adopted roughly as written —
  but the brief's *stated* motivation (typed state is good hygiene) is not why it was worth doing.
  The reason is that we were already encoding provenance, badly, in sentinel confidence values, and
  the ball's bool was silently conflating "fitted arc" with "no idea". The brief pointed at the right
  place for the wrong reason; the value came from measuring what was actually there.
- The one that survived is also a warning about *our own* measurement. R3's salvage first benchmarked
  at −43 % and landed at −9 % once the noise model stopped flattering it. A benchmark we wrote to
  justify a change will find what we ask it to find unless we attack its assumptions first.

## Alternatives considered

- **Keep the rejections inline in `roadmap.md`** — rejected: the roadmap is a forward-looking plan and
  gets rewritten each iteration; rejections need a stable address to be cited from.
- **One ADR per rejection** — rejected: most entries are a paragraph, and the value is in seeing them
  in one table with a common conditionality column.
- **Port v2 §9 verbatim** — rejected: it would import their goal (analytics accuracy) along with their
  verdicts. Several entries are right for us for *different* reasons, and one (UE5 for geometry) they
  accept and we defer. The per-entry "ours" column is the point.
