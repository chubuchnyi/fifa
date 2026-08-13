# A first camera from named lines — the ceiling is measured, and it clears

**2026-08-13.** Steps 1–2 of `camlab/docs/findings/automating-the-anchor-2026-08-13.md`, which
proposes replacing the human who drags camlab's first camera into place with a labeller that names
its detected line segments.

Before asking *can a model name them*, this asks the cheaper question that cancels everything
downstream if the answer is no: **given perfect names, does the rest of the path produce a camera
camlab can use?**

It does, by a wide margin. The rest of this file is that measurement and the four defects found
getting to it — all four in the new AVATAR code, none in camlab, which was not modified.

---

## The result

`broadcast` frame 0, camlab's own 9 detected segments, labels taken from the answer key
(§"How the key is built"), ranked by camlab's own paint:

| | median | worst line | scored samples | focal | position |
|---|---|---|---|---|---|
| **anchor from labels, after refit** | **0.84 px** | 5.4 px | 268 | 4279 | (5.11, 71.96, 17.19) |
| its half-turn twin | 0.87 px | 4.0 px | 268 | 4478 | (−5.11, −71.96, 17.19) |
| the camera camlab believes (`camera_smooth` f0) | 0.94 px | 3.42 px | 270 | 4251 | (−2.36, −68.00, 16.62) |
| next-best wrong assignment | 18.12 px | 213.9 px | 237 | 3240 | (−14.27, −69.57, 20.83) |

**The automatic anchor is not worse than the camera camlab currently believes for that frame**, and
the separation from the first wrong answer is 20×. The whole pool is ten cameras, not twenty
thousand.

### All four clips where a key can be built

An answer key needs a camera camlab already believes, so only its solved clips are testable — and
those are also the easy ones. Every number is the median against camlab's paint, after its own
refit:

| clip, frame | anchor from labels | camera camlab believes | verdict |
|---|---|---|---|
| `broadcast` f0 — 1920×1080, professional | **0.84 px** / 268 | 0.94 px / 270 | better |
| `fan` f8 — 1080×608, phone in the stands, floodlit night | **0.72 px** / 299 | 1.13 px / 304 | better |
| `CRO_MOR_194948` f0 — the clip solved from an operator's own anchor | **1.30 px** / 317 | 1.33 px / 319 | equal |
| `NET_ARG_225042` f25 | 2.06 px / 622 | 1.34 px / 620 | worse, well inside the 20 px band |

On `fan`, **eleven of the twelve** shortlisted assignments land between 0.72 and 1.78 px at
essentially the same camera (position within ~2 m, focal 3093–3457), from raw aims of 1.5–8.7 px.
That is not degeneracy — it is the useful robustness result: **the labels do not have to be exactly
right**, because several different label-consistent assignments funnel into one basin under the
refit.

No half-turn twin appears in `fan`'s shortlist, unlike `broadcast`'s. Not investigated; do not read
it as the degeneracy being weaker on that clip.

### Two frames where it cannot work at all, and the rule that follows

- **`NET_ARG_225042` f0** — nine segments, six of them real markings, and **five in one family
  against one in the other**. A homography needs two correspondences per family, so no labelling
  however perfect can seed this frame. Not a defect: the frame simply shows one line along the
  pitch's long axis.
- **`g15449383` f0** — five segments, **one** of which is a marking. camlab's own STATUS already
  refuses to call this clip solved (2 markings, 21 % of frames with no paint across).

**So scan frames; do not fix on frame 0.** The anchor is needed on *one* frame of a clip, and on
`NET_ARG` the family balance goes 1+5 at f0, **2+3 at f10, 3+5 at f25**, 1+3 at f40. Frame 0 is a
convention, not a requirement, and choosing it cost this clip its anchor until the scan was run.

Reproduce::

    cd ~/camlab && .venv/bin/python -m uvicorn camlab.server.app:app --port 8899 &
    .venv/bin/python scripts/bench_line_labeller.py prepare --clip broadcast --frame 0
    .venv/bin/python scripts/write_camlab_anchor.py --clip broadcast --frame 0 \
        --labels out/labeller/broadcast_f0/labels_oracle.json --dry-run

## Why this is the right thing to have measured first

camlab's bootstrap fails on five of six anchors by **abstaining** — *"no plausible camera at all"* —
and its own `bootstrap-progress.md` locates the failure precisely: the generator is right (4680
physically plausible cameras in 12 s) and the chooser lands 113 m out. Names delete the chooser
rather than improve it: `_homography_from_lines` needs four correspondences, two per parallel
family, and a label says which world line each detected segment may be.

So the value of a labeller is bounded above by whether four correct names are enough. They are.

## How the key is built, and why it needs no hand labelling

camlab already has clips whose camera it believes. Project the pitch model through that camera and
every detected segment gets its true class for free — `is-a-model-worth-training.md`'s
self-labelling idea run backwards: instead of making training data, it makes an **answer key**.

`scripts/bench_line_labeller.py prepare` fetches camlab's own segments and its own projected model
from `GET /api/run/{clip}/lines/{n}`, assigns each segment its nearest projected marking within
`EXPLAIN_PX = 12.0`, and writes the key beside a numbered image the labeller sees instead.

Two checks that it is a key and not an artefact of our own assignment rule:

- on `broadcast` f0 the six explained segments sit **0.09–1.14 px** from their markings — there is
  no ambiguity to resolve;
- on `fan` f8 it independently reproduces camlab's own count: **7 of 9 explained, 2 not**, against
  `11-is-blocked-by-14-2026-08-12.md`'s *"two of nine detected segments are 55–60 px from any
  marking, and both lie along the join between the grass and the advertising hoarding."*

## The four defects, in the order they bit

Each one presented as "no camera exists", which is also how camlab's own bootstrap fails — worth
remembering before concluding that a search is hopeless.

**1. A copied routine left its helper behind.** `homography_from_lines` was copied from camlab
verbatim; `_line`, which normalises a homogeneous line to `|n| = 1`, was not. Image lines are built
from coordinates of order 10³ and world lines from order 10¹, so the two sides entered the DLT six
orders of magnitude apart and the SVD solved whichever rows were loudest. **All 384**
label-consistent hypotheses came back 771–3221 px from reproducing their own homography. Nothing
raised.

**2. A reflected homography is a valid-looking camera.** A line correspondence is sign-free — `l`
and `−l` are the same line — so a line-based DLT admits reflections, and `_decompose` orthogonalises
`[r1, r2, r1×r2]` through an SVD, which turns a reflection into a *proper* rotation without
complaint. `det(H)` cannot be the test, because `H → λH` scales the determinant by `λ³` and the sign
is not scale-invariant. Reprojection can: a reflected solve does not reproduce its own homography.

**3. …but measured over the whole model, that test rejects good cameras.** Near the horizon a
tiny difference in H becomes hundreds of pixels, so the max over all 1446 model points read 771 px
for a camera whose probe points land within 2–5 px in frame. Restrict it to points inside the
frame, exactly as `camera_from_calibration` does with `_probe_points`.

**4. Ranking raw aims by paint asks the wrong question.** A four-line fit through *detected*
segments is an aim, not a solve. camlab documents the human's second click — *"aim it roughly, the
solver finishes it. A rough aim at 445 px comes back at 4.7 px on `broadcast` frame 0"* — and the
same is true here: **327.59 px worst / 6.07 median raw → 4.04 / 0.84 after `POST /refine/{n}`.**
`refit._accept` takes the fit only if the worst offset fell and no correspondence was lost, so the
refit can refuse but not damage.

Two further traps inside the ranking itself, both worth stating because both looked like results:

- `worst_line_px` is a max over markings, and on this frame it was set by an **arc scored on 14
  samples** while the same camera's median was 6.07 px. Rank on the median, read the max.
- "require support ≥ half the best candidate's" inverted the test: two candidates with the focal
  pinned at the **300 px search floor** framed the whole pitch, scored **1087 samples** — four
  times the true camera's support — and set a bar that excluded the right answer. The reference has
  to be the seed camera's own support on that frame, and a focal sitting on a bound is rejected
  outright. *A bound that is being hit is a finding, not a setting* (camlab, `inherited-claims.md`).

## What camlab supplies, and what was changed in it

**Nothing was changed in camlab.** The whole loop is its existing HTTP surface:

| step | endpoint |
|---|---|
| the frame the labeller sees | `GET /api/run/{clip}/frame/{n}` |
| **its own** detected segments + the projected model | `GET /api/run/{clip}/lines/{n}` |
| the world markings, in camlab's frame | `GET /api/pitch` |
| the principal point the camera was solved under | `GET /api/run/{clip}/camera` |
| finish the aim | `POST /api/run/{clip}/refine/{n}` |
| judge it | `GET /api/run/{clip}/residual/{n}` |
| the anchor itself | written to `runs/<clip>/camera_manual.json`, the store `solve/hand.py` reads |

Because `hand_candidates` *"refuses to rank by source"*, an anchor written by a script competes with
the seed on the paint like any other, and `solve_carry.py` already prints when none of them won.
Nothing had to be added for that to be safe.

The one thing copied is the line-to-line DLT and its `_line` helper, with an origin header
(ADR-0013 §5). The pitch model is **not** copied — it comes over the API, so no convention can
drift between the repos. The principal point likewise: `fan`'s optical axis is at `cy = −334`, not
304, and `plane_camera._measure_focal` and `_k_inv` already take `cx, cy` — only
`camera_from_calibration` hardcodes the image centre, and this path does not go through it.

## What is not claimed

- **That a vision model can produce these labels.** Not measured here. This is the ceiling, and the
  ceiling clearing is what makes the question worth asking.
- **That the half-turn is solved.** It is not, and the two best cameras above are exactly each
  other's negation, scoring 0.84 and 0.87 px. No paint metric will ever separate them; the script
  detects the twin pair and says so instead of letting sort order look like a decision. It needs
  the labeller's left/right call or camlab's `flip 180°`.
- **That two frames of two clips generalise.** Both are clips camlab has already solved, which is
  what makes an answer key possible and also what makes them the easy cases. The seven clips camlab
  cannot start have not been touched, and by construction no key can be built for them.
- **That the labels used here are realistic.** They are perfect by construction.
