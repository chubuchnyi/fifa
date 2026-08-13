# What a vision model can and cannot say about a detected pitch line

**2026-08-13.** The second half of the §7 measurement in
`camlab/docs/findings/automating-the-anchor-2026-08-13.md`. The ceiling — *given perfect names,
does the anchor path work* — is measured separately in
[`anchor-from-labels-2026-08-13.md`](anchor-from-labels-2026-08-13.md) and it clears on four
clips. This file is the other half: **can a model produce those names?**

Answer: **not the names. But it produces the one thing camlab needs more, and it produces it
cleanly.**

The labeller was a fresh agent with no access to this repo's context, given one image and the
vocabulary. The answer key was built by projecting the pitch model through a camera camlab already
believes, and the labeller never saw it.

---

## The result, split by what was asked

`broadcast` f0, 1920×1080, nine detected segments, one full-frame image:

| what was asked | score |
|---|---|
| **is this a marking at all** (`not_a_marking`) | **3 / 3** |
| type, for lines along the pitch's long axis | 3 / 3 |
| type, for lines across the pitch | **0 / 2** |
| exact type overall | 6 / 9 |
| **correct family overall** | **3 / 9** — family 0: 3, family 1: **0** |

**Verdict: FAIL** on the stated bar (two correct-family correspondences per family), and the
failure is entirely on one axis: every across-pitch line was called a long-axis type.

`fan` f8, 1080×608, phone from the stands, same prompt: **1 / 9 exact, 1 / 9 family**. Worse in the
same direction — it inverted the meaning of the two families wholesale.

## The two halves are worth separating, because one of them is the valuable one

**It is right about paint.** On `broadcast` it called segments 3, 7 and 9 `not_a_marking` — and its
own note says why: *"Segments 3, 7 and 9 lie on that penalty ARC, not on any straight marking…
The arc is painted but has no category in the list, so I returned not_a_marking to keep them out of
a line fit."* That is the correct call for the right reason; the key excludes arcs because
`straight_markings()` does. It also refused the classic trap deliberately: *"Segment 6 hugs the
grass/hoarding join, which is the classic false positive, but it is the touchline: the ball and at
least five players sit below it."*

**This matters more than the naming**, because it is `#14`'s open precision half, and camlab has
already measured what it is worth: feeding the generator the seven real segments instead of nine
moves the best hypothesis in the pool from **11.9 m to 3.7 m** and the focal from **28 % wrong to
2.1 %**. And `11-is-blocked-by-14-2026-08-12.md` says the dependency runs exactly that way.

**It is wrong about orientation.** Which image direction runs along the pitch is a question it
answers by inference rather than by sight, and it gets it wrong. `fan`'s labeller said so plainly:
*"The labelling rests on a nesting argument rather than on directly readable paint."*

## So does geometry supply the family instead? Measured: no, not reliably

camlab splits families by vanishing-point consensus (`solve/bootstrap.split_families`), which is
exactly the question the model fails. Copied here and run against the key:

| | with all segments | with the true non-markings removed first |
|---|---|---|
| `fan` f8 | A = 3×family 1 + both hoarding segments, B = 4×family 0 | **pure** (4 + 3) |
| `broadcast` f0 | mixed both ways | **still mixed** — [0,1,0] against [0,1,0] |

So on `broadcast` neither the model nor the geometric split gives the family. Which is fine,
because **nothing has to**: camlab's `hypotheses()` enumerates the family swap and the
order-preserving assignments itself. The family is a thing to enumerate, not to ask for.

## What this says the design should be

Stop asking the model to name lines. Ask it the one question it answers well and no classical
feature here has managed — **paint or not paint** — and let camlab do the rest:

1. **model**: is there white paint under this segment, or is it the hoarding join / a mow stripe /
   an arc? (measured 3/3 on `broadcast`)
2. **camlab**: family split and order-preserving enumeration over the survivors — its
   `bootstrap_clip.py`, which already returns 1.0 px on `fan` f0 and abstains on five of six anchors
   for exactly the reason step 1 removes;
3. **camlab**: refit, then its paint decides.

That is a much smaller ask than an eight-class vocabulary, and it maps one-to-one onto this repo's
own stated dependency: `#11` is blocked by `#14`.

**And it is where the one camlab change becomes necessary.** `bootstrap_clip.py` takes no segment
filter and is not on the HTTP surface. The change is small — accept a list of segment indices to
keep — and its test is already stated: *does removing the false segments let the bootstrap solve
the five anchors it currently abstains on?* Until that is run, nothing should be added to camlab.

## Two defects in the instrument, both fixed

- **A label box covered another and a segment vanished from the run.** On `broadcast` boxes 4 and 5
  landed 23 px apart at the left edge; the labeller reported *"There is no number 4 anywhere in the
  frame"* and returned eight labels for nine segments. `draw()` now keeps boxes clear of each other
  and inside the image. The scorer counted the missing one as wrong, so the 6/9 above is not
  inflated by it — but a silently dropped segment is the kind of thing that would have been read as
  a model failure.
- **The full frame is not enough of an instrument.** Two earlier runs stalled trying to crop the
  image themselves, and the one that finished said it could not read the paint. `prepare` now also
  writes `crops.png`, a contact sheet of one upscaled tile per segment with the segment drawn thin
  and **dashed** so the pixels underneath stay visible. Supplementary to the overview, never a
  replacement: a crop cannot answer "which marking is this".

## What is not claimed

- **That crops fix it.** The re-run with tiles was in flight when this was written; the paint/not-
  paint score above is from the full frame alone.
- **That one labeller is the labeller.** One agent, one prompt, two clips. A different prompt or a
  model given the tiles may do better, and the `not_a_marking` result is 3 of 3 on one frame —
  a promising number, not an established one.
- **That the vocabulary is right.** It has no value for arcs, and the labeller had to work around
  that. `straight_markings()` excludes arcs too, so the pipeline is consistent, but
  `11-is-blocked-by-14` already notes that arcs are modelled for scoring and never fed to the
  bootstrap.
- **Anything about `which_end`.** Both labellers said the half-turn is **not** determinable from
  their frame, and gave good reasons — sponsor boards and second-tier signage repeat all the way
  round the bowl, and neither frame contains a scoreboard, dugout, tunnel or roof line. That is a
  useful negative: the half-turn will not be solved by asking about one arbitrary frame.
