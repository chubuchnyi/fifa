# What a vision model can and cannot say about a detected pitch line

**2026-08-13.** The second half of the §7 measurement in
`camlab/docs/findings/automating-the-anchor-2026-08-13.md`. The ceiling — *given perfect names,
does the anchor path work* — is measured separately in
[`anchor-from-labels-2026-08-13.md`](anchor-from-labels-2026-08-13.md) and it clears on four
clips. This file is the other half: **can a model produce those names?**

Answer: **yes — but only once it is given close-ups. The instrument was the binding constraint, not
the model.** Same clip, same frame, same question, same vocabulary:

| | exact type | correct family | bar | end to end |
|---|---|---|---|---|
| `fan` f8, **full frame only** | 1 / 9 | 1 / 9 | FAIL | no camera found at all |
| `fan` f8, **plus a contact sheet of close-ups** | **7 / 9** | **6 / 9** (3 + 3) | **PASS** | **0.88 px**, against 1.13 for camlab's own |

The labeller was a fresh agent with no access to this repo's context. The answer key was built by
projecting the pitch model through a camera camlab already believes, and the labeller never saw it.

**The end-to-end line is the one that matters.** Those 7-of-9 labels went through
`write_camlab_anchor.py` unedited and produced an anchor which, after camlab's own auto-fit, scores
**median 0.88 px on 300 samples against the 1.13 px of `camera_smooth` f8** — the camera camlab
currently believes for that frame. Two labels were wrong and it did not matter, because the
enumeration and the refit absorb them.

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

## What the close-ups changed, and why

`prepare` now writes `crops.png` beside the overview: one upscaled tile per segment, with the
segment drawn thin and **dashed** so the pixels underneath stay visible. On the re-run the labeller
stopped inferring and started looking, and its notes say exactly that:

> Tile 2: the dashes sit exactly on the junction between the grass and the dark 'PLAY WITH'
> advertising boards, and a separate real white line runs parallel about 60-80 px below them on the
> grass — so the dashes are on the hoarding base, not paint.

> Tile 7: the dashes cross the goal at crossbar/net height and continue along the top edge of the
> boards, well above the base of the posts.

Both are correct, and they are the two segments camlab's `#14` uses as its standing example of the
hoarding join. Its closing geometry check — *"1, 3, 4, 9 all run along the pitch's long axis; 2, 5,
6, 7, 8 run across"* — is now **exactly** the key's family split, where the full-frame run had it
backwards.

**So the earlier "it cannot see the orientation" conclusion was wrong about the cause.** It could
not see the *paint*, and with nothing to see it fell back on a geometric argument that happened to
be inverted. Given the pixels, both halves came right at once.

## What this says the design should be

Ask for the type, and give the model close-ups to answer with. That is what the two scripts now do,
and it needs **no change in camlab at all** — which was the constraint.

Two things stay true and are worth keeping as fallbacks rather than as the plan:

- **Paint-or-not-paint is the robust core.** It scored 3/3 on `broadcast` from the full frame alone
  and 2/2 on `fan` with tiles, and camlab has measured what it alone is worth: feeding the
  generator the seven real segments instead of nine moves the best hypothesis from **11.9 m to
  3.7 m** and the focal from **28 % wrong to 2.1 %** (`11-is-blocked-by-14-2026-08-12.md`). If the
  type labels degrade on a harder clip, this is the half to fall back to.
- **Falling back to it would need the one camlab change**, because `bootstrap_clip.py` takes no
  segment filter and is not on the HTTP surface. It is **not needed now** and should not be made
  until a clip actually demands it.

## The two labels it still got wrong, and why they did not matter

Segment 4 (`penalty_area_side`, 1.2 px from its marking) was called `not_a_marking` — a false
negative on a real line in the far field, where the labeller saw *"only a tone change from one
mowing band to the next"*. Segment 9 (`goal_area_side`) was called `touchline` — wrong instance,
**right family**, which is the cheap kind of error because `hypotheses()` enumerates instances
within a family anyway.

That is the robustness result stated precisely: **two of nine labels wrong, and the anchor still
beat the seed.** The pool went from 40 candidates under perfect labels to 24 under these, and the
top of the ranking is the same camera.

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

- **That this generalises.** One agent, one prompt, **one** frame with tiles. `broadcast` has not
  been re-run with close-ups, and the seven clips camlab cannot start have not been touched — and
  by construction no answer key can be built for them, so on those the only judge is camlab's paint.
- **That 7/9 is the number.** It is one draw. The useful invariant is not the label score but the
  end-to-end one: two wrong labels of nine still produced an anchor better than the seed.
- **That the vocabulary is right.** It has no value for arcs, and the labeller had to work around
  that. `straight_markings()` excludes arcs too, so the pipeline is consistent, but
  `11-is-blocked-by-14` already notes that arcs are modelled for scoring and never fed to the
  bootstrap.
- **Anything about `which_end`.** Both labellers said the half-turn is **not** determinable from
  their frame, and gave good reasons — sponsor boards and second-tier signage repeat all the way
  round the bowl, and neither frame contains a scoreboard, dugout, tunnel or roof line. That is a
  useful negative: the half-turn will not be solved by asking about one arbitrary frame.
