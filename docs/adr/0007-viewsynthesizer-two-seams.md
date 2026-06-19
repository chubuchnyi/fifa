# ADR-0007 — `ViewSynthesizer` on one port, two integration seams (A render / B amplify+inpaint)

- **Status:** Accepted
- **Date:** 2026-06-19
- **Deciders:** architecture
- **Related:** ADR-0001 (hexagonal core), ADR-0002 (source of truth), ADR-0005 (canonical model), ADR-0006 (providers); C6, FR-29, FR-30, FR-31, FR-32, R-14, R-15, R-16, UX-7; roadmap M2/M3

## Context

Generative novel-view (video-diffusion) re-shoots an episode from a new viewpoint while keeping
photorealism. It is useful in two structurally different ways that nonetheless share a backend
family:

- **Seam A — alternative render pass:** a photoreal video along a *limited* new camera trajectory
  around the broadcast view (FR-29). Good for moderate cinematic moves.
- **Seam B — data amplifier + inpaint:** synthesise N extra views from one camera to feed
  reconstruction as pseudo-multi-view (FR-30), and inpaint unseen sides of subjects for the avatar
  pipeline (FR-31).

Three risks bound it: frustum-overlap limits make strong deviations hallucinate (R-14); its output
is **pixels, not geometry**, so it cannot be edited (R-15); and crowds/fast motion drift in
identity (R-16).

## Decision

1. **One `ViewSynthesizer` port exposes both seams** (`core/ports/view_synthesizer.py`):
   - **Seam A:** `render_orbit(clip, target_camera, scene_hints=None) -> SynthViewRef`.
   - **Seam B:** `amplify(clip, n_views, deviation) -> list[SynthViewRef]` and
     `inpaint_occlusions(subject_views) -> SynthViewRef`.
2. **`SynthViewRef` (`core/scene/assets.py`) is the single output record** for both seams. It
   carries the `seam` (`A_render` / `B_amplify` / `B_inpaint`), the **prescribed** camera
   trajectory (`estimated=False`), `frustum_overlap ∈ [0,1]`, the producing `model`, and an
   `editable` flag.
3. **Seam-A output is video, never editable** (R-15): `editable=False` always; an adapter may wrap
   it as a `RenderPass`, but there is no path from those pixels back to the SMPL/curve source of
   truth (ADR-0002). Arbitrary free-viewpoint stays on the splat/avatar render path.
4. **Seam-B output is consumed by `EnvReconstructor`/`AvatarBuilder`** as pseudo-multi-view input,
   **gated by `frustum_overlap`** — low overlap means likely hallucination and is limited or
   dropped (R-14, R-16).
5. **It is a model provider like any other** (ADR-0006) and its passes are **cached** (ADR-0004,
   FR-32) because they are expensive and generative.

## Consequences

**Positive**
- One contract, one provenance/caching story, for two very different uses.
- The non-editable nature of seam A is encoded in the type (`editable=False`), not just docs (R-15).
- Frustum-overlap gating is a first-class field, so quality limits are enforceable in UX (R-14/R-16).

**Negative / costs**
- Two different consumption paths on one port can confuse callers; the `seam` discriminator and
  docs must carry that weight.
- Generative passes are expensive and identity-drift-prone on crowded 30 s clips (R-16) — requires
  preview/overlay gating and selective application, never blind trust for critical positions.

## Alternatives considered

- **Two separate ports (one per seam)** — rejected: same backend family, would duplicate the
  provenance (ADR-0006) and caching (ADR-0004) wiring.
- **Treat seam-A video as editable geometry** — rejected: it is pixels, not SMPL (R-15); editing
  must stay on the source of truth (ADR-0002).
- **Skip view-synthesis, rely only on splat/avatar** — rejected: loses the moderate-orbit photoreal
  shortcut and the mono→multi-view amplification that helps reconstruction quality.
