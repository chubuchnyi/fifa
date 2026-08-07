# pitch3d — Risk map (R-1 … R-16)

Risks from the TZ (§8) mapped to the **architectural hook** that contains each one. The point of
this table is that no risk is "handled in code somewhere" — each has a *named seam* in the design
(a port, a field, a layer, or a flow) you can point to. See [`architecture.md`](architecture.md) §10
for the condensed version and the ADRs for the decisions behind each hook.

Legend for status: 🟢 contained by a shipped seam in this scaffold · 🟡 seam exists, real
mitigation lands with the milestone · 🔴 inherent limit, bounded by UX/scope, never "solved".

---

## Photoreal & reconstruction quality

| # | Risk | Architectural hook | Status |
|---|---|---|---|
| **R-1** | Photoreal from one viewpoint is **synthesis, not a copy** — unseen detail is invented | Output is honestly a reconstruction: avatars/env are `RenderAssetRef`s with provenance (ADR-0006); seam-B inpaint is flagged plausible-not-exact (R-16); the edit truth stays SMPL/curves | 🔴 |
| **R-2** | **Per-subject avatar cost ×24** people | Avatars are built **on demand** as cached `RenderAssetRef`s (ADR-0004); default strategy is textured SMPL-X (#1), generative/Gaussian (#2/#3) applied selectively (roadmap M2/M3) | 🟡 |
| **R-4** | **Mono ball height** is ambiguous | `BallTrack.height_confidence` is a **first-class per-frame field**; 3D lift is core ballistics (ground contacts via homography, airborne via gravity parabola); low confidence surfaces in the attention list (UX-4) | 🟢 |
| **R-5** | **Occlusions / ID switches** between subjects | Stable `track_id` from the `Tracker` port (FR-6); per-frame/joint `ConfidenceMap`; seam-B `inpaint_occlusions` for unseen sides (FR-31) | 🟡 |
| **R-7** | **Foot-sliding** (root drift vs contact) | Root anchored by the field homography (FR-8); foot-contact handling in HMR; `REFIT` correction mode can re-fit with contact constraints (FR-22c) | 🟡 |
| **R-11** | **Auto-stage quality** is imperfect | The whole point of the non-destructive edit loop (ADR-0002): proposal → operator/agent fixes → propagate; the attention list (UX-4) routes effort to the worst frames | 🟢 |
| **R-12** | **Domain shift** (broadcast styles, kits, lighting) | Models are swappable per deployment behind `ModelProvider` (ADR-0006, NFR-6); no model choice is baked into core | 🟡 |

## Environment & render integration

| # | Risk | Architectural hook | Status |
|---|---|---|---|
| **R-8** | **3DGS needs camera motion** to reconstruct env | `EnvReconstructor` may consume seam-B `amplify` views (mono → pseudo-multi-view, FR-30) as synthetic baseline; generative stadium is the documented fallback (FR-11) | 🟡 |
| **R-9** | **Splat-render integration** complexity | Photoreal render is **external, imported back** behind `RenderPass` (ADR-0003); `bpy` quarantined in `adapters/blender`; core never imports a renderer | 🟡 |
| **R-3** | **edit↔render desync** | Single source of truth (ADR-0002): every render representation is re-driven from `resolved`; there is **no edit path** from pixels back to geometry (architecture §7.1) | 🟢 |

## Interaction & operations

| # | Risk | Architectural hook | Status |
|---|---|---|---|
| **R-6** ⚠️ | **Homography drift** across frames — *see the naming clash note below* | `FieldCalibration` carries **per-frame confidence**; a temporal-smoothing slot is reserved; drift surfaces via `field_homography_conf` | 🟢 |
| **R-10** | **Non-blocking UI** under heavy stages | Everything heavy is a queued **job** (ADR-0004, UX-8); the fake queue runs in-process, a real worker is an adapter swap | 🟢 |
| **R-13** | **Content rights / licensing** (models, footage) | `ModelInfo.license` + `est_cost_usd` recorded per artifact and summed in the `RunLog` (ADR-0006, UX-7, NFR-8); license visible before use | 🟢 |

## ViewSynthesizer (v0.3) — the generative novel-view boundary

| # | Risk | Architectural hook | Status |
|---|---|---|---|
| **R-14** | **Frustum-overlap limit** — strong deviation hallucinates | `SynthViewRef.frustum_overlap ∈ [0,1]` gates seam-B application; **seam A only for moderate moves**, arbitrary free camera stays on the splat/avatar path (ADR-0007) | 🟢 |
| **R-15** | **Output is pixels, not geometry** — cannot be edited | `SynthViewRef.editable = False` for seam A is encoded in the **type**, not just docs; no path from those pixels into the SMPL/curve truth (ADR-0007, ADR-0002) | 🟢 |
| **R-16** | **Crowds / fast motion** drift identity; inpaint is plausible, not exact | Quality gating via preview + reprojection overlay, bounded `deviation`, **selective** application; never trusted for critical positions (ADR-0007) | 🔴 |

---

**Cross-cutting invariant.** Every risk above is contained *without* the heavy stack: the seams are
ports, fields and layers in pure `core` (ADR-0001), so the mitigations are exercised by the fakes,
the dry-run, and the test suite with **no GPU, no Blender, no models, no LLM**.

---

## Naming clash: two things are called "R-6"

**R-6 in THIS table** is the risk-register entry above — homography drift, numbered from TZ §8.

**"R-6" in `CLAUDE.md`, `docs/STATUS.md` and `adr/0011`** means something else entirely: the
honesty rule *"mark, never hide"* — a subject the tracker lost but that is certainly present is
interpolated at low confidence and flagged, never blinked out. That usage is the load-bearing
one in day-to-day work and is not going to be renumbered; this note exists so a reader who
arrives from either direction is not quietly misled. Noted 2026-08-07.

The statuses in the table above were scored against the 2026-06-19 scaffold and have **not**
been re-scored since — a 🟢 here means "a seam exists", not "judged good by eye".
