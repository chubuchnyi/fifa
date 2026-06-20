# pitch3d — Roadmap (M0 → M3)

Maps the TZ milestones (§10) to concrete packages/tickets. **Mono through every milestone.**
Each milestone ends with a **working end-to-end artifact on one real clip**. Real multi-camera
(M4) is out of scope — only the data-model seam exists.

Legend: ✅ exists in this scaffold · 🟡 stub/contract only · ⬜ not started.

---

## M0 — Skeleton (this scaffold)

**Goal:** the hexagonal skeleton compiles, the scene model round-trips, a dry-run runs end-to-end
on fakes, glTF export path exists.

| Ticket | Package | State |
|---|---|---|
| M0-1 Canonical scene model + JSON serialization | `core/scene` | ✅ |
| M0-2 Port ABCs (all of §6) | `core/ports` | ✅ |
| M0-3 Correction engine (4 modes) + rotations | `core/correction` | ✅ |
| M0-4 Orchestration: Stage enum, Pipeline, cache-key, ball lift | `core/orchestration` | ✅ |
| M0-5 Fakes (incl. FakeViewSynthesizer A+B), in-proc queue, cache | `adapters/fakes` | ✅ |
| M0-6 Real-adapter + exporter stubs (`NotImplementedError`) | `adapters/{models,viewsynth,blender,render,export}` | 🟡 |
| M0-7 CLI dry-run + wiring | `app` | ✅ |
| M0-8 Core tests green without GPU/Blender | `tests` | ✅ |
| M0-9 Observation port + viewpoint math + scene summary (LLM feedback) | `core/ports`, `core/agent` | ✅ |
| M0-10 MCP tool catalog (import-free) + `FakeSceneObserver` | `adapters/mcp`, `adapters/fakes` | ✅ |
| M0-11 Live MCP `serve()` (needs `mcp` extra + app controller) | `adapters/mcp` | ✅ |

**Artifact:** `python -m pitch3d.app.cli` runs source → stages(fakes) → scene → edit →
propagate → render(fake) → export, and `pytest` is green. The LLM-feedback loop
(`observe → reason → Correction → resolve → observe`) is exercisable on fakes — no GPU/LLM.

---

## M1 — Editable loop (the vertical slice) 🟡

**Goal (TZ M1):** a 3D scene on a real clip that can be **edited**, with edits propagated.
Proxy only — no photoreal yet. This is the first milestone that replaces fakes with real models.

**Status:** every perception/render/export port now has its real adapter wired behind the same
split pattern (pure half unit-tested via an injected stub backend; heavy half lazy-imported and
gated behind its extra). Each is selectable per-port — `default_ports(detector="rfdetr", …)` and
the matching CLI flags (`--detector/--tracker/--calibrator/--pose/--ball/--render/--export`). The
two dependency-free reals (overlay render, glTF/npz export) run end-to-end on a real clip today
with no GPU; the heavy perception reals need their extra + weights at call time. The live MCP
`serve()` is wired too — its tool→use-case→content-block dispatch is real and unit-tested on the
fakes (the SDK stdio loop is lazy-imported, gated behind the `mcp` extra). The Blender proxy is
wired as well: a pure `ProxyPlan` builder (unit-tested, never imports `bpy`) feeds an
out-of-process `blender --background` runner that writes the editable `.blend` (root/ball as
F-curves, β + body pose as channels) and renders proxy `SCENE_3D` viewpoints for the LLM loop —
gated on a Blender binary (`$PITCH3D_BLENDER`/PATH), Workbench/CPU, no GPU. Confidence
highlighting (overlay/radar markers fade toward a warning colour as confidence drops, UX-3/FR-16)
and the top-down tactical **radar** VIEW (a camera-free 2D minimap surfaced through
`observe(include_radar=…)`) are real and dependency-free too. Remaining for M1: the *interactive*
in-Blender radar placement affordance (a GUI drag that maps onto a `ROOT_TRANSLATION` correction,
step 10) and the heavy perception weights at call time (GPU).

### Vertical slice (one clip, end to end)
1. **Ingest** a real broadcast clip via the FFmpeg adapter → `Source` (fps/res/timecode). `adapters/models` (io)
2. **Episode** select (manual range first; action-spotting later). `core/scene`, `app`
3. **Detect** players/keepers/refs/ball per frame — RF-DETR adapter. `adapters/models` (replaces `FakeDetector`) — 🟡 *wired*: `RFDETRDetector` (pure map/threshold/assembly, unit-tested) over an injected `RFDETRBackend` (torch/cv2/rfdetr, `cv` extra). Live inference needs the extra + weights + GPU.
4. **Track + teams** — ByteTrack/BoT-SORT + team clustering. `adapters/models` — 🟡 *wired*: `ByteTrackTracker` (pure association + appearance team clustering, unit-tested via injected stub); live ByteTrack needs the `cv` extra.
5. **Field homography** — keypoint model + `findHomography` + temporal smoothing → world anchor. `adapters/models` — 🟡 *wired*: `KeypointFieldCalibrator` (pure numpy DLT homography + smoothing, unit-tested); live keypoint net needs the `cv` extra + weights.
6. **HMR → SMPL-X** — GVHMR/WHAM; **root from homography**, foot-contact (anti-foot-slide). `adapters/models` — 🟡 *wired*: `GVHMRPoseEstimator` (pure homography root-grounding + geometric refit, unit-tested); live HMR needs the `hmr` extra + weights + GPU.
7. **Ball** — TrackNet 2D + **core ballistic 3D lift** (already implemented) with height confidence. `adapters/models` + `core/orchestration` — 🟡 *wired*: `TrackNetBallTracker` (pure threshold + linear gap-fill with honest zero-confidence fills, unit-tested); live TrackNet needs the `ball` extra.
8. **Assemble Scene** — proposal layer + confidence map. `core/orchestration` — ✅
9. **Proxy + overlay** — SMPL proxy in Blender; **reprojection overlay**; confidence highlighting. `adapters/render`, `adapters/blender` — ✅ *all real, no GPU*: `ReprojectionOverlayRenderPass` (pure pinhole projection + visibility + stdlib PNG, no extra) and the Blender proxy (`build_proxy_plan` → out-of-process `blender --background` → editable `.blend` / proxy `SCENE_3D` PNGs, Workbench/CPU). **Confidence highlighting** is real (`confidence_to_color` blends each marker toward a red warning colour as per-frame confidence drops; full confidence is a no-op so existing scenes render unchanged — UX-3/FR-16).
10. **Edit** — pose bones/β, ball/root as F-curves, 2D radar VIEW + placement. `adapters/blender` ↔ `core/correction` — 🟡 *editing surface + radar VIEW are real*: `BlenderProxyBuilder` writes the editable `.blend` with root/ball location + axis-angle **F-curves** and β + per-joint body pose as keyframed channels (baked from the resolved proposal ⊕ corrections); edits there map back to `Correction`s, the source of truth (ADR-0002). The top-down **radar** is real on the read side — `render_radar` (pure numpy + stdlib PNG) draws a pitch + team-coloured subject dots + ball from the resolved world XY, surfaced as an `ObservationImage(kind=RADAR)` via `observe(include_radar=…)`. The *interactive* in-Blender radar **placement** drag (which maps onto a `ROOT_TRANSLATION` correction) stays ⬜ — it is a GUI affordance.
11. **Propagate** — offset / interp / re-fit / smoothing with preview + undo (re-fit calls `PoseEstimator.refit`). `core/correction` — ✅
12. **Export** — glTF + SMPL-X `.npz` + intermediate JSON. `adapters/export` — 🟡 *wired*: `GltfExporter` — SMPL-X `.npz` (resolved per subject) + canonical JSON are real (numpy/stdlib, no extra); glTF/GLB assembly (Z-up→Y-up) is real and unit-tested, the `pygltflib` serialization gated behind the `export` extra.

**Exit criteria (TZ AC-1, AC-2, AC-3):** reprojection overlay matches the source on wide shots;
operator fixes a pose and propagates it three ways with preview+undo; ball/player curves edit and
show up in 3D and export.

**Engineering notes**
- Replace one fake at a time; the dry-run wiring is the integration harness.
- Each real adapter must satisfy the same port test the fake passes.
- **Pattern (set by the detector):** split each real adapter into a *pure* half (maps the model's
  raw output into the canonical types — tested with no GPU via an injected backend) and a *heavy*
  half (decode + inference, lazy torch/cv2, gated behind its extra). Select it in wiring per-port
  (`default_ports(detector="rfdetr")`) so the swap is isolated.
- `re-fit` is the first place a model is called from inside a correction — keep it behind the port.

---

## M2 — Photoreal layer ⬜  (+ ViewSynthesizer **seam A**)

**Goal (TZ M2):** photoreal render of the edited scene; edit↔render stays in sync.

| Ticket | Package |
|---|---|
| M2-1 Env reconstruction (3DGS/NeRF by camera motion; generative stadium fallback) | `adapters/models` (`EnvReconstructor`) |
| M2-2 Avatars strategy #1 (textured SMPL-X) + #2 (generative, Rodin-class API) | `adapters/models` (`AvatarBuilder`) |
| M2-3 `SplatAvatarRenderPass` — assemble photoreal frame from `resolved` | `adapters/render` |
| M2-4 Edit↔render sync wiring (resolved drives every render rep) | `app`, `adapters/render` |
| **M2-5 ViewSynthesizer seam A** — `render_orbit` as a `RenderPass` for limited orbits | `adapters/viewsynth`, `adapters/render` |
| M2-6 Render preview (fast low-q) for both paths | `adapters/render` |

**Exit criteria (TZ AC-4, AC-5a):** render pass yields a photoreal frame; editing an SMPL pose
re-projects into photoreal with no manual redo. ViewSynthesizer seam A yields a photoreal
limited-orbit video from the source clip, **cached**, and clearly flagged "video, not editable".

**Boundary reminder (R-14/R-15):** seam A only for moderate moves; arbitrary free camera stays on
the splat/avatar path.

---

## M3 — Quality & polish ⬜  (+ ViewSynthesizer **seam B**)

**Goal (TZ M3):** raise reconstruction quality and operator efficiency.

| Ticket | Package |
|---|---|
| M3-1 Per-subject Gaussian avatars (#3) selectively | `adapters/models` (`AvatarBuilder`) |
| M3-2 Constraint-guided re-fit hardened (PromptHMR-class) | `adapters/models` (`PoseEstimator.refit`) |
| **M3-3 ViewSynthesizer seam B** — `amplify` (mono → pseudo-multi-view) feeding env/avatar recon | `adapters/viewsynth`, `core/orchestration` |
| **M3-4 ViewSynthesizer seam B** — `inpaint_occlusions` for unseen player sides | `adapters/viewsynth`, `adapters/models` |
| M3-5 Confidence map + "needs attention" prioritization UI | `adapters/blender`, `core/scene` |
| M3-6 Versioning / named snapshots / rollback | `core/scene`, `app` |
| M3-7 Web export (three.js / R3F) | `adapters/export` |

**Exit criteria (TZ AC-5b, AC-6, AC-7):** seam B emits N synthetic views accepted by
reconstruction as multi-view input; export + three.js viewer open externally without scale/coord
loss; core still passes tests with fakes (incl. `FakeViewSynthesizer`) **without GPU/Blender**.

---

## Agent / MCP automation track (cross-cutting, ADR-0008) ⬜

The north-star: **automate operator work with an LLM** that drives the same use-cases a human
does, over MCP, with visual feedback. This is not a milestone — it *rides on top of* each one,
gaining capability as the render path matures. The contracts (port, catalog, viewpoint math,
fakes) ship in **M0**; the loop gets sharper feedback as fakes are replaced.

| Capability | Lands with | Package | State |
|---|---|---|---|
| A-1 `SceneObserver` port + `Observation` (3D/overlay/UI + summary) | M0 | `core/ports/observation` | ✅ |
| A-2 Pure viewpoint math: `look_at`, `standard_viewpoints`, centroid | M0 | `core/agent` | ✅ |
| A-3 `scene_summary` from the UX-4 attention list (text feedback) | M0 | `core/agent` | ✅ |
| A-4 MCP `tool_catalog` (import-free, 12 use-cases as data) | M0 | `adapters/mcp` | ✅ |
| A-5 `FakeSceneObserver` (stdlib PNGs, no renderer) | M0 | `adapters/fakes` | ✅ |
| A-6 Live MCP `serve()` over the app controller (`mcp` extra) | M1 | `adapters/mcp`, `app` | ✅ dispatch (tool→use-case→text/image blocks) real + unit-tested on fakes; SDK stdio loop lazy-imported, gated behind the `mcp` extra |
| A-7 `observe` returns real `FRAME_OVERLAY` (reprojection) + proxy `SCENE_3D` | M1 | `adapters/blender`, `adapters/render` | ✅ reprojection `RenderPass` real (`ReprojectionOverlayRenderPass`); proxy `SCENE_3D` real via `BlenderSceneObserver` (out-of-process `blender --background`, Workbench/CPU, gated on a Blender binary) — `--observer blender` |
| A-8 `observe` returns **photoreal** `SCENE_3D` from canonical viewpoints | M2 | `adapters/render` | ⬜ |
| A-9 Orbit viewpoints via ViewSynthesizer seam A in `observe` | M2 | `adapters/viewsynth` | ⬜ |
| A-10 Agent autonomy hardening: bounded edits, attention-driven targeting, eval harness | M3 | `app`, `core/agent` | ⬜ |

**Invariants (hold at every milestone):** the agent edits *only* via `Correction` tools
(`apply_offset|keyframes|smoothing|refit`), never raw geometry; `resolve()` is the sole path to
pixels; feedback images come from the *same* `RenderPass` a human sees. The loop is testable on
fakes with **no GPU, no Blender, no LLM** (TZ AC-7 extends to the agent seam).

**Exit criteria (per stage):** *(M0)* catalog + observer satisfy their contracts in tests and the
dry-run emits inspectable snapshots; *(M1)* an MCP client opens an episode, calls `observe`, applies
an offset, and sees the overlay change; *(M2)* `observe` returns photoreal multi-view of the edited
scene; *(M3)* an agent fixes a seeded-wrong pose end-to-end, verified by the attention list clearing.

---

## M4 (optional) — Real multi-camera ⬜

Out of scope. The data model already treats cameras as a **list** and homography as a degenerate
calibration, so adding synchronized calibrated sources is an adapter + orchestration change, not a
core rewrite. Note: seam-B amplification already exercises the same multi-view input path with
*synthetic* cameras.

---

## Open dependencies / decisions to revisit per milestone

- **M2:** which generative avatar API is primary (realism vs cost vs rig control)? Splat render
  inside Blender vs external renderer (ADR-0003 default: external + import).
- **M3:** which ViewSynthesizer backend per seam (seam B favors 3D-consistency, e.g.
  GEN3C/TrajectoryCrafter; seam A favors orbit fidelity, e.g. ReCamMaster) — see ADR-0007.
- **Cross-cutting:** SMPL-X (hands/face) vs SMPL/SMPL-H (body only) — affects pose dimensions;
  generative-API budget per episode bounds the share of per-subject/ViewSynthesizer work.
- **Agent track (ADR-0008):** which MCP host(s) to target first (Claude CLI/Desktop); how many
  viewpoints per `observe` is the right cost/feedback trade-off; whether the agent eval harness
  (A-10) seeds known-wrong scenes or replays operator-fixed episodes.
