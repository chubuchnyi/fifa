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

**Artifact:** `python -m pitch3d.app.cli` runs source → stages(fakes) → scene → edit →
propagate → render(fake) → export, and `pytest` is green.

---

## M1 — Editable loop (the vertical slice) ⬜

**Goal (TZ M1):** a 3D scene on a real clip that can be **edited**, with edits propagated.
Proxy only — no photoreal yet. This is the first milestone that replaces fakes with real models.

### Vertical slice (one clip, end to end)
1. **Ingest** a real broadcast clip via the FFmpeg adapter → `Source` (fps/res/timecode). `adapters/models` (io)
2. **Episode** select (manual range first; action-spotting later). `core/scene`, `app`
3. **Detect** players/keepers/refs/ball per frame — RF-DETR adapter. `adapters/models` (replaces `FakeDetector`)
4. **Track + teams** — ByteTrack/BoT-SORT + team clustering. `adapters/models`
5. **Field homography** — keypoint model + `findHomography` + temporal smoothing → world anchor. `adapters/models`
6. **HMR → SMPL-X** — GVHMR/WHAM; **root from homography**, foot-contact (anti-foot-slide). `adapters/models`
7. **Ball** — TrackNet 2D + **core ballistic 3D lift** (already implemented) with height confidence. `adapters/models` + `core/orchestration`
8. **Assemble Scene** — proposal layer + confidence map. `core/orchestration`
9. **Proxy + overlay** — SMPL proxy in Blender; **reprojection overlay**; confidence highlighting. `adapters/blender`
10. **Edit** — pose bones/β, ball/root as F-curves, 2D radar placement. `adapters/blender` ↔ `core/correction`
11. **Propagate** — offset / interp / re-fit / smoothing with preview + undo (re-fit calls `PoseEstimator.refit`). `core/correction`
12. **Export** — glTF + SMPL-X `.npz` + intermediate JSON. `adapters/export`

**Exit criteria (TZ AC-1, AC-2, AC-3):** reprojection overlay matches the source on wide shots;
operator fixes a pose and propagates it three ways with preview+undo; ball/player curves edit and
show up in 3D and export.

**Engineering notes**
- Replace one fake at a time; the dry-run wiring is the integration harness.
- Each real adapter must satisfy the same port test the fake passes.
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
