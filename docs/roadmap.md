# pitch3d — Roadmap (M0 → M4)

Maps the TZ milestones (§10) to concrete packages/tickets. **Mono through every milestone.**
Each milestone ends with a **working end-to-end artifact on one real clip**. Real multi-camera
(M4) is out of scope — only the data-model seam exists.

> **Positioning & competitors:** [`competitive-landscape.md`](competitive-landscape.md) — where we
> sit vs. commercial/academic systems, the honest moat (combination + single-broadcast-cam framing,
> not any single pillar), and the multi-sport (tennis → basketball → hockey) extension plan.

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

**Status (snapshot — the live state, blockers and next steps live in
[`m1-status-and-plan.md`](m1-status-and-plan.md)).** Every perception/render/export port now has a
real adapter behind the same split (pure half unit-tested via an injected stub; heavy half
lazy-imported behind its extra), selectable per-port via `default_ports(...)` and the
`--detector/--tracker/--calibrator/--pose/--ball/--render/--export` flags.

- **Real, no GPU, today:** overlay render (+ confidence highlighting, UX-3/FR-16), glTF/`.npz`
  export, the camera-free top-down **radar** VIEW, the live MCP `serve()` dispatch (SDK stdio loop
  behind the `mcp` extra), and the Blender proxy `.blend`/`SCENE_3D` builder (needs a Blender
  binary, Workbench/CPU).
- **Real on CPU, validated on a real clip:** RF-DETR detection (~15–17 players/frame) → ByteTrack →
  teams A/B (HSV), plus the pose adapter's *pure* root-grounding (16 tracks grounded to world metres
  from the field homography). The interactive-placement seam (a live-Blender drag → `ROOT_TRANSLATION`
  `Correction` through the **same** `apply_offset` the LLM drives, ADR-0008/0010) is host-side real
  and unit-tested over a socket pair.
- **Device knob:** `--device {cpu,cuda}` forwards to every real adapter, defaulting to `cpu` (the
  local validation profile, ADR-0009); production flips to `cuda` with no code change.
- **Still box-gated (stubs):** the pose and ball *heavy* backends. The chosen nets are **SMPLest-X**
  (pose) and **WASB** (ball); they inject on the box by dotted path
  (`--pose-backend`/`--ball-backend pkg.module:Factory` — config-not-fork, research code stays out
  of core, ADR-0006). The injection seam itself is wired + headlessly tested. Also pending: the live
  **GUI** Blender session (`launch_live_session` + depsgraph watcher, step 10) — it needs a display.

### Vertical slice (one clip, end to end)
1. **Ingest** a real broadcast clip via the FFmpeg adapter → `Source` (fps/res/timecode). `adapters/models` (io)
2. **Episode** select (manual range first; action-spotting later). `core/scene`, `app`
3. **Detect** players/keepers/refs/ball per frame — RF-DETR adapter. `adapters/models` (replaces `FakeDetector`) — 🟢 *CPU-validated (P2.3)*: `RFDETRDetector` (pure map/threshold/assembly, unit-tested) over an injected `RFDETRBackend` (torch/cv2/rfdetr, `cv` extra). The `--detector-classes {coco,sports}` knob picks the class map: `coco` (default — the free, downloadable base weights; person→player, sports_ball→ball) or `sports` (the Roboflow checkpoint via `--detector-weights`, which splits keepers/refs apart). Same adapter=production / root=validation default as `device` (ADR-0009). Real run on CPU detected ~15–17 players/frame on a broadcast clip.
4. **Track + teams** — ByteTrack/BoT-SORT + team clustering. `adapters/models` — 🟢 *CPU-validated (P2.3)*: `ByteTrackTracker` (pure association + appearance team clustering, unit-tested via injected stub); the live `ByteTrackBackend` (`cv` extra) needs **no learned weights or GPU** (supervision's ByteTrack + HSV torso sampling). Real run on CPU: detections → 16 stable tracks across 8 frames → teams A/B.
5. **Field homography** — keypoint model + robust homography + temporal smoothing → world anchor. `adapters/models` — 🟢 *wired + robustified + B1-measured*: `KeypointFieldCalibrator` fits each frame with **RANSAC + confidence-weighted normalized DLT** (rejects mislocalised landmarks, the dominant real-broadcast failure) and scores confidence on the inliers × inlier-fraction (R-6 honesty); pure numpy, unit-tested. Validated on real PnLCalib output (messi frame: rejecting 2/18 outliers cut reprojection RMS 3.87 m → 0.36 m). Live keypoint net is injected by dotted path with `--calibrator-backend pkg.module:Factory` (ADR-0006) — the PnLCalib HRNet, now **benchmarked end-to-end on SoccerNet `calibration-2023`** (B1): completeness 0.745, **median 0.236 m** on the frames it locks on. A 2026-06-24 kp-threshold sweep (400 frames) showed the heatmap gate is a **precision/recall dial** — lowering it lifts completeness 0.745→0.918 but net line-accuracy stays flat ~0.61, so the real lever is **better landmarks**, not the gate (portrait / drone footage is still out-of-distribution for the net). A 2026-06-25 A/B of PnLCalib's full camera module (`--solver camera`: points **and** lines) vs the bare DLT confirmed the *solver* is not the lever — median a wash (0.170→0.179 m), completeness unchanged (~0.74) — but it does tighten line registration (line_acc@5px 0.757→0.811) and the error tail (p95 6.41→1.15 m); the remaining lever is **completeness** (the ~¼ of frames the detector drops).
6. **HMR → SMPL-X** — GVHMR/WHAM; **root from homography**, foot-contact (anti-foot-slide). `adapters/models` — 🟢 *pure half CPU-validated (P2.4)*: `GVHMRPoseEstimator` grounds each subject's world root on the pitch from the field homography + assembles SMPL-X `SubjectMotion` + applies geometric refit (numpy, unit-tested). Validated on CPU on REAL ByteTrack tracks — 16 subjects grounded into world metres (Z = pelvis height). The **live `GVHMRBackend` is an unwired, GPU-bound stub**: GVHMR is a research repo (not pip), so the `hmr` extra ships only torch/smplx/chumpy, not the network/weights; `estimate_bodies` raises an actionable `NotImplementedError` (use `--pose fake`, or inject a vendored network by dotted path with `--pose-backend pkg.module:Factory` — the on-box config-not-fork seam, ADR-0006).
7. **Ball** — TrackNet 2D + **core ballistic 3D lift** (already implemented) with height confidence. `adapters/models` + `core/orchestration` — 🟡 *pure half only*: `TrackNetBallTracker` (threshold + linear gap-fill with honest zero-confidence fills, unit-tested) runs end to end on the fake. The **live `TrackNetBackend` is an unimplemented stub** — unlike the rfdetr/bytetrack reals, the `ball` extra ships only torch (no TrackNet weights/decoder), so `detect_ball` raises an actionable `NotImplementedError` (use `--ball fake`, or inject a vendored network by dotted path with `--ball-backend pkg.module:Factory`, ADR-0006) until the network is wired. **Option to bench (2026):** *"Where Is The Ball"* (CVPR'25 CVsports) — a **calibration-free** monocular 2D→3D ball lift (predicts height; trained in sim, soccer-tested) as an alternative to our hand-rolled ballistic lift; evaluate behind the same seam once code releases.
8. **Assemble Scene** — proposal layer + confidence map. `core/orchestration` — ✅
9. **Proxy + overlay** — SMPL proxy in Blender; **reprojection overlay**; confidence highlighting. `adapters/render`, `adapters/blender` — ✅ *all real, no GPU*: `ReprojectionOverlayRenderPass` (pure pinhole projection + visibility + stdlib PNG, no extra) and the Blender proxy (`build_proxy_plan` → out-of-process `blender --background` → editable `.blend` / proxy `SCENE_3D` PNGs, Workbench/CPU). **Confidence highlighting** is real (`confidence_to_color` blends each marker toward a red warning colour as per-frame confidence drops; full confidence is a no-op so existing scenes render unchanged — UX-3/FR-16).
10. **Edit** — pose bones/β, ball/root as F-curves, 2D radar VIEW + placement. `adapters/blender` ↔ `core/correction` — 🟡 *editing surface + radar VIEW are real*: `BlenderProxyBuilder` writes the editable `.blend` with root/ball location + axis-angle **F-curves** and β + per-joint body pose as keyframed channels (baked from the resolved proposal ⊕ corrections); edits there map back to `Correction`s, the source of truth (ADR-0002). The top-down **radar** is real on the read side — `render_radar` (pure numpy + stdlib PNG) draws a pitch + team-coloured subject dots + ball from the resolved world XY, surfaced as an `ObservationImage(kind=RADAR)` via `observe(include_radar=…)`. The *interactive* **placement** drag is now 🟡 *seam real, GUI pending (P1)*: the host side is complete and unit-tested over a socket pair — `radar_to_world` (the inverse of `world_to_radar`, turning a dragged radar pixel back into world XY), the `subject_<id>` id↔name contract (`subject_object_name`/`parse_subject_name`, one source of truth shared by the builder and the bridge), and `serve_edits`/`apply_drag`, which diff a dropped root against the *resolved* position and commit the offset through the **same** `apply_offset` use-case the MCP agent calls (human ≡ LLM, ADR-0008/0010). The live `launch_live_session` (GUI Blender, no `--background`) + the in-Blender `_live.py` depsgraph watcher stay ⬜ — they need a display + a Blender binary, so they are not run in CI (the same honest limitation as the GVHMR/TrackNet reals).
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

## M2 — Photoreal layer 🟡  (+ ViewSynthesizer **seam A**)

**Goal (TZ M2):** photoreal render of the edited scene; edit↔render stays in sync.

**Scope decision (2026-06-25, option A — measured photoreal):** M2's photoreal target is a *real
renderer* (Blender/Cycles) of the **measured** scene — textured, LBS-posed SMPL-X avatars on a grass
pitch under scene lighting, with unmeasured regions honestly marked. **Broadcast-convincing** fidelity
(clothing geometry, full-body texture inpaint, stadium) needs *generative* fill and is **M3** (R-8,
per M2-0). The numpy vertex-splat pass (M2-3) is debug-grade scaffolding, not the photoreal deliverable.

| Ticket | Package |
|---|---|
| **M2-0 Realism approach spike** — decide the realism strategy before building avatars: textured SMPL-X via *measured pixel-projection* (primary) vs generative avatars (#2); explicitly assess & rule on image upscaler + image-to-3D (see note below); honesty gate = measured over hallucinated. Feeds M2-2. | `adapters/models`, doc |
| M2-1 Env reconstruction (3DGS/NeRF by camera motion; generative stadium fallback) | `adapters/models` (`EnvReconstructor`) |
| M2-2 Avatars strategy #1 (textured SMPL-X) + #2 (generative, Rodin-class API) | `adapters/models` (`AvatarBuilder`) |
| M2-3 `SplatAvatarRenderPass` — assemble a measured **vertex-splat** debug frame from `resolved` (scaffolding, not photoreal — see M2-7) | `adapters/render` |
| M2-4 Edit↔render sync wiring (resolved drives every render rep) | `app`, `adapters/render` |
| **M2-5 ViewSynthesizer seam A** — `render_orbit` as a `RenderPass` for limited orbits | `adapters/viewsynth`, `adapters/render` |
| M2-6 Render preview (fast low-q) for both paths | `adapters/render` |
| **M2-7 Real photoreal renderer** — Blender/Cycles `RenderPass` over the *resolved* scene (ADR-0003 external + import); replaces the numpy splat pass as the photoreal path (splat stays the no-dep debug viz) | `adapters/render`, `adapters/blender` |
| **M2-8 Posed, textured avatar** — `SmplxTextureBackend.observe()` per-frame SMPL-X LBS (geometry follows the edited pose, incl. limbs) + surface texture/material from the measured pixel-projection (extend M2-2 #1 per-vertex → texture; R-6 marks unmeasured) | `adapters/models` (`hmr`/`avatar` extra, GPU) |
| **M2-9 Measured env materials + lighting** — grass PBR for the measured pitch + scene lighting/HDRI; stadium stays gated/marked (R-8 → M3) | `adapters/models`, `adapters/blender` |
| **M2-10 Edit↔render sync + photoreal `observe`** — a pose edit (root *and* limbs via LBS) re-projects into the Cycles frame with no manual redo (AC-4 proper); `observe` returns photoreal multi-view (A-8); seam-A orbit becomes a re-render of the 3D scene at orbit cameras, no generative (A-9) | `app`, `adapters/render`, `adapters/viewsynth` |

**Exit criteria (TZ AC-4, AC-5a — measured-photoreal reading, option A):** a *real renderer*
(Blender/Cycles) yields a photoreal frame of the resolved scene — textured, LBS-posed avatars on a
grass pitch under scene lighting, unmeasured regions honestly marked (**not** the numpy vertex-splat
viz); editing an SMPL pose (root *and* limbs) re-projects into that photoreal frame with no manual
redo; `observe` returns photoreal multi-view; seam A yields a photoreal limited-orbit by re-rendering
the 3D scene at orbit cameras, **cached**, clearly flagged "video, not editable". **Honest ceiling:**
measured-only ≈ textured mannequins (single broadcast cam ⇒ half the body unmeasured; naked SMPL-X ⇒
no clothing geometry; no stadium) — broadcast-convincing fidelity is M3.

**Boundary reminder (R-14/R-15):** seam A only for moderate moves; arbitrary free camera stays on
the splat/avatar path.

**Realism approach — investigation (M2-0, R-6 honesty, added 2026-06-24):** the realism gap is *3D
appearance*, not pixel count, so the levers by leverage are: **(1)** texture/appearance on the
existing tracked SMPL-X by **projecting the player's real broadcast pixels** onto the mesh (M2-2 #1)
— keeps temporal coherence + rigging and is *measured* (honest); **(2)** Blender materials/lighting
(grass, PBR kit, shadows, HDRI) — cheap, high payoff; **(3)** clothed-human geometry beyond the naked
SMPL-X body (ECON/ICON-class), *not* generic image-to-3D. **Assessed & de-prioritized:** *image
upscalers* (a 2D-pixel tool — they hallucinate detail, can hurt the geometric perception nets, and
output resolution is set in Blender anyway) and *image-to-3D-object* models (single-shot, un-rigged,
not temporally coherent → they break our tracked SMPL-X; useful only for static props like
goals/stands, which broadcast rarely shows cleanly). **Honesty gate (R-6):** generative
upscale/image-to-3D *hallucinate unmeasured* detail, conflicting with the project's mark-don't-
fabricate stance — prefer pixel-projection (measured) over generative synthesis, and if generative
avatars (#2) are used, flag them as synthesized. Realism is a *presentation* layer, sequenced after
M1's measurement accuracy (calibration/pose), which is the core product value.

**Cross-input — SAM-Body4D / photoreal-rendering brief** (`upscail/sam-body4d-and-photoreal-
rendering-brief.md`, reviewed 2026-06-25): three takeaways adopted, one subordinated. **Adopt:**
**(a)** homography as a *validator-anchor*, not just calibration — the PnLCalib ground plane (M1)
bounds every heavier appearance/HMR layer (a measured texture or an inferred body part that violates
pitch geometry is rejected — "a leg can't pass through the pitch"), operationalising the measured-
over-hallucinated gate; **(b)** a failure-mode eval harness *first* — quantify error by mode
(occlusion / athletic pose / small scale / blur) on real clips before building avatars, then
fine-tune only the dominant-error component; **(c)** the brief's ports table matches ours — keep each
backend a swappable adapter. **Subordinate (R-6):** the brief defaults to *generative* avatars
(feed-forward Gaussian LRMs PF-LHM/IDOL/LHM, multi-view diffusion, kit-number hallucination); M2-0
keeps measured pixel-projection **primary** and routes those candidates to M3 (#3 / seam-B), flagged
synthesized. (The brief's stated "current stack" assumes GVHMR/WHAM + ByteTrack; the actually-wired
calibration is PnLCalib — it is a *target* design, not the repo state.)

**Progress — M2-2 #1 wired E2E (2026-06-25):** the measured textured-SMPL-X `AvatarBuilder`
(projection → front-face/z-buffer visibility → per-vertex colour averaging, R-6: never-seen verts
stay `measured=0`) is now wired through the pipeline, not just unit-tested. New controller stage
`Application.build_avatars` consumes the *resolved* scene's subjects and attaches one
`RenderAssetRef` per subject to the scene; the CLI dry-run runs it as stage 9 (`reconstruct → … →
resolve → observe → **avatar** → render → export`). An injectable, dependency-free
`SyntheticAvatarMeshBackend` (no SMPL-X, no GPU) drives the *real* measured path E2E so CI/`--avatar
textured` exercises projection without weights — `--avatar-backend
pitch3d.adapters.models.avatar:SyntheticAvatarMeshBackend` yields a genuine 2/3-coverage PLY (one
vertex off-frame, honestly unmeasured). The heavy SMPL-X meshing half stays gated behind the
`avatar` extra.

**Progress — M2-3 `SplatAvatarRenderPass` wired E2E (2026-06-25):** the first render pass that
consumes the M2-2 assets. It reads each avatar's vertex-coloured PLY (new paired
`read_vertex_colored_ply`), places the canonical mesh at the subject's *resolved* per-frame root,
projects every vertex through the camera and paints **z-buffered colour splats** onto per-frame
PNGs (pure numpy + the stdlib PNG encoder shared with the overlay pass — no Blender/GPU). R-6
travels into the picture: a `measured=0` vertex renders in a distinct *unmeasured tint*, not its
fabricated placeholder, so unobserved body regions are *visibly* marked. Wired as `--render splat`
(`default_ports(render="splat")`); the dry-run `--avatar textured … --render splat` renders the
measured meshes E2E (decoded frame: measured colour + black + the R-6 tint, never a faked colour).
Honest limits, deferred to heavier upgrades: vertex splats (no triangle fill) and root-translation
placement only (no per-frame limb re-posing — needs the SMPL-X model). **Next: M2-4** (edit↔render
sync) / **M2-1** env reconstruction.

**Progress — M2-4 edit↔render sync wired E2E (2026-06-25):** the splat pass now places the
canonical mesh by the subject's **full resolved root rigid transform** — `global_orient` (axis-angle
→ rotation matrix via the existing `axis_angle_to_matrix`) **⊕** root translation — instead of
translation only. Because the placement reads the *resolved* pose, a `ROOT_ORIENTATION` **or**
`ROOT_TRANSLATION` correction re-projects straight into the rendered frame with **no avatar rebuild**
(TZ AC-5a "editing an SMPL pose re-projects into photoreal with no manual redo"). Proven at the
framebuffer level: a 90° root-orientation edit swings an off-root vertex to its new pixel and vacates
the old one while the PLY's mtime is unchanged (asset never rebuilt); a translation edit slides the
same vertex — both through the `resolved → render` path only. Audit confirms *resolved drives every
render rep*: splat via `_resolved_roots`→`resolve_subject_motion`, the overlay reprojects resolved
roots/ball, and observe snapshots the resolved scene — M2-4 closed the one gap (root orientation was
previously dropped). Honest deferred limit: `POSE_BODY_JOINT` and `SHAPE_BETA` edits still need the
heavy SMPL-X LBS model and are *not* faked — the dependency-free pass honours the rigid root only.
**Next: M2-1** env reconstruction.

**Progress — M2-1 measured-pitch env wired E2E (2026-06-25):** the environment we genuinely
*measure* is the **pitch plane**, so M2-1 ships exactly that and gates the hallucinated rest. New
pure-numpy core geometry `core/scene/pitch.py` (`pitch_line_world_points`) generates the standard
Laws-of-the-Game markings — outer rectangle, halfway line, centre circle/spot, both penalty &
goal areas, penalty spots and the "D" arcs — in world meters on `Z = plane_z`. A new
`MeasuredPitchEnvReconstructor` (`--env pitch`) emits them as a vertex-coloured PLY with **every
vertex `measured=1`** (`coverage=1.0`): it is a measured template, anchored by the M1 calibration,
and is the M2-0 *validator anchor* ("a leg can't pass through the pitch"). It is labelled with a new
honest `ENV_PITCH_MESH` asset kind — **not** the `env_splat` it is not. The same `SplatAvatarRenderPass`
now splats world-space **env meshes** by identity (no per-subject root) under the *shared* z-buffer,
so the pitch grounds the avatars (a near player occludes a far line); `measured=0` env verts would
ride the same R-6 tint, though the pitch is fully measured. New controller stage
`Application.build_env` attaches one `RenderAssetRef` to the scene; the dry-run runs it as stage 9b
(`… → avatar → **env** → render → export`). E2E: `--env pitch --render splat` renders the markings —
a decoded frame shows ~24k white pitch pixels over the background (manifest `env_vertices=1444`),
while `--env fake` returns a placeholder marker the splat pass honestly skips (`env_meshes=0`).
Honest deferred (R-8): a photoreal 3DGS/NeRF stadium from camera motion, or a *generative* stadium
when motion is insufficient, stay gated in `SplatEnvReconstructor` until their milestone — those
hallucinate unmeasured stands. **Next: M2-5** (ViewSynthesizer seam A).

**Progress — M2-5 ViewSynthesizer seam A wired E2E (2026-06-25):** seam A feeds the *eye*, never
the reconstruction — it re-shoots the source clip along a bounded orbit and returns a photoreal
**video, not editable**. New pure camera math `core/agent/viewpoints.py:bounded_orbit_camera`
pans the BROADCAST framing by a *bounded* azimuth arc around the action centroid (one prescribed
`estimated=False` pose per source frame, ground radius + height preserved so it is an orbit not a
dolly), **hard-capped at ±45°** (R-15: a moderate re-aim, never free-viewpoint). The `adapters/render`
stub is now real: `ViewSynthOrbitRenderPass` delegates to `ViewSynthesizer.render_orbit` and wraps
the `SynthViewRef` as a `RenderResult(is_video=True)` via the pure `orbit_render_result` (which
carries `frustum_overlap` + the non-editable flag into the note, so an orbit re-shoot can never
masquerade as an editable result). New controller use-case `Application.render_orbit` is the
authoritative path: it builds the orbit over the **registered clip** (uri/fps the resolved-scene
contract can't carry), calls the synthesizer, **content-addresses + caches** the ref (ADR-0004 — a
second call is a cache hit, no recompute), and attaches it to `scene.synth_views` (deduped on id).
Wiring adds the `--render orbit` selector (sharing one synthesizer instance with the use-case) and
the dry-run runs seam A as stage 10b (`… → render → **seam-A** → export`). E2E: stage 10b reports
`A_render overlap=0.85 editable=False — video, not editable`, `cached + deduped: 1 synth_view after
2 render_orbit call(s)`; `--render orbit` makes the main render `is_video=True`. Honest deferred
(R-8, ADR-0007): the real generative backends (ReCamMaster/GEN3C/TrajectoryCrafter-class) stay gated
in `GenerativeViewSynthesizer`; only the dependency-free fake re-shoots here. **Next: M2-6**.

**Progress — M2-6 fast low-q preview wired E2E (2026-06-25):** "preview before the expensive
final" (UX-9) is implemented as one honest lever — **resolution**. New `RenderQuality.scale`
(preview = 0.5 → ¼ the pixels, final = 1.0) drives new pure camera math `CameraIntrinsics.scaled`
/ `CameraTrack.scaled`, which downscale **only the intrinsics**: per-frame pose, frame indices and
frame *count* are preserved, so the same content rasterises at fewer pixels (no geometry faked,
just cheaper). `scale(1.0)` returns `self`, so a FINAL render keeps the caller's exact camera
(identity intact). Both TZ paths consume it: `SplatAvatarRenderPass.render` renders into the
downscaled framebuffer (manifest/note report the actual size), and the seam-A pair
(`ViewSynthOrbitRenderPass.render` + `Application.render_orbit`) downscales the bounded orbit for a
cheaper generative re-shoot and threads `quality` into both the **scene hint** (a real synthesizer
can also drop steps) and the **cache key** — so the preview and the final are *distinct* ADR-0004
entries and neither expensive pass recomputes for the same inputs. The debug `ReprojectionOverlay`
pass is deliberately left full-res (it is the reprojection inspector, not a TZ render path). E2E:
`--render splat` writes the preview manifest `size=640x360 quality=preview` (the 1280×720 broadcast
camera at ¼ the pixels); `--render orbit` re-shoots the orbit at 640×360 with the 6-frame count
preserved and `is_video=True`. Honest scope (R-6): the lever changes *resolution only* — final-grade
fidelity (texture, AA, real generative steps) still rides the heavy backends, which stay gated.
**Honest status (re-assessed 2026-06-25): the M2 GOAL — a photoreal render — is NOT met.** M2-0…M2-6
delivered the render-pass *architecture* (seam, edit↔render sync on the rigid root, content-addressed
cache, preview-downscale, measured pitch-env, seam-A contract) plus a measured **vertex-splat**
visualization — debug-grade, pure-numpy, *not* a photoreal renderer. The avatars are untextured and
not LBS-posed, no real renderer is wired, and the seam-A "video" is a fake re-shoot. M2-0 routing
*generative* to M3 does **not** make the measured path photoreal — that still needs **M2-7…M2-10**
(real Cycles renderer + posed textured avatars + env materials + edit↔render/observe through it).
**M2 stays 🟡 because the photoreal output does not exist — not because the generative backends are
gated.** Measured ceiling: textured mannequins on grass; broadcast-convincing fidelity is M3. Next
within M2: **M2-7**.

---

## M3 — Quality & polish ⬜  (+ ViewSynthesizer **seam B**)

**Goal (TZ M3):** raise reconstruction quality and operator efficiency.

| Ticket | Package |
|---|---|
| M3-1 Per-subject Gaussian avatars (#3) selectively — candidates: feed-forward Gaussian LRMs (**IDOL** <1 s, **LHM**, **PF-LHM** pose-free/multi-image) for the bulk; per-subject **GaussianAvatar/GART** for hero shots. All flagged *synthesized* (R-6). | `adapters/models` (`AvatarBuilder`) |
| M3-2 Constraint-guided re-fit hardened (PromptHMR-class); **cluster-occlusion robustness** option — amodal occlusion completion (**Diffusion-VAS**) + pixel-level identity (**SAM-3** masklets) gated to occluded segments, validated against the homography anchor (from the brief) | `adapters/models` (`PoseEstimator.refit`) |
| **M3-3 ViewSynthesizer seam B** — `amplify` (mono → pseudo-multi-view) feeding env/avatar recon | `adapters/viewsynth`, `core/orchestration` |
| **M3-4 ViewSynthesizer seam B** — `inpaint_occlusions` for unseen player sides (multi-view human-diffusion candidates: **PSHuman** / **SiTH** / **AniGS**; text-conditioned **TeCH** for kit+number) | `adapters/viewsynth`, `adapters/models` |
| M3-5 Confidence map + "needs attention" prioritization UI | `adapters/blender`, `core/scene` |
| M3-6 Versioning / named snapshots / rollback | `core/scene`, `app` |
| M3-7 Web export (three.js / R3F) | `adapters/export` |
| **M3-8 Learned motion-prior smoothing (option)** — a learned denoiser/motion-prior as an alternative to moving-average, behind the existing `smoothing`/`refit` Correction seam. Leading candidates: **HTD-Refine**, **StableMotion** (purpose-built denoisers); MotionBricks = generative fallback | `adapters/models` ↔ `core/correction` |

**Exit criteria (TZ AC-5b, AC-6, AC-7):** seam B emits N synthetic views accepted by
reconstruction as multi-view input; export + three.js viewer open externally without scale/coord
loss; core still passes tests with fakes (incl. `FakeViewSynthesizer`) **without GPU/Blender**.

**On M3-8 (option, not committed):** the *denoiser*-style models fit our smoothing/refit seam best —
**HTD-Refine** (arXiv 2605.26879, post-hoc SMPL-X velocity/accel refiner, −58–77% foot-slide/jitter)
and **StableMotion** (arXiv 2505.03154, diffusion mocap cleanup from *unpaired* corrupt data,
**soccer-proven** −68% pops / −81% frozen frames). NVIDIA **MotionBricks** (SIGGRAPH 2026) is a
*generative* model (constraints→motion), not a from-video estimator — a weaker fit as a denoiser, a
fallback only. None replaces the HMR backend (B2 pick stays SMPLest-X + SMART); all inject behind the
`smoothing`/`refit` port (ADR-0006) as an isolated swap. Open before adopting: licences (NVIDIA-
research; HTD-Refine / StableMotion code pending), **SMPL-X** in/out compatibility, and football-
motion coverage. Tracked as research, off the M1 critical path.

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

- **M2:** which generative avatar API is primary (realism vs cost vs rig control)? Resolve via the
  **M2-0 realism spike** (measured pixel-projection over generative hallucination, R-6 honesty gate);
  the generative path itself (deferred to M3) ranks per the SAM-Body4D brief — PF-LHM (pose-free,
  multi-image) default, IDOL/LHM alternates. Splat render inside Blender vs external renderer
  (ADR-0003 default: external + import).
- **M3:** which ViewSynthesizer backend per seam (seam B favors 3D-consistency, e.g.
  GEN3C/TrajectoryCrafter; seam A favors orbit fidelity, e.g. ReCamMaster) — see ADR-0007.
- **Cross-cutting:** SMPL-X (hands/face) vs SMPL/SMPL-H (body only) — affects pose dimensions;
  generative-API budget per episode bounds the share of per-subject/ViewSynthesizer work.
- **Agent track (ADR-0008):** which MCP host(s) to target first (Claude CLI/Desktop); how many
  viewpoints per `observe` is the right cost/feedback trade-off; whether the agent eval harness
  (A-10) seeds known-wrong scenes or replays operator-fixed episodes.
