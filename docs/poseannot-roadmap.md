# poseannot — roadmap

Ordered by dependency + eye-value delivered.  Each version is one
merge-able chunk; nothing here is decorative.

## v0 — read-only foundation  ✅ shipped 2026-07-07 (`354cfee`)

Everything you need to eyeball where the pipeline gets it right and where
it fails.  No editing yet.

- FastAPI + JWT + cookie auth (`admin` / `physics` default, rotate before deploy).
- `/api/scene`, `/api/frame/{n}`, `/api/subject/{tid}/mesh|joints|joints2d`,
  `/api/camera/{frame}`.
- CVAT-style layout: toolbar / sidebar / main canvas / right panel / timeline strip.
- 2D overlay: SVG bones + circles projected via server; auto-scales with the video.
- 3D view: Three.js orbit camera, translucent mesh + joint spheres + bones,
  centred on selected pelvis.
- Keyboard: `J`/`→` next frame, `K`/`←` prev.
- Deploy path documented for RunPod.

Verified end-to-end via `curl` + composite overlay screenshot; SMPL-X
joints land on the actual players in the source video.

## v1 — body_pose editing  ← next

**Deliverable:** rotate any of the 21 body joints, save, see the change
propagate through the 2D overlay and stay after reload.

- Three.js pick pass: click a joint sphere → highlight + spawn a rotation
  gizmo (three-axis handles around that joint's world position).
- Drag a gizmo handle → convert screen-space drag to an axis-angle delta
  in the joint's parent frame → POST to backend →
  backend re-runs FK for that (subject, frame) → returns updated verts +
  reprojected joints2d → client redraws.
- `POST /api/edit` writes one `Correction` row to `edits.json` (see
  architecture doc for the format).
- `edits.json` is loaded at start and merged into the backend's scene
  state, so refreshes preserve work.
- Timeline cells turn orange for frames with human edits.
- Undo button (Ctrl-Z) pops the last edit for the current (subject, frame).

**Non-goals for v1:**
- No propagation across frames (v2).
- No physics re-run on save (v2).
- No multi-user coordination (v3).

**Risks:**
- Screen-to-3D drag maths on a rotation gizmo is fiddly — expect one
  session on axis-projection UX.
- SMPL-X body_pose is per-joint axis-angle in the *parent bone frame*.
  Users think in world axes; we need a mode toggle (local vs world) so
  the same drag doesn't behave differently depending on the pose.
- FK per edit is fast (~40 ms) but Three.js `BufferGeometry` rebuild is
  the bottleneck; if it feels laggy we'll switch to `Position.needsUpdate`
  in place.

## v2 — propagation + physics-aware save

**Deliverable:** edit one keyframe → the neighbouring frames follow with
correct temporal continuity; physics gates run after so we don't
introduce jerk / inversions.

- "Propagate to next N frames" button next to Save.  Backend applies the
  edit with a time-decay weight (linear or ease-out) to the neighbour
  frames' body_pose, respecting existing edits.
- After propagation, backend runs the current physics stack over the
  affected subject only (targeted, not scene-wide) to smooth jerks.
- Timeline shows a *span* highlight for the propagated range.
- Undo scopes: single frame vs full propagation span.

## v3 — multi-user + review

Only when the team grows past one operator.

- Frame lock: while user A holds `subj 15, frame 30`, user B sees a
  read-only banner on that pane.
- Review flow: `mark verified` toggles a per-frame verified flag → cell
  turns green in the timeline; verified spans are the export target.
- Export: `POST /api/export` fixes a set of verified frames' resolved
  motion back to `scene.json` as immutable "reviewed corrections" and
  clears them from `edits.json` (compaction).

## v4 — beyond one clip

Multi-clip navigation, per-project user roles, and stitching between
clips.  Explicitly out of scope until v0-v3 are load-bearing in day-to-
day work.

## Bake-off / comparison features (cross-cutting)  ← backlog

Added 2026-07-08 from the pose A/B bake-off (SMPLest-X vs SAM 3D Body). These are eval aids,
not part of the v0→v4 editing ladder, but they live in poseannot because that's where the human
judges pose quality on the real frames.

- **2D overlay source toggle — "original PE" ⇄ "3D reconstruction".**  One switch on the 2D
  frame that flips the overlay between:
  * **original PE** — the raw per-frame 2D perception straight off the pixels (e.g. YOLO-pose /
    the detector+keypoints stage), i.e. what the pose net *saw* before any 3D lift/grounding.
    Prototype exists standalone: `/tmp/real_pe_overlay.py` (YOLO-pose on the real frames).
  * **3D reconstruction** — the current behaviour: our SMPL-X body projected back through the
    calibrated camera.
  Purpose: separate *perception* error from *reconstruction/calibration* error by eye. Needs a
  per-frame "raw 2D keypoints" source in `scene.json` (or a sidecar) the server can serve.
- **Backend A/B compare (SMPLest-X vs SAM 3D Body) in one view.**  Each backend already produces
  its own clip bundle (`poseannot/clips/A_smplestx`, later `…/B_sam3dbody`); the clip switcher
  loads either. Next step: a side-by-side / A-B flip on the same frame so the two skeletons can be
  judged on identical pixels without reloading. Standalone stand-in today:
  `scripts/overlay_from_scene.py` renders each scene's skeleton on the real frames via the exact
  poseannot projection.

## Pipeline-side backlog (feeds the scene.json poseannot consumes)

- **SAM 3D Body pose backend (variant B).**  `src/pitch3d/adapters/models/sam3dbody_backend.py`
  behind the `HMRBackend` port; MHR→SMPL-X via Meta's converter. **Blocked** on the HF-gated
  `facebook/sam-3d-body-dinov3` checkpoint (user must accept licence + `hf download`). Swappable
  via `POSE_BACKEND` in `scripts/pod_real_e2e.sh`.
- **camera→world orientation lift (shared blocker).**  Both pose backends return `global_orient`
  in the **camera** frame; the camera→world rotation is never applied, so ~35 % of subjects read
  inverted in world. The pure half already owns world *translation* (foot→homography); this is the
  matching rotation. Highest-value correctness fix for the pose track.

## Technical debt to watch

- **FK cache is in-process memory only.**  A pod restart re-warms in ~55 s
  for the target clip.  If we scale past one worker, add a Redis/disk
  cache keyed on `(scene_hash, subject_hash, frame)`.
- **Three.js from CDN** — pin the version tag (`0.170.0`), don't chase
  latest.  Compatibility churn is real.
- **`bcrypt` truncates at 72 bytes** — documented in `auth.py`.  If we
  ever add multi-user signup with long passwords, cover this in the UI.
- **`video.py` uses one shared `cv2.VideoCapture`** under a threading
  lock.  Fine for one clip / one process; will need a per-worker capture
  pool if we go multi-worker.
