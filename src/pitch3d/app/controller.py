"""Application controller — the use-cases the CLI and the MCP server both drive (ADR-0008).

One method per MCP tool (``run_reconstruction``, ``observe``, ``get_attention``, ``apply_*``,
``set_correction_enabled``, ``preview``, ``render``, ``export``), so the LLM agent and the human
operator exercise identical logic. Methods accept JSON-friendly inputs (dict targets, ``[start,
end]`` ranges) and return scene/domain objects. All editing goes through :class:`Correction`s and
``resolve_scene``; render/observe consume the *resolved* scene only — never the correction stack
directly (single source of truth, ADR-0002).
"""

from __future__ import annotations

import hashlib
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.agent import bounded_orbit_camera, scene_summary, standard_viewpoints
from ..core.config import PhysicsConfig
from ..core.config.gates import IdentityConfig
from ..core.correction.coherence import (
    CoherenceConfig,
    CoherenceReport,
    add_temporal_coherence,
)
from ..core.correction.collision import CollisionReport, collision_gate
from ..core.correction.contact_lock import contact_lock_gate
from ..core.correction.contact_probe import (
    ContactProbeReport,
    FootPositionProvider,
    contact_probe,
)
from ..core.correction.engine import (
    make_keyframes,
    make_offset,
    make_refit,
    make_smoothing,
    preview_subject_motion,
    resolve_ball,
    resolve_subject_motion,
)
from ..core.correction.facing_align import facing_align_gate
from ..core.correction.foot_floor import FootFloorReport, foot_floor_gate
from ..core.correction.foot_plant import FootPlantReport, foot_plant_gate
from ..core.correction.gravity_project import (
    gravity_project_gate,
)
from ..core.correction.inertia_smooth import inertia_smooth_gate
from ..core.correction.jerk_clamp import jerk_clamp_gate
from ..core.correction.joint_kinematics import JointKinematicReport, joint_kinematic_gate
from ..core.correction.joint_smooth import joint_smooth_gate
from ..core.correction.kinematics import (
    KinematicConfig,
    KinematicReport,
    kinematic_gate,
)
from ..core.correction.momentum_smooth import (
    momentum_smooth_gate,
)
from ..core.correction.orient_verticality import (
    orient_verticality_gate,
)
from ..core.correction.orientation import OrientationReport, orientation_gate
from ..core.correction.pose_motion_sync import (
    pose_motion_sync_gate,
)
from ..core.orchestration import (
    HandoverConfig,
    HandoverReport,
    ReconstructionPipeline,
    StitchConfig,
    StitchReport,
    assemble_scene,
    merge_handovers,
    resolve_scene,
)
from ..core.orchestration.identity import AppearanceProvider
from ..core.ports.export import ExportFormat, ExportResult
from ..core.ports.io import ClipRef, CropRef
from ..core.ports.observation import Observation, Viewpoint
from ..core.ports.render import RenderQuality, RenderResult
from ..core.scene.assets import RenderAssetRef, SynthViewRef, SynthViewSeam
from ..core.scene.camera import CameraSource, CameraTrack
from ..core.scene.layers import Correction, CorrectionTarget, TargetKind
from ..core.scene.plane_camera import PlaneCameraFit, camera_from_calibration
from ..core.scene.review import AttentionItem, attention_list
from ..core.scene.scene import Episode, EpisodeSource, Scene, Source, SourceKind
from ..core.scene.units import TimeBase
from ..core.scene.versioning import Snapshot, SnapshotStore, scene_fingerprint

if TYPE_CHECKING:  # avoid a wiring↔controller import cycle
    from .wiring import AppPorts


@dataclass
class Application:
    """In-memory project + ports; the single control surface for CLI and MCP."""

    ports: AppPorts
    out_dir: Path = field(default_factory=lambda: Path("out"))

    _sources: dict[str, Source] = field(default_factory=dict, repr=False)
    _episodes: dict[str, Episode] = field(default_factory=dict, repr=False)
    _clips: dict[str, ClipRef] = field(default_factory=dict, repr=False)
    _scenes: dict[str, Scene] = field(default_factory=dict, repr=False)
    _scene_clip: dict[str, ClipRef] = field(default_factory=dict, repr=False)
    _scene_stitch: dict[str, StitchReport | None] = field(default_factory=dict, repr=False)
    _scene_coherence: dict[str, CoherenceReport | None] = field(default_factory=dict, repr=False)
    _scene_handover: dict[str, HandoverReport | None] = field(default_factory=dict, repr=False)
    _scene_kinematics: dict[str, KinematicReport | None] = field(default_factory=dict, repr=False)
    _scene_camera_fit: dict[str, PlaneCameraFit] = field(default_factory=dict, repr=False)
    _scene_contact: dict[str, ContactProbeReport | None] = field(default_factory=dict, repr=False)
    _snapshots: SnapshotStore = field(default_factory=SnapshotStore, repr=False)
    _ids: dict[str, itertools.count[int]] = field(default_factory=dict, repr=False)

    def _next(self, kind: str) -> int:
        self._ids.setdefault(kind, itertools.count(1))
        return next(self._ids[kind])

    # --- project setup --------------------------------------------------------
    def register_clip(self, clip: ClipRef, *, name: str | None = None) -> Episode:
        """Register a clip as a Source + Episode the agent/CLI can reconstruct."""
        src = self._sources.get(clip.source_id) or Source(
            id=clip.source_id, uri=clip.uri, kind=SourceKind.VIDEO,
            time_base=TimeBase(fps=clip.fps), width=clip.width, height=clip.height,
            frame_count=int(clip.frames[-1]) + 1 if clip.n_frames else 0,
        )
        self._sources[src.id] = src
        ep = Episode(
            id=f"ep-{self._next('episode')}",
            source_id=src.id,
            start_frame=int(clip.frames[0]) if clip.n_frames else 0,
            end_frame=int(clip.frames[-1]) if clip.n_frames else 0,
            origin=EpisodeSource.MANUAL,
            name=name,
        )
        self._episodes[ep.id] = ep
        self._clips[ep.id] = clip
        return ep

    def list_episodes(self) -> list[Episode]:
        return list(self._episodes.values())

    # --- reconstruction -------------------------------------------------------
    def run_reconstruction(
        self,
        episode_id: str,
        *,
        on_ground: np.ndarray | None = None,
        params: dict | None = None,
        stitch_cfg: StitchConfig | None = None,
        coherence_cfg: CoherenceConfig | None = None,
        handover_cfg: HandoverConfig | None = None,
        kinematic_cfg: KinematicConfig | None = None,
        identity_cfg: IdentityConfig | None = None,
        appearance_provider: AppearanceProvider | None = None,
        profile_provider: Any = None,
        auto_tune_sink: Any = None,
        physics_cfg: PhysicsConfig | None = None,
        pelvis_target_provider: Any = None,
        foot_position_provider: FootPositionProvider | None = None,
    ) -> str:
        """Run DETECT→TRACK→(stitch)→CALIBRATE→POSE→BALL, assemble the scene, return its id.

        ``stitch_cfg`` (default ``None`` = off) enables structural track-continuity stitching
        between TRACK and POSE; the report is kept for :meth:`stitch_report`.

        ``coherence_cfg`` (default ``None`` = off) densifies short interior pose gaps and
        appends auto temporal-smoothing corrections after assembly; the report is kept for
        :meth:`coherence_report`. It runs before the camera so the static track spans the
        now-dense frame set.

        ``kinematic_cfg`` (default ``None`` = off) runs the M3-9 plausibility gate after
        coherence: impossible root speed/accel is clamped via per-subject KEYFRAME_INTERP
        corrections, teleports are marked in :meth:`kinematic_report` (never erased, R-6).

        ``profile_provider`` (default ``None``) — optional callable
        ``Subject → PlayerProfile | None``. When present the M3-9 gate uses each
        subject's per-player ceilings from the profile store (T4.b). The gate
        also emits ``ProfileUpdateProposal`` on the report; when
        ``auto_tune_sink`` is provided it is called as
        ``auto_tune_sink(scene, kinematic_report)`` after the gate to persist
        the observations through :func:`apply_profile_updates` (T4.b + T4.c).
        """
        clip = self._clips[episode_id]
        ep = self._episodes[episode_id]
        p = self.ports
        pipeline = ReconstructionPipeline(
            detector=p.detector, tracker=p.tracker, calibrator=p.calibrator,
            pose=p.pose, ball=p.ball, cache=p.cache, queue=p.queue,
            model_version=p.model_version, stitch_cfg=stitch_cfg,
            identity_cfg=identity_cfg, appearance_provider=appearance_provider,
        )
        result = pipeline.run(clip, on_ground=on_ground, params=params)
        scene_id = f"scene-{self._next('scene')}"
        # R-6: `assemble_scene` falls back to `team_id=None` **and** `role=PLAYER` when a motion's
        # track id is missing from `result.tracks.tracklets`, and those two defaults are
        # indistinguishable from measured values in the output. That is how #137 stayed invisible
        # for a whole pod run: 23 of 27 subjects came out `role=player team=None` while the scene's
        # own `teams` block held both teams with member-averaged colours. A null team is not
        # cosmetic either — `StitchConfig.require_same_team` treats `None` as a wildcard, so an
        # unlabelled subject is stitchable to anyone. Say it out loud instead.
        _posed = set(result.motions)
        _tracked = {tl.track_id for tl in result.tracks.tracklets}
        if orphan := sorted(_posed - _tracked):
            print(
                f"== unmatched: {len(orphan)} of {len(_posed)} posed subject(s) have no tracklet "
                f"in the result — they take role=player and team=None by DEFAULT, not by "
                f"measurement: {orphan[:12]}{' …' if len(orphan) > 12 else ''}"
            )
        scene = assemble_scene(
            result, scene_id=scene_id, episode_id=ep.id, source_id=ep.source_id
        )
        coherence_rep: CoherenceReport | None = None
        if coherence_cfg is not None:
            scene, coherence_rep = add_temporal_coherence(scene, coherence_cfg, fps=clip.fps)
        # П3/П2: two ids on one human. This runs HERE and not in the stitcher because the
        # criterion only survives on the post-pose population — pre-pose the same rule joins two
        # different shirts 4 times in 6 (W3, `scripts/bench_handover_stitch.py`). It reads
        # `provenance`, so it must follow coherence, and it runs before the physics gates so they
        # see one body instead of two.
        handover_rep: HandoverReport | None = None
        if handover_cfg is not None and handover_cfg.enabled:
            if coherence_cfg is None:
                raise ValueError(
                    "handover_cfg needs coherence: the criterion reads `provenance`, which only "
                    "exists after add_temporal_coherence. Pass coherence_cfg too (--coherence)."
                )
            scene, handover_rep = merge_handovers(scene, handover_cfg)
        kinematic_rep: KinematicReport | None = None
        if kinematic_cfg is not None:
            scene, kinematic_rep = kinematic_gate(
                scene, kinematic_cfg, fps=clip.fps,
                profile_provider=profile_provider,
            )
            if auto_tune_sink is not None and kinematic_rep is not None:
                auto_tune_sink(scene, kinematic_rep)

        # T1a / T6.a / T1b / T1c / T3 — the rest of the physics gate chain.
        # Enabled per PhysicsConfig; skipped when physics_cfg is None (backwards compat).
        foot_floor_rep: FootFloorReport | None = None
        foot_plant_rep: FootPlantReport | None = None
        joint_rep: JointKinematicReport | None = None
        orientation_rep: OrientationReport | None = None
        collision_rep: CollisionReport | None = None
        if physics_cfg is not None:
            scene, foot_floor_rep = foot_floor_gate(scene, physics_cfg.foot_floor)
            scene, foot_plant_rep = foot_plant_gate(
                scene, physics_cfg.foot_plant,
                pelvis_target_provider=pelvis_target_provider,
            )
            scene, joint_rep = joint_kinematic_gate(
                scene, physics_cfg.joint, fps=clip.fps,
            )
            scene, orientation_rep = orientation_gate(
                scene, physics_cfg.orientation, fps=clip.fps,
            )
            scene, collision_rep = collision_gate(scene, physics_cfg.collision)
            # Step 4b — low-pass root translation (broad CoM smoothing).
            if physics_cfg.momentum_smooth.enabled:
                scene, _ = momentum_smooth_gate(
                    scene, physics_cfg.momentum_smooth,
                    foot_position_provider=foot_position_provider,
                    fps=clip.fps,
                )
            # Orient verticality: force body-up to world-up on HMR-flipped
            # frames. Runs BEFORE facing_align so facing sees upright bodies.
            if physics_cfg.orient_verticality.enabled:
                scene, _ = orient_verticality_gate(
                    scene, physics_cfg.orient_verticality,
                )
            # Pose-motion sync (procedural walk cycle on desynced frames).
            if physics_cfg.pose_motion_sync.enabled:
                scene, _ = pose_motion_sync_gate(
                    scene, physics_cfg.pose_motion_sync, fps=clip.fps,
                )
            # Facing align: rotate body yaw to motion direction.
            if physics_cfg.facing_align.enabled:
                scene, _ = facing_align_gate(
                    scene, physics_cfg.facing_align, fps=clip.fps,
                )
            # Inertia smooth: bound angular acceleration.
            if physics_cfg.inertia_smooth.enabled:
                scene, _ = inertia_smooth_gate(
                    scene, physics_cfg.inertia_smooth, fps=clip.fps,
                )
            # Jerk clamp: iterative low-pass to bound peak jerk in XY/Z.
            # Runs BEFORE contact_lock so the lock is the final authority on
            # foot anchoring — jerk_clamp would otherwise smooth the anchor
            # back out of position.
            if physics_cfg.jerk_clamp.enabled:
                scene, _ = jerk_clamp_gate(
                    scene, physics_cfg.jerk_clamp, fps=clip.fps,
                )
            # Step 3b — contact-lock foot slide during stance runs. Last
            # authority on foot XY during contact frames.
            if physics_cfg.contact_probe.enabled and foot_position_provider is not None:
                scene, contact_lock_rep = contact_lock_gate(
                    scene, physics_cfg.contact_probe, foot_position_provider,
                )
                # Probe re-runs on the CORRECTED scene so the report is honest
                contact_rep = contact_probe(
                    scene, physics_cfg.contact_probe, foot_position_provider,
                )
                self._scene_contact[scene_id] = contact_rep
            # Gravity project: airborne Z onto ballistic parabola — final
            # authority on vertical motion inside airborne runs.
            if physics_cfg.gravity_project.enabled and foot_position_provider is not None:
                scene, _ = gravity_project_gate(
                    scene, physics_cfg.gravity_project, foot_position_provider,
                    fps=clip.fps,
                )
            # Joint smooth: per-joint low-pass on body_pose (twitch removal).
            if physics_cfg.joint_smooth.enabled:
                scene, _ = joint_smooth_gate(
                    scene, physics_cfg.joint_smooth, fps=clip.fps,
                )

        # R-6 applied to the camera (#140). `_measured_camera` refuses whenever the field
        # calibration will not reduce to one camera — correct, and deliberate (#61). What was
        # wrong is that the substitution below was SILENT: on disk a stand-in and a solve were
        # indistinguishable, and nine of nine scenes turned out to carry the stand-in, including
        # the one the #135 eye labels were made on. Mark it, in the scene and on the console.
        measured = self._measured_camera(scene, clip)
        fit = self._scene_camera_fit.get(scene.id)
        if measured is not None:
            scene.camera = replace(
                measured, source=CameraSource.PLANE_FIT,
                fit_reprojection_px=None if fit is None else float(fit.reprojection_px),
                fit_focal_px=None if fit is None else float(fit.focal_px),
            )
            print(f"== camera: MEASURED from the field calibration — focal "
                  f"{scene.camera.intrinsics.fx:.1f} px, reprojection "
                  f"{scene.camera.fit_reprojection_px or float('nan'):.1f} px")
        else:
            scene.camera = replace(
                self._static_camera(scene), source=CameraSource.STATIC_FALLBACK,
                fit_reprojection_px=None if fit is None else float(fit.reprojection_px),
                fit_focal_px=None if fit is None else float(fit.focal_px),
            )
            why = ("no field calibration" if fit is None else
                   f"the plane fit refused: focal {fit.focal_px:.1f} px would reproject at "
                   f"{fit.reprojection_px:.1f} px")
            print(f"== camera: SYNTHETIC FALLBACK, {why}. This is NOT the clip's camera "
                  f"(focal {scene.camera.intrinsics.fx:.1f} px @ "
                  f"{scene.camera.intrinsics.width}x{scene.camera.intrinsics.height}) — nothing "
                  f"comparing this scene to the source pixels is meaningful. #140")
        self._scenes[scene_id] = scene
        self._scene_clip[scene_id] = clip
        self._scene_stitch[scene_id] = result.stitch
        self._scene_coherence[scene_id] = coherence_rep
        self._scene_handover[scene_id] = handover_rep
        self._scene_kinematics[scene_id] = kinematic_rep
        return scene_id

    def stitch_report(self, scene_id: str) -> StitchReport | None:
        """The continuity-stitch report for a scene, or ``None`` if stitching was off."""
        return self._scene_stitch.get(scene_id)

    def coherence_report(self, scene_id: str) -> CoherenceReport | None:
        """The temporal-coherence report for a scene, or ``None`` if coherence was off."""
        return self._scene_coherence.get(scene_id)

    def handover_report(self, scene_id: str) -> HandoverReport | None:
        """The handover-merge report for a scene, or ``None`` if the pass was off."""
        return self._scene_handover.get(scene_id)

    def kinematic_report(self, scene_id: str) -> KinematicReport | None:
        """The M3-9 kinematic-gate report for a scene, or ``None`` if the gate was off."""
        return self._scene_kinematics.get(scene_id)

    def get_scene(self, scene_id: str) -> Scene:
        return self._scenes[scene_id]

    def resolved(self, scene_id: str) -> Scene:
        """The resolved scene (proposal ⊕ corrections), REFIT-aware."""
        scene = self._scenes[scene_id]
        return resolve_scene(
            scene, refit_port=self.ports.pose, clip=self._scene_clip.get(scene_id),
            motion_prior=self.ports.motion_prior,
        )

    # --- visual + textual feedback (the LLM loop) -----------------------------
    def observe(
        self,
        scene_id: str,
        *,
        frame: int | None = None,
        viewpoints: list[str] | None = None,
        n_orbit: int = 0,
        include_ui: bool = False,
        include_radar: bool = False,
        quality: str = "preview",
    ) -> Observation:
        """Snapshot the resolved 3D from several viewpoints (+ overlay/radar/UI) with a summary.

        Pixels come from the *resolved* scene (single source of truth), but the textual
        summary is built from the *stored* scene so the agent sees the live correction count
        and proposal context (the resolved copy bakes the stack empty).
        """
        resolved = self.resolved(scene_id)
        which = [Viewpoint(v) for v in viewpoints] if viewpoints else None
        views = standard_viewpoints(resolved, frame=frame, which=which, n_orbit=n_orbit)
        return self.ports.observer.observe(
            resolved, views, frame=frame, include_ui=include_ui, include_radar=include_radar,
            quality=RenderQuality(quality), summary=scene_summary(self._scenes[scene_id]),
        )

    def get_attention(self, scene_id: str, *, max_items: int = 10) -> list[AttentionItem]:
        return attention_list(self._scenes[scene_id], max_items=max_items)

    # --- edits (always non-destructive Corrections) ---------------------------
    def apply_offset(self, scene_id: str, target, frame_range, delta, *, note: str | None = None) -> Correction:
        corr = make_offset(self._corr_id(), _target(target), frame_range, np.asarray(delta, float), note=note)
        return self._add(scene_id, corr)

    def apply_keyframes(self, scene_id: str, target, frame_range, key_frames, key_values, *, interp: str = "linear", note: str | None = None) -> Correction:
        corr = make_keyframes(self._corr_id(), _target(target), frame_range, np.asarray(key_frames, int), np.asarray(key_values, float), interp=interp, note=note)
        return self._add(scene_id, corr)

    def apply_smoothing(self, scene_id: str, target, frame_range, *, window: int = 5, method: str = "moving_average", sigma: float = 1.0, note: str | None = None) -> Correction:
        corr = make_smoothing(self._corr_id(), _target(target), frame_range, window=window, method=method, sigma=sigma, note=note)
        return self._add(scene_id, corr)

    def apply_refit(self, scene_id: str, target, frame_range, constraints: dict | None = None, *, note: str | None = None) -> Correction:
        corr = make_refit(self._corr_id(), _target(target), frame_range, constraints, note=note)
        return self._add(scene_id, corr)

    def set_correction_enabled(self, scene_id: str, correction_id: str, enabled: bool) -> Correction:
        for c in self._scenes[scene_id].corrections:
            if c.id == correction_id:
                c.enabled = bool(enabled)
                return c
        raise KeyError(f"no correction {correction_id} in {scene_id}")

    # --- versioning (M3-6: named snapshots + rollback) ------------------------
    def snapshot(self, scene_id: str, name: str, *, note: str | None = None) -> Snapshot:
        """Checkpoint the live scene under ``name`` (overwrites a same-named snapshot)."""
        scene = self.get_scene(scene_id)
        return self._snapshots.take(
            scene, name, note=note, created_at=datetime.now(UTC).isoformat()
        )

    def list_snapshots(self, scene_id: str) -> list[Snapshot]:
        """Named checkpoints taken for ``scene_id`` (oldest-first by name registration)."""
        self.get_scene(scene_id)  # validate existence
        return self._snapshots.list(scene_id)

    def rollback(self, scene_id: str, name: str) -> Scene:
        """Replace the live scene with a fresh copy of the named snapshot; return it.

        The snapshot stays intact, so the same checkpoint can be rolled back to repeatedly even
        after further edits (ADR-0002: this restores the whole correction stack, not raw geometry).
        """
        self.get_scene(scene_id)  # validate existence
        restored = self._snapshots.restore(scene_id, name)
        self._scenes[scene_id] = restored
        return restored

    def scene_fingerprint(self, scene_id: str) -> str:
        """Content-addressed digest of the live scene (ADR-0004) — equal content, equal digest."""
        return scene_fingerprint(self.get_scene(scene_id))

    # --- preview (FR-23: resolve as-if, without storing) ----------------------
    def preview(self, scene_id: str, candidate: Correction) -> dict[str, Any]:
        """Resolve as if ``candidate`` were applied; return the max change vs. current resolved."""
        scene = self._scenes[scene_id]
        clip = self._scene_clip.get(scene_id)
        tgt = candidate.target
        if tgt.kind == TargetKind.BALL_POSITION:
            if scene.ball is None:
                base = prev = np.empty((0, 3))
            else:
                existing = scene.corrections_for(None)
                base = resolve_ball(scene.ball, existing).positions_3d
                prev = resolve_ball(scene.ball, [*existing, candidate]).positions_3d
        else:
            tid = tgt.subject_track_id
            if tid is None:
                raise ValueError("non-ball correction target is missing a subject_track_id")
            subj = scene.subject(tid)
            existing = scene.corrections_for(tid)
            base_m = resolve_subject_motion(subj.proposal, existing, refit_port=self.ports.pose, clip=clip, motion_prior=self.ports.motion_prior)
            prev_m = preview_subject_motion(subj.proposal, existing, candidate, refit_port=self.ports.pose, clip=clip, motion_prior=self.ports.motion_prior)
            base, prev = _targeted_array(base_m, tgt), _targeted_array(prev_m, tgt)
        max_change = float(np.max(np.abs(np.asarray(prev) - np.asarray(base)))) if np.size(base) else 0.0
        return {
            "scene_id": scene_id,
            "target_kind": tgt.kind.value,
            "subject_track_id": tgt.subject_track_id,
            "frame_range": [candidate.frame_range.start, candidate.frame_range.end],
            "max_abs_change": max_change,
            "committed": False,
        }

    # --- render + export (consume the resolved scene) -------------------------
    def render(self, scene_id: str, *, quality: str = "preview") -> RenderResult:
        scene = self.resolved(scene_id)
        camera = scene.camera or self._static_camera(scene)
        return self.ports.render.render(scene, camera, RenderQuality(quality))

    def export(self, scene_id: str, fmt: str, out_path: str) -> ExportResult:
        scene = self.resolved(scene_id)
        return self.ports.exporter.export(scene, ExportFormat(fmt), out_path)

    # --- photoreal assets (avatars) -------------------------------------------
    def build_avatars(
        self, scene_id: str, *, ref_crops: dict[int, Sequence[CropRef]] | None = None
    ) -> list[RenderAssetRef]:
        """Build a per-subject avatar asset and attach it to the scene's ``render_assets``.

        Consumes the *resolved* scene's subjects (single source of truth), so edits/refits are
        reflected in the geometry the builder sees. The configured :class:`AvatarBuilder` yields a
        :class:`RenderAssetRef` per subject — a measured textured-SMPL-X PLY for ``--avatar
        textured`` (vertices never seen front-facing stay ``measured=0``, R-6) or a marker for
        ``--avatar fake``. Each ref is attached to the *stored* scene (replacing any same-id ref) so
        render/export/observe can consume it. ``ref_crops`` maps a subject ``track_id`` to its
        reference crops (the heavy backend's sampling hint); the pipeline does not source crops yet,
        so an empty list is passed by default. The scene camera + clip are passed through so a
        measured builder can sample the subject's real broadcast pixels (M2-8b).
        """
        resolved = self.resolved(scene_id)
        crops = ref_crops or {}
        camera = resolved.camera or self._static_camera(resolved)
        clip = self._scene_clip.get(scene_id)
        stored = self._scenes[scene_id]
        # Seam-B (ADR-0007, FR-30/31): feed amplified pseudo-multi-views (shared across subjects) +
        # this subject's inpainted unseen sides into the builder as extra viewpoints (AC-5b).
        amplified = [v for v in stored.synth_views if v.seam is SynthViewSeam.B_AMPLIFY]
        refs = []
        for subject in resolved.subjects:
            inpaint = [
                v for v in stored.synth_views
                if v.seam is SynthViewSeam.B_INPAINT and v.subject_track_id == subject.track_id
            ]
            refs.append(
                self.ports.avatar.build(
                    subject, crops.get(subject.track_id, []),
                    synth_views=(amplified + inpaint) or None, camera=camera, clip=clip,
                )
            )
        new_ids = {ref.id for ref in refs}
        stored.render_assets = [a for a in stored.render_assets if a.id not in new_ids] + refs
        return refs

    # --- measured environment (pitch) -----------------------------------------
    def build_env(self, scene_id: str) -> RenderAssetRef:
        """Reconstruct the scene's environment asset and attach it to ``render_assets``.

        The configured :class:`EnvReconstructor` emits one environment :class:`RenderAssetRef`: the
        measured, calibration-anchored pitch markings for ``--env pitch`` (every vertex
        ``measured=1`` — the M2-0 validator anchor, "a leg can't pass through the pitch"), or a
        placeholder marker for ``--env fake``. The ref is attached to the *stored* scene (replacing
        any same-id ref) so render/observe/export can consume it, mirroring :meth:`build_avatars`.
        """
        resolved = self.resolved(scene_id)
        clip = self._scene_clip[scene_id]
        camera = resolved.camera or self._static_camera(resolved)
        stored = self._scenes[scene_id]
        # Seam-B (ADR-0007, FR-30): feed amplified pseudo-multi-views into the reconstructor as
        # extra calibrated viewpoints (AC-5b); inpaint views are subject-only, not environment.
        amplified = [v for v in stored.synth_views if v.seam is SynthViewSeam.B_AMPLIFY]
        ref = self.ports.env.reconstruct(clip, camera, synth_views=amplified or None)
        stored.render_assets = [a for a in stored.render_assets if a.id != ref.id] + [ref]
        return ref

    # --- ViewSynthesizer seam A (limited orbit, video — NOT editable) ----------
    def render_orbit(
        self,
        scene_id: str,
        *,
        max_deviation_deg: float = 20.0,
        quality: str = "preview",
        scene_hints: dict | None = None,
    ) -> SynthViewRef:
        """Re-shoot the source clip along a bounded orbit (ADR-0007 seam A), cached.

        Builds a *prescribed* limited-orbit camera (a moderate azimuth re-aim around the action
        centroid, hard-capped at 45° — R-14/R-15) over the registered clip's frames, downscales it
        for a fast low-q ``quality="preview"`` re-shoot (UX-9; ``"final"`` is full-res), then asks
        the :class:`ViewSynthesizer` to render it. The result is a photoreal **video, not editable**
        (``SynthViewRef.editable=False``): seam A feeds the eye, never the reconstruction. The ref
        is content-addressed and cached (ADR-0004) — keyed on the orbit *and* the quality, so the
        preview and the final are distinct entries and neither expensive pass recomputes for the
        same inputs — and attached to the *stored* scene's ``synth_views`` (replacing any same-id
        ref). ``frustum_overlap`` on the returned ref falls as the orbit strays, so a caller can
        still gate how far it trusts the re-shoot.
        """
        resolved = self.resolved(scene_id)
        clip = self._scene_clip[scene_id]
        frames = clip.frames
        q = RenderQuality(quality)
        orbit_cam = bounded_orbit_camera(
            resolved, frames, max_deviation_deg=max_deviation_deg
        ).scaled(q.scale)
        span = f"{int(frames[0])}-{int(frames[-1])}" if frames.shape[0] else "empty"
        # Content-address on the *resolved* scene too: a seam-A backend that re-renders the 3D
        # scene (A-9) must re-shoot after any edit/asset rebuild, so an edit busts the cache while
        # an unedited re-call still hits it. A generative backend ignores the fingerprint's effect.
        key = self.ports.cache.key_for(
            "viewsynth_orbit",
            f"{clip.source_id}:{span}",
            {
                "max_deviation_deg": float(max_deviation_deg),
                "quality": q.value,
                "scene": _orbit_fingerprint(resolved),
            },
            self.ports.viewsynth.info().name,
        )
        ref = self.ports.cache.get(key)
        if ref is None:
            # Pass the resolved scene as a 3D hint so a re-render backend (CyclesViewSynthesizer,
            # A-9) can re-shoot the actual geometry; generative backends ignore it and synthesize
            # from the clip (ADR-0007). The orbit camera is prescribed (estimated=False).
            hints = {
                **(scene_hints or {}),
                "quality": q.value,
                "scene": resolved,
                "max_deviation_deg": float(max_deviation_deg),
            }
            ref = self.ports.viewsynth.render_orbit(clip, orbit_cam, hints)
            self.ports.cache.put(key, ref)
        stored = self._scenes[scene_id]
        stored.synth_views = [v for v in stored.synth_views if v.id != ref.id] + [ref]
        return ref

    # --- ViewSynthesizer seam B (data amplifier — feeds reconstruction) --------
    def amplify_views(
        self, scene_id: str, *, n_views: int = 4, deviation: float = 0.3
    ) -> list[SynthViewRef]:
        """Synthesize ``n_views`` pseudo-multi-views from the mono clip (ADR-0007 seam B, FR-30).

        Asks the :class:`ViewSynthesizer` for prescribed off-axis viewpoints around the source
        camera (``deviation`` ∈ [0,1] bounds how far they stray; ``frustum_overlap`` falls with it,
        R-14/R-16). Unlike seam A these are **data, not eye-candy**: they are attached to the stored
        scene's ``synth_views`` so :meth:`build_env`/:meth:`build_avatars` feed them to the
        reconstructor as extra calibrated input (AC-5b). Content-addressed + cached (ADR-0004) on
        the clip and the orbit, so a re-call with the same bounds reuses the synthesized set.
        """
        self.get_scene(scene_id)  # validate existence
        clip = self._scene_clip[scene_id]
        key = self.ports.cache.key_for(
            "viewsynth_amplify",
            clip.source_id,
            {"n_views": int(n_views), "deviation": float(deviation)},
            self.ports.viewsynth.info().name,
        )
        refs = self.ports.cache.get(key)
        if refs is None:
            refs = self.ports.viewsynth.amplify(clip, n_views, deviation)
            self.ports.cache.put(key, refs)
        stored = self._scenes[scene_id]
        new_ids = {r.id for r in refs}
        stored.synth_views = [v for v in stored.synth_views if v.id not in new_ids] + list(refs)
        return refs

    def inpaint_subject(
        self, scene_id: str, track_id: int, *, ref_crops: Sequence[CropRef] | None = None
    ) -> SynthViewRef:
        """Synthesize subject ``track_id``'s unseen sides from its crops (ADR-0007 seam B, FR-31).

        Asks the :class:`ViewSynthesizer` to hallucinate the occluded/back side of one subject from
        the broadcast crops it *was* seen in (a placeholder crop stands in until the pipeline
        sources real ones). The result is tagged ``B_INPAINT`` for that ``track_id`` and attached
        to the stored scene, so :meth:`build_avatars` feeds it to *that* subject's avatar build
        only — plausible, not exact (R-16), never relied on for analysing critical positions.
        """
        self.get_scene(scene_id)  # validate existence
        crops = list(ref_crops) if ref_crops is not None else [self._placeholder_crop(track_id)]
        ref = self.ports.viewsynth.inpaint_occlusions(crops)
        stored = self._scenes[scene_id]
        stored.synth_views = [v for v in stored.synth_views if v.id != ref.id] + [ref]
        return ref

    # --- internals ------------------------------------------------------------
    def _corr_id(self) -> str:
        return f"corr-{self._next('correction')}"

    def _add(self, scene_id: str, corr: Correction) -> Correction:
        self._scenes[scene_id].corrections.append(corr)
        return corr

    def _placeholder_crop(self, track_id: int) -> CropRef:
        """A zero stand-in crop for a subject's pixels until the pipeline sources real ones."""
        return CropRef(subject_track_id=track_id, uri="", frame=0, bbox_xyxy=np.zeros(4))

    def _scene_frames(self, scene: Scene) -> np.ndarray:
        if scene.subjects:
            return scene.subjects[0].proposal.pose.frames
        if scene.ball is not None:
            return scene.ball.frames
        return np.arange(1)

    def _measured_camera(self, scene: Scene, clip: ClipRef) -> CameraTrack | None:
        """The camera the scene's own calibration came from, or ``None`` if it did not come from one.

        ``None`` is the common answer and not a failure: a calibration of *free* per-frame
        homographies is not a camera at any focal (#107), so the caller falls back to the synthetic
        track. Refusing beats emitting a plausible wrong camera — that is the #61 defect, where a
        scene carried two cameras 12686 px apart and nothing noticed for months.
        """
        if scene.field is None or scene.field.calibration is None:
            return None
        fit = camera_from_calibration(
            scene.field.calibration, width=clip.width, height=clip.height
        )
        self._scene_camera_fit[scene.id] = fit
        return fit.camera

    def camera_fit(self, scene_id: str) -> PlaneCameraFit | None:
        """Why a scene's camera is measured or synthetic — focal and reprojection, in pixels."""
        return self._scene_camera_fit.get(scene_id)

    def _static_camera(self, scene: Scene) -> CameraTrack:
        """A static broadcast camera replicated over the whole clip (fallback render path).

        Used only where :meth:`_measured_camera` refuses. It is *not* the clip's camera and cannot
        be compared pixel-to-pixel with the source — anything doing that must check
        :meth:`camera_fit` first.
        """
        frames = self._scene_frames(scene)
        cam0 = standard_viewpoints(scene, which=[Viewpoint.BROADCAST], frame=int(frames[0]))[0].camera
        t = frames.shape[0]
        return CameraTrack(
            intrinsics=cam0.intrinsics,
            frames=frames,
            rotation_quat=np.tile(cam0.rotation_quat[0], (t, 1)),
            translation=np.tile(cam0.translation[0], (t, 1)),
            estimated=True,
        )


def _orbit_fingerprint(scene: Scene) -> str:
    """A short content hash of the resolved scene's render-relevant state (poses + assets).

    Keys the seam-A orbit cache so a re-render backend (A-9, :class:`CyclesViewSynthesizer`)
    re-shoots whenever an edit or asset rebuild changes what the orbit would show, while an
    unedited re-call still hits the cache (ADR-0004).
    """
    h = hashlib.sha256()
    for s in sorted(scene.subjects, key=lambda s: s.track_id):
        p = s.proposal.pose
        for arr in (p.frames, p.global_orient, p.body_pose, p.transl, s.proposal.shape.betas):
            h.update(np.ascontiguousarray(arr).tobytes())
    for a in sorted(scene.render_assets, key=lambda a: a.id):
        h.update(f"{a.id}|{a.uri}|{a.kind.value}".encode())
    return h.hexdigest()[:16]


def _target(spec) -> CorrectionTarget:
    """Accept a CorrectionTarget or a dict (MCP) and return a CorrectionTarget."""
    if isinstance(spec, CorrectionTarget):
        return spec
    return CorrectionTarget(
        kind=TargetKind(spec["kind"]),
        subject_track_id=spec.get("subject_track_id"),
        joint_index=spec.get("joint_index"),
    )


def _targeted_array(motion, target: CorrectionTarget) -> np.ndarray:
    """The array a correction target addresses, for preview diffing."""
    if target.kind == TargetKind.POSE_BODY_JOINT:
        return motion.pose.body_pose[:, target.joint_index, :]
    if target.kind == TargetKind.ROOT_ORIENTATION:
        return motion.pose.global_orient
    if target.kind == TargetKind.ROOT_TRANSLATION:
        return motion.pose.transl
    if target.kind == TargetKind.SHAPE_BETA:
        return motion.shape.betas
    raise ValueError(f"no previewable array for target {target.kind}")
