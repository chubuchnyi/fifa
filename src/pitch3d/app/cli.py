"""CLI dry-run — the whole golden path on fakes, no GPU/Blender/models (ADR-0008).

Runs the same use-cases the MCP agent drives (one :class:`Application` method per MCP tool),
so the human and the LLM exercise identical logic. The flow demonstrates the LLM feedback
loop end to end:

    register clip → reconstruct → OBSERVE → get_attention → reason → preview (FR-23) →
    apply Correction → resolve → OBSERVE again → render → export

Everything writes inspectable artifacts under ``--out-dir`` (tiny PNG snapshots, a render
manifest, canonical-JSON export) and the process exits 0 when the path completes.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..adapters.io import FFmpegIngestor
from ..core.agent import EditBudget, auto_correct
from ..core.correction.anchor import validate_against_anchor
from ..core.correction.engine import make_offset, make_smoothing
from ..core.orchestration import StitchConfig, describe_calibration_solve
from ..core.ports.io import ClipRef
from ..core.ports.observation import Observation
from ..core.scene.layers import CorrectionTarget, TargetKind
from .controller import Application
from .wiring import build_app, default_ports


def _synthetic_clip(*, n_frames: int, width: int = 1280, height: int = 720, fps: float = 25.0) -> ClipRef:
    """A dependency-free clip reference the fakes can 'reconstruct' deterministically."""
    return ClipRef(
        source_id="demo-src",
        uri="memory://demo.mp4",
        frames=np.arange(n_frames),
        width=width,
        height=height,
        fps=fps,
    )


def _truncate_to_first_shot(clip: ClipRef) -> ClipRef:
    """Stop the pipeline reconstructing two cameras as one episode (#132).

    A broadcast clip cuts between cameras; the target clip cuts at frame 236. Tracking identities
    and solving one camera across that boundary silently blends two views, and until now nothing
    checked. Every run was safe only because they all took 48-60 frames from the start.

    R-6 says mark, never hide: the cut is printed with its frame and the clip is trimmed to the
    shot the run starts in, rather than either failing outright or quietly producing a blend.
    ``--no-shot-guard`` turns this off for a caller who genuinely wants the whole file.
    """
    try:
        from ..adapters.models.shot_detect import clip_histograms
        from ..core.orchestration.shots import find_shot_cuts, shot_bounds, shot_containing
    except ImportError:  # pragma: no cover - cv2 absent (fakes-only runs)
        return clip
    frames = np.asarray(clip.frames)
    if frames.size == 0:
        return clip
    try:
        hists = clip_histograms(clip.uri, n_frames=int(frames.size), start=int(frames[0]))
    except Exception as exc:  # pragma: no cover - unreadable/synthetic uri
        print(f"== shot guard: could not scan {clip.uri} ({exc}); continuing unchecked")
        return clip
    cuts = find_shot_cuts(hists)
    if not cuts:
        return clip
    n = int(hists.shape[0])
    bounds = shot_bounds(n, cuts)
    # Offsets are relative to the frames we scanned, which start at clip.frames[0], not at 0.
    base = int(frames[0])
    print(f"== shot guard: {len(bounds)} shots, cut(s) at {[c + base for c in cuts]} — "
          f"{', '.join(f'{a + base}-{b + base}' for a, b in bounds)}")
    _first, last = shot_containing(cuts, n, 0)  # the shot this run starts in
    keep = last + 1
    if keep >= frames.size:
        return clip
    print(f"== shot guard: TRUNCATING to frames {base}-{base + last}, "
          f"{keep} of {frames.size}. Reconstructing across a cut blends two cameras into "
          f"one episode; pass --no-shot-guard to override.")
    return replace(clip, frames=frames[:keep])


def _airborne_on_ground(n_frames: int) -> np.ndarray:
    """Ground contact only at the first/last frame ⇒ a bracketed airborne arc in between.

    The ball lift fits a gravity parabola across the arc and dips ``height_confidence`` toward
    the apex (the honest mono ambiguity, R-4), which is exactly what makes a ``low_ball_height``
    item show up in the attention list — giving the agent something concrete to react to.
    """
    og = np.ones(n_frames, dtype=bool)
    if n_frames > 2:
        og[1:-1] = False
    return og


def _print_observation(obs: Observation, *, label: str) -> None:
    print(f"\n[{label}] {len(obs.images)} snapshot(s), scene={obs.scene_id} frame={obs.frame}")
    for img in obs.images:
        vp = img.viewpoint.value if img.viewpoint else "-"
        print(f"    {img.kind.value:<14} vp={vp:<9} {img.uri}")
    print("    --- summary ---")
    for line in obs.summary.splitlines():
        print(f"    {line}")


def run_dry_run(
    *, out_dir: Path, n_frames: int, n_subjects: int, export_format: str,
    clip_path: str | None = None, detector: str = "fake", tracker: str = "fake",
    calibrator: str = "fake", pose: str = "fake", ball: str = "fake", env: str = "fake",
    avatar: str = "fake", render: str = "fake", export: str = "fake", observer: str = "fake",
    viewsynth: str = "fake", amplify_views: int = 0, amplify_deviation: float = 0.3,
    device: str = "cpu", detector_weights: str | None = None, detector_classes: str = "coco",
    pose_backend: str | None = None, ball_backend: str | None = None,
    calibrator_backend: str | None = None, tracker_backend: str | None = None,
    avatar_backend: str | None = None, occlusion_backend: str | None = None,
    motion_prior: str = "fake", camera_carry: int = 8,
    stitch: bool = True, coherence: bool = False, physics: bool = False,
    physics_profile: str = "default", physics_config: str | None = None,
    player_profiles_dir: str | None = None, player_priors: str | None = None,
    auto_tune: bool = False, ball_id: str = "match_ball_1",
    identity: bool = False,
    demo_edits: bool = True,
    shot_guard: bool = True,
    kit_split: bool = True,
) -> int:
    """Drive the full reconstruction→edit→resolve→render→export path; return an exit code.

    Each adapter argument selects the fake (default, dependency-free) or its real adapter, so the
    same golden path runs entirely on fakes, fully real, or any mix (FR-2..28, ADR-0008). The
    dependency-free reals (``render="overlay"``, ``export="gltf"``) run here with no GPU; the heavy
    perception reals (rfdetr/bytetrack/keypoints/gvhmr/tracknet) raise an actionable extras error
    at call time when their weights/extra are absent. ``device`` (default ``"cpu"``) is forwarded to
    every real adapter so the concept is validated here without a GPU; pass ``"cuda"`` elsewhere.
    """
    out_dir = Path(out_dir)
    ports = default_ports(
        out_dir=out_dir, n_subjects=n_subjects, detector=detector, tracker=tracker,
        calibrator=calibrator, pose=pose, ball=ball, env=env, avatar=avatar, render=render,
        export=export, observer=observer, viewsynth=viewsynth, device=device,
        detector_weights=detector_weights,
        detector_classes=detector_classes, pose_backend=pose_backend, ball_backend=ball_backend,
        calibrator_backend=calibrator_backend, tracker_backend=tracker_backend,
        avatar_backend=avatar_backend, occlusion_backend=occlusion_backend,
        motion_prior=motion_prior, camera_carry=camera_carry, kit_split=kit_split,
    )
    app: Application = build_app(out_dir=out_dir, ports=ports)

    # 1) Project setup: register a clip as an episode the agent/CLI can reconstruct.
    #    --clip ingests a real file via ffprobe (M1 step 1); else a dependency-free synthetic clip.
    if clip_path is not None:
        clip = FFmpegIngestor().clip(clip_path, max_frames=n_frames)
        print(f"== ingested {clip_path}: {clip.width}x{clip.height} @ {clip.fps:.3f}fps, "
              f"{clip.n_frames} frame(s)")
        if shot_guard:
            clip = _truncate_to_first_shot(clip)
    else:
        clip = _synthetic_clip(n_frames=n_frames)
    n = clip.n_frames
    episode = app.register_clip(clip, name="demo episode")
    print(f"== registered {episode.id} ({episode.n_frames} frames) from {clip.uri}")

    # 2) Reconstruction: DETECT→TRACK→(stitch)→CALIBRATE→POSE→BALL, assemble the proposal scene.
    stitch_cfg = StitchConfig() if stitch else None
    # Physics thresholds live in config/physics.yaml — never a hidden Python constant.
    # Precedence: base → profile → env vars (PITCH3D_KIN_*, PITCH3D_COH_*) → CLI --physics-config.
    from ..core.config import load_physics_config
    _phys = (
        load_physics_config(path=physics_config, profile=physics_profile)
        if (coherence or physics)
        else None
    )
    coherence_cfg = _phys.coherence if (coherence and _phys is not None) else None
    kinematic_cfg = _phys.kinematic if (physics and _phys is not None) else None
    if _phys is not None:
        print(f"== physics config: {_phys.summary()}  from {_phys.source_path}")

    # Identity gate wiring: GTA-style split + cross-track merge with the HSV
    # appearance provider (numpy-only starter — swap to OSNet/CLIP-ReIdent later).
    identity_cfg = None
    appearance_provider = None
    if identity and _phys is not None:
        from ..adapters.models.appearance_hsv import make_hsv_appearance_provider
        identity_cfg = _phys.identity
        # Enable regardless of the profile's default (base=false) — the flag is
        # the operator's explicit opt-in; profile still tunes the thresholds.
        if not identity_cfg.enabled:
            from dataclasses import replace as _replace
            identity_cfg = _replace(identity_cfg, enabled=True)
        appearance_provider = make_hsv_appearance_provider(clip)
        print(f"== identity: enabled dbscan_eps={identity_cfg.dbscan_eps} "
              f"merge={identity_cfg.merge_enabled} "
              f"(threshold={identity_cfg.merge_cosine_threshold})")

    # T4.b/T4.c wiring: per-player profile provider + auto-tune sink.
    # Enabled only when --player-profiles-dir is given AND --physics is on
    # (the gate is where the provider gets consulted).
    profile_provider = None
    auto_tune_sink = None
    _priors_cache = {"priors": None, "store": None, "subject_lookup": None, "ball_lookup": None}
    if physics and player_profiles_dir is not None:
        from ..adapters.profiles import LocalJsonPlayerStore
        from ..core.scene.player_profile import (
            Position,
            apply_profile_updates,
            default_player_profile,
            load_priors,
        )
        priors = load_priors(player_priors)
        store = LocalJsonPlayerStore(player_profiles_dir)
        _priors_cache["priors"] = priors
        _priors_cache["store"] = store
        print(f"== profiles: dir={player_profiles_dir!r} priors={priors.policy}")

        def _team_key(subject) -> tuple[str, int, Position]:
            team = str(subject.team_id if subject.team_id is not None else "UNK")
            jersey = int(subject.jersey_number if subject.jersey_number is not None
                         else subject.track_id)
            return team, jersey, Position.UNKNOWN

        def profile_provider(subject):
            team, jersey, pos = _team_key(subject)
            got = store.load_player(team, jersey)
            return got if got is not None else default_player_profile(
                team, jersey, pos, priors=priors,
            )

        if auto_tune:
            def auto_tune_sink(scene, report):
                subject_lookup = {
                    int(s.track_id): _team_key(s) for s in scene.subjects
                }
                counts = apply_profile_updates(
                    store, priors, subject_lookup, report.profile_updates,
                    ball_id_lookup={-1: ball_id},
                )
                print(f"== auto-tune: {counts} "
                      f"({len(report.profile_updates)} proposal(s))")

    # T6a v2 wiring: measured pelvis-above-foot via SMPL-X FK. Falls back to the
    # constant target from cfg when the model isn't available.
    pelvis_target_provider = None
    if physics and _phys is not None and _phys.foot_plant.enabled:
        try:
            from ..adapters.models.smplx_foot_z import make_smplx_foot_z_provider
            pelvis_target_provider = make_smplx_foot_z_provider()
            if pelvis_target_provider is not None:
                print("== foot_plant: SMPL-X FK provider ON (measured per-subject offset)")
            else:
                print("== foot_plant: SMPL-X model not found — using cfg.target_pelvis_m")
        except ImportError:
            print("== foot_plant: SMPL-X FK provider unavailable — using cfg.target_pelvis_m")

    # Foot-position provider (step 3b + 4b need per-frame (T, 3) foot world XY).
    foot_position_provider = None
    if physics and _phys is not None and _phys.contact_probe.enabled:
        try:
            from ..adapters.models.smplx_foot_pos import (
                make_smplx_foot_position_provider,
            )
            foot_position_provider = make_smplx_foot_position_provider()
            if foot_position_provider is not None:
                print("== contact/momentum: SMPL-X foot-position provider ON")
            else:
                print("== contact/momentum: SMPL-X model not found — probes skipped")
        except ImportError:
            print("== contact/momentum: SMPL-X provider unavailable — probes skipped")

    scene_id = app.run_reconstruction(
        episode.id, on_ground=_airborne_on_ground(n),
        stitch_cfg=stitch_cfg, coherence_cfg=coherence_cfg, kinematic_cfg=kinematic_cfg,
        identity_cfg=identity_cfg, appearance_provider=appearance_provider,
        profile_provider=profile_provider, auto_tune_sink=auto_tune_sink,
        physics_cfg=_phys, pelvis_target_provider=pelvis_target_provider,
        foot_position_provider=foot_position_provider,
    )
    scene = app.get_scene(scene_id)
    # Subject 0 need not span the clip — R-6 keeps a short track rather than dropping it — so the
    # observation frame is the middle of that subject's OWN track, not the middle of the clip.
    _s0_frames = scene.subjects[0].proposal.pose.frames
    mid_frame = int(_s0_frames[len(_s0_frames) // 2])
    print(f"== reconstructed {scene_id}: {len(scene.subjects)} subject(s), "
          f"ball={'yes' if scene.ball is not None else 'no'}")
    print(f"== calibration: {describe_calibration_solve(scene.field.calibration)}")
    sr = app.stitch_report(scene_id)
    if sr is not None:
        print(f"== continuity: {sr.n_in}→{sr.n_out} tracklets "
              f"({len(sr.merges)} merge(s), {len(sr.dropped)} blip(s) dropped)")
    cr = app.coherence_report(scene_id)
    if cr is not None:
        print(f"== coherence: bridged {cr.filled_frames} gap frame(s) across "
              f"{cr.subjects_filled}/{cr.n_subjects} subject(s), "
              f"extended {cr.extended_frames} edge frame(s) across "
              f"{cr.subjects_extended}/{cr.n_subjects} subject(s), "
              f"+{cr.corrections_added} auto-smoothing correction(s)")
    kr = app.kinematic_report(scene_id)
    if kr is not None:
        print(f"== physics: speed viol {kr.speed_viol_before}→{kr.speed_viol_after}, "
              f"accel viol {kr.accel_viol_before}→{kr.accel_viol_after}, "
              f"max dev {kr.max_dev_m:.2f}m, "
              f"+{kr.corrections_added} kinematic correction(s) on "
              f"{kr.subjects_corrected}/{kr.n_subjects} subject(s), "
              f"{len(kr.teleports)} teleport(s) marked")
        for tp in kr.teleports[:5]:  # marked, not erased (R-6) — identity/stitch review queue
            print(f"    teleport: subject {tp.track_id} @f{tp.frame} "
                  f"jump {tp.jump_m:.2f}m ({tp.speed_mps:.1f} m/s)")

    # 3) OBSERVE (initial): multi-viewpoint 3D + frame overlay + radar + UI + textual summary.
    obs_before = app.observe(
        scene_id, frame=mid_frame, n_orbit=2, include_ui=True, include_radar=True,
        quality="preview",
    )
    _print_observation(obs_before, label="observe:before")

    # 4-8e) The dry-run edit walkthrough exercises the OBSERVE→edit→resolve seams, but steps 7
    #     (offset) and 8c (refit) COMMIT demo corrections into the scene — fine for the synthetic
    #     golden path, wrong for a real deliverable. --no-demo-edits keeps the scene measured-only.
    if demo_edits:
        # 4) The agent reads the prioritized 'needs attention' list (UX-4).
        attention = app.get_attention(scene_id, max_items=5)
        print(f"\n== attention: {len(attention)} item(s)")
        for it in attention:
            where = f"subject {it.track_id}" if it.track_id is not None else "ball/global"
            frame = f" frame {it.frame}" if it.frame is not None else ""
            print(f"    [{it.score:.2f}] {it.reason} ({where}{frame}) — {it.detail}")

        # 5) Reason → pick an edit. Nudge the first subject's root up over the first half.
        tid = scene.subjects[0].track_id
        frames = scene.subjects[0].proposal.pose.frames
        frame_range = (int(frames[0]), int(frames[len(frames) // 2]))
        delta = np.array([0.0, 0.0, 0.10])  # lift the root 10 cm (Z-up, meters)
        target = CorrectionTarget(kind=TargetKind.ROOT_TRANSLATION, subject_track_id=tid)

        # 6) PREVIEW (FR-23): resolve AS IF the candidate were applied — must NOT mutate the scene.
        candidate = make_offset("candidate", target, frame_range, delta, note="dry-run nudge")
        n_before = len(scene.corrections)
        pv = app.preview(scene_id, candidate)
        assert len(app.get_scene(scene_id).corrections) == n_before, "preview must not mutate"
        print(f"\n== preview (not committed): max_abs_change={pv['max_abs_change']:.4f} m "
              f"over frames {pv['frame_range']} — corrections still {n_before}")

        # 7) Commit the correction (the only way edits ever enter the scene — ADR-0002).
        corr = app.apply_offset(scene_id, target, frame_range, delta, note="dry-run nudge")
        print(f"== committed {corr.id} ({corr.mode.value}) → "
              f"{len(app.get_scene(scene_id).corrections)} correction(s)")

        # 8) OBSERVE (after): the loop is closed — the agent sees the consequence of its edit.
        obs_after = app.observe(scene_id, frame=mid_frame, quality="preview")
        _print_observation(obs_after, label="observe:after")

        # 8c) M3-2: constraint-guided RE-FIT hardened on the measured homography anchor +
        #     validation. Re-fit the first subject's first half locked to its MEASURED ground
        #     track (bbox-foot → world via the homography), then validate the resolved root still
        #     sits on that anchor. A generative cluster-occlusion completion (--occlusion-backend,
        #     gated R-8) would be the thing under scrutiny here; a completion that drifts
        #     off-anchor is flagged, not trusted (R-6).
        subj0 = app.get_scene(scene_id).subject(tid)
        anchor_xy = subj0.proposal.pose.transl[:, :2].copy()  # measured homography ground track
        in_range = (frames >= frame_range[0]) & (frames <= frame_range[1])
        refit_corr = app.apply_refit(
            scene_id, target, frame_range, {"foot_anchor": anchor_xy[in_range]},
            note="dry-run refit locked to the measured homography anchor",
        )
        resolved0 = app.resolved(scene_id).subject(tid)
        report = validate_against_anchor(
            resolved0.proposal.pose.frames, resolved0.proposal.pose.transl, anchor_xy,
        )
        print(
            f"\n== M3-2 refit[{refit_corr.id}]: locked {int(in_range.sum())} frame(s) to the "
            f"measured anchor → on-anchor {report.n_valid}/{report.n_frames} "
            f"(max residual {report.worst_residual_m:.3f} m, {report.n_off_anchor} off-anchor, R-6)"
        )

        # 8d) M3-8: a learned temporal denoiser offered through the SAME smoothing Correction seam
        #     (method="learned"). PREVIEWED (FR-23, not committed) so the gated learned model
        #     swaps in without changing this path — here the GPU-free fake prior denoises the
        #     subject's stepped root path. The pure moving_average/gaussian methods need no prior;
        #     the learned model (HTD-Refine/StableMotion) is gated (R-8, --motion-prior learned).
        smooth_cand = make_smoothing(
            "cand-learned-smooth", target, (int(frames[0]), int(frames[-1])), method="learned",
            note="dry-run learned motion-prior denoise (preview only)",
        )
        n_corr = len(app.get_scene(scene_id).corrections)
        try:
            pv_smooth = app.preview(scene_id, smooth_cand)
            assert len(app.get_scene(scene_id).corrections) == n_corr, "preview must not mutate"
            span = pv_smooth["frame_range"][1] - pv_smooth["frame_range"][0] + 1
            print(
                f"== M3-8 learned-smoothing[preview]: MotionPrior "
                f"'{ports.motion_prior.info().name}' denoised subject {tid} root over {span} "
                f"frame(s) → max_abs_change {pv_smooth['max_abs_change']:.4f} m (not committed; "
                f"learned model gated R-8 — --motion-prior learned)"
            )
        except (NotImplementedError, RuntimeError) as exc:
            print(
                f"\n== M3-8 learned-smoothing[preview]: skipped — learned model not wired "
                f"({type(exc).__name__}); use --motion-prior fake (GPU-free gaussian)"
            )

        # 8e) A-10: BOUNDED, ATTENTION-DRIVEN AUTONOMY. The agent doesn't just preview one edit —
        #     it reads the attention list, targets the worst off-anchor subject, and applies
        #     bounded anchor-pull corrections until attention clears (R-6 measured proof) — all on
        #     a LOCAL scene copy, so nothing here pollutes the export. We seed a wrong pose (2 m
        #     off-anchor) on the measured ground track and watch the loop fix it within its
        #     EditBudget leash.
        base = app.resolved(scene_id)
        anchors = {s.track_id: s.proposal.pose.transl[:, :2].copy() for s in base.subjects}
        seeded = replace(
            base,
            corrections=[
                make_offset(
                    "seed-wrong-pose", target, (int(frames[0]), int(frames[-1])),
                    np.array([2.0, 0.0, 0.0]), note="seeded 2 m off-anchor reconstruction error",
                )
            ],
        )
        _fixed, areport = auto_correct(seeded, anchors, budget=EditBudget(max_abs_change_m=1.0))
        print(
            f"\n== A-10 autonomy[eval]: seeded subject {tid} 2 m off-anchor → "
            f"attention {areport.attention_before}→{areport.attention_after} in "
            f"{areport.edits_applied} bounded edit(s) (≤1.0 m), cleared={areport.cleared} "
            f"(local copy; export untouched)"
        )
    else:
        print("\n== demo edits: OFF (--no-demo-edits) — no dry-run offset/refit committed; "
              "the scene carries only measured proposals + auto corrections")

    # 8b) SEAM B (ADR-0007, FR-30): the data amplifier. Synthesize N pseudo-multi-views from the
    #     single broadcast camera and attach them to the scene; the env/avatar reconstruction below
    #     consumes them as extra calibrated input (AC-5b). Unlike seam A these are data, not video —
    #     their frustum_overlap falls as the synthetic camera strays (R-14/R-16). 0 disables.
    if amplify_views > 0:
        try:
            amp = app.amplify_views(scene_id, n_views=amplify_views, deviation=amplify_deviation)
            print(
                f"\n== seam B[amplify]: {len(amp)} pseudo-view(s) @ "
                f"deviation={amplify_deviation} (overlap={amp[0].frustum_overlap:.2f}) "
                f"→ fed to env/avatar reconstruction"
            )
        except (NotImplementedError, RuntimeError) as exc:
            print(
                f"\n== seam B[amplify]: skipped — generative backend not wired "
                f"({type(exc).__name__}); reconstruction proceeds on the mono view (use "
                f"--viewsynth fake)"
            )

    # 9) AVATAR: build a per-subject render asset, attached to the scene (photoreal stage #1). The
    #    gated heavy path (`--avatar textured` with no backend) is reported, not fatal — honest
    #    about what it can't measure yet (R-6); the golden path still renders/exports.
    try:
        avatar_refs = app.build_avatars(scene_id)
        print(f"\n== avatar[{avatar}]: built {len(avatar_refs)} asset(s) → {out_dir / 'assets'}")
        for r in avatar_refs:
            x = r.extra
            if "coverage" in x:
                detail = f"coverage={x['coverage']:.2f} {x['n_measured']}/{x['n_vertices']} verts"
            else:
                detail = f"ref_crops={x.get('ref_crops', 0)}"
            if x.get("synth_views"):
                detail += f", synth_views={x['synth_views']} (seam B)"
            print(f"    subject {r.subject_track_id}: {r.kind.value} — {detail}")
    except (NotImplementedError, RuntimeError) as exc:
        print(
            f"\n== avatar[{avatar}]: skipped — heavy backend not wired "
            f"({type(exc).__name__}); inject --avatar-backend or use --avatar fake"
        )

    # 9b) ENV: reconstruct the measured environment and attach it. With `--env pitch` the splat
    #     render grounds the avatars on the calibration-anchored pitch markings (M2-1, the M2-0
    #     validator anchor); the fake stays honest about being a placeholder, not a measured mesh.
    env_ref = app.build_env(scene_id)
    x = env_ref.extra
    detail = (
        f"coverage={x['coverage']:.2f} {x['n_vertices']} verts (all measured)"
        if "coverage" in x else "placeholder marker"
    )
    if x.get("synth_views"):
        detail += f", synth_views={x['synth_views']} (seam B)"
    print(f"\n== env[{env}]: {env_ref.kind.value} — {detail} → {out_dir / 'assets'}")

    # 10) Render the RESOLVED scene (proposal ⊕ corrections), single source of truth.
    render_result = app.render(scene_id, quality="preview")
    print(
        f"\n== render: {render_result.n_frames} frame(s), "
        f"is_video={render_result.is_video} → {render_result.uri}"
    )

    # 10b) SEAM A (ADR-0007): re-shoot the clip along a bounded orbit → a photoreal *video, not
    #      editable*. Content-addressed + cached (the 2nd call hits the cache, no recompute) and
    #      deduped on the scene's synth_views, so the eye-candy path can never masquerade as an
    #      editable reconstruction (R-15).
    synth = app.render_orbit(scene_id, max_deviation_deg=20.0)
    app.render_orbit(scene_id, max_deviation_deg=20.0)  # idempotent: cache hit, no new synth_view
    n_views = len(app.get_scene(scene_id).synth_views)
    print(
        f"\n== seam A[orbit]: {synth.seam.value} overlap={synth.frustum_overlap:.2f} "
        f"editable={synth.editable} — {synth.note} → {synth.uri}"
    )
    print(f"    cached + deduped: {n_views} synth_view(s) after 2 render_orbit call(s)")

    # 11) Export (canonical JSON is a real round-trip; other formats are honest fakes).
    export_path = out_dir / "export" / f"scene.{export_format}"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    result = app.export(scene_id, export_format, str(export_path))
    print(f"== export[{result.fmt.value}]: {', '.join(result.paths) or '(no paths)'}")

    lineup = {
        "detect": detector, "track": tracker, "calibrate": calibrator, "pose": pose,
        "ball": ball, "env": env, "avatar": avatar, "render": render, "export": export,
        "observe": observer, "viewsynth": viewsynth, "motion_prior": motion_prior,
    }
    real = [f"{k}={v}" for k, v in lineup.items() if v != "fake"]
    fake = [k for k, v in lineup.items() if v == "fake"]
    print("\nOK — reconstruct → observe → attention → preview → edit → resolve → "
          "observe → avatar → render → seam-A → export completed.")
    print(f"  device: {device}")
    print(f"  real adapters: {', '.join(real) if real else '(none — fully on fakes)'}")
    print(f"  fake adapters: {', '.join(fake) if fake else '(none — fully real)'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pitch3d",
        description="Run the pitch3d dry-run (full golden path on fake adapters).",
    )
    parser.add_argument("--out-dir", default="out/dry-run", help="where artifacts are written")
    parser.add_argument("--frames", type=int, default=12,
                        help="clip length (synthetic) or max frames sampled from --clip")
    parser.add_argument("--subjects", type=int, default=4, help="number of fake subjects")
    parser.add_argument("--format", default="json", help="export format (json is a real round-trip)")
    parser.add_argument("--clip", default=None,
                        help="ingest a real video via ffprobe (M1 step 1); default: synthetic clip")
    parser.add_argument("--detector", default="fake", choices=["fake", "rfdetr"],
                        help="detection adapter; 'rfdetr' needs the cv extra + weights + GPU")
    parser.add_argument("--tracker", default="fake", choices=["fake", "bytetrack"],
                        help="tracking adapter; 'bytetrack' needs the cv extra")
    parser.add_argument("--calibrator", default="fake", choices=["fake", "keypoints"],
                        help="field calibrator; 'keypoints' needs the cv extra + weights")
    parser.add_argument("--pose", default="fake", choices=["fake", "gvhmr"],
                        help="pose estimator; 'gvhmr' live backend is an unwired GPU-bound stub "
                             "(the pure root-grounding half is real; use 'fake')")
    parser.add_argument("--ball", default="fake", choices=["fake", "tracknet"],
                        help="ball tracker; 'tracknet' live backend is an unwired stub "
                             "(the pure threshold/gap-fill half is real; use 'fake')")
    parser.add_argument("--env", default="fake", choices=["fake", "pitch"],
                        help="environment reconstructor; 'pitch' = measured calibration-anchored "
                             "pitch markings as a vertex-coloured PLY (M2-1, rendered by --render "
                             "splat). 3DGS/NeRF/generative stadium stay gated (R-8)")
    parser.add_argument("--avatar", default="fake", choices=["fake", "textured", "gaussian"],
                        help="avatar builder; 'textured' = measured pixel-projection onto the "
                             "tracked SMPL-X (M2-0 primary), 'gaussian' = measured 3DGS splats "
                             "anchored per vertex (M3-1 #3). Both measured halves are real + "
                             "GPU-free; the SMPL-X meshing / generative refiner heavy halves are "
                             "gated (inject --avatar-backend or use 'fake')")
    parser.add_argument("--render", default="fake",
                        choices=["fake", "overlay", "splat", "cycles", "orbit"],
                        help="render pass; 'overlay' reprojects PNGs, 'splat' rasterises the "
                             "measured avatar meshes (both real + dependency-free), 'cycles' "
                             "renders those meshes photoreal via Blender/Cycles (M2-7, needs "
                             "$PITCH3D_BLENDER), 'orbit' is the ViewSynthesizer seam-A "
                             "limited-orbit re-shoot (video, not editable)")
    parser.add_argument("--export", default="fake", choices=["fake", "gltf", "threejs"],
                        help="exporter; 'gltf'/'threejs' are real (npz+JSON+glTF; web viewer is "
                             "dependency-free, use --format threejs)")
    parser.add_argument("--observer", default="fake", choices=["fake", "blender", "cycles"],
                        help="scene observer; 'blender' renders real proxy SCENE_3D, 'cycles' "
                             "renders photoreal SCENE_3D per viewpoint (M2-10/A-8); both need "
                             "$PITCH3D_BLENDER or 'blender' on PATH")
    parser.add_argument("--viewsynth", default="fake", choices=["fake", "cycles", "generative"],
                        help="view synthesizer (both ADR-0007 seams); 'cycles' re-renders the "
                             "reconstructed 3D scene at the orbit cameras (non-generative, "
                             "M2-10/A-9, needs $PITCH3D_BLENDER); 'fake' is the deterministic "
                             "stand-in; 'generative' is the real diffusion backend, gated (R-8)")
    parser.add_argument("--amplify-views", type=int, default=0, metavar="N",
                        help="seam-B data amplifier: synthesize N pseudo-multi-views from the mono "
                             "clip and feed them to env/avatar reconstruction (0 disables; AC-5b)")
    parser.add_argument("--amplify-deviation", type=float, default=0.3, metavar="D",
                        help="seam-B camera offset bound D in [0,1]; frustum overlap falls as it "
                             "grows (R-14/R-16). Only used when --amplify-views > 0")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="inference device for real perception adapters "
                             "(default: cpu, the local concept-validation profile)")
    parser.add_argument("--detector-weights", default=None,
                        help="optional RF-DETR weights path/identifier (else the base weights)")
    parser.add_argument("--detector-classes", default="coco", choices=["coco", "sports"],
                        help="RF-DETR class map: 'coco' (free base weights, person->player; "
                             "default) or 'sports' (Roboflow checkpoint via --detector-weights)")
    parser.add_argument("--pose-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own HMRBackend by dotted path (on-box GVHMR "
                             "wiring, no fork); requires --pose gvhmr")
    parser.add_argument("--ball-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own BallDetectionBackend (TrackNet); "
                             "requires --ball tracknet")
    parser.add_argument("--calibrator-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own KeypointBackend; "
                             "requires --calibrator keypoints")
    parser.add_argument("--camera-carry", type=int, default=8, metavar="N",
                        help="R2 camera propagation: re-estimate each frame's homography from its "
                             "+-N neighbours carried on Lucas-Kanade inter-frame motion (CPU, no "
                             "weights). Removes 92%% of scene swim for ~0.004m of paint accuracy "
                             "(#104). 0 disables. Only used by --calibrator keypoints")
    parser.add_argument("--tracker-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own TrackingBackend; "
                             "requires --tracker bytetrack")
    parser.add_argument("--avatar-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own AvatarMeshBackend (SMPL-X meshing + frame "
                             "sampling); requires --avatar textured or gaussian")
    parser.add_argument("--occlusion-backend", default=None, metavar="pkg.module:Factory",
                        help="inject a bring-your-own OcclusionBackend (Diffusion-VAS amodal + "
                             "SAM-3 masklets, M3-2) used by REFIT 'complete_occlusions'; "
                             "requires --pose gvhmr")
    parser.add_argument("--motion-prior", default="fake", metavar="fake|learned|pkg:Factory",
                        help="learned temporal denoiser for TEMPORAL_SMOOTHING method='learned' "
                             "(M3-8): 'fake' (real GPU-free gaussian), 'learned' (gated "
                             "HTD-Refine/StableMotion, R-8) or a dotted-path BYO MotionPrior")
    parser.add_argument("--no-stitch", dest="stitch", action="store_false",
                        help="disable track-continuity stitching (ON by default): without it, "
                             "occluded players re-enter as NEW track ids and spawn phantom "
                             "bodies (the #202 swarm)")
    parser.add_argument("--no-kit-split", dest="kit_split", action="store_false",
                        help="disable the #132 team-change track split (ON by default): "
                             "without it a crossing can hand one track id to a different "
                             "human, which keeps its avatar, kit and motion history")
    parser.add_argument("--no-shot-guard", dest="shot_guard", action="store_false",
                        help="reconstruct across camera cuts (guard is ON by default): a "
                             "broadcast clip cuts between cameras, and tracking + calibrating "
                             "through one blends two views into a single 'episode'")
    parser.add_argument("--coherence", action="store_true",
                        help="bridge short interior pose gaps (slerp/lerp) + add auto "
                             "temporal-smoothing corrections (off by default)")
    parser.add_argument("--physics", action="store_true",
                        help="M3-9 kinematic plausibility gate (off by default): clamp root "
                             "speed/accel to human limits via auto corrections, mark teleports "
                             "for identity review. Thresholds come from config/physics.yaml — "
                             "select a named profile with --physics-profile or a custom file "
                             "with --physics-config. Env vars (PITCH3D_KIN_MAX_SPEED/MAX_ACCEL/"
                             "TELEPORT, PITCH3D_COH_*) still override.")
    parser.add_argument("--physics-profile", default="default",
                        help="named profile from config/physics.yaml (default/conservative/"
                             "strict/no_smoothing/future_full)")
    parser.add_argument("--physics-config", default=None,
                        help="path to an alternate physics config (defaults to the shipped "
                             "config/physics.yaml)")
    parser.add_argument("--player-profiles-dir", default=None, metavar="DIR",
                        help="local directory for per-player + per-ball profile JSONs (T4). "
                             "When set alongside --physics, the M3-9 gate uses each subject's "
                             "peak_speed_mps / peak_accel_mps2 from the stored profile "
                             "(seeded from config/player_priors.yaml on first sighting).")
    parser.add_argument("--player-priors", default=None, metavar="PATH",
                        help="alternate player-priors YAML (defaults to shipped "
                             "config/player_priors.yaml)")
    parser.add_argument("--auto-tune", action="store_true",
                        help="after the M3-9 gate, feed the p95 speed/accel observations from "
                             "the RESOLVED motion through update_field and save the mutated "
                             "profiles. Applies the seven-layer filter (§4.4) at the "
                             "persistence seam. Requires --player-profiles-dir.")
    parser.add_argument("--ball-id", default="match_ball_1", metavar="ID",
                        help="identifier used to key the ball profile in the store (default "
                             "one profile per match)")
    parser.add_argument("--identity", action="store_true",
                        help="enable identity_gate (GTA-style intra-track split + "
                             "cross-track merge). Requires --physics for the config. Wires the "
                             "HSV appearance provider automatically; swap to a real Re-ID "
                             "backbone via a follow-up.")
    parser.add_argument("--no-demo-edits", dest="demo_edits", action="store_false",
                        help="skip the dry-run edit walkthrough (steps 4-8e) so no demo "
                             "offset/refit correction is committed — use for real deliverables")
    args = parser.parse_args(argv)

    return run_dry_run(
        out_dir=Path(args.out_dir),
        n_frames=args.frames,
        n_subjects=args.subjects,
        export_format=args.format,
        clip_path=args.clip,
        detector=args.detector,
        tracker=args.tracker,
        calibrator=args.calibrator,
        pose=args.pose,
        ball=args.ball,
        env=args.env,
        avatar=args.avatar,
        render=args.render,
        export=args.export,
        observer=args.observer,
        viewsynth=args.viewsynth,
        amplify_views=args.amplify_views,
        amplify_deviation=args.amplify_deviation,
        device=args.device,
        detector_weights=args.detector_weights,
        detector_classes=args.detector_classes,
        pose_backend=args.pose_backend,
        ball_backend=args.ball_backend,
        calibrator_backend=args.calibrator_backend,
        tracker_backend=args.tracker_backend,
        avatar_backend=args.avatar_backend,
        occlusion_backend=args.occlusion_backend,
        motion_prior=args.motion_prior,
        camera_carry=args.camera_carry,
        stitch=args.stitch,
        shot_guard=args.shot_guard,
        kit_split=args.kit_split,
        coherence=args.coherence,
        physics=args.physics,
        physics_profile=args.physics_profile,
        physics_config=args.physics_config,
        player_profiles_dir=args.player_profiles_dir,
        player_priors=args.player_priors,
        auto_tune=args.auto_tune,
        ball_id=args.ball_id,
        identity=args.identity,
        demo_edits=args.demo_edits,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
