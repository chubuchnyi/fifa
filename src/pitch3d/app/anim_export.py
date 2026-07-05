"""Anim export use-case: canonical scene JSON → Blender-ready animation directory.

The still-image cousin (scripts/smplx_export_meshes.py) poses ONE frame from the bodies-only
`smplx_npz` export. This one reads the **full canonical scene** (`--format json`, which alone
carries the ball) and forwards every subject through SMPL-X across *all* frames — the input
half of the deliverable video path (`scripts/blender_animate.py` renders the output half).

Artifacts written to ``--out`` (each validated + recorded in ``manifest.json``, the versioned
contract in :mod:`pitch3d.adapters.blender.anim_contract` — the renderer refuses anything else):

- ``anim_subject_<track_id>.npz`` — verts (T,V,3) world z-up, faces, kit ``color``, frames,
  entry/exit ``alpha``; plus measured per-vertex texture and jersey-number plate when available.
- ``ball.npz`` — resolved ball track (the correction stack applies here, like the exporters).
- ``cameras.npz`` — the VIRTUAL OPERATOR (:mod:`pitch3d.core.scene.cameras`): fixed mounts
  inside the stadium bowl with per-frame look-at + fov that pan/zoom with the action. The
  renderer aims its cameras from this file instead of deriving static ones from scene bbox
  (which framed the whole bowl from outside — the failed v2 framing).
- ``pitch.npz`` / ``stadium.npz`` / ``lighting.npz`` — measured markings+goals; crowd bowl and
  floodlight colour (both gated on the source clip: nothing measured, nothing invented).

CLI flags override env, env overrides the repo-root ``.env`` (see .env.example):

  --scene         PITCH3D_SCENE_JSON     canonical scene JSON (pipeline --format json export)
  --out           PITCH3D_ANIM_OUT       output dir
  --smplx-models  PITCH3D_SMPLX_MODELS   dir containing smplx/SMPLX_NEUTRAL.npz
  --source-video  PITCH3D_STADIUM_VIDEO  broadcast clip driving every MEASURED appearance
  --fade-frames   PITCH3D_FADE_FRAMES    entry/exit opacity ramp (0 = opaque)
  --canonical-up  PITCH3D_CANONICAL_UP   fake/canonical exports are y-up, not camera-frame

Run: ``.venv/bin/python -m pitch3d.app.anim_export`` (or the scripts/anim_export.py shim).
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any

import numpy as np

from pitch3d.adapters.blender.anim_contract import MANIFEST_NAME, SCHEMA_VERSION, write_manifest
from pitch3d.adapters.io.frames import resolve_source_path
from pitch3d.adapters.render.overlay import appearance_alpha
from pitch3d.core.correction.engine import resolve_ball, resolve_subject_motion
from pitch3d.core.scene.cameras import plan_virtual_cameras
from pitch3d.core.scene.pitch import goal_frame_geometry, pitch_line_ribbons
from pitch3d.core.scene.serialization import load_scene
from pitch3d.core.scene.stadium import (
    adboard_loop_uvs,
    adboard_ring_geometry,
    bowl_tile_loop_uvs,
    fill_holes_by_copy,
    stadium_bowl_geometry,
)
from pitch3d.env import load_env

STADIUM_REPEAT_AROUND = 40.0  # tile mode: crowd-tile copies laid around the loop (mirror-tiled)
STADIUM_REPEAT_UP = 4.0       # tile mode: copies up the rake
# Quilt (width, height): sized so ONE quilt texel ≈ one 720p screen pixel on the far stand.
# Measured 2026-07-04: at 8192x512 the broadcast framing magnified the quilt ~1.75x — the
# tile's genuine 4 px per-fan grain smeared into 7 px marble (clip: 2.7 px at 720p-equivalent).
CROWD_QUILT_SIZE = (16384, 1024)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    env = os.environ
    p = argparse.ArgumentParser(
        prog="pitch3d.app.anim_export",
        description="Forward a canonical scene through SMPL-X into a Blender-ready npz dir.",
    )
    p.add_argument("--scene", default=env.get("PITCH3D_SCENE_JSON", "out/anim/export/scene.json"))
    p.add_argument("--out", default=env.get("PITCH3D_ANIM_OUT", "out/anim/mesh"))
    p.add_argument("--smplx-models", default=env.get("PITCH3D_SMPLX_MODELS", "SMPL-X/models"))
    p.add_argument("--source-video", default=env.get("PITCH3D_STADIUM_VIDEO", ""))
    p.add_argument(
        "--crowd-mode",
        choices=("quilt", "tile"),
        default=env.get("PITCH3D_CROWD_MODE", "quilt"),
        help="stands texture: one large non-repeating quilt (auto default) or the legacy "
        "small-tile mirror repeat",
    )
    p.add_argument(
        "--crowd-seed", type=int, default=int(env.get("PITCH3D_CROWD_SEED", "0"))
    )
    p.add_argument(
        "--crowd-structure", type=int,
        default=int(env.get("PITCH3D_CROWD_STRUCTURE", "1")),
        help="overlay tier walkway/railing/aisles/top-fade on the quilt (0 = raw crowd)",
    )
    p.add_argument(
        "--crowd-fan-scale", type=float,
        default=float(env.get("PITCH3D_CROWD_FAN_SCALE", "1.0")),
        help="quilt px per measured-tile px (1.0 = native clip grain)",
    )
    p.add_argument(
        "--crowd-contrast", type=float,
        default=float(env.get("PITCH3D_CROWD_CONTRAST", "1.0")),
        help="fans-vs-background luma contrast about the quilt median (1.0 = as measured)",
    )
    p.add_argument(
        "--board-height", type=float,
        default=float(env.get("PITCH3D_BOARD_HEIGHT", "1.0")),
        help="LED ad-board ring height in metres (0 disables the ring)",
    )
    p.add_argument(
        "--board-offset", type=float,
        default=float(env.get("PITCH3D_BOARD_OFFSET", "5.0")),
        help="ad-board distance outside the touch/goal lines, metres",
    )
    p.add_argument("--fade-frames", type=int, default=int(env.get("PITCH3D_FADE_FRAMES", "4")))
    p.add_argument(
        "--canonical-up",
        action=argparse.BooleanOptionalAction,
        default=env.get("PITCH3D_CANONICAL_UP", "0") == "1",
    )
    return p.parse_args(argv)


def _rotation(canonical_up: bool) -> np.ndarray:
    """SMPL-X output frame → our z-up world.

    Same orientation gotcha as smplx_export_meshes.py: real SMPLest-X output is camera-frame
    (y-down) → new = [x, z, -y]; a fake/canonical export needs the plain y-up → z-up map.
    """
    if canonical_up:
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
    return np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)


def _purge_stale(out_dir: str) -> None:
    """Make the mesh dir reflect EXACTLY this scene.

    The pod-side out dir lives on the persistent volume and is reused across runs, so a track
    that existed only in a PRIOR run (e.g. anim_subject_22.npz) would otherwise linger and be
    rendered as a phantom body. The manifest goes too: a crashed export must leave no valid
    manifest behind, so the renderer refuses the half-written directory.
    """
    stale = glob.glob(os.path.join(out_dir, "anim_subject_*.npz")) + [
        os.path.join(out_dir, name)
        for name in ("ball.npz", "pitch.npz", "stadium.npz", "boards.npz", "lighting.npz",
                     "cameras.npz", MANIFEST_NAME)
    ]
    for path in stale:
        if os.path.exists(path):
            os.remove(path)


def _boost_rgb(rgb, sat: float, val: float) -> np.ndarray:
    """Saturation/value gain in HSV space, clipped to [0, 1]."""
    import colorsys

    r, g, b = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    return np.asarray(
        colorsys.hsv_to_rgb(h, min(s * sat, 1.0), min(v * val, 1.0)), dtype=np.float32
    )


#: Fallback skin albedo when a team's skin was never measured (broadcast-distance tan).
_TAN_SKIN_RGB = (0.70, 0.52, 0.38)


def _env_rgb(name: str) -> np.ndarray | None:
    """Parse env var ``name`` as ``r,g,b`` floats in [0,1]; None when unset/empty."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    r, g, b = (float(x) for x in raw.replace(",", " ").split())
    return np.asarray((r, g, b), dtype=np.float32)


def _rgb_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RGB→(hue deg, sat, val) for float samples in [0,1]."""
    rgb = np.asarray(rgb, dtype=float).reshape(-1, 3)
    mx, mn = rgb.max(1), rgb.min(1)
    d = mx - mn
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    h = np.zeros(len(rgb))
    nz = d > 1e-12
    for i, (num, off) in enumerate(((g - b, 0.0), (b - r, 2.0), (r - g, 4.0))):
        pick = nz & (rgb.argmax(1) == i)
        h[pick] = (num[pick] / d[pick] + off) * 60.0
    s = np.divide(d, mx, out=np.zeros_like(d), where=mx > 1e-12)
    return h % 360.0, s, mx


def _zone_color_estimate(samples: np.ndarray) -> np.ndarray | None:
    """Robust kit colour from mis-projection-polluted vertex samples (None = too few clean).

    A broadcast-distance player is 20–40 px tall, so a couple of px of camera/mesh error land
    many leg vertices on the background — measured 2026-07-04 on the target clip: raw pooled
    medians were grass-green for EVERY zone of BOTH teams, and after dropping grass they were
    line/LED white. Plain medians can't survive a >50% pollutant, so: (1) drop grass (the
    stripe-metric gate, uint8-HSV H 30–70 ⇒ 60–140°, S>60/255, V>50/255); (2) if enough of the
    remainder is saturated, the kit is coloured — take the hue-histogram MODE and the median of
    its ±25° neighbourhood (the measure_team_hsv trick: white lines/glare are desaturated and
    drop out); (3) otherwise the kit itself is neutral (white socks) — plain median, where
    conflating it with white lines is harmless.
    """
    rgb = np.asarray(samples, dtype=float).reshape(-1, 3)
    if rgb.size:
        h, s, v = _rgb_hsv(rgb)
        grass = (h >= 60.0) & (h <= 140.0) & (s > 60.0 / 255.0) & (v > 50.0 / 255.0)
        rgb, h, s = rgb[~grass], h[~grass], s[~grass]
    if rgb.shape[0] < 40:
        return None
    sat = s >= 0.25
    if float(sat.mean()) >= 0.3:
        hh = h[sat]
        hist, edges = np.histogram(hh, bins=36, range=(0.0, 360.0))
        peak = float(edges[int(hist.argmax())]) + 5.0
        near = np.abs((hh - peak + 180.0) % 360.0 - 180.0) <= 25.0
        return np.median(rgb[np.flatnonzero(sat)[near]], axis=0)
    return np.median(rgb, axis=0)


def _team_zone_medians(posed: list[dict], zones: np.ndarray) -> dict[str, dict[int, Any]]:
    """Median measured RGB per (team, garment zone), pooled across every subject of the team.

    One player's shorts may be occluded or a handful of noisy pixels, but a real team wears ONE
    kit — pooling ~10 players' measured vertices through the pollution-robust
    :func:`_zone_color_estimate` gives a stable estimate AND the faithful one. Returns
    ``{team_key: {zone: (3,) rgb | None}}``; None where a zone never collected 40+ clean
    samples (the composer falls back down the kit).
    """
    from pitch3d.adapters.models.avatar import KIT_SHORTS, KIT_SKIN, KIT_SOCKS

    out: dict[str, dict[int, Any]] = {}
    for team in {str(d["subj"].team_id or "") for d in posed}:
        per_zone: dict[int, Any] = {}
        for zone in (KIT_SHORTS, KIT_SOCKS, KIT_SKIN):
            samples = [
                d["vcolor"][d["measured"] & (zones == zone)]
                for d in posed
                if str(d["subj"].team_id or "") == team and d["measured"] is not None
            ]
            pooled = np.concatenate(samples, axis=0) if samples else np.zeros((0, 3))
            per_zone[zone] = _zone_color_estimate(pooled)
        out[team] = per_zone
    return out


def _compose_kit_vcolor(
    zones: np.ndarray,
    shirt_rgb: np.ndarray,
    medians: dict[str, dict[int, Any]],
    team_key: str,
    kit_sat: float,
    kit_val: float,
) -> np.ndarray:
    """Zone-flat kit for one subject: measured medians dressed to presentation brightness.

    Shirt = the boosted team colour (the identity anchor downstream v2v pins on); shorts/socks =
    boosted per-team measured medians, falling back DOWN the kit (shorts→shirt, socks→shorts —
    one-colour kits are the common case); skin = value-lifted measured median (the night grade
    crushes it) else a tan prior; boots near-black. Manual overrides (auto+manual, R-6):
    ``PITCH3D_SHORTS_RGB_<team>`` / ``PITCH3D_SOCKS_RGB_<team>`` / ``PITCH3D_SKIN_RGB``.
    """
    from pitch3d.adapters.models.avatar import (
        KIT_BOOTS,
        KIT_SHIRT,
        KIT_SHORTS,
        KIT_SKIN,
        KIT_SOCKS,
    )

    suffix = f"_{team_key}" if team_key else ""
    med = medians.get(team_key, {})
    shorts = _env_rgb(f"PITCH3D_SHORTS_RGB{suffix}")
    if shorts is None:
        m = med.get(KIT_SHORTS)
        shorts = (
            _boost_rgb(m, kit_sat, kit_val) if m is not None else np.asarray(shirt_rgb, np.float32)
        )
    socks = _env_rgb(f"PITCH3D_SOCKS_RGB{suffix}")
    if socks is None:
        m = med.get(KIT_SOCKS)
        socks = _boost_rgb(m, kit_sat, kit_val) if m is not None else shorts
    skin = _env_rgb("PITCH3D_SKIN_RGB")
    if skin is None:
        m = med.get(KIT_SKIN)
        skin = _boost_rgb(m, 1.0, kit_val) if m is not None else np.asarray(_TAN_SKIN_RGB, np.float32)
    by_zone = {
        KIT_SKIN: skin,
        KIT_SHIRT: np.asarray(shirt_rgb, dtype=np.float32),
        KIT_SHORTS: shorts,
        KIT_SOCKS: socks,
        KIT_BOOTS: np.asarray((0.05, 0.05, 0.05), np.float32),
    }
    vcolor = np.empty((zones.shape[0], 3), dtype=np.float32)
    for zone, rgb in by_zone.items():
        vcolor[zones == zone] = rgb
    return vcolor


def _export_subjects(
    scene,
    *,
    models_dir: str,
    out_dir: str,
    rot: np.ndarray,
    fade_frames: int,
    source_video: str,
    source_ok: bool,
    entries: dict[str, list[str]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pose every subject through SMPL-X → anim_subject_*.npz; return (frames, pelvis) tracks."""
    import matplotlib
    import smplx
    import torch

    from pitch3d.adapters.models.avatar import (
        KIT_SHORTS,
        KIT_SKIN,
        KIT_SOCKS,
        measured_texture_from_clip,
        smplx_kit_zones,
    )

    # Rendered clip span = union of every present frame (subjects + ball) — the same range
    # blender_animate.py iterates. A subject touching this span's edge was clipped by the
    # window, not a genuine entry/exit, so it is NOT faded. Frames are resolve-invariant (the
    # engine never inserts rows), so proposal frames are exact here.
    present = [np.asarray(s.proposal.pose.frames, dtype=int) for s in scene.subjects]
    if scene.ball is not None and np.asarray(scene.ball.frames).size:
        present.append(np.asarray(scene.ball.frames, dtype=int))
    clip_first = int(min(int(f[0]) for f in present if f.size))
    clip_last = int(max(int(f[-1]) for f in present if f.size))

    # Team colours when the tracker classified them (team A vs B reads clearly in the render);
    # otherwise fall back to a distinct per-subject palette.
    team_color = {t.id: t.color_rgb for t in scene.teams if t.color_rgb is not None}
    palette = matplotlib.colormaps["tab10"](np.linspace(0, 1, 10))[:, :3]

    # Kit-colour boost (v2, 2026-07-03 §6): team colours are MEASURED off a floodlit-night clip,
    # so they come out dark/desaturated — and at 7-11 % texture coverage ~90 % of vertices carry
    # this flat kit colour, which is why bodies read grey in the render AND why v2v conditioning
    # cannot lock team identity. Boost saturation/value at this presentation layer only (the
    # measured scene value stays untouched, R-6). Override PITCH3D_KIT_SAT / PITCH3D_KIT_VAL;
    # 1.0 1.0 disables.
    kit_sat = float(os.environ.get("PITCH3D_KIT_SAT", "1.6"))
    kit_val = float(os.environ.get("PITCH3D_KIT_VAL", "1.4"))
    if kit_sat != 1.0 or kit_val != 1.0:
        for t_id, rgb in sorted(team_color.items()):
            print(f"kit boost team {t_id}: {np.round(np.asarray(rgb), 3)} -> "
                  f"{np.round(_boost_rgb(rgb, kit_sat, kit_val), 3)} "
                  f"(sat x{kit_sat}, val x{kit_val})")

    # Kit zones (v2 face/limb lever, 2026-07-04 §6): batch #1 rendered every player as a
    # whole-body team-colour "morphsuit" — the measured texture is too dark/noisy at broadcast
    # distance to carry the kit LAYOUT, and the flat fallback has none. The layout now comes from
    # the body model (smplx_kit_zones), the colours stay measured (per-team zone medians) — which
    # needs TEAM-pooled statistics, hence two passes: pose+sample everyone, then compose+write.
    # PITCH3D_KIT_ZONES=0 restores the old whole-body fill.
    kit_zones_on = os.environ.get("PITCH3D_KIT_ZONES", "1") != "0"

    zones: np.ndarray | None = None
    posed: list[dict[str, Any]] = []
    for i, subj in enumerate(scene.subjects):
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        betas = np.asarray(motion.shape.betas, dtype=np.float32)
        frames = np.asarray(motion.pose.frames)
        n_frames = int(frames.shape[0])
        model = smplx.create(
            models_dir,
            model_type="smplx",
            gender="neutral",
            num_betas=int(betas.shape[0]),
            use_pca=False,
            flat_hand_mean=True,
            batch_size=n_frames,
        )
        with torch.no_grad():
            out = model(
                betas=torch.tensor(np.tile(betas[None], (n_frames, 1)), dtype=torch.float32),
                global_orient=torch.tensor(motion.pose.global_orient, dtype=torch.float32),
                body_pose=torch.tensor(
                    np.asarray(motion.pose.body_pose).reshape(n_frames, -1), dtype=torch.float32
                ),
            )
        if zones is None:
            zones = smplx_kit_zones(
                model.lbs_weights.numpy(), model.J_regressor.numpy(), model.v_template.numpy()
            )
        transl = np.asarray(motion.pose.transl, dtype=np.float32)  # (T,3) z-up world
        verts = out.vertices.numpy() @ rot.T + transl[:, None, :]  # (T,V,3)
        color = _boost_rgb(
            team_color.get(subj.team_id, palette[i % 10]), kit_sat, kit_val
        )
        alpha = appearance_alpha(frames, clip_first, clip_last, fade_frames)  # (T,) in [0,1]

        # Measured per-vertex body texture (M2-8b): sample each player's real broadcast pixels
        # onto its posed mesh through the solved camera. Kept RAW here (no fallback fill) — the
        # zone medians must pool only genuinely observed pixels; the measured flag is the honest
        # R-6 channel either way.
        vcolor = None
        measured = None
        if source_ok:
            vcolor, measured = measured_texture_from_clip(
                verts, model.faces, scene.camera, [int(f) for f in frames], source_video
            )
        joints = None
        if subj.jersey_number is not None:
            joints = out.joints.numpy() @ rot.T + transl[:, None, :]  # (T, J, 3) z-up world
        posed.append(
            dict(
                subj=subj,
                verts=verts,
                faces=model.faces,
                frames=frames,
                alpha=alpha,
                color=color,
                vcolor=vcolor,
                measured=measured,
                joints=joints,
                transl=transl,
            )
        )

    zone_medians: dict[str, dict[int, Any]] = {}
    if kit_zones_on and zones is not None:
        zone_medians = _team_zone_medians(posed, zones)
        for team, med in sorted(zone_medians.items()):
            pretty = {
                name: (np.round(med[z], 3).tolist() if med[z] is not None else None)
                for name, z in (("shorts", KIT_SHORTS), ("socks", KIT_SOCKS), ("skin", KIT_SKIN))
            }
            print(f"kit zones team {team or '?'}: measured medians {pretty}")

    tracks: list[tuple[np.ndarray, np.ndarray]] = []
    for d in posed:
        subj = d["subj"]
        verts, frames, alpha, color = d["verts"], d["frames"], d["alpha"], d["color"]
        transl, vcolor, measured = d["transl"], d["vcolor"], d["measured"]
        n_frames = int(frames.shape[0])
        if kit_zones_on and zones is not None:
            vcolor = _compose_kit_vcolor(
                zones, color, zone_medians, str(subj.team_id or ""), kit_sat, kit_val
            )
            if measured is None:
                measured = np.zeros(zones.shape[0], dtype=bool)
        elif vcolor is not None:
            vcolor[~measured] = color

        # Shirt number plate (#numbers, v1): a per-frame upper-back anchor + outward "back"
        # direction so the renderer can place a number without any SMPL-X knowledge. The back
        # normal is the *posterior* horizontal direction, -(facing), facing = eyes−head
        # (SMPL-X joints 23/24 are the eyeballs, in front of the head joint 15).
        num_extra: dict[str, Any] = {}
        if subj.jersey_number is not None:
            joints = d["joints"]
            spine3, neck, head = joints[:, 9], joints[:, 12], joints[:, 15]
            facing = 0.5 * (joints[:, 23] + joints[:, 24]) - head
            facing[:, 2] = 0.0
            fn = np.linalg.norm(facing, axis=1, keepdims=True)
            facing = np.divide(facing, fn, out=np.zeros_like(facing), where=fn > 1e-6)
            back = -facing  # (T,3) unit posterior horizontal; zero rows where degenerate
            # Mid-upper-back height (spine3-weighted so it sits between the shoulder blades)
            # pushed ~0.19 m out along the posterior normal: smaller offsets leave the plate
            # buried in the curved back mesh, 0.19 floats the digits a few cm proud.
            anchor = 0.62 * spine3 + 0.38 * neck + 0.19 * back
            lum = float(0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2])
            num_rgb = np.array([0.04, 0.04, 0.07] if lum > 0.5 else [0.97, 0.97, 0.97], np.float32)
            num_extra = dict(
                jersey_number=np.asarray(int(subj.jersey_number), dtype=np.int64),
                back_anchor=anchor.astype(np.float32),
                back_dir=back.astype(np.float32),
                number_rgb=num_rgb,
            )

        tex_extra: dict[str, Any] = {}
        if vcolor is not None and measured is not None:
            tex_extra = dict(vcolor=vcolor.astype(np.float32), measured=measured.astype(np.uint8))
        if zones is not None:
            # Garment zone per vertex — lets the renderer make its team mask shirt-aware so the
            # generative repaint keys on shirts, not whole bodies.
            tex_extra["zones"] = zones.astype(np.uint8)
        fname = f"anim_subject_{subj.track_id}.npz"
        dst = os.path.join(out_dir, fname)
        np.savez(
            dst,
            verts=verts.astype(np.float32),
            faces=d["faces"].astype(np.int32),
            color=color,
            frames=frames.astype(np.int64),
            alpha=alpha.astype(np.float32),
            # Team id ("A"/"B", "" when untracked) — lets the renderer draw per-team AOV masks
            # (--team-mask) that downstream hue re-pinning keys on after generative finishing.
            team=str(subj.team_id or ""),
            **num_extra,
            **tex_extra,
        )
        entries[fname] = sorted(
            ["verts", "faces", "color", "frames", "alpha", "team", *num_extra, *tex_extra]
        )
        tracks.append((frames.astype(np.int64), transl.astype(np.float64)))

        span = float(np.linalg.norm(transl.max(0) - transl.min(0)))
        num_msg = ""
        if num_extra:
            bz = float(np.abs(num_extra["back_dir"][:, 2]).mean())  # ~0 ⇒ horizontal (sane)
            num_msg = f" number={int(num_extra['jersey_number'])} back_dir|z|~{bz:.2f}"
        tex_msg = f" tex={measured.mean() * 100:.0f}%" if measured is not None else ""
        print(
            f"subject_{subj.track_id}: team={subj.team_id} frames={n_frames} "
            f"transl_span={span:.2f}m{num_msg}{tex_msg} -> {fname}"
        )
    return tracks


def _export_ball(scene, out_dir: str, entries: dict[str, list[str]]):
    if scene.ball is None:
        print("ball: none in scene (skipping ball.npz)")
        return None
    ball = resolve_ball(scene.ball, scene.corrections_for(None))
    dst = os.path.join(out_dir, "ball.npz")
    np.savez(
        dst,
        frames=np.asarray(ball.frames, dtype=np.int64),
        positions_3d=np.asarray(ball.positions_3d, dtype=np.float32),
        height_confidence=np.asarray(ball.height_confidence, dtype=np.float32),
    )
    entries["ball.npz"] = ["frames", "height_confidence", "positions_3d"]
    print(f"ball: {int(np.asarray(ball.frames).shape[0])} frames -> ball.npz")
    return ball


def _export_cameras(
    subject_tracks: list[tuple[np.ndarray, np.ndarray]],
    ball,
    out_dir: str,
    entries: dict[str, list[str]],
) -> None:
    """Plan the virtual operator over the union of present frames → cameras.npz."""
    frame_sets = [frames for frames, _ in subject_tracks]
    ball_frames = np.zeros(0, dtype=np.int64)
    if ball is not None:
        ball_frames = np.asarray(ball.frames, dtype=np.int64)
        if ball_frames.size:
            frame_sets = [*frame_sets, ball_frames]
    union = np.unique(np.concatenate(frame_sets)).astype(np.int64)
    roots = np.full((union.shape[0], len(subject_tracks), 3), np.nan)
    for i, (frames, pelvis) in enumerate(subject_tracks):
        roots[np.searchsorted(union, frames), i] = pelvis
    ball_arr = None
    if ball_frames.size:
        ball_arr = np.full((union.shape[0], 3), np.nan)
        ball_arr[np.searchsorted(union, ball_frames)] = np.asarray(ball.positions_3d, dtype=float)
    planned = plan_virtual_cameras(roots, ball_arr, union)
    data: dict[str, Any] = {"names": np.array(list(planned)), "frames": union}
    for name, track in planned.items():
        data[f"{name}_pos"] = track.position.astype(np.float32)
        data[f"{name}_look"] = track.look_at.astype(np.float32)
        data[f"{name}_fov_deg"] = track.fov_x_deg.astype(np.float32)
    dst = os.path.join(out_dir, "cameras.npz")
    np.savez(dst, **data)
    entries["cameras.npz"] = sorted(data)
    bc = planned["broadcast"]
    print(
        f"cameras: {','.join(planned)} over {union.shape[0]} frames; "
        f"broadcast fov {bc.fov_x_deg.min():.1f}-{bc.fov_x_deg.max():.1f}deg -> cameras.npz"
    )


def _export_pitch(scene, out_dir: str, entries: dict[str, list[str]]) -> None:
    # Measured pitch markings + goal frames as world geometry (#205). The dims are the field's
    # own (the homography anchors bodies to THIS template), so lines/goals line up with the
    # subjects once placement is correct.
    dims = scene.field.dimensions
    pv, pf = pitch_line_ribbons(dims)
    gv, gf = goal_frame_geometry(dims)
    dst = os.path.join(out_dir, "pitch.npz")
    np.savez(
        dst,
        pitch_verts=pv.astype(np.float32),
        pitch_faces=pf.astype(np.int32),
        goal_verts=gv.astype(np.float32),
        goal_faces=gf.astype(np.int32),
    )
    entries["pitch.npz"] = ["goal_faces", "goal_verts", "pitch_faces", "pitch_verts"]
    print(
        f"pitch: {pf.shape[0]} line-tris + {gf.shape[0]} goal-tris "
        f"({dims.length:g}x{dims.width:g} m) -> pitch.npz"
    )


def _export_stadium(
    scene,
    out_dir: str,
    source_video: str,
    source_ok: bool,
    entries: dict[str, list[str]],
    *,
    crowd_mode: str = "quilt",
    crowd_seed: int = 0,
    crowd_structure: bool = True,
    crowd_fan_scale: float = 1.0,
    crowd_contrast: float = 1.0,
) -> None:
    # Hybrid stadium backdrop (M2): procedural bowl + REAL appearance from THIS clip — a
    # *tinted mosaic* (crisp crowd texture x per-vertex measured tint, mirror copy-fill for the
    # unseen side). No clip → we cannot measure crowd colour, so omit the bowl, not invent it.
    # AUTO default texture is the QUILT (one large non-repeating stitch of measured crowd crops,
    # uint8, continuous 0-1 unwrap, REPEAT) — repeating the small tile reads as a kaleidoscope
    # once sharpened. MANUAL: --crowd-mode tile restores the legacy 40x4 mirror mosaic,
    # --crowd-seed re-rolls the quilt.
    if not source_ok:
        print("stadium: no source clip (--source-video / PITCH3D_STADIUM_VIDEO), skipping "
              "stadium.npz")
        return
    from pitch3d.adapters.render.stadium_backdrop import (
        assemble_crowd_quilt,
        bake_backdrop_colors,
        extract_crowd_tile,
    )

    dims = scene.field.dimensions
    sv, sf, sp = stadium_bowl_geometry(dims)
    scolors, scov = bake_backdrop_colors(scene.camera, sv, source_video)
    sfilled, _ = fill_holes_by_copy(sv, scolors, scov)
    stile = extract_crowd_tile(scene.camera, sv, sp, scov, source_video)
    tile_gain = 1.0
    if crowd_mode == "quilt":
        qw, qh = CROWD_QUILT_SIZE
        quilt = assemble_crowd_quilt(
            stile, width=qw, height=qh, seed=crowd_seed, fan_scale=crowd_fan_scale,
            contrast=crowd_contrast,
        )
        if crowd_structure:
            from pitch3d.adapters.render.stadium_backdrop import apply_stand_structure

            raw_mean = float(quilt.mean())
            quilt = apply_stand_structure(quilt)
            # The renderer normalises the tile to unit mean (scale-invariant), so darkening
            # part of the texture would silently BRIGHTEN the seated rows past the tuned
            # emission. tile_gain hands the renderer the mean drop to compensate with.
            tile_gain = float(quilt.mean()) / max(raw_mean, 1e-6)
        tex = (quilt * 255.0 + 0.5).astype(np.uint8)
        suv = bowl_tile_loop_uvs(sf, sp, repeat_around=1.0, repeat_up=1.0)
        tile_ext = "REPEAT"
    else:
        tex = stile.astype(np.float32)
        suv = bowl_tile_loop_uvs(
            sf, sp, repeat_around=STADIUM_REPEAT_AROUND, repeat_up=STADIUM_REPEAT_UP
        )
        tile_ext = "MIRROR"
    dst = os.path.join(out_dir, "stadium.npz")
    np.savez(
        dst,
        verts=sv.astype(np.float32),
        faces=sf.astype(np.int32),
        colors=sfilled.astype(np.float32),
        uv=suv.astype(np.float32),
        tile=tex,
        tile_ext=tile_ext,
        tile_gain=np.float32(tile_gain),
    )
    entries["stadium.npz"] = [
        "colors", "faces", "tile", "tile_ext", "tile_gain", "uv", "verts",
    ]
    print(
        f"stadium: {sv.shape[0]} verts {sf.shape[0]} tris; covered "
        f"{int(scov.sum())}/{scov.size} ({scov.mean() * 100:.0f}%); "
        f"{crowd_mode} {tex.shape[1]}x{tex.shape[0]} (ext {tile_ext}) -> stadium.npz"
    )


def _export_boards(
    scene, out_dir: str, entries: dict[str, list[str]], *, height: float, offset: float,
    source_video: str = "", source_ok: bool = False,
) -> None:
    # Broadcast perimeter furniture (v2 lever, 2026-07-03): grass → white LED boards → dark
    # walkway → crowd is the night-broadcast silhouette the finisher knows; without it the
    # grass runs straight into the crowd wall and Wan paints mush at the boundary. The RING is
    # a geometric prior (needs no clip; --board-height 0 / PITCH3D_BOARD_HEIGHT=0 disables it);
    # with a clip the sponsor LED strip is MEASURED off it and wrapped around the ring
    # (2026-07-04), scaled so one texture repeat covers strip-aspect × board-height metres.
    if height <= 0.0:
        print("boards: --board-height 0, skipping boards.npz")
        return
    walkway = _env_rgb("PITCH3D_WALKWAY_RGB")  # manual override of the dark-grey default
    ring_kwargs = {} if walkway is None else {"gap_color": tuple(float(c) for c in walkway)}
    bv, bf, bc = adboard_ring_geometry(
        scene.field.dimensions, offset=offset, height=height, **ring_kwargs
    )
    payload: dict[str, Any] = {
        "verts": bv.astype(np.float32),
        "faces": bf.astype(np.int32),
        "colors": bc.astype(np.float32),
    }
    keys = ["colors", "faces", "verts"]
    note = "flat prior"
    if source_ok:
        from pitch3d.adapters.render.stadium_backdrop import extract_board_strip, strip_emission

        band = bv[: bv.shape[0] // 2]  # board band verts; walkway band follows
        frame_env = os.environ.get("PITCH3D_BOARD_FRAME", "")
        try:
            strip, pick = extract_board_strip(
                scene.camera, band, source_video,
                frame_override=int(frame_env) if frame_env else None,
            )
            emission = strip_emission(strip)
            bottoms = band.reshape(-1, 2, 3)[:, 0, :]
            hops = np.diff(np.vstack([bottoms, bottoms[:1]]), axis=0)
            perim = float(np.linalg.norm(hops, axis=1).sum())
            aspect = strip.shape[1] / strip.shape[0]
            repeat = max(1.0, round(perim / (aspect * height)))
            payload.update(
                tile=strip.astype(np.float32),
                uv=adboard_loop_uvs(bottoms.shape[0], repeat_around=repeat).astype(np.float32),
                tile_ext="REPEAT",
                emission=np.float32(emission),
            )
            keys += ["emission", "tile", "tile_ext", "uv"]
            note = (f"measured LED strip {strip.shape[1]}x{strip.shape[0]} x{repeat:.0f} around "
                    f"(clip frame {pick}, emission x{emission:.2f})")
        except ValueError as exc:
            note = f"flat prior (strip failed: {exc})"
    np.savez(os.path.join(out_dir, "boards.npz"), **payload)
    entries["boards.npz"] = keys
    print(f"boards: ring {bv.shape[0]} verts {bf.shape[0]} tris "
          f"(h={height} m at +{offset} m; {note}) -> boards.npz")


def _export_lighting(
    scene, out_dir: str, source_video: str, source_ok: bool, entries: dict[str, list[str]]
) -> None:
    # Lighting from the clip (v2 lever 3, AUTO baseline): the target is a floodlit NIGHT match,
    # so "light from the clip" is the floodlights' measured COLOUR (white-patch illuminant off
    # bright near-neutral surfaces), not a daytime sun. --light-* flags on the renderer are the
    # manual override half.
    if not source_ok:
        print("lighting: no source clip (--source-video / PITCH3D_STADIUM_VIDEO), skipping "
              "lighting.npz")
        return
    from pitch3d.adapters.render.lighting import estimate_lighting_from_clip

    light = estimate_lighting_from_clip(source_video, scene.camera.frames)
    dst = os.path.join(out_dir, "lighting.npz")
    np.savez(dst, **light)
    entries["lighting.npz"] = sorted(light)
    lr = light["light_rgb"]
    print(
        f"lighting: floodlight rgb=[{lr[0]:.3f}, {lr[1]:.3f}, {lr[2]:.3f}] "
        f"(night model: {int(light['sun_count'])} soft suns) -> lighting.npz"
    )


def main(argv: list[str] | None = None) -> int:
    load_env()  # machine paths come from the repo-root .env, never hard-coded
    args = _parse_args(argv)
    source_ok = bool(args.source_video) and os.path.exists(resolve_source_path(args.source_video))
    os.makedirs(args.out, exist_ok=True)
    _purge_stale(args.out)

    scene = load_scene(args.scene)
    assert scene.subjects, f"no subjects in {args.scene}"

    entries: dict[str, list[str]] = {}
    subject_tracks = _export_subjects(
        scene,
        models_dir=args.smplx_models,
        out_dir=args.out,
        rot=_rotation(args.canonical_up),
        fade_frames=args.fade_frames,
        source_video=args.source_video,
        source_ok=source_ok,
        entries=entries,
    )
    ball = _export_ball(scene, args.out, entries)
    _export_cameras(subject_tracks, ball, args.out, entries)
    _export_pitch(scene, args.out, entries)
    _export_stadium(
        scene, args.out, args.source_video, source_ok, entries,
        crowd_mode=args.crowd_mode, crowd_seed=args.crowd_seed,
        crowd_structure=bool(args.crowd_structure),
        crowd_fan_scale=args.crowd_fan_scale,
        crowd_contrast=args.crowd_contrast,
    )
    _export_boards(
        scene, args.out, entries, height=args.board_height, offset=args.board_offset,
        source_video=args.source_video, source_ok=source_ok,
    )
    _export_lighting(scene, args.out, args.source_video, source_ok, entries)

    write_manifest(args.out, entries)
    print(f"manifest: {len(entries)} artifacts (schema v{SCHEMA_VERSION}) -> {MANIFEST_NAME}")
    print(f"ANIM_EXPORT_OK ({len(scene.subjects)} subjects -> {args.out})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
