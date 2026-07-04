#!/usr/bin/env python3
"""Shirt numbers for the CURRENT scene generation: read off the clip, pin into scene.json.

Track IDs are per-recon, so v1's manual reads (2026-06-28, #10/#25/#20/#12) died with that
scene. This tool makes re-assignment a ~10-min repeatable step per recon (R-6: only pin
numbers a human/LLM actually read; everyone else stays None):

  sheets  — per-subject contact sheets of upscaled torso crops from the source clip
            (same solved camera + 180-roll convention as the body-texture sampler)
  set     — pin track=number pairs into scene.json (raw-JSON patch) + provenance sidecar

Usage:
  python scripts/jersey_numbers.py sheets --scene out/.../scene.json \
      --clip samples/video/Colombia-1-0-Congo-DR1080p.mp4 --out out/backs
  python scripts/jersey_numbers.py set --scene out/.../scene.json 3=10 7=25 12=20
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pitch3d.core.correction.engine import resolve_subject_motion  # noqa: E402
from pitch3d.core.scene.projection import (  # noqa: E402
    camera_pose,
    project_world_points_with_depth,
)
from pitch3d.core.scene.serialization import load_scene  # noqa: E402

TILE_H = 380  # upscaled tile height, px — big enough to read 20-30 px source digits


def _even_rows(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def cmd_sheets(args: argparse.Namespace) -> int:
    import cv2
    from PIL import Image, ImageDraw

    from pitch3d.adapters.io.frames import iter_clip_frames

    scene = load_scene(args.scene)
    cam = scene.camera
    k = cam.intrinsics
    w, h = int(k.width), int(k.height)

    # (subject, raw frame) -> world pelvis; decode each wanted raw frame once
    wants: dict[int, list[tuple[int, np.ndarray]]] = {}
    order: list[int] = []
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        frames = np.asarray(motion.pose.frames, dtype=int)
        transl = np.asarray(motion.pose.transl, dtype=float)
        rows = _even_rows(frames.shape[0], args.per_subject)
        order.append(int(subj.track_id))
        for r in rows:
            wants.setdefault(int(frames[r]), []).append((int(subj.track_id), transl[r]))

    rot0, _ = camera_pose(cam, sorted(wants)[0])
    upside_down = float(-rot0[1, 2]) < 0.0

    tiles: dict[int, list[Image.Image]] = {t: [] for t in order}
    for idx, bgr in iter_clip_frames(args.clip, sorted(wants)):
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        if upside_down:
            bgr = cv2.rotate(bgr, cv2.ROTATE_180)
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        for track, pelvis in wants[int(idx)]:
            head = pelvis + np.array([0.0, 0.0, 1.85])
            uv, _z, vis = project_world_points_with_depth(
                cam, int(idx), np.stack([pelvis, head])
            )
            if not bool(vis.all()):
                continue
            (u0, v0), (u1, v1) = uv  # v1 = head (smaller v), v0 = feet-ish
            hh = abs(float(v0 - v1))
            if hh < args.min_px:  # too small to ever read a digit
                continue
            cx = 0.5 * (u0 + u1)
            top, bot = min(v0, v1) - 0.10 * hh, max(v0, v1) + 0.05 * hh
            half_w = 0.55 * hh
            x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
            y0, y1 = int(max(0, top)), int(min(h, bot))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            # The frame is in the solved camera's ROLLED convention (uv match it); rotate the
            # tile back upright for human reading.
            crop = Image.fromarray(rgb[y0:y1, x0:x1]).rotate(180)
            s = TILE_H / crop.height
            tile = crop.resize((max(1, int(crop.width * s)), TILE_H), Image.LANCZOS)
            d = ImageDraw.Draw(tile)
            d.rectangle((0, 0, 52, 16), fill=(0, 0, 0))
            d.text((3, 2), f"f{idx}", fill=(255, 255, 0))
            tiles[track].append(tile)

    os.makedirs(args.out, exist_ok=True)
    team = {
        s.track_id: (s.team_id or "?") for s in scene.subjects
    }
    made = 0
    for track in order:
        tt = tiles[track]
        if not tt:
            print(f"subject {track:>3} team {team[track]}: no usable crops (too small/offscreen)")
            continue
        sheet = Image.new("RGB", (sum(t.width for t in tt) + 4 * len(tt), TILE_H + 24), (15, 15, 15))
        d = ImageDraw.Draw(sheet)
        d.text((4, TILE_H + 4), f"subject {track}  team {team[track]}", fill=(255, 255, 0))
        x = 0
        for t in tt:
            sheet.paste(t, (x, 0))
            x += t.width + 4
        path = os.path.join(args.out, f"subject_{track:02d}.png")
        sheet.save(path)
        made += 1
        print(f"subject {track:>3} team {team[track]}: {len(tt)} crops -> {path}")
    print(f"sheets: {made}/{len(order)} subjects")
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    pins: dict[int, int] = {}
    for pair in args.pins:
        t, n = pair.split("=", 1)
        pins[int(t)] = int(n)

    with open(args.scene, encoding="utf-8") as fh:
        raw = json.load(fh)
    subs = raw["fields"]["subjects"]
    hit = []
    for s in subs:
        f = s.get("fields", s)
        t = int(f["track_id"])
        if t in pins:
            f["jersey_number"] = pins[t]
            hit.append(t)
    missing = sorted(set(pins) - set(hit))
    if missing:
        print(f"ERROR: tracks not in scene: {missing}", file=sys.stderr)
        return 1
    with open(args.scene, "w", encoding="utf-8") as fh:
        json.dump(raw, fh)
    sidecar = os.path.join(os.path.dirname(args.scene) or ".", "numbers.json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "read": "manual high-confidence (R-6: unread stay None)",
                "date": _dt.date.today().isoformat(),
                "assignments": {str(t): pins[t] for t in sorted(pins)},
            },
            fh,
            indent=2,
        )
    print(f"pinned {len(hit)} numbers into {args.scene}; provenance -> {sidecar}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sh = sub.add_parser("sheets")
    sh.add_argument("--scene", required=True)
    sh.add_argument("--clip", required=True)
    sh.add_argument("--out", default="out/backs")
    sh.add_argument("--per-subject", type=int, default=10)
    sh.add_argument("--min-px", type=float, default=45.0)
    sh.set_defaults(fn=cmd_sheets)
    st = sub.add_parser("set")
    st.add_argument("--scene", required=True)
    st.add_argument("pins", nargs="+", help="track=number pairs, e.g. 3=10 7=25")
    st.set_defaults(fn=cmd_set)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
