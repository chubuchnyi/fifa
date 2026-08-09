#!/usr/bin/env python3
"""Do our players move vertically like real ones? Measured against WorldPose GT.

The board has carried "largest root-Z excursion in a whole scene is 0.082 m against ~0.23 m for a
real player" for four plan revisions. That comparison is not sound: it puts a **60-frame scene's
maximum** against GT's **1032-frame median**. Excursion grows with window length, so the two
numbers are not the same statistic and the gap they imply is not the gap that exists.

This measures both sides in the same window, on the same statistic, and separates two defects that
the single number hides:

  1. the root Z is sometimes a **literal constant** — ``pose.py:334`` substitutes the nominal
     ``pelvis_height_m`` whenever the backend returns no ``pelvis_above_foot`` (SMPL-X FK), and
     nothing records that it did;
  2. where it does vary, it varies **too abruptly** — per-frame steps well outside what a real
     footballer's pelvis does, and on *measured* frames, so it is not gap-filling.

GT is WorldPose: 89 clips, 22 players each, ``transl (players, T, 3)`` with axis 2 up, the same
convention as ours.

    PYTHONPATH=src python scripts/bench_vertical_motion.py [scene.json ...]

CPU, seconds. Reads WorldPose/poses/*.npz (gitignored) and any scenes named on the command line.

Written 2026-08-09. Findings: docs/findings/vertical-motion-2026-08-09.md
"""

from __future__ import annotations

import glob
import sys

import numpy as np

GT_GLOB = "WorldPose/poses/*.npz"
DEFAULT_SCENES = (
    "out/cue/scene_off.json",
    "out/res_ab236/f236_res896.json",
)
UP = 2  # WorldPose transl axis 2 is the pelvis height; ours is world Z, same axis


def gt_tracks() -> list[np.ndarray]:
    paths = sorted(glob.glob(GT_GLOB))
    if not paths:
        print(f"No GT at {GT_GLOB} — WorldPose is gitignored (docs/models-dir.md).",
              file=sys.stderr)
        raise SystemExit(1)
    out = []
    for p in paths:
        for pz in np.load(p)["transl"][:, :, UP]:
            pz = pz[np.isfinite(pz)]
            if len(pz) >= 30:
                out.append(pz)
    return out


def table_excursion(gt: list[np.ndarray], windows=(60, 236, 1032)) -> dict[int, float]:
    """Excursion is window-dependent, which is what makes the headline comparison unsound."""
    print(f"\n== 1. Real players: root-Z excursion depends on the window ({len(gt)} tracks) ==")
    print(f"{'window (frames)':<18s}{'p05':>9s}{'median':>10s}{'p90':>9s}{'p99':>9s}")
    med = {}
    for w in windows:
        v = [np.ptp(t[s : s + w]) for t in gt for s in range(0, len(t) - w + 1, w)]
        v = np.asarray(v)
        med[w] = float(np.percentile(v, 50))
        print(
            f"{w:<18d}{np.percentile(v, 5):9.3f}{med[w]:10.3f}"
            f"{np.percentile(v, 90):9.3f}{np.percentile(v, 99):9.3f}"
        )
    print("  A 60-frame max against a 1032-frame median is a 7x window mismatch, not a defect.")
    return med


def table_step(gt: list[np.ndarray]) -> float:
    """Per-frame step separates real crouching (smooth) from noise (spiky)."""
    d = np.concatenate([np.abs(np.diff(t)) for t in gt])
    p90 = float(np.percentile(d, 90))
    print("\n== 2. Real players: per-frame |dZ| ==")
    print(f"  median {np.median(d):.4f} m   p90 {p90:.4f} m   p99 {np.percentile(d, 99):.4f} m")
    print("  Real vertical motion is built from small steps. Anything spikier is not football.")
    return p90


def report_scene(path: str, gt_med: dict[int, float], gt_p90: float) -> None:
    from pitch3d.core.scene.serialization import load_scene

    try:
        scene = load_scene(path)
    except Exception as exc:  # noqa: BLE001 — a missing artifact must not kill the sweep
        print(f"\n  {path}: unreadable ({type(exc).__name__})")
        return

    exc_, const, steps, spikes_by_prov = [], 0, [], {}
    n_frames = 0
    for sub in scene.subjects:
        z = np.asarray(sub.proposal.pose.transl, dtype=float)[:, 2]
        if len(z) < 5:
            continue
        n_frames = max(n_frames, len(z))
        exc_.append(np.ptp(z))
        if np.std(z) < 1e-9:
            const += 1
        dz = np.abs(np.diff(z))
        steps.append(dz)
        prov = [str(p) for p in np.asarray(sub.proposal.pose.provenance, dtype=object)]
        for i, d in enumerate(dz):
            k = prov[i + 1] if i + 1 < len(prov) else "?"
            tot, big = spikes_by_prov.get(k, (0, 0))
            spikes_by_prov[k] = (tot + 1, big + int(d > gt_p90))

    exc_ = np.asarray(exc_)
    steps = np.concatenate(steps) if steps else np.array([0.0])
    w = min(gt_med, key=lambda k: abs(k - n_frames))
    ratio = np.median(exc_) / gt_med[w]

    print(f"\n  {path}   {len(exc_)} subjects, {n_frames} frames")
    print(
        f"    excursion  median {np.median(exc_):.3f} m  vs GT@{w} {gt_med[w]:.3f} m"
        f"   -> {ratio:.1f}x {'OVER' if ratio > 1 else 'under'}"
    )
    print(f"    per-frame  median {np.median(steps):.4f}  p90 {np.percentile(steps, 90):.4f}"
          f"  (GT p90 {gt_p90:.4f})")
    if const:
        print(f"    ** {const}/{len(exc_)} subjects have EXACTLY constant Z — the nominal"
              f" pelvis_height_m fallback (pose.py:334), unrecorded")
    if spikes_by_prov:
        worst = ", ".join(
            f"{k} {big / tot * 100:.0f}%" for k, (tot, big) in sorted(spikes_by_prov.items())
        )
        print(f"    steps above GT p90, by provenance: {worst}")


def main(argv: list[str]) -> int:
    gt = gt_tracks()
    med = table_excursion(gt)
    p90 = table_step(gt)
    print("\n== 3. Ours, same statistic, window-matched ==")
    for path in argv[1:] or list(DEFAULT_SCENES):
        report_scene(path, med, p90)
    print("\nWrite-up: docs/findings/vertical-motion-2026-08-09.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
