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
import sys
from pathlib import Path

import numpy as np

from ..core.correction.engine import make_offset
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
    *, out_dir: Path, n_frames: int, n_subjects: int, export_format: str
) -> int:
    """Drive the full reconstruction→edit→resolve→render→export path; return an exit code."""
    out_dir = Path(out_dir)
    ports = default_ports(out_dir=out_dir, n_subjects=n_subjects)
    app: Application = build_app(out_dir=out_dir, ports=ports)

    # 1) Project setup: register a clip as an episode the agent/CLI can reconstruct.
    clip = _synthetic_clip(n_frames=n_frames)
    episode = app.register_clip(clip, name="demo episode")
    print(f"== registered {episode.id} ({episode.n_frames} frames) from {clip.uri}")

    # 2) Reconstruction: DETECT→TRACK→CALIBRATE→POSE→BALL, assemble the proposal scene.
    scene_id = app.run_reconstruction(episode.id, on_ground=_airborne_on_ground(n_frames))
    scene = app.get_scene(scene_id)
    mid_frame = int(scene.subjects[0].proposal.pose.frames[n_frames // 2])
    print(f"== reconstructed {scene_id}: {len(scene.subjects)} subject(s), "
          f"ball={'yes' if scene.ball is not None else 'no'}")

    # 3) OBSERVE (initial): multi-viewpoint 3D + frame overlay + UI + textual summary.
    obs_before = app.observe(
        scene_id, frame=mid_frame, n_orbit=2, include_ui=True, quality="preview"
    )
    _print_observation(obs_before, label="observe:before")

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
    print(f"== committed {corr.id} ({corr.mode.value}) → {len(app.get_scene(scene_id).corrections)} correction(s)")

    # 8) OBSERVE (after): the loop is closed — the agent sees the consequence of its edit.
    obs_after = app.observe(scene_id, frame=mid_frame, quality="preview")
    _print_observation(obs_after, label="observe:after")

    # 9) Render the RESOLVED scene (proposal ⊕ corrections), single source of truth.
    render = app.render(scene_id, quality="preview")
    print(f"\n== render: {render.n_frames} frame(s), is_video={render.is_video} → {render.uri}")

    # 10) Export (canonical JSON is a real round-trip; other formats are honest fakes).
    export_path = out_dir / "export" / f"scene.{export_format}"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    result = app.export(scene_id, export_format, str(export_path))
    print(f"== export[{result.fmt.value}]: {', '.join(result.paths) or '(no paths)'}")

    print("\nDRY-RUN OK — reconstruct → observe → attention → preview → edit → resolve → "
          "observe → render → export completed on fakes.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pitch3d",
        description="Run the pitch3d dry-run (full golden path on fake adapters).",
    )
    parser.add_argument("--out-dir", default="out/dry-run", help="where artifacts are written")
    parser.add_argument("--frames", type=int, default=12, help="synthetic clip length")
    parser.add_argument("--subjects", type=int, default=4, help="number of fake subjects")
    parser.add_argument("--format", default="json", help="export format (json is a real round-trip)")
    args = parser.parse_args(argv)

    return run_dry_run(
        out_dir=Path(args.out_dir),
        n_frames=args.frames,
        n_subjects=args.subjects,
        export_format=args.format,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
