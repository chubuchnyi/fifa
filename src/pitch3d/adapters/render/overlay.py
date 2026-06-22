"""Reprojection-overlay render pass — the first *real* RenderPass (M1, FR-14, UX-3).

No Blender, no splats, no GPU: it reprojects the **resolved** scene's 3D world points (each
subject's grounded root + the ball) back into image space through the estimated camera and
draws them as markers on a per-frame PNG. That makes the whole mono pipeline visually
inspectable now — and gives the LLM observation loop (ADR-0008) something concrete to look at
— while the photoreal splat/avatar pass stays a later adapter swap (roadmap M2).

The projection maths (quaternion world→camera rotation + pinhole projection + visibility) is
pure numpy and fully unit-tested; the raster is a tiny stdlib PNG encoder (same trick as
:class:`~pitch3d.adapters.fakes.observer.FakeSceneObserver`). The pass reads only resolved
state and never mutates the scene (RenderPass contract).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ...core.correction.engine import resolve_ball, resolve_subject_motion
from ...core.ports.render import RenderPass, RenderQuality, RenderResult
from ...core.scene.camera import CameraTrack
from ...core.scene.scene import Scene
from ...core.scene.subject import Role, Subject, Team

_BALL_COLOR = (255, 215, 0)       # gold
_REFEREE_COLOR = (255, 80, 200)   # pink — referees carry no team colour
_PLAYER_COLOR = (60, 170, 255)    # blue — default when a team has no colour set
_BACKGROUND = (18, 22, 18)        # dark pitch green
_LOW_CONF_COLOR = (255, 60, 60)   # red — markers fade toward this as confidence drops


def confidence_to_color(
    base: tuple[int, int, int], conf: float, *, low: tuple[int, int, int] = _LOW_CONF_COLOR
) -> tuple[int, int, int]:
    """Blend ``base`` toward ``low`` as confidence drops: ``conf=1`` → base, ``conf=0`` → low.

    Confidence highlighting (UX-3, FR-16): a low-confidence marker is pulled toward a warning
    colour so the operator/LLM can *see* where the reconstruction is unsure (e.g. the ball at a
    flight apex, R-4). Full confidence leaves the colour untouched, so a scene with no confidence
    map renders exactly as before.
    """
    c = float(np.clip(conf, 0.0, 1.0))
    ch = [int(round(low[i] * (1.0 - c) + base[i] * c)) for i in range(3)]
    return (ch[0], ch[1], ch[2])


def quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
    """World→camera rotation ``(3, 3)`` from a (w, x, y, z) quaternion (normalised first)."""
    q = np.asarray(quat, dtype=float).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array(
        [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]],
        dtype=float,
    )


def _frame_row(frames: np.ndarray, frame_index: int) -> int:
    """Nearest camera row for a frame index (exact match when present)."""
    i = int(np.searchsorted(frames, frame_index))
    return min(max(i, 0), frames.shape[0] - 1)


def project_world_points(
    camera: CameraTrack, frame_index: int, world_xyz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to image pixels at a frame; flag which are actually visible.

    Applies the per-frame world→camera pose (``X_c = R @ X_w + t``) then the pinhole intrinsics.
    A point is visible only if it is in front of the camera (``Z_c > 0``) *and* lands inside the
    image rectangle. Returns ``(uv (N, 2), visible (N,) bool)``.
    """
    pts = np.asarray(world_xyz, dtype=float).reshape(-1, 3)
    row = _frame_row(camera.frames, frame_index)
    rot = quat_to_rotation_matrix(camera.rotation_quat[row])
    cam = pts @ rot.T + camera.translation[row]
    z = cam[:, 2]
    in_front = z > 1e-6
    safe_z = np.where(in_front, z, 1.0)
    k = camera.intrinsics
    u = k.fx * cam[:, 0] / safe_z + k.cx
    v = k.fy * cam[:, 1] / safe_z + k.cy
    on_image = (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    return np.column_stack([u, v]), in_front & on_image


def _encode_png(fb: np.ndarray) -> bytes:
    """Encode an ``(H, W, 3)`` uint8 framebuffer as a PNG (stdlib only — no PIL)."""
    height, width = fb.shape[:2]

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + typ + data
            + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit RGB
    rows = fb.reshape(height, width * 3)
    raw = np.hstack([np.zeros((height, 1), dtype=np.uint8), rows]).tobytes()  # filter byte per row
    idat = zlib.compress(raw, 9)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _subject_color(subject: Subject, teams: list[Team]) -> tuple[int, int, int]:
    if subject.role == Role.REFEREE:
        return _REFEREE_COLOR
    team = next((t for t in teams if t.id == subject.team_id), None)
    if team is not None and team.color_rgb is not None:
        rgb = [int(np.clip(round(c * 255), 0, 255)) for c in team.color_rgb]
        return (rgb[0], rgb[1], rgb[2])
    return _PLAYER_COLOR


@dataclass
class _Marker:
    """One resolved world-space track to draw: its frames, world points, colour, confidence."""

    frames: np.ndarray
    points: np.ndarray
    color: tuple[int, int, int]
    conf: np.ndarray  # (T,) per-frame confidence in [0, 1]; 1.0 = full (colour untouched)


@dataclass
class ReprojectionOverlayRenderPass(RenderPass):
    """Reproject resolved 3D roots + ball onto per-frame PNGs (FR-14) — pure numpy + stdlib.

    Attributes:
        out_dir: Root for the per-render frame directory.
        marker_radius: Half-size (px) of the square drawn at each projected point.
    """

    out_dir: Path = field(default_factory=lambda: Path("out/render"))
    marker_radius: int = 3

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        scene: Scene,
        camera_path: CameraTrack,
        quality: RenderQuality = RenderQuality.PREVIEW,
    ) -> RenderResult:
        k = camera_path.intrinsics
        width, height = int(k.width), int(k.height)
        target = self.out_dir / f"{scene.id}_{quality.value}"
        target.mkdir(parents=True, exist_ok=True)
        markers = _resolved_markers(scene)

        for i, frame in enumerate(camera_path.frames.tolist()):
            fb = np.empty((height, width, 3), dtype=np.uint8)
            fb[:] = _BACKGROUND
            pts, colors = _points_at_frame(markers, int(frame))
            if pts.shape[0]:
                uv, visible = project_world_points(camera_path, int(frame), pts)
                for (u, v), ok, color in zip(uv, visible, colors, strict=True):
                    if ok:
                        _draw_marker(fb, u, v, color, self.marker_radius)
            (target / f"frame_{i:05d}.png").write_bytes(_encode_png(fb))

        (target / "manifest.txt").write_text(
            f"scene={scene.id} subjects={len(scene.subjects)} "
            f"ball={'yes' if scene.ball is not None else 'no'} "
            f"frames={camera_path.n_frames} size={width}x{height} quality={quality.value}\n",
            encoding="utf-8",
        )
        return RenderResult(
            uri=str(target),
            n_frames=camera_path.n_frames,
            quality=quality,
            is_video=False,
            camera=camera_path,
            note=f"reprojection overlay {width}x{height}",
        )


def _resolved_markers(scene: Scene) -> list[_Marker]:
    """Resolve every subject's motion + the ball once (proposal ⊕ corrections, copy-safe).

    Module-level so any top-down/2D consumer (e.g. the tactical radar) can reuse the same
    resolved world points + confidence-tinted colours without a camera or a RenderPass instance.
    """
    conf_map = scene.confidence.subject_frame_conf if scene.confidence is not None else {}
    markers: list[_Marker] = []
    for subj in scene.subjects:
        motion = resolve_subject_motion(subj.proposal, scene.corrections_for(subj.track_id))
        frames = motion.pose.frames
        markers.append(
            _Marker(
                frames, motion.pose.transl, _subject_color(subj, scene.teams),
                _frame_conf(conf_map.get(subj.track_id), frames.shape[0]),
            )
        )
    if scene.ball is not None:
        ball = resolve_ball(scene.ball, scene.corrections_for(None))
        markers.append(
            _Marker(
                ball.frames, ball.positions_3d, _BALL_COLOR,
                _frame_conf(ball.height_confidence, ball.frames.shape[0]),
            )
        )
    return markers


def _frame_conf(conf: np.ndarray | None, n: int) -> np.ndarray:
    """Per-frame confidence aligned to ``n`` frames; full confidence when absent or mismatched."""
    if conf is None:
        return np.ones(n)
    arr = np.asarray(conf, dtype=float).reshape(-1)
    return arr if arr.shape[0] == n else np.ones(n)


def _points_at_frame(
    markers: list[_Marker], frame: int
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """Collect the world points (and confidence-tinted colours) present at ``frame``."""
    pts: list[np.ndarray] = []
    colors: list[tuple[int, int, int]] = []
    for m in markers:
        hit = np.nonzero(m.frames == frame)[0]
        if hit.size:
            row = int(hit[0])
            pts.append(m.points[row])
            colors.append(confidence_to_color(m.color, float(m.conf[row])))
    arr = np.asarray(pts, dtype=float).reshape(-1, 3) if pts else np.zeros((0, 3))
    return arr, colors


def _draw_marker(
    fb: np.ndarray, x: float, y: float, color: tuple[int, int, int], radius: int
) -> None:
    """Paint a filled square at ``(x, y)``, clipped to the framebuffer bounds."""
    height, width = fb.shape[:2]
    cx, cy = int(round(x)), int(round(y))
    x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
    if x1 > x0 and y1 > y0:
        fb[y0:y1, x0:x1] = color
