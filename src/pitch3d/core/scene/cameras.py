"""Virtual-operator camera planning for the deliverable video render (pure core).

The render script used to derive four STATIC cameras from the bbox of everything it loaded;
with the full 105x68 m pitch folded into that bbox every camera framed the whole stadium bowl
from outside and the players became 5-10 px specks (the failed v2 eye-judgement). Real broadcast
cameras do the opposite: a FIXED mount inside the stadium that PANs (follows the action) and
ZOOMs (fits the action) — they never dolly. This module plans exactly that, per frame, from the
reconstructed world positions alone, so the framing logic is unit-testable numpy instead of
untested Blender-side code.

Outputs are per-camera tracks over the global frame index: one fixed ``position``, a smoothed
``look_at`` path, and a smoothed horizontal ``fov_x_deg`` — everything the renderer needs to
aim ``bpy`` cameras. Mounts are placed like a real rig relative to the pitch and the seating
bowl (main stand halfway line, pitchside touchline, behind the action-side goal), clamped to
stay INSIDE the bowl envelope so no camera ever sees the stadium from outside again.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from pitch3d.core.scene.units import FieldDimensions

#: Bowl envelope used to place mounts (mirrors core.scene.stadium.stadium_bowl_geometry
#: defaults: footprint = pitch/2 + apron, seating rises ``rise`` and runs ``run`` per row).
BOWL_APRON = 7.0
BOWL_RISE = 0.80
BOWL_RUN = 0.90
BOWL_ROWS = 20


@dataclass(frozen=True)
class CameraTrack:
    """A planned virtual camera: fixed mount, per-frame aim + zoom."""

    name: str
    position: np.ndarray  # (3,) fixed mount, world metres z-up
    look_at: np.ndarray  # (F, 3)
    fov_x_deg: np.ndarray  # (F,) horizontal field of view
    frames: np.ndarray  # (F,) global frame indices the rows correspond to

    def rotation(self, row: int) -> np.ndarray:
        """World-from-camera basis at ``row``: columns = right, up, backward (-forward)."""
        fwd = self.look_at[row] - self.position
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right = right / max(float(np.linalg.norm(right)), 1e-9)
        up = np.cross(right, fwd)
        return np.stack([right, up, -fwd], axis=1)


def smooth_moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Zero-phase (centred) moving average over axis 0, edge-padded so the ends don't lag."""
    v = np.asarray(values, dtype=float)
    w = max(1, int(window))
    if w % 2 == 0:
        w += 1
    if w == 1 or v.shape[0] < 2:
        return v.copy()
    half = w // 2
    pad = [(half, half)] + [(0, 0)] * (v.ndim - 1)
    padded = np.pad(v, pad, mode="edge")
    kernel = np.ones(w) / w
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="valid"), 0, padded)
    return out


def _ffill_rows(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Forward- then back-fill invalid rows so smoothing never eats a NaN."""
    out = np.array(values, dtype=float)
    idx = np.where(valid)[0]
    if idx.size == 0:
        raise ValueError("no frame has any tracked entity — cannot plan cameras")
    last = idx[0]
    for i in range(out.shape[0]):
        if valid[i]:
            last = i
        else:
            out[i] = out[last]
    for i in range(idx[0] - 1, -1, -1):
        out[i] = out[idx[0]]
    return out


def action_track(
    roots: np.ndarray,
    ball: np.ndarray | None = None,
    *,
    look_window: int = 9,
    zoom_window: int = 25,
    pad_m: float = 3.0,
    min_radius_m: float = 6.0,
    bulk_q: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed action ``centroid (F, 3)`` + horizontal ``radius (F,)`` from world tracks.

    ``roots`` is (F, N, 3) pelvis positions with NaN rows where a subject is absent; ``ball``
    is (F, 3) with NaN where absent. The ball counts as part of the action (it IS the action).
    Both aim and zoom are straggler-robust, like a real operator: the centroid is the
    component-wise MEDIAN of whoever is present (one goalkeeper idling 50 m from the play must
    not drag the aim), and the radius covers the BULK of the action — the ``bulk_q`` quantile
    of horizontal distances plus ``pad_m`` — not the farthest entity (same keeper must not
    zoom the broadcast out to a stadium-wide shot). Near-quantile entities still fit thanks to
    pad + the fov fill margin; true stragglers are deliberately cropped, exactly like real TV.
    Look and zoom get separate smoothing windows: the aim may follow at ~1/3 s while the zoom
    must breathe slower (~1 s) or the render pumps.
    """
    roots = np.asarray(roots, dtype=float)
    pts = roots
    if ball is not None:
        pts = np.concatenate([roots, np.asarray(ball, dtype=float)[:, None, :]], axis=1)
    present = ~np.isnan(pts[..., 0])
    any_present = present.any(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are ffilled below
        centroid = np.nanmedian(np.where(present[..., None], pts, np.nan), axis=1)
        centroid = _ffill_rows(centroid, any_present)
        d = np.linalg.norm(pts[..., :2] - centroid[:, None, :2], axis=-1)
        radius = np.nanquantile(np.where(present, d, np.nan), bulk_q, axis=1)
    radius = _ffill_rows(radius[:, None], any_present)[:, 0]
    radius = np.maximum(radius + pad_m, min_radius_m)
    return (
        smooth_moving_average(centroid, look_window),
        smooth_moving_average(radius, zoom_window),
    )


def _fov_x_deg(
    position: np.ndarray,
    look: np.ndarray,
    radius: np.ndarray,
    *,
    aspect: float,
    fill: float = 0.80,
    lo: float = 8.0,
    hi: float = 50.0,
    head_m: float = 2.4,
    zoom_window: int = 25,
) -> np.ndarray:
    """Per-frame horizontal fov that fits the action disc on BOTH image axes.

    Horizontal need: the disc's radius at the slant distance. Vertical need: the disc
    foreshortened by the depression angle plus standing-player headroom, converted to a
    horizontal angle via the aspect ratio (tan(fov_x/2) = aspect * tan(fov_y/2)).
    """
    offset = look - position[None, :]
    dist = np.maximum(np.linalg.norm(offset, axis=1), 1e-6)
    sin_depression = np.abs(position[2] - look[:, 2]) / dist
    half_x = np.arctan(radius / dist)
    half_y = np.arctan((radius * sin_depression + head_m) / dist)
    half = np.maximum(half_x, np.arctan(aspect * np.tan(half_y)))
    fov = np.degrees(2.0 * half) / fill
    return smooth_moving_average(np.clip(fov, lo, hi), zoom_window)


@dataclass(frozen=True)
class OperatorRig:
    """Where the fixed mounts sit, relative to pitch dims + the bowl envelope."""

    dims: FieldDimensions = field(default_factory=FieldDimensions)
    aspect: float = 16.0 / 9.0
    broadcast_out_m: float = 10.0  # outward from the bowl footprint, along the rake
    broadcast_height_m: float = 12.0
    sideline_height_m: float = 2.5
    goal_out_m: float = 5.0
    goal_height_m: float = 9.0
    top_height_m: float = 95.0

    def bowl_footprint(self) -> tuple[float, float]:
        return (
            self.dims.length / 2.0 + BOWL_APRON,
            self.dims.width / 2.0 + BOWL_APRON,
        )


def plan_virtual_cameras(
    roots: np.ndarray,
    ball: np.ndarray | None,
    frames: np.ndarray,
    rig: OperatorRig | None = None,
) -> dict[str, CameraTrack]:
    """Plan the four deliverable cameras from world tracks over global ``frames``.

    - ``broadcast``: main-stand mount over the halfway line (the classic elevated side view),
      panning/zooming with the action.
    - ``sideline``: low pitchside mount on the same side, x anchored near the action's median.
    - ``goal``: behind the goal of the half where the action lives.
    - ``top``: static overhead schematic framing the whole pitch (fixed fov).

    All mounts stay inside the bowl envelope: outward offsets are capped at the bowl depth and
    heights at the seating surface sight-lines, so no camera renders the stadium from outside.
    """
    rig = rig or OperatorRig()
    frames = np.asarray(frames, dtype=np.int64)
    centroid, radius = action_track(roots, ball)
    hx, hy = rig.bowl_footprint()
    bowl_depth = BOWL_ROWS * BOWL_RUN
    half_len = rig.dims.length / 2.0

    out = min(rig.broadcast_out_m, bowl_depth - 1.0)
    broadcast_pos = np.array([0.0, -(hy + out), rig.broadcast_height_m])
    x_anchor = float(np.clip(np.median(centroid[:, 0]), -half_len / 2.0, half_len / 2.0))
    sideline_pos = np.array([x_anchor, -(rig.dims.width / 2.0 + 2.0), rig.sideline_height_m])
    goal_sign = 1.0 if float(np.median(centroid[:, 0])) >= 0.0 else -1.0
    goal_x = goal_sign * (hx + min(rig.goal_out_m, bowl_depth - 1.0))
    goal_pos = np.array([goal_x, 0.0, rig.goal_height_m])
    top_pos = np.array([0.0, 0.0, rig.top_height_m])

    look = centroid.copy()
    look[:, 2] = 1.0  # aim at torso height, not at the grass

    tracks: dict[str, CameraTrack] = {}
    for name, pos in (("broadcast", broadcast_pos), ("sideline", sideline_pos), ("goal", goal_pos)):
        tracks[name] = CameraTrack(
            name=name,
            position=pos,
            look_at=look,
            fov_x_deg=_fov_x_deg(pos, look, radius, aspect=rig.aspect),
            frames=frames,
        )

    n = frames.shape[0]
    pitch_half_y_as_x = (rig.dims.width / 2.0) * rig.aspect
    top_half = max(half_len, pitch_half_y_as_x) * 1.06
    top_fov = np.degrees(2.0 * np.arctan(top_half / rig.top_height_m))
    tracks["top"] = CameraTrack(
        name="top",
        position=top_pos,
        look_at=np.tile(np.array([[0.0, 0.0, 0.0]]), (n, 1)),
        fov_x_deg=np.full(n, top_fov),
        frames=frames,
    )
    return tracks


def project_normalized(
    track: CameraTrack, row: int, points: np.ndarray, aspect: float
) -> np.ndarray:
    """Project world ``points (P, 3)`` through the planned camera at ``row`` → (P, 2) in [-1, 1]
    image units (x right, y up; |coord| <= 1 means inside frame). The renderer applies the same
    look-at + horizontal-fov convention, so this is the contract's framing check."""
    basis = track.rotation(row)  # columns right, up, -fwd
    rel = (np.asarray(points, dtype=float) - track.position) @ basis
    depth = -rel[:, 2]
    depth = np.where(depth <= 1e-9, np.nan, depth)
    tan_x = np.tan(np.radians(track.fov_x_deg[row]) / 2.0)
    x = rel[:, 0] / depth / tan_x
    y = rel[:, 1] / depth / (tan_x / aspect)
    return np.stack([x, y], axis=1)
