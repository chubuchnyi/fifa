"""SoccerNet camera-calibration GT loader — the real-broadcast calibration benchmark (B1).

The SoccerNet *camera-calibration* challenge (``calibration-2023``) ships real broadcast frames
with per-image **pitch-marking** ground truth: for each of the 26 semantic pitch-line classes
visible in a frame, the annotated 2D image points along that line. Unlike the broadcast *video*
(NDA-gated), this calibration data is **openly downloadable** — see
``scripts/get_soccernet_calibration.py``. That makes it the missing B1 asset: a landscape 16:9
broadcast set with pitch calibration GT, on which to score our :class:`FieldCalibrator` against
*independent* footage (``samples/video/clip.mp4`` has faint markings and yields zero landmarks).

This module is the **pure** half — no image decode, no cv2, no GPU, no download:

* :func:`pitch_plane_lines` re-derives the world coordinates of the 17 *straight, pitch-plane*
  line classes from the FIFA pitch geometry (lengths in metres, origin at the centre mark, the
  ``Z = 0`` lawn plane). Class **names** follow SoccerNet's published convention so we interoperate
  with their annotation format; the **coordinates** are the laws-of-the-game facts, independently
  derived (the SoccerNet repo carries no licence, so nothing is copied). Curved elements (the
  three circles) and the out-of-plane goal frames (posts/crossbars at ``Z ≠ 0``) are deliberately
  excluded: only the planar straight lines constrain an image→world *homography*.
* :func:`load_calib_annotation` parses one ``<frame>.json`` (``{class: [{"x","y"}, ...]}`` with
  coordinates **normalised to [0, 1]**) into a :class:`CalibFrameGT`, scaling the normalised points
  to the pixel space of the actual image ``(width, height)`` and attaching each line's world
  segment. It needs only the JSON + the image dimensions, so it is unit-testable from a hand-built
  fixture with no asset on disk.

The heavy directory walk that reads real JPG sizes lives in :func:`load_calib_dir` (kept minimal,
behind a lazy image-size read), so the parsing/geometry logic above stays asset-free and tested.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.ports.io import ClipRef

# FIFA laws-of-the-game pitch metrics (metres). Defaults are the SoccerNet template (105 x 68).
_PENALTY_AREA_LENGTH = 16.5
_PENALTY_AREA_WIDTH = 40.32
_GOAL_AREA_LENGTH = 5.5
_GOAL_AREA_WIDTH = 18.32


def pitch_plane_lines(
    length: float = 105.0, width: float = 68.0
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """World segments of the 17 straight, ``Z = 0`` pitch lines → ``{class: (A(2,), B(2,))}``.

    Coordinate frame matches SoccerNet's template: origin at the centre mark, ``X`` along the pitch
    length in ``[-length/2, +length/2]`` (left goal at ``-length/2``), ``Y`` along the width in
    ``[-width/2, +width/2]`` with the *top* touchline at ``-width/2`` and *bottom* at ``+width/2``.
    Units are metres. Only straight lines on the lawn plane are returned — exactly the lines that
    constrain an image→world homography (circles are curved; goal frames sit at ``Z ≠ 0``).
    """
    hl, hw = length / 2.0, width / 2.0
    pa_x = -hl + _PENALTY_AREA_LENGTH  # big-rect "main" line, left side
    ga_x = -hl + _GOAL_AREA_LENGTH  # small-rect "main" line, left side
    pa_hw = _PENALTY_AREA_WIDTH / 2.0
    ga_hw = _GOAL_AREA_WIDTH / 2.0

    def seg(ax: float, ay: float, bx: float, by: float) -> tuple[np.ndarray, np.ndarray]:
        return np.array([ax, ay], dtype=float), np.array([bx, by], dtype=float)

    return {
        # Touchlines + halfway line.
        "Side line top": seg(-hl, -hw, hl, -hw),
        "Side line bottom": seg(-hl, hw, hl, hw),
        "Side line left": seg(-hl, -hw, -hl, hw),
        "Side line right": seg(hl, -hw, hl, hw),
        "Middle line": seg(0.0, -hw, 0.0, hw),
        # Penalty ("big rect") boxes.
        "Big rect. left top": seg(-hl, -pa_hw, pa_x, -pa_hw),
        "Big rect. left bottom": seg(-hl, pa_hw, pa_x, pa_hw),
        "Big rect. left main": seg(pa_x, -pa_hw, pa_x, pa_hw),
        "Big rect. right top": seg(-pa_x, -pa_hw, hl, -pa_hw),
        "Big rect. right bottom": seg(-pa_x, pa_hw, hl, pa_hw),
        "Big rect. right main": seg(-pa_x, -pa_hw, -pa_x, pa_hw),
        # Goal ("small rect") boxes.
        "Small rect. left top": seg(-hl, -ga_hw, ga_x, -ga_hw),
        "Small rect. left bottom": seg(-hl, ga_hw, ga_x, ga_hw),
        "Small rect. left main": seg(ga_x, -ga_hw, ga_x, ga_hw),
        "Small rect. right top": seg(-ga_x, -ga_hw, hl, -ga_hw),
        "Small rect. right bottom": seg(-ga_x, ga_hw, hl, ga_hw),
        "Small rect. right main": seg(-ga_x, -ga_hw, -ga_x, ga_hw),
    }


@dataclass
class CalibLineGT:
    """One annotated straight pitch line in one frame.

    Attributes:
        name: SoccerNet line class (e.g. ``"Big rect. left main"``).
        image_uv: ``(K, 2)`` annotated points in **image pixels** (normalised GT × ``(W, H)``).
        world_a: ``(2,)`` one endpoint of the line's known world segment (metres, ``Z = 0``).
        world_b: ``(2,)`` the other endpoint.
    """

    name: str
    image_uv: np.ndarray
    world_a: np.ndarray
    world_b: np.ndarray


@dataclass
class CalibFrameGT:
    """Pitch-marking GT for a single broadcast frame (one SoccerNet calibration image).

    Attributes:
        frame: Frame index used to key into a :class:`~pitch3d.core.scene.field.FieldCalibration`.
        image_path: Path to the broadcast JPG (the calibrator decodes this; empty in unit tests).
        width: Image width in pixels — the px space the GT points and the homography share.
        height: Image height in pixels.
        lines: The annotated straight pitch lines present in this frame.
    """

    frame: int
    image_path: str
    width: int
    height: int
    lines: list[CalibLineGT]

    @property
    def n_lines(self) -> int:
        return len(self.lines)

    @property
    def n_points(self) -> int:
        return int(sum(line.image_uv.shape[0] for line in self.lines))


def load_calib_annotation(
    annotation: dict | str | Path,
    width: int,
    height: int,
    *,
    frame: int = 0,
    image_path: str = "",
    length: float = 105.0,
    pitch_width: float = 68.0,
) -> CalibFrameGT:
    """Parse one SoccerNet calibration annotation into a :class:`CalibFrameGT`.

    ``annotation`` is either the already-loaded ``{class: [{"x","y"}, ...]}`` dict or a path to the
    ``<frame>.json`` holding it. Point coordinates are **normalised to [0, 1]** (SoccerNet's
    format), so they are scaled by ``(width, height)`` into the pixel space the calibrator's
    homography lives in. Only the straight, pitch-plane classes from :func:`pitch_plane_lines` are
    kept (with ≥ 2 points); circles, goal frames and unknown classes are dropped — they do not
    constrain a planar homography. ``width``/``height`` must be the dimensions of the actual image
    the calibrator processed, so GT points and the predicted homography agree on pixel scale.
    """
    if isinstance(annotation, (str, Path)):
        with open(annotation) as fh:
            data = json.load(fh)
    else:
        data = annotation

    template = pitch_plane_lines(length=length, width=pitch_width)
    scale = np.array([width, height], dtype=float)
    lines: list[CalibLineGT] = []
    for name, world in template.items():
        pts = data.get(name)
        if not pts or len(pts) < 2:
            continue
        uv = np.array([[p["x"], p["y"]] for p in pts], dtype=float) * scale
        lines.append(CalibLineGT(name=name, image_uv=uv, world_a=world[0], world_b=world[1]))
    return CalibFrameGT(
        frame=frame, image_path=image_path, width=int(width), height=int(height), lines=lines
    )


def synthetic_calib_frames(
    n_frames: int = 4,
    *,
    seed: int = 0,
    width: int = 960,
    height: int = 540,
    n_samples: int = 5,
    length: float = 105.0,
    pitch_width: float = 68.0,
) -> tuple[list[CalibFrameGT], np.ndarray]:
    """Asset-free synthetic frames + their **true** image→world homographies (no decode, no GPU).

    Builds a plausible broadcast view by mapping a world rectangle to an image trapezoid (a
    per-frame jittered convex quad), samples points along each visible straight pitch line, projects
    into the image to play the role of SoccerNet annotations. Returns ``(frames, homographies)``
    where ``homographies[i]`` is the exact image→world ``H`` for ``frames[i]`` — so scoring those
    homographies yields ~0 error, and perturbing them grows it. This is the CLI/harness self-test
    that runs *here and now*, the calibration analogue of ``generate_scene`` for pose.
    """
    from ..adapters.models.calibration import solve_homography  # pure numpy, lazy to avoid coupling

    rng = np.random.default_rng(seed)
    template = pitch_plane_lines(length=length, width=pitch_width)
    world_corners = np.array([[-40.0, -30.0], [40.0, -30.0], [40.0, 30.0], [-40.0, 30.0]])
    base_quad = np.array([[0.30, 0.20], [0.70, 0.20], [0.95, 0.85], [0.05, 0.85]])
    scale = np.array([width, height], dtype=float)

    def project(g: np.ndarray, pts: np.ndarray) -> np.ndarray:
        hom = np.hstack([pts, np.ones((pts.shape[0], 1))]) @ g.T
        return hom[:, :2] / hom[:, 2:3]

    frames: list[CalibFrameGT] = []
    homs: list[np.ndarray] = []
    for f in range(n_frames):
        img_quad = (base_quad + rng.uniform(-0.03, 0.03, size=(4, 2))) * scale
        g = solve_homography(world_corners, img_quad)  # world → image
        h = np.linalg.inv(g)  # image → world (the FieldCalibration convention)
        lines: list[CalibLineGT] = []
        for name, (a, b) in template.items():
            t = np.linspace(0.15, 0.85, n_samples)
            world_pts = a + t[:, None] * (b - a)
            img_pts = project(g, world_pts)
            inb = (
                (img_pts[:, 0] >= 0)
                & (img_pts[:, 0] < width)
                & (img_pts[:, 1] >= 0)
                & (img_pts[:, 1] < height)
            )
            if int(inb.sum()) >= 2:
                lines.append(CalibLineGT(name=name, image_uv=img_pts[inb], world_a=a, world_b=b))
        frames.append(CalibFrameGT(frame=f, image_path="", width=width, height=height, lines=lines))
        homs.append(h)
    return frames, np.stack(homs)


def _image_size(path: Path) -> tuple[int, int]:  # pragma: no cover - thin lazy image-dim read
    """Return ``(width, height)`` of an image without decoding pixels (PIL, lazy import)."""
    from PIL import Image

    with Image.open(path) as im:
        return int(im.width), int(im.height)


_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def load_calib_dir(
    frames_dir: str | Path,
    *,
    min_lines: int = 4,
    limit: int | None = None,
    length: float = 105.0,
    pitch_width: float = 68.0,
) -> list[CalibFrameGT]:  # pragma: no cover - walks a real on-disk split
    """Load a SoccerNet calibration split directory (``<id>.jpg`` + ``<id>.json``) → frame GTs.

    Each returned :class:`CalibFrameGT`'s ``frame`` is its index into the **sorted image list** —
    the exact convention :func:`pitch3d.adapters.models.detection._iter_frames` uses to decode a
    directory clip — so a ``ClipRef`` built from these frames (see :func:`as_clip`) hands the
    calibrator the pixels that match each GT. Images without a usable annotation (missing JSON or
    fewer than ``min_lines`` straight pitch lines, mirroring SoccerNet's "> 4 line annotations"
    evaluation set) are skipped, but the surviving indices still point at the right images. Reads
    each kept image's real pixel size so GT and the homography share pixel scale. ``limit`` caps the
    count (a quick pod smoke).
    """
    frames_dir = Path(frames_dir)
    images = sorted(p.name for p in frames_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    out: list[CalibFrameGT] = []
    for i, img_name in enumerate(images):
        img_path = frames_dir / img_name
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            continue
        w, h = _image_size(img_path)
        gt = load_calib_annotation(
            json_path, w, h, frame=i, image_path=str(img_path),
            length=length, pitch_width=pitch_width,
        )
        if gt.n_lines >= min_lines:
            out.append(gt)
        if limit is not None and len(out) >= limit:
            break
    return out


def as_clip(
    frames: list[CalibFrameGT], frames_dir: str | Path, *, source_id: str = "soccernet-calib"
) -> ClipRef:
    """Build a :class:`ClipRef` over a directory of SoccerNet frames for the calibrator to decode.

    ``frames`` come from :func:`load_calib_dir` (their ``frame`` fields index the sorted image
    list); the clip's ``frames`` are exactly those indices, so the calibrator decodes the images
    the GT describes. Each SoccerNet image is an independent broadcast view, so the calibrator must
    solve each frame on its own (run with temporal smoothing off).
    """
    idx = np.array([f.frame for f in frames], dtype=int)
    w = frames[0].width if frames else 0
    h = frames[0].height if frames else 0
    return ClipRef(
        source_id=source_id, uri=str(frames_dir), frames=idx, width=w, height=h, fps=25.0
    )
