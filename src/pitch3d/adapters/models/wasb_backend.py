"""WASB ball-detection backend — the heavy half of the ball path (M1, FR-9).

:class:`~pitch3d.adapters.models.ball.TrackNetBallTracker` owns the *pure*
threshold + gap-fill logic; it consumes per-frame ball peaks through the
:class:`~pitch3d.adapters.models.ball.BallDetectionBackend` protocol. This module is one such
backend: it runs **WASB** (``nttcom/WASB-SBDT``, MIT — the HRNet "Widely Applicable Strong
Baseline" heatmap tracker) on a clip and returns the canned
:class:`~pitch3d.adapters.models.ball.RawBallDetections` hand-off.

WASB is a research repo, not a pip package, so everything heavy (torch, the repo's
``models``/``detectors``/``trackers`` packages, the Hydra configs, the soccer checkpoint) is
imported/loaded **lazily** on first :meth:`WASBBallBackend.detect_ball`. Importing this module is
cheap and torch-free, matching the rfdetr/bytetrack/smplest-x adapters.

Wire it in at the composition root::

    --ball tracknet --ball-backend pitch3d.adapters.models.wasb_backend:make --device cuda

The zero-arg :func:`make` factory (the contract ``_resolve_backend`` expects) reads its config
from the environment, since the seam instantiates the factory with no arguments:

* ``PITCH3D_WASB_REPO``  — checkout dir (default ``/workspace/repos/WASB-SBDT``)
* ``PITCH3D_WASB_CKPT``  — weight (default ``/workspace/weights/wasb/wasb_soccer_best.pth.tar``)
* ``PITCH3D_WASB_DATASET`` — Hydra dataset group for the input geometry (default ``soccer``)
* ``PITCH3D_DEVICE``     — inference device (default ``cuda``; WASB's detector is GPU-only)

**Inference shape (verified against WASB ``GET_STARTED.md`` @ 923462cacdeb).** WASB's soccer model
is HRNet with ``frames_in = frames_out = 3`` and a ``512x288`` input. Per non-overlapping window of
three frames the detector emits one heatmap per input frame; the post-processor thresholds blobs
(``concomp``) into candidate ``(x, y)`` peaks already mapped back to *original* image pixels via the
inverse affine, and :class:`OnlineTracker` links the per-frame candidates into a single ball track
with a visibility flag. We surface that visibility as confidence ``1.0`` (visible) / ``0.0``
(occluded), and the pure half interpolates the occluded gaps (R-6).

**Validation status (honest).** The pure windowing (:func:`_window_starts`) and per-frame assembly
(:func:`_assemble_detections`) are unit-tested; importing this module is torch-free. The GPU path
(:meth:`_load` Hydra build + :meth:`_preprocess_window` warpAffine + ``run_tensor`` + tracker) is
**API-faithful but not yet executed on hardware** — the pod that hosts WASB is the first place it
runs. Stage the repo + weight with ``scripts/stage_wasb_weight.sh`` and confirm on the first pod run
(see ``docs/runpod-agent-setup.md``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ...core.ports.io import ClipRef
from .ball import RawBallDetections
from .detection import _iter_frames

#: WASB soccer HRNet input (model.inp_width x model.inp_height); output heatmap is the same size.
_INPUT_WH = (512, 288)
#: WASB stacks ``frames_in`` consecutive RGB frames as the network's channel input.
_FRAMES_IN = 3


def _window_starts(n_frames: int, frames_in: int = _FRAMES_IN, step: int = _FRAMES_IN) -> list[int]:
    """Start positions of length-``frames_in`` windows tiling ``n_frames`` at ``step`` stride.

    Pure (no torch/cv2). The default ``step == frames_in`` tiles the clip into the
    non-overlapping windows WASB's documented soccer setting (``detector.step=3``) uses. A final
    window is always clamped to the end so the last frame is covered even when the stride
    overshoots; clips shorter than one window yield a single (right-padded) window at 0.
    """
    if n_frames <= 0:
        return []
    if n_frames <= frames_in:
        return [0]
    starts = list(range(0, n_frames - frames_in + 1, max(1, step)))
    if starts[-1] != n_frames - frames_in:
        starts.append(n_frames - frames_in)
    return starts


def _assemble_detections(
    frame_indices: list[int],
    xy_by_pos: dict[int, tuple[float, float]],
    visi_by_pos: dict[int, bool],
) -> RawBallDetections:
    """Stack per-frame tracker outputs into :class:`RawBallDetections` (pure, no torch/cv2).

    A visible frame carries its tracked ``(x, y)`` at confidence ``1.0``; an occluded frame keeps a
    placeholder ``(0, 0)`` at confidence ``0.0`` so the pure half (``TrackNetBallTracker``) treats
    it as a miss and interpolates the position (R-6).
    """
    n = len(frame_indices)
    points = np.zeros((n, 2), dtype=float)
    scores = np.zeros(n, dtype=float)
    for pos in range(n):
        if visi_by_pos.get(pos, False):
            points[pos] = xy_by_pos[pos]
            scores[pos] = 1.0
    return RawBallDetections(
        frames=np.asarray(frame_indices, dtype=int), points_xy=points, scores=scores
    )


@dataclass
class WASBBallBackend:
    """Run WASB per clip → per-frame ball peaks (image px), keyed by source frame index.

    Attributes:
        repo_dir: WASB-SBDT checkout (provides ``models``/``detectors``/``trackers``/``configs``
            under ``<repo>/src``).
        weights: Soccer checkpoint (``*.pth.tar``) passed to ``detector.model_path``.
        dataset: Hydra ``dataset`` config group (selects the input geometry; ``soccer`` for us).
        device: Inference device (WASB's detector asserts ``cuda``).
        step: Window stride; ``frames_in`` (default) gives the documented non-overlapping tiling.
    """

    repo_dir: str = "/workspace/repos/WASB-SBDT"
    weights: str = "/workspace/weights/wasb/wasb_soccer_best.pth.tar"
    dataset: str = "soccer"
    device: str = "cuda"
    step: int = _FRAMES_IN
    _detector: Any = field(default=None, init=False, repr=False)
    _tracker: Any = field(default=None, init=False, repr=False)
    _transform: Any = field(default=None, init=False, repr=False)
    _cfg: Any = field(default=None, init=False, repr=False)

    def detect_ball(self, clip: ClipRef) -> RawBallDetections:  # pragma: no cover - heavy GPU path
        self._load()

        frame_indices = [int(f) for f in clip.frames.tolist()]
        rgb_frames = [
            np.ascontiguousarray(image_bgr[:, :, ::-1])  # WASB reads RGB (PIL); _iter_frames is BGR
            for _, image_bgr in _iter_frames(clip)
        ]
        n = len(rgb_frames)
        if n == 0:
            return _assemble_detections([], {}, {})

        per_pos_dets: dict[int, list] = {pos: [] for pos in range(n)}
        for start in _window_starts(n, _FRAMES_IN, self.step):
            window = [min(start + k, n - 1) for k in range(_FRAMES_IN)]
            imgs_t, affine_mats = self._preprocess_window([rgb_frames[p] for p in window])
            results, _ = self._detector.run_tensor(imgs_t, affine_mats)
            for j in range(_FRAMES_IN):
                if start + j < n:  # skip clamp-padded slots at the clip tail
                    per_pos_dets[start + j].extend(results[0][j])

        self._tracker.refresh()
        xy_by_pos: dict[int, tuple[float, float]] = {}
        visi_by_pos: dict[int, bool] = {}
        for pos in range(n):
            out = self._tracker.update(per_pos_dets[pos])
            xy_by_pos[pos] = (float(out["x"]), float(out["y"]))
            visi_by_pos[pos] = bool(out["visi"])
        return _assemble_detections(frame_indices, xy_by_pos, visi_by_pos)

    def _preprocess_window(self, rgb_frames):  # pragma: no cover - heavy path (cv2 + torch)
        """``frames_in`` RGB frames → ``(1, 3*frames_in, 288, 512)`` tensor + inverse affine.

        Faithful to ``dataloaders.dataset_loader``: each frame is warped to ``512x288`` by an
        affine about the image centre (scale = long edge), ToTensor'd and ImageNet-normalised, then
        channel-stacked. The matching ``inv=1`` affine maps heatmap pixels back to the original
        frame and is handed to ``run_tensor`` keyed by output scale ``0``.
        """
        import cv2
        import torch
        from PIL import Image
        from utils.image import get_affine_transform

        imgs_t = []
        affine_inv = None
        for rgb in rgb_frames:
            h, w = rgb.shape[:2]
            c = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            s = max(h, w) * 1.0
            trans = get_affine_transform(c, s, 0, list(_INPUT_WH), inv=0)
            warped = cv2.warpAffine(rgb, trans, _INPUT_WH, flags=cv2.INTER_LINEAR)
            imgs_t.append(self._transform(Image.fromarray(warped)))
            if affine_inv is None:
                affine_inv = get_affine_transform(c, s, 0, list(_INPUT_WH), inv=1)
        imgs = torch.cat(imgs_t, dim=0)[None].to(self.device)  # (1, 3*frames_in, 288, 512)
        affine_mats = {0: torch.from_numpy(np.asarray(affine_inv, dtype=np.float32))[None]}
        return imgs, affine_mats

    def _load(self) -> None:  # pragma: no cover - heavy GPU path
        if self._detector is not None:
            return
        import sys

        from hydra import compose, initialize_config_dir

        src = os.path.join(os.path.abspath(self.repo_dir), "src")
        if not os.path.isdir(src):
            raise RuntimeError(
                f"WASB checkout not found at {src!r}. Set PITCH3D_WASB_REPO, or clone "
                "https://github.com/nttcom/WASB-SBDT and stage the soccer weight "
                "(scripts/stage_wasb_weight.sh)."
            )
        if src not in sys.path:
            sys.path.insert(0, src)

        from dataloaders import build_img_transforms
        from detectors import build_detector
        from trackers import build_tracker

        with initialize_config_dir(version_base=None, config_dir=os.path.join(src, "configs")):
            cfg = compose(
                config_name="eval",
                overrides=[
                    f"dataset={self.dataset}",
                    "model=wasb",
                    f"detector.model_path={self.weights}",
                    f"detector.step={self.step}",
                    f"runner.device={self.device}",
                    "runner.gpus=[0]",  # single-GPU pod; the repo default is [0,1,2,3]
                ],
            )
        cfg["output_dir"] = cfg.get("output_dir") or os.path.join(src, "outputs")
        self._cfg = cfg
        self._detector = build_detector(cfg)
        self._tracker = build_tracker(cfg)
        _, self._transform = build_img_transforms(cfg)


def make() -> WASBBallBackend:
    """Zero-arg factory for ``--ball-backend``; config comes from the environment."""
    return WASBBallBackend(
        repo_dir=os.environ.get("PITCH3D_WASB_REPO", "/workspace/repos/WASB-SBDT"),
        weights=os.environ.get(
            "PITCH3D_WASB_CKPT", "/workspace/weights/wasb/wasb_soccer_best.pth.tar"
        ),
        dataset=os.environ.get("PITCH3D_WASB_DATASET", "soccer"),
        device=os.environ.get("PITCH3D_DEVICE", "cuda"),
    )
