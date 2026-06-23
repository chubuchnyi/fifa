"""PnLCalib pitch-keypoint backend — the real ``KeypointBackend`` for the B1 SoccerNet eval.

Wraps the public **PnLCalib** field-calibration network (two HRNet heads — pitch *keypoints* and
pitch *lines*; Gutiérrez-Pérez & Agudo, 2024) as a
:class:`pitch3d.adapters.models.calibration.KeypointBackend`. Per frame it runs both heads,
decodes pitch landmarks with PnLCalib's own heatmap post-processing, and pairs each detected
landmark id with its known pitch-plane world coordinate (metres, centre-origin, ``Z = 0``). The
robust homography solve (RANSAC + confidence-weighted DLT + temporal smoothing) is done downstream
by :class:`KeypointFieldCalibrator`; this adapter only emits the image↔world correspondences.

It is the heavy backend injected through the dotted-path seam (ADR-0006)::

    scripts/run_calib_eval.py --dataset soccernet --frames-dir <split> \\
        --backend pitch3d.adapters.models.pnlcalib_backend:make

PnLCalib's code + weights are **not vendored** into this repo (its licence is restrictive and the
weights are ~265 MB each); they live on the GPU box and are imported at call time. The zero-arg
``make()`` factory (the seam calls it with no args, see ``app.wiring._resolve_backend``) is
configured entirely from the environment, so the box points the adapter at its checkout + weights
without editing any wiring:

    * ``PNLCALIB_REPO`` — PnLCalib checkout (``/workspace/repos/PnLCalib``)
    * ``PNLCALIB_WEIGHTS_KP`` — keypoint-head weights (``/workspace/weights/pnlcalib/SV_kp``)
    * ``PNLCALIB_WEIGHTS_LINES`` — line-head weights (``/workspace/weights/pnlcalib/SV_lines``)
    * ``PNLCALIB_DEVICE`` — torch device (``cuda:0``)
    * ``PNLCALIB_KP_THRESHOLD`` — keypoint heatmap gate (``0.3434``, PnLCalib's own default)
    * ``PNLCALIB_LINE_THRESHOLD`` — line heatmap gate (``0.7867``, PnLCalib's own default)

The world tables (``keypoint_world_coords_2D`` / ``…aux…``, already centre-origin in PnLCalib's
``utils.utils_calib``) are **imported from the installed PnLCalib**, never copied, so the id→world
mapping always matches the weights actually loaded. All heavy imports (torch, cv2, PnLCalib) are
lazy, so importing this module is cheap and dependency-free — only :meth:`detect_keypoints` pulls
the stack. PnLCalib needs ``shapely`` (``pip install shapely`` into the box venv).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ...core.ports.io import ClipRef
from .calibration import FrameKeypoints

# PnLCalib resizes every frame to this fixed input before the HRNet heads; its heatmap decoding
# returns landmark coords in this space, and ``complete_keypoints(normalize=True)`` divides by it.
_INPUT_W = 960
_INPUT_H = 540
# PnLCalib keypoint ids are 1-based: ids 1..57 index the main table, ids ≥58 the aux table.
_N_MAIN = 57


@dataclass
class _PnLCalibBackend:
    """Run PnLCalib's keypoint + line heads and emit per-frame image↔world landmark matches."""

    repo: str
    weights_kp: str
    weights_lines: str
    device: str = "cuda:0"
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867
    _state: dict = field(default_factory=dict, init=False, repr=False)

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        """Detect pitch landmarks per frame of ``clip`` (a video file or a directory of frames)."""
        s = self._load()
        torch, cv2, Image = s["torch"], s["cv2"], s["Image"]

        out: list[FrameKeypoints] = []
        for idx, bgr in s["iter_frames"](clip):
            h_orig, w_orig = bgr.shape[:2]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = s["to_tensor"](Image.fromarray(rgb)).float().unsqueeze(0)
            if tensor.shape[-1] != _INPUT_W:
                tensor = s["resize"](tensor)
            tensor = tensor.to(self.device)
            with torch.no_grad():
                heatmaps = s["model"](tensor)
                heatmaps_l = s["model_l"](tensor)
            kp_raw = s["get_kp"](heatmaps[:, :-1, :, :])
            line_raw = s["get_line"](heatmaps_l[:, :-1, :, :])
            kp_list = s["coords_to_dict"](kp_raw, threshold=self.kp_threshold)
            line_list = s["coords_to_dict"](line_raw, threshold=self.line_threshold)
            kp_dict, _ = s["complete"](
                kp_list[0], line_list[0], w=_INPUT_W, h=_INPUT_H, normalize=True
            )

            uv, world, conf = [], [], []
            for kid, d in kp_dict.items():
                wp = s["kw"][kid - 1] if kid <= _N_MAIN else s["ka"][kid - 1 - _N_MAIN]
                uv.append([d["x"] * w_orig, d["y"] * h_orig])  # normalised → original image px
                world.append([float(wp[0]), float(wp[1])])  # metres, centre-origin, Z=0
                conf.append(float(d.get("p", 1.0)))
            out.append(
                FrameKeypoints(
                    frame=int(idx),
                    image_uv=np.asarray(uv, dtype=float).reshape(-1, 2),
                    world_xy=np.asarray(world, dtype=float).reshape(-1, 2),
                    confidence=np.asarray(conf, dtype=float).reshape(-1),
                )
            )
        return out

    def _load(self) -> dict:
        """Lazy-build the two HRNet models + bind PnLCalib helpers; cached after first call."""
        if self._state:
            return self._state

        import sys

        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)  # PnLCalib imports as top-level `model` / `utils`

        import cv2
        import torch
        import torchvision.transforms as transforms
        import torchvision.transforms.functional as tvf
        import yaml
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        from PIL import Image
        from utils.utils_calib import (
            keypoint_aux_world_coords_2D,  # already centre-origin: [x-52.5, y-34]
            keypoint_world_coords_2D,
        )
        from utils.utils_heatmap import (
            complete_keypoints,
            coords_to_dict,
            get_keypoints_from_heatmap_batch_maxpool,
            get_keypoints_from_heatmap_batch_maxpool_l,
        )

        from .detection import _iter_frames

        cfg = yaml.safe_load(open(os.path.join(self.repo, "config", "hrnetv2_w48.yaml")))
        cfg_l = yaml.safe_load(open(os.path.join(self.repo, "config", "hrnetv2_w48_l.yaml")))
        model = get_cls_net(cfg)
        model.load_state_dict(torch.load(self.weights_kp, map_location=self.device))
        model.to(self.device).eval()
        model_l = get_cls_net_l(cfg_l)
        model_l.load_state_dict(torch.load(self.weights_lines, map_location=self.device))
        model_l.to(self.device).eval()

        self._state = {
            "torch": torch,
            "cv2": cv2,
            "Image": Image,
            "to_tensor": tvf.to_tensor,
            "resize": transforms.Resize((_INPUT_H, _INPUT_W)),
            "model": model,
            "model_l": model_l,
            "get_kp": get_keypoints_from_heatmap_batch_maxpool,
            "get_line": get_keypoints_from_heatmap_batch_maxpool_l,
            "coords_to_dict": coords_to_dict,
            "complete": complete_keypoints,
            "kw": keypoint_world_coords_2D,
            "ka": keypoint_aux_world_coords_2D,
            "iter_frames": _iter_frames,
        }
        return self._state


def make() -> _PnLCalibBackend:
    """Zero-arg factory (ADR-0006 seam) → an env-configured PnLCalib ``KeypointBackend``."""
    repo = os.environ.get("PNLCALIB_REPO", "/workspace/repos/PnLCalib")
    return _PnLCalibBackend(
        repo=repo,
        weights_kp=os.environ.get("PNLCALIB_WEIGHTS_KP", "/workspace/weights/pnlcalib/SV_kp"),
        weights_lines=os.environ.get(
            "PNLCALIB_WEIGHTS_LINES", "/workspace/weights/pnlcalib/SV_lines"
        ),
        device=os.environ.get("PNLCALIB_DEVICE", "cuda:0"),
        kp_threshold=float(os.environ.get("PNLCALIB_KP_THRESHOLD", "0.3434")),
        line_threshold=float(os.environ.get("PNLCALIB_LINE_THRESHOLD", "0.7867")),
    )
