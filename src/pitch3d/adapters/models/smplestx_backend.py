"""SMPLest-X HMR backend — the heavy half of the pose path (M1, FR-8, B3).

:class:`~pitch3d.adapters.models.pose.GVHMRPoseEstimator` owns the *pure* grounding/assembly
logic; it consumes camera-space SMPL-X articulation through the :class:`HMRBackend` protocol.
This module is one such backend: it runs **SMPLest-X** (ViT-Huge SMPL-X mesh recovery) on each
tracklet's per-frame boxes and returns the canned :class:`RawBodyMotion` hand-off — no YOLO,
because we feed OUR ByteTrack boxes instead of the upstream detector.

SMPLest-X is a research repo, not a pip package, so everything heavy (torch, the repo's
``main``/``models``/``human_models`` packages, the 8 GB checkpoint, the SMPL-X model files)
is imported/loaded **lazily** on first ``estimate_bodies`` call. Importing this module is cheap
and torch-free, matching the rfdetr/bytetrack adapters.

Wire it in at the composition root::

    --pose gvhmr --pose-backend pitch3d.adapters.models.smplestx_backend:make --device cuda

The zero-arg :func:`make` factory (the contract ``_resolve_backend`` expects) reads its config
from the environment, since the seam instantiates the factory with no arguments:

* ``PITCH3D_SMPLESTX_REPO``  — checkout dir (default ``/workspace/repos/SMPLest-X``)
* ``PITCH3D_SMPLESTX_CKPT``  — model dir under ``pretrained_models`` (default ``smplest_x_h``)
* ``PITCH3D_DEVICE``         — inference device (default ``cuda``)

Coordinate note: SMPLest-X returns ``global_orient`` in its own **camera** frame. Per the seam's
design the pure half supplies the world *translation* (foot point → pitch homography) but leaves
articulation as-is, so the body orientation is camera-relative until a camera→world rotation is
applied downstream. That alignment is out of scope for this backend.

T2 foot-plane anchor: alongside the articulation this backend also fills
:attr:`RawBodyMotion.pelvis_above_foot` — a per-frame pelvis-above-foot height from a quick
SMPL-X FK at zero global orient (:meth:`SMPLestXBackend._pelvis_above_foot`). The pure half uses
it as the grounded root Z instead of the fixed nominal, so the pelvis tracks crouch/stride.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Tracks
from .detection import _iter_frames
from .pose import RawBodyMotion

#: SMPL-X body articulation = 21 joints (excludes hands/jaw/eyes); fallback pose shape.
_N_BODY_JOINTS = 21

#: SMPL-X joint indices nearest the ground (ankles + toe bases) — the foot-contact proxy (T2).
_FOOT_JOINTS = (7, 8, 10, 11)

#: Pelvis-above-foot height (m) for the rare degenerate first crop, before any FK value lands.
_FALLBACK_PELVIS_H = 0.92


@dataclass
class SMPLestXBackend:
    """Run SMPLest-X per tracklet box → camera-space SMPL-X params, keyed by ``track_id``.

    Attributes:
        repo_dir: SMPLest-X checkout (provides ``main``/``models``/``human_models`` packages
            plus ``pretrained_models/<ckpt_name>/`` and ``human_models/human_model_files``).
        ckpt_name: Sub-dir of ``pretrained_models`` holding ``config_base.py`` and
            ``<ckpt_name>.pth.tar``.
        device: Inference device.
    """

    repo_dir: str = "/workspace/repos/SMPLest-X"
    ckpt_name: str = "smplest_x_h"
    device: str = "cuda"
    _model: object = field(default=None, init=False, repr=False)
    _cfg: object = field(default=None, init=False, repr=False)
    _to_tensor: object = field(default=None, init=False, repr=False)
    _smplx: object = field(default=None, init=False, repr=False)

    def estimate_bodies(  # pragma: no cover - heavy GPU path
        self, clip: ClipRef, tracks: Tracks
    ) -> dict[int, RawBodyMotion]:
        import torch

        self._load()

        # frame -> [(track_id, bbox_xyxy)] for people only (the ball has its own tracker).
        per_frame: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for tl in tracks.tracklets:
            if tl.cls == "ball":
                continue
            for f, bb in zip(tl.frames.tolist(), tl.bboxes_xyxy, strict=True):
                per_frame[int(f)].append((tl.track_id, bb))

        acc: dict[int, dict[str, list]] = defaultdict(
            lambda: {"frames": [], "root": [], "body": [], "betas": [], "pelvis_h": []}
        )
        last_good: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

        for frame_idx, image_bgr in _iter_frames(clip):
            rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])  # SMPLest-X expects RGB (load_img)
            height, width = rgb.shape[:2]
            for track_id, bbox_xyxy in per_frame.get(frame_idx, []):
                res = self._infer_crop(torch, rgb, bbox_xyxy, width, height)
                betas = None
                if res is None:  # degenerate box: hold the last good pose (rare)
                    root, body, pelvis_h = last_good.get(
                        track_id,
                        (np.zeros(3), np.zeros((_N_BODY_JOINTS, 3)), _FALLBACK_PELVIS_H),
                    )
                else:
                    root, body, betas, pelvis_h = res
                    last_good[track_id] = (root, body, pelvis_h)
                acc[track_id]["frames"].append(frame_idx)
                acc[track_id]["root"].append(root)
                acc[track_id]["body"].append(body)
                acc[track_id]["pelvis_h"].append(pelvis_h)
                if betas is not None:
                    acc[track_id]["betas"].append(betas)

        out: dict[int, RawBodyMotion] = {}
        for track_id, a in acc.items():
            if not a["frames"]:
                continue
            frames = np.asarray(a["frames"], dtype=int)
            order = np.argsort(frames)  # _align_rows needs ascending, covering frames
            betas = (
                np.mean(np.stack(a["betas"]), axis=0)
                if a["betas"]
                else np.zeros(10)
            )
            out[track_id] = RawBodyMotion(
                track_id=track_id,
                frames=frames[order],
                global_orient=np.stack(a["root"])[order],
                body_pose=np.stack(a["body"])[order],
                betas=betas,
                pelvis_above_foot=np.asarray(a["pelvis_h"], dtype=float)[order],
            )
        return out

    def _infer_crop(self, torch, rgb, bbox_xyxy, width, height):  # pragma: no cover - heavy path
        """One box → (root (3,), body (21,3), betas (n,), pelvis_h) axis-angle, ``None`` if degenerate.

        ``pelvis_h`` is the pelvis-above-foot height (m) for the T2 foot-plane anchor (see
        :meth:`_pelvis_above_foot`).
        """
        from utils.data_utils import generate_patch_image, process_bbox

        x1, y1, x2, y2 = (float(v) for v in np.asarray(bbox_xyxy, dtype=float).reshape(4))
        bbox_xywh = np.array([x1, y1, abs(x2 - x1), abs(y2 - y1)], dtype=np.float32)
        bbox = process_bbox(
            bbox=bbox_xywh,
            img_width=width,
            img_height=height,
            input_img_shape=self._cfg.model.input_img_shape,
            ratio=getattr(self._cfg.data, "bbox_ratio", 1.25),
        )
        if bbox is None:
            return None
        patch, _, _ = generate_patch_image(
            cvimg=rgb,
            bbox=bbox,
            scale=1.0,
            rot=0.0,
            do_flip=False,
            out_shape=self._cfg.model.input_img_shape,
        )
        img = self._to_tensor(patch.astype(np.float32)) / 255
        img = img.to(self.device)[None, :, :, :]
        with torch.no_grad():
            res = self._model({"img": img}, {}, {}, "test")
        root = res["smplx_root_pose"].detach().cpu().numpy()[0].reshape(3)
        body = res["smplx_body_pose"].detach().cpu().numpy()[0].reshape(-1, 3)
        betas = res["smplx_shape"].detach().cpu().numpy()[0].reshape(-1)
        pelvis_h = self._pelvis_above_foot(torch, body, betas)
        return root, body, betas, pelvis_h

    def _pelvis_above_foot(self, torch, body, betas) -> float:  # pragma: no cover - heavy path
        """Pelvis-above-foot height (m) from SMPL-X FK at **zero** global orient (T2 anchor).

        Re-poses the predicted body in its canonical (body-up) frame — articulation kept, world
        tilt dropped — and returns the vertical (SMPL-X native +Y) drop from the pelvis to the
        lowest foot/ankle joint (:data:`_FOOT_JOINTS`). :class:`GVHMRPoseEstimator._ground_root`
        consumes it as the per-frame root Z, so a crouch/stride lowers the pelvis as it should.
        R-6 caveat: zeroing global orient assumes body-up ≈ world-up, so a markedly leaning torso
        biases the height; the foot proxy is the ankle/toe joint, ~ankle-height above the sole.
        """
        model = self._smplx_layer()
        b = np.zeros(10)
        src = np.asarray(betas, dtype=float).reshape(-1)
        b[: min(src.shape[0], 10)] = src[:10]
        with torch.no_grad():
            out = model(
                betas=torch.as_tensor(b[None], dtype=torch.float32),
                global_orient=torch.zeros(1, 3),
                body_pose=torch.as_tensor(
                    np.asarray(body, dtype=float).reshape(1, -1), dtype=torch.float32
                ),
            )
        j = out.joints.detach().cpu().numpy()[0]  # (J, 3) SMPL-X native: x-left, y-up, z-front
        return float(j[0, 1] - j[list(_FOOT_JOINTS), 1].min())

    def _smplx_layer(self):  # pragma: no cover - heavy path
        """Lazily build a standalone SMPL-X layer for the T2 FK (same .npz the net uses)."""
        if self._smplx is None:
            import smplx

            root = os.environ.get(
                "PITCH3D_SMPLX_MODEL_PATH",
                os.path.join(self.repo_dir, "human_models", "human_model_files"),
            )
            self._smplx = smplx.create(
                root, model_type="smplx", gender="neutral",
                use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=1,
            )
        return self._smplx

    def _load(self) -> None:  # pragma: no cover - heavy GPU path
        if self._model is not None:
            return
        import sys
        import tempfile

        import torchvision.transforms as transforms

        repo = os.path.abspath(self.repo_dir)
        if not os.path.isdir(repo):
            raise RuntimeError(
                f"SMPLest-X checkout not found at {repo!r}. Set PITCH3D_SMPLESTX_REPO, or clone "
                "https://github.com/SMPLCap/SMPLest-X and stage pretrained_models/ + "
                "human_models/human_model_files/."
            )
        if repo not in sys.path:
            sys.path.insert(0, repo)

        cwd = os.getcwd()
        try:
            os.chdir(repo)  # SMPLest-X resolves a few paths relative to its root
            from human_models.human_models import SMPLX
            from main.base import Tester
            from main.config import Config

            ckpt_dir = os.path.join(repo, "pretrained_models", self.ckpt_name)
            cfg = Config.load_config(os.path.join(ckpt_dir, "config_base.py"))
            logdir = os.path.join(tempfile.gettempdir(), "pitch3d_smplestx_log")
            cfg.update_config(
                {
                    "model": {
                        "pretrained_model_path": os.path.join(
                            ckpt_dir, f"{self.ckpt_name}.pth.tar"
                        ),
                        "human_model_path": os.path.join(
                            repo, "human_models", "human_model_files"
                        ),
                    },
                    "log": {
                        "exp_name": "pitch3d_infer",
                        "output_dir": logdir,
                        "model_dir": logdir,
                        "log_dir": logdir,
                        "result_dir": logdir,
                    },
                }
            )
            cfg.prepare_log()
            SMPLX(cfg.model.human_model_path)  # build the human-model singleton the net consumes
            tester = Tester(cfg)
            tester._make_model()  # builds graph + remaps/loads the checkpoint (strict=False)
            self._model = tester.model
            self._cfg = cfg
            self._to_tensor = transforms.ToTensor()
        finally:
            os.chdir(cwd)


def make() -> SMPLestXBackend:
    """Zero-arg factory for ``--pose-backend``; config comes from the environment."""
    return SMPLestXBackend(
        repo_dir=os.environ.get("PITCH3D_SMPLESTX_REPO", "/workspace/repos/SMPLest-X"),
        ckpt_name=os.environ.get("PITCH3D_SMPLESTX_CKPT", "smplest_x_h"),
        device=os.environ.get("PITCH3D_DEVICE", "cuda"),
    )
