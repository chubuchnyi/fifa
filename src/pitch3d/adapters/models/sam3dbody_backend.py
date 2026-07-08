"""SAM 3D Body HMR backend — variant B of the pose bake-off (alt to SMPLest-X, M1/FR-8).

Like :mod:`~pitch3d.adapters.models.smplestx_backend`, this is a heavy
:class:`~pitch3d.adapters.models.pose.HMRBackend` behind the same port, so the pure
grounding/assembly half (:class:`GVHMRPoseEstimator`) is unchanged and the two nets can be
compared by swapping ONLY ``--pose-backend`` (identical detect/track/calibrate upstream).

**SAM 3D Body (3DB)** (Meta, github.com/facebookresearch/sam-3d-body) is a *promptable*
single-image full-body mesh recovery model. We feed OUR ByteTrack boxes as the box prompt
(``process_one_image(img, bboxes=...)``) and skip its bundled detector/segmentor/FOV modules.

Coordinate/representation note (the reason this backend is NOT a drop-in like SMPLest-X):
3DB predicts the body on the **Momentum Human Rig (MHR)**, *not* SMPL-X. The rest of pitch3d
(scene, physics, anim_export, FK) is SMPL-X. So each prediction is bridged MHR→SMPL-X with
Meta's own converter (``facebookresearch/MHR`` ``tools/mhr_smpl_conversion``:
:class:`Conversion.convert_sam3d_output_to_smpl`), a per-person PyTorch mesh fit. As with
SMPLest-X, the resulting ``global_orient`` is in the model's **camera** frame — the world
*translation* is the pure half's foot→homography job, and the camera→world rotation lift is
still owed downstream (shared blocker with variant A).

Everything heavy (torch, the sam_3d_body package, the MHR package + converter, the gated
checkpoint) imports/loads **lazily** on first ``estimate_bodies`` — importing this module is
cheap and torch-free, matching the other real adapters.

Native ABI pin (verified 2026-07-08 on RTX PRO 4500 Blackwell / torch 2.8.0+cu128):
the MHR rig load (``pymomentum.geometry.Character.load_fbx`` inside ``MHR.from_files``)
**segfaults** with ``pymomentum-gpu`` wheels >=0.1.97 — their bundled libtorch ABI does
not match torch 2.8. Pin ``pymomentum-gpu==0.1.90.post0`` (the newest that loads cleanly).
The solver extension also needs ``LD_LIBRARY_PATH`` to include ``<torch>/lib`` (it links
``libtorch.so`` at import). Both are wired in ``scripts/run_sam3dbody.sh``.

Wire it in at the composition root::

    --pose gvhmr --pose-backend pitch3d.adapters.models.sam3dbody_backend:make --device cuda

The zero-arg :func:`make` factory reads its config from the environment:

* ``PITCH3D_SAM3D_REPO``   — sam-3d-body checkout (default ``/workspace/repos/sam-3d-body``)
* ``PITCH3D_MHR_REPO``     — MHR checkout providing ``mhr`` + the converter
                             (default ``/workspace/repos/MHR``)
* ``PITCH3D_SAM3D_CKPT``   — 3DB checkpoint ``model.ckpt`` (gated on HuggingFace —
                             ``facebook/sam-3d-body-dinov3``; request access + ``hf download``)
* ``PITCH3D_SAM3D_MHR_ASSET`` — MHR asset ``mhr_model.pt`` shipped alongside the checkpoint
* ``PITCH3D_SMPLX_MODELS`` — SMPL-X model dir for the fit target + the T2 FK
                             (default the SMPLest-X ``human_model_files`` dir)
* ``PITCH3D_DEVICE``       — inference device (default ``cuda``)

T2 foot-plane anchor: exactly as in the SMPLest-X backend, this fills
:attr:`RawBodyMotion.pelvis_above_foot` from a zero-global-orient SMPL-X FK so the grounded
root Z tracks crouch/stride — computed with the SAME helper so A and B differ only in
articulation, not in how the root is placed.
"""

from __future__ import annotations

import contextlib
import os
import sys
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

#: Pelvis-above-foot height (m) for the rare degenerate crop, before any FK value lands.
_FALLBACK_PELVIS_H = 0.92


@dataclass
class SAM3DBodyBackend:
    """Run SAM 3D Body per tracklet box → MHR → SMPL-X (camera-space), keyed by ``track_id``.

    Attributes:
        repo_dir: sam-3d-body checkout (provides the ``sam_3d_body`` package + ``tools``).
        mhr_repo_dir: MHR checkout (provides the ``mhr`` package and the
            ``tools/mhr_smpl_conversion`` converter used to bridge MHR→SMPL-X).
        ckpt_path: 3DB checkpoint (``model.ckpt``); gated on HuggingFace.
        mhr_asset_path: MHR asset (``mhr_model.pt``) shipped with the checkpoint.
        smplx_model_dir: SMPL-X ``human_model_files`` dir (fit target + T2 FK).
        device: Inference device.
        bbox_thr: forwarded to ``process_one_image`` (unused when boxes are provided, kept
            for parity with the demo).
        convert_batch: batch size for the MHR→SMPL-X PyTorch fit (all crops at once).
    """

    repo_dir: str = "/workspace/repos/sam-3d-body"
    mhr_repo_dir: str = "/workspace/repos/MHR"
    ckpt_path: str = (
        "/workspace/repos/sam-3d-body/checkpoints/sam-3d-body-dinov3/model.ckpt"
    )
    mhr_asset_path: str = (
        "/workspace/repos/sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
    )
    smplx_model_dir: str = "/workspace/repos/SMPLest-X/human_models/human_model_files"
    device: str = "cuda"
    bbox_thr: float = 0.5
    convert_batch: int = 256
    _estimator: object = field(default=None, init=False, repr=False)
    _converter: object = field(default=None, init=False, repr=False)
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

        # One 3DB pass per frame (all its boxes at once), collecting every crop's raw MHR
        # output with a parallel (track_id, frame) index so a single batched MHR→SMPL-X fit
        # can be scattered back afterwards.
        sam3d_outputs: list[dict] = []
        index: list[tuple[int, int]] = []  # (track_id, frame_idx) per row of sam3d_outputs
        for frame_idx, image_bgr in _iter_frames(clip):
            entries = per_frame.get(frame_idx, [])
            if not entries:
                continue
            ids = [tid for tid, _ in entries]
            boxes = np.stack(
                [np.asarray(bb, dtype=np.float32).reshape(4) for _, bb in entries]
            )
            rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])  # process_one_image wants RGB
            outs = self._estimator.process_one_image(
                rgb, bboxes=boxes, bbox_thr=self.bbox_thr, use_mask=False,
                inference_type="body",  # broadcast soccer is body-only; skip the hand decoder
            )
            # With explicit boxes there is no NMS/reordering: outs[i] ↔ boxes[i] ↔ ids[i].
            for tid, out in zip(ids, outs, strict=True):
                sam3d_outputs.append(out)
                index.append((tid, frame_idx))

        if not sam3d_outputs:
            return {}

        params = self._convert_to_smplx(sam3d_outputs)  # dict of (P, …) SMPL-X params

        # Scatter the per-row SMPL-X params back onto each tracklet, then the T2 FK height.
        acc: dict[int, dict[str, list]] = defaultdict(
            lambda: {"frames": [], "root": [], "body": [], "betas": [], "pelvis_h": []}
        )
        for row, (tid, frame_idx) in enumerate(index):
            root = np.asarray(params["global_orient"][row], dtype=float).reshape(3)
            body = np.asarray(params["body_pose"][row], dtype=float).reshape(-1, 3)[
                :_N_BODY_JOINTS
            ]
            betas = np.asarray(params["betas"][row], dtype=float).reshape(-1)
            acc[tid]["frames"].append(frame_idx)
            acc[tid]["root"].append(root)
            acc[tid]["body"].append(body)
            acc[tid]["betas"].append(betas)
            acc[tid]["pelvis_h"].append(self._pelvis_above_foot(torch, body, betas))

        out: dict[int, RawBodyMotion] = {}
        for track_id, a in acc.items():
            if not a["frames"]:
                continue
            frames = np.asarray(a["frames"], dtype=int)
            order = np.argsort(frames)  # _align_rows needs ascending, covering frames
            betas = np.mean(np.stack(a["betas"]), axis=0)
            out[track_id] = RawBodyMotion(
                track_id=track_id,
                frames=frames[order],
                global_orient=np.stack(a["root"])[order],
                body_pose=np.stack(a["body"])[order],
                betas=betas,
                pelvis_above_foot=np.asarray(a["pelvis_h"], dtype=float)[order],
            )
        return out

    def _convert_to_smplx(self, sam3d_outputs):  # pragma: no cover - heavy path
        """Batched MHR→SMPL-X fit → dict with ``global_orient``/``body_pose``/``betas``.

        Delegates to Meta's converter (``convert_sam3d_output_to_smpl``, PyTorch method), then
        normalises the returned SMPL-X parameter dict to numpy arrays. The converter accepts the
        raw ``process_one_image`` dicts directly (it reads ``mhr_model_params``/``pred_vertices``/
        ``pred_cam_t`` and handles the SAM3D↔MHR camera flip internally).
        """
        with self._conv_cwd():
            res = self._converter.convert_sam3d_output_to_smpl(
                sam3d_outputs=sam3d_outputs,
                return_smpl_meshes=False,
                return_smpl_parameters=True,
                return_smpl_vertices=False,
                return_fitting_errors=False,
                batch_size=self.convert_batch,
            )
        raw = res.result_parameters

        def _np(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        params = {k: _np(v) for k, v in raw.items()}
        if "global_orient" not in params or "body_pose" not in params:
            raise RuntimeError(
                "MHR→SMPL-X converter returned unexpected keys "
                f"{sorted(params)}; expected 'global_orient'+'body_pose'(+'betas')."
            )
        n = params["global_orient"].shape[0]
        params.setdefault("betas", np.zeros((n, 10), dtype=float))
        return params

    def _pelvis_above_foot(self, torch, body, betas) -> float:  # pragma: no cover - heavy path
        """Pelvis-above-foot height (m) from SMPL-X FK at zero global orient (T2 anchor).

        Identical measurement to the SMPLest-X backend, so variants A and B ground the root the
        same way and differ only in the predicted articulation.
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
        if self._smplx is None:
            import smplx

            self._smplx = smplx.create(
                self.smplx_model_dir, model_type="smplx", gender="neutral",
                use_pca=False, flat_hand_mean=True, num_betas=10, batch_size=1,
            )
        return self._smplx

    def _load(self) -> None:  # pragma: no cover - heavy GPU path
        if self._estimator is not None:
            return
        import torch

        for path in (self.repo_dir, self.mhr_repo_dir,
                     os.path.join(self.mhr_repo_dir, "tools", "mhr_smpl_conversion")):
            ap = os.path.abspath(path)
            if not os.path.isdir(ap):
                raise RuntimeError(
                    f"SAM 3D Body dependency not found at {ap!r}. Clone "
                    "github.com/facebookresearch/{sam-3d-body,MHR} into PITCH3D_SAM3D_REPO / "
                    "PITCH3D_MHR_REPO."
                )
            if ap not in sys.path:
                sys.path.insert(0, ap)
        if not os.path.isfile(self.ckpt_path):
            raise RuntimeError(
                f"SAM 3D Body checkpoint not found at {self.ckpt_path!r}. It is GATED on "
                "HuggingFace: request access at huggingface.co/facebook/sam-3d-body-dinov3, "
                "then `hf download facebook/sam-3d-body-dinov3 --local-dir "
                "<repo>/checkpoints/sam-3d-body-dinov3` (needs an authenticated HF token)."
            )

        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body

        device = torch.device(self.device)
        model, model_cfg = load_sam_3d_body(
            self.ckpt_path, device=device, mhr_path=self.mhr_asset_path
        )
        # No detector/segmentor/FOV: we supply boxes and the pure half owns world placement.
        self._estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model, model_cfg=model_cfg,
            human_detector=None, human_segmentor=None, fov_estimator=None,
        )
        self._converter = self._build_converter(device)

    def _import_conversion(self):  # pragma: no cover - heavy path
        """Import Meta's ``Conversion``, forcing its bare ``from utils import ...`` (and the
        ``pytorch_fitting``/``conversion`` chain) to resolve against the converter dir.

        Those modules use un-namespaced top-level names, so whichever ``utils`` another
        dependency imported first wins in ``sys.modules`` — a non-deterministic collision
        that fails the fit with ``cannot import name ... from 'utils'``. Pin the converter
        dir to the front of ``sys.path`` and evict any cached generic-named module that is
        not the converter's own, so the import re-resolves correctly.
        """
        conv_dir = os.path.abspath(
            os.path.join(self.mhr_repo_dir, "tools", "mhr_smpl_conversion")
        )
        if not sys.path or sys.path[0] != conv_dir:
            sys.path.insert(0, conv_dir)
        for name in ("conversion", "pytorch_fitting", "utils", "constants",
                     "rotation_utils", "smpl_fitting", "mhr_fitting"):
            mod = sys.modules.get(name)
            if mod is None:
                continue
            f = getattr(mod, "__file__", None)
            if not f or not os.path.abspath(f).startswith(conv_dir):
                del sys.modules[name]
        from conversion import Conversion
        return Conversion

    def _build_converter(self, device):  # pragma: no cover - heavy path
        import smplx

        from mhr.mhr import MHR

        Conversion = self._import_conversion()
        mhr_model = MHR.from_files(lod=1, device=device)
        # ``smplx.create`` appends the ``smplx/`` model-type subdir under the given
        # root (matching SMPLest-X's own loader), so ``smplx_model_dir`` points at
        # ``human_model_files`` and the layer resolves ``smplx/SMPLX_NEUTRAL.npz``.
        smplx_model = smplx.create(
            self.smplx_model_dir, model_type="smplx", gender="neutral",
            use_pca=False, flat_hand_mean=True,
        ).to(device)
        # ``Conversion`` loads mapping/mask assets via cwd-relative ``./assets/...``
        # paths at BOTH construction and conversion time, so pin cwd for each.
        with self._conv_cwd():
            return Conversion(
                mhr_model=mhr_model, smpl_model=smplx_model, method="pytorch"
            )

    @contextlib.contextmanager
    def _conv_cwd(self):  # pragma: no cover - heavy path
        """Pin cwd to the MHR conversion-tool dir so its ``./assets/*`` reads resolve."""
        conv_dir = os.path.join(self.mhr_repo_dir, "tools", "mhr_smpl_conversion")
        prev_cwd = os.getcwd()
        try:
            os.chdir(conv_dir)
            yield
        finally:
            os.chdir(prev_cwd)


def make() -> SAM3DBodyBackend:
    """Zero-arg factory for ``--pose-backend``; config comes from the environment."""
    repo = os.environ.get("PITCH3D_SAM3D_REPO", "/workspace/repos/sam-3d-body")
    ckpt_default = os.path.join(repo, "checkpoints", "sam-3d-body-dinov3", "model.ckpt")
    mhr_asset_default = os.path.join(
        repo, "checkpoints", "sam-3d-body-dinov3", "assets", "mhr_model.pt"
    )
    return SAM3DBodyBackend(
        repo_dir=repo,
        mhr_repo_dir=os.environ.get("PITCH3D_MHR_REPO", "/workspace/repos/MHR"),
        ckpt_path=os.environ.get("PITCH3D_SAM3D_CKPT", ckpt_default),
        mhr_asset_path=os.environ.get("PITCH3D_SAM3D_MHR_ASSET", mhr_asset_default),
        smplx_model_dir=os.environ.get(
            "PITCH3D_SMPLX_MODELS",
            "/workspace/repos/SMPLest-X/human_models/human_model_files",
        ),
        device=os.environ.get("PITCH3D_DEVICE", "cuda"),
    )
