"""PnLCalib pitch backend — the real ``KeypointBackend`` **and** ``HomographyBackend`` (B1 eval).

Wraps the public **PnLCalib** field-calibration network (two HRNet heads — pitch *keypoints* and
pitch *lines*; Gutiérrez-Pérez & Agudo, 2024). One backend, two solver paths so the eval can A/B
them through the same dotted-path seam:

* :meth:`detect_keypoints` (``KeypointBackend``) — emits raw image↔world landmark correspondences
  **plus point-on-line observations** taken from the line head, and lets the downstream
  :class:`KeypointFieldCalibrator` fit a planar homography by weighted DLT. The line head runs on
  every frame anyway (it is what completes the occluded keypoints), so its straight-line detections
  are evidence already paid for; feeding them in as ``lᵀ·H·x = 0`` rows costs one matrix row each
  and keeps frames solvable when few keypoint *intersections* are visible.
* :meth:`calibrate_frames` (``HomographyBackend``) — runs PnLCalib's **full camera module**
  (:class:`FramebyFrameCalib` — points **and** lines, mode + RANSAC voting, optional PnL line
  refinement) on the box and emits a ready image→world homography per frame, which
  :class:`CameraModuleFieldCalibrator` only scores + smooths. This is the stronger lever.

Both share the same front half (:meth:`_infer_frame`: resize → both HRNet heads → heatmap decode).
World coords are metres, centre-origin, ``Z = 0`` — identical frame for both paths, so their output
homographies are directly comparable.

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
    * ``PNLCALIB_PNL_REFINE`` — camera-path PnL line refinement (``1``/true default; only affects
      :meth:`calibrate_frames`)
    * ``PNLCALIB_USE_LINES`` — emit point-on-line constraints from the line head (``1``/true
      default; only affects :meth:`detect_keypoints`)

The world tables (``keypoint_world_coords_2D`` / ``…aux…``, already centre-origin in PnLCalib's
``utils.utils_calib``) are **imported from the installed PnLCalib**, never copied, so the id→world
mapping always matches the weights actually loaded. All heavy imports (torch, cv2, PnLCalib) are
lazy, so importing this module is cheap and dependency-free — only the work methods
(:meth:`detect_keypoints` / :meth:`calibrate_frames`) pull the stack. PnLCalib needs ``shapely``
(``pip install shapely`` into the box venv).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from ...core.ports.io import ClipRef
from ...core.scene.pitch import pitch_line_coefficients
from .calibration import (
    FrameHomography,
    FrameKeypoints,
    image_to_world_from_cam_params,
    point_line_residual,
    solve_homography_ransac,
)

# PnLCalib resizes every frame to this fixed input before the HRNet heads; its heatmap decoding
# returns landmark coords in this space, and ``complete_keypoints(normalize=True)`` divides by it.
_INPUT_W = 960
_INPUT_H = 540
# PnLCalib keypoint ids are 1-based: ids 1..57 index the main table, ids ≥58 the aux table.
_N_MAIN = 57


def _unletterbox_point(d: dict, scale: float, pad_x: int, pad_y: int,
                       w_orig: int, h_orig: int) -> bool:
    """Map one decoded ``{"x","y"}`` back to the ORIGINAL frame in place; False if it is outside.

    The heads see a padded, aspect-preserved copy, and `complete_keypoints(normalize=True)` divides
    by the *input* size — so a point comes back normalised against the padded image. Undo the pad
    and the scale, then renormalise against the original, after which every consumer that
    multiplies by ``w_orig`` is correct without knowing any of this happened.

    A point landing in the padding is dropped rather than clamped: it is outside the picture, and a
    clamped landmark is a correspondence to a place the camera never saw.
    """
    if "x" not in d or "y" not in d:
        return True
    x = (float(d["x"]) * _INPUT_W - pad_x) / scale
    y = (float(d["y"]) * _INPUT_H - pad_y) / scale
    if not (0.0 <= x <= w_orig and 0.0 <= y <= h_orig):
        return False
    d["x"], d["y"] = x / w_orig, y / h_orig
    return True

# SoccerNet line class → world ``(a, b, c)``, ``a² + b² = 1``. PnLCalib's line head emits SoccerNet
# class names, so a detection keys straight into this; unknown names (circles, goal frames) are
# dropped because they do not constrain a planar homography.
_PITCH_LINES = pitch_line_coefficients()

# Median point-on-line residual (m) above which the line detections are judged to disagree with the
# keypoints' own world frame, and dropped. A correct frame lands well inside a metre; a mirrored or
# mis-scaled one lands tens of metres out, so anything in between is already worth refusing.
_LINE_FRAME_TOL_M = 3.0


@dataclass
class _PnLCalibBackend:
    """Run PnLCalib's keypoint + line heads and emit per-frame image↔world landmark matches."""

    repo: str
    weights_kp: str
    weights_lines: str
    device: str = "cuda:0"
    kp_threshold: float = 0.3434
    line_threshold: float = 0.7867
    pnl_refine: bool = True
    use_lines: bool = True
    _state: dict = field(default_factory=dict, init=False, repr=False)

    def _infer_frame(  # pragma: no cover - heavy path
        self, s: dict, bgr: np.ndarray
    ) -> tuple[dict, dict, int, int]:
        """Run both HRNet heads on one BGR frame → ``(kp_dict, lines_dict, w_orig, h_orig)``.

        The shared front half of both backends: resize → keypoint + line heads → PnLCalib heatmap
        decode → ``complete_keypoints(normalize=True)``. The dicts are **normalised** ``[0, 1]`` (as
        PnLCalib's own ``inference.py`` produces them): :meth:`detect_keypoints` multiplies them to
        original px itself, while :meth:`calibrate_frames` hands them to a ``denormalize=True``
        camera that scales them by the original frame dims. ``lines_dict`` feeds both paths: the
        camera module consumes it directly, the DLT path turns it into point-on-line constraints.
        """
        torch, cv2, Image = s["torch"], s["cv2"], s["Image"]
        h_orig, w_orig = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = s["to_tensor"](Image.fromarray(rgb)).float().unsqueeze(0)

        # LETTERBOX, not stretch. `transforms.Resize((540, 960))` squashes whatever it is given
        # into 16:9, so a 1080x1920 portrait clip reached the heads at 0.5x across and 0.28x down —
        # a body of evidence the network was never trained on, and every number produced from a
        # portrait clip was under that handicap. Fit inside the input instead and pad; on a 16:9
        # clip `scale` fits exactly and both pads are 0, so the broadcast path is bit-unchanged.
        scale = min(_INPUT_W / w_orig, _INPUT_H / h_orig)
        new_w, new_h = max(1, round(w_orig * scale)), max(1, round(h_orig * scale))
        pad_x, pad_y = (_INPUT_W - new_w) // 2, (_INPUT_H - new_h) // 2
        if (new_w, new_h) != (w_orig, h_orig):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
        canvas = torch.zeros((1, tensor.shape[1], _INPUT_H, _INPUT_W), dtype=tensor.dtype)
        canvas[:, :, pad_y:pad_y + new_h, pad_x:pad_x + new_w] = tensor
        tensor = canvas.to(self.device)

        with torch.no_grad():
            heatmaps = s["model"](tensor)
            heatmaps_l = s["model_l"](tensor)
        kp_raw = s["get_kp"](heatmaps[:, :-1, :, :])
        line_raw = s["get_line"](heatmaps_l[:, :-1, :, :])
        kp_list = s["coords_to_dict"](kp_raw, threshold=self.kp_threshold)
        line_list = s["coords_to_dict"](line_raw, threshold=self.line_threshold)
        kp_dict, lines_dict = s["complete"](
            kp_list[0], line_list[0], w=_INPUT_W, h=_INPUT_H, normalize=True
        )
        # Undo the letterbox HERE, so the dicts are normalised against the ORIGINAL frame — which
        # is what this method's docstring already promised and what every consumer below assumes
        # when it multiplies by `w_orig` or hands them to a `denormalize=True` camera.
        box = (scale, pad_x, pad_y, w_orig, h_orig)
        for kid in [k for k, d in kp_dict.items()
                    if isinstance(d, dict) and not _unletterbox_point(d, *box)]:
            kp_dict.pop(kid, None)
        for name, pts in list(lines_dict.items()):
            if isinstance(pts, (list, tuple)):
                lines_dict[name] = [d for d in pts if isinstance(d, dict)
                                    and _unletterbox_point(d, *box)]
        return kp_dict, lines_dict, int(w_orig), int(h_orig)

    def _line_observations(
        self, lines_dict: dict, w_orig: int, h_orig: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Line-head detections → point-on-line observations ``(uv (M,2), abc (M,3), conf (M,))``.

        Each detected straight pitch line contributes its endpoints as points known to lie on that
        line's world equation — evidence the DLT path used to throw away, even though the line head
        already ran. Classes absent from the pitch template are skipped, which is exactly the right
        filter: the circles are curved and the goal frames sit at ``Z ≠ 0``, so neither constrains a
        planar homography.
        """
        uv: list[list[float]] = []
        abc: list[np.ndarray] = []
        conf: list[float] = []
        for name, pts in lines_dict.items():
            line = _PITCH_LINES.get(str(name))
            if line is None or not isinstance(pts, (list, tuple)):
                continue
            for d in pts:
                try:
                    x, y = float(d["x"]), float(d["y"])
                except (TypeError, KeyError, ValueError):
                    continue
                uv.append([x * w_orig, y * h_orig])  # normalised → original image px
                abc.append(line)
                conf.append(float(d.get("p", 1.0)) if isinstance(d, dict) else 1.0)
        return (
            np.asarray(uv, dtype=float).reshape(-1, 2),
            np.asarray(abc, dtype=float).reshape(-1, 3),
            np.asarray(conf, dtype=float).reshape(-1),
        )

    def detect_keypoints(self, clip: ClipRef) -> list[FrameKeypoints]:
        """Detect pitch landmarks per frame of ``clip`` (a video file or a directory of frames)."""
        s = self._load()

        out: list[FrameKeypoints] = []
        for idx, bgr in s["iter_frames"](clip):
            kp_dict, lines_dict, w_orig, h_orig = self._infer_frame(s, bgr)

            uv, world, conf = [], [], []
            for kid, d in kp_dict.items():
                wp = s["kw"][kid - 1] if kid <= _N_MAIN else s["ka"][kid - 1 - _N_MAIN]
                uv.append([d["x"] * w_orig, d["y"] * h_orig])  # normalised → original image px
                # Y negated out of PnLCalib's top-down template into our Z-up world (#118, see
                # `calibration.TEMPLATE_TO_WORLD`), so this path and the camera-module path below
                # stay in one frame — the whole point of scoring them against each other.
                world.append([float(wp[0]), -float(wp[1])])  # metres, centre-origin, Z=0
                conf.append(float(d.get("p", 1.0)))
            image_uv = np.asarray(uv, dtype=float).reshape(-1, 2)
            world_xy = np.asarray(world, dtype=float).reshape(-1, 2)

            l_uv, l_abc, l_conf = (
                self._line_observations(lines_dict, w_orig, h_orig)
                if self.use_lines
                else (np.empty((0, 2)), np.empty((0, 3)), np.empty(0))
            )
            if l_uv.size and not self._lines_agree(image_uv, world_xy, l_uv, l_abc):
                l_uv, l_abc, l_conf = np.empty((0, 2)), np.empty((0, 3)), np.empty(0)

            out.append(
                FrameKeypoints(
                    frame=int(idx),
                    image_uv=image_uv,
                    world_xy=world_xy,
                    confidence=np.asarray(conf, dtype=float).reshape(-1),
                    line_uv=l_uv if l_uv.size else None,
                    line_abc=l_abc if l_uv.size else None,
                    line_confidence=l_conf if l_uv.size else None,
                )
            )
        return out

    def _lines_agree(
        self, image_uv: np.ndarray, world_xy: np.ndarray, l_uv: np.ndarray, l_abc: np.ndarray
    ) -> bool:
        """Do the line detections sit on the pitch the *keypoints* describe? Cached per run.

        The keypoints' world coords come from PnLCalib's own table; the line classes' world
        equations come from our pitch template. Those are two independently-authored statements
        about the same pitch, so a mismatched axis convention would be silent and catastrophic —
        every line constraint pulling toward a mirrored pitch. So: fit points-only, measure the
        median point-on-line residual, and only trust the lines if it is plausible (a wrong frame
        lands tens of metres out, a right one well inside the pitch). Auto-detect with a manual
        override — ``PNLCALIB_USE_LINES=0`` disables the whole path.
        """
        cached = self._state.get("lines_agree")
        if cached is not None:
            return bool(cached)
        if image_uv.shape[0] < 6:
            return False  # too thin to judge on; wait for a better frame
        try:
            h, _ = solve_homography_ransac(image_uv, world_xy, threshold=1.0)
            med = float(np.median(point_line_residual(h, l_uv, l_abc)))
        except (ValueError, np.linalg.LinAlgError):
            return False
        ok = bool(np.isfinite(med) and med < _LINE_FRAME_TOL_M)
        self._state["lines_agree"] = ok
        print(
            f"[pnlcalib] line-constraint frame check: median residual {med:.2f} m "
            f"→ {'using' if ok else 'DISABLED, tolerance ' + str(_LINE_FRAME_TOL_M) + ' m'}",
            flush=True,
        )
        return ok

    def calibrate_frames(self, clip: ClipRef) -> list[FrameHomography]:
        """Full camera-module solve per frame of ``clip`` → image→world homographies.

        The camera-module counterpart of :meth:`detect_keypoints`: instead of emitting raw
        correspondences for a downstream DLT, it feeds the detected keypoints **and pitch lines**
        into PnLCalib's :class:`FramebyFrameCalib` and runs ``heuristic_voting`` (mode + RANSAC
        sweep, optional PnL line refinement), then converts the winning camera parameters to an
        image→world homography in the same centre-origin metric frame the DLT path uses. A frame
        whose solve fails (no consensus, or a degenerate/singular camera) is emitted with
        ``homography=None`` so the calibrator can surface it as zero-confidence drift (R-6), never a
        crash that aborts the whole clip.
        """
        s = self._load()
        cam_cls = s["cam_cls"]

        out: list[FrameHomography] = []
        for idx, bgr in s["iter_frames"](clip):
            kp_dict, lines_dict, w_orig, h_orig = self._infer_frame(s, bgr)
            n_landmarks = len(kp_dict)
            homography: np.ndarray | None = None
            rep_err: float | None = None
            try:
                cam = cam_cls(iwidth=w_orig, iheight=h_orig, denormalize=True)
                cam.update(kp_dict, lines_dict)
                result = cam.heuristic_voting(refine_lines=self.pnl_refine)
                if result is not None:
                    rep_err = float(result["rep_err"])
                    homography = image_to_world_from_cam_params(result["cam_params"])
            except Exception:  # noqa: BLE001 — degenerate view → unsolved frame, not a clip-killer
                homography, rep_err = None, None
            out.append(
                FrameHomography(
                    frame=int(idx),
                    homography=homography,
                    rep_err_px=rep_err,
                    n_landmarks=int(n_landmarks),
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
            FramebyFrameCalib,  # full camera-module solve: points + lines, mode/RANSAC sweep
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
            "cam_cls": FramebyFrameCalib,
            "kw": keypoint_world_coords_2D,
            "ka": keypoint_aux_world_coords_2D,
            "iter_frames": _iter_frames,
        }
        return self._state


def make(
    *,
    kp_threshold: float | None = None,
    line_threshold: float | None = None,
) -> _PnLCalibBackend:
    """Zero-arg factory (ADR-0006 seam) → an env-configured PnLCalib ``KeypointBackend``.

    The dotted-path seam calls this with no arguments, so configuration comes from the environment
    (see the module docstring). ``kp_threshold`` / ``line_threshold`` may be passed explicitly to
    override the ``PNLCALIB_*_THRESHOLD`` env defaults — used by the calibration eval's threshold
    sweep to trade completeness against accuracy (precedence: explicit kwarg > env > PnLCalib
    default). Building the backend is torch-free; the heavy stack loads only on first detect call.
    """
    repo = os.environ.get("PNLCALIB_REPO", "/workspace/repos/PnLCalib")
    kp_th = (
        kp_threshold
        if kp_threshold is not None
        else float(os.environ.get("PNLCALIB_KP_THRESHOLD", "0.3434"))
    )
    line_th = (
        line_threshold
        if line_threshold is not None
        else float(os.environ.get("PNLCALIB_LINE_THRESHOLD", "0.7867"))
    )
    return _PnLCalibBackend(
        repo=repo,
        weights_kp=os.environ.get("PNLCALIB_WEIGHTS_KP", "/workspace/weights/pnlcalib/SV_kp"),
        weights_lines=os.environ.get(
            "PNLCALIB_WEIGHTS_LINES", "/workspace/weights/pnlcalib/SV_lines"
        ),
        device=os.environ.get("PNLCALIB_DEVICE", "cuda:0"),
        kp_threshold=kp_th,
        line_threshold=line_th,
        pnl_refine=_env_flag("PNLCALIB_PNL_REFINE", default=True),
        use_lines=_env_flag("PNLCALIB_USE_LINES", default=True),
    )


def _env_flag(name: str, *, default: bool) -> bool:
    """Read a boolean env var (``1/0``, ``true/false``, ``yes/no``); unset → ``default``."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
