"""Cutie mask propagation — the temporal cue #133 measured a need for.

Three cheap fixes came back null against the same failure: players lose their identity at a
crossing, 96 % of the time with an unclaimed detection sitting a median 6-23 px away
(`scripts/identity_failure_kind.py`). The boxes are there; IoU plus a Kalman prediction cannot say
which one belongs to which track. McByte's answer is a mask that is **propagated from earlier
frames**, so it is independent of the detection being judged — a per-frame SAM mask prompted by the
box cannot substitute, because it is derived from the very box under test.

This wraps `Cutie <https://github.com/hkchengrex/Cutie>`_ (MIT) as that propagator:

* :meth:`CutiePropagator.seed` — start tracking a set of labelled masks on a frame.
* :meth:`CutiePropagator.step` — advance one frame, returning a label image where each pixel
  carries the id of the object it belongs to (0 = background).

Kept free of any tracker coupling: it takes images and masks and returns label images. The
association cue that consumes it lives with the tracker, and the seeding is the caller's business —
we seed with the SAM we already have rather than pulling in `segment_anything` for a second copy.

Heavy imports are lazy, matching the other adapters. Config comes from the environment:

* ``PITCH3D_CUTIE_REPO``    — checkout dir (default ``backends/McByte/mask_propagation/Cutie``)
* ``PITCH3D_CUTIE_WEIGHTS`` — ``cutie-base-mega.pth`` (default: the repo's ``weights/``)
* ``PITCH3D_DEVICE``        — inference device (default ``cpu``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CutiePropagator:
    """Propagate labelled masks across frames; one label image per frame.

    Attributes:
        repo_dir: Cutie checkout (provides the ``cutie`` package and its hydra config tree).
        weights: Path to ``cutie-base-mega.pth``. Verified md5 ``a6071de6…`` / 140 443 788 B —
            a truncated download of this file still *exists* and still has a plausible size, and
            fails only when torch tries to read the zip. Check the length, not just the path.
        device: Inference device.
        max_side: Longest image side fed to the network. Cutie is resolution-hungry and our
            players are ~35 px wide, so downscaling costs exactly the detail the cue is for;
            0 disables the resize.
    """

    repo_dir: str = "backends/McByte/mask_propagation/Cutie"
    weights: str | None = None
    device: str = "cpu"
    max_side: int = 0
    _core: object = field(default=None, init=False, repr=False)
    _torch: object = field(default=None, init=False, repr=False)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def _load(self) -> None:  # pragma: no cover - heavy path (needs the checkout + weights)
        if self._core is not None:
            return
        import sys

        import torch

        repo = os.path.abspath(self.repo_dir)
        if not os.path.isdir(repo):
            raise RuntimeError(
                f"Cutie checkout not found at {repo!r}. Set PITCH3D_CUTIE_REPO, or clone "
                "https://github.com/tstanczyk95/McByte (it vendors Cutie under "
                "mask_propagation/Cutie) and stage weights/cutie-base-mega.pth."
            )
        weights = self.weights or os.path.join(repo, "weights", "cutie-base-mega.pth")
        if not os.path.isfile(weights):
            raise RuntimeError(f"Cutie weights not found at {weights!r}")

        if repo not in sys.path:
            sys.path.insert(0, repo)
        from cutie.inference.inference_core import InferenceCore
        from cutie.inference.utils.args_utils import get_dataset_cfg
        from cutie.model.cutie import CUTIE
        from hydra import compose, initialize_config_dir
        from omegaconf import open_dict

        # initialize_config_dir, not initialize(): the latter resolves config_path relative to the
        # *calling module*, which would point inside pitch3d rather than the checkout.
        with initialize_config_dir(version_base="1.3.2",
                                   config_dir=os.path.join(repo, "cutie", "config"),
                                   job_name="pitch3d_cutie"):
            cfg = compose(config_name="eval_config")
        with open_dict(cfg):
            cfg["weights"] = weights
        get_dataset_cfg(cfg)  # mutates cfg — Cutie's own loader relies on this side effect

        net = CUTIE(cfg).to(self.device).eval()
        net.load_weights(torch.load(weights, map_location=self.device, weights_only=False))
        self._core = InferenceCore(net, cfg=cfg)
        self._torch = torch

    # ── use ──────────────────────────────────────────────────────────────────
    def _to_tensor(self, image_bgr: np.ndarray):  # pragma: no cover - heavy path
        torch = self._torch
        rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.to(self.device)

    def seed(  # pragma: no cover - heavy path
        self, image_bgr: np.ndarray, labels: np.ndarray, object_ids: list[int]
    ) -> np.ndarray:
        """Begin propagating ``labels`` (an int label image) and return the network's own view.

        ``object_ids`` lists the non-zero ids present, in the order Cutie should hold them.
        """
        self._load()
        torch = self._torch
        mask = torch.from_numpy(np.asarray(labels, dtype=np.int64)).to(self.device)
        with torch.inference_mode():
            prob = self._core.step(self._to_tensor(image_bgr), mask, objects=list(object_ids))
        return self._labels_from(prob)

    def step(self, image_bgr: np.ndarray) -> np.ndarray:  # pragma: no cover - heavy path
        """Advance one frame with no new evidence; return the propagated label image."""
        self._load()
        torch = self._torch
        with torch.inference_mode():
            prob = self._core.step(self._to_tensor(image_bgr))
        return self._labels_from(prob)

    def _labels_from(self, prob) -> np.ndarray:  # pragma: no cover - heavy path
        """Cutie returns per-object probabilities with channel 0 = background; argmax to labels.

        The returned ids are Cutie's own object indices, which are the ``object_ids`` handed to
        :meth:`seed` — the caller keeps the map from those to its track ids.
        """
        return self._core.output_prob_to_mask(prob).cpu().numpy().astype(np.int32)


def make() -> CutiePropagator:
    """Zero-arg factory; config from the environment, matching the other adapters."""
    return CutiePropagator(
        repo_dir=os.environ.get(
            "PITCH3D_CUTIE_REPO", "backends/McByte/mask_propagation/Cutie"
        ),
        weights=os.environ.get("PITCH3D_CUTIE_WEIGHTS"),
        device=os.environ.get("PITCH3D_DEVICE", "cpu"),
    )
