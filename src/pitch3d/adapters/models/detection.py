"""RF-DETR detector — the first real adapter (M1, FR-5).

Self-hosted default for `Detector`: RF-DETR (`roboflow/sports`), Apache-2.0, behind the
optional ``cv`` extra. The adapter is split in two so its *logic* is testable with **no
torch, no cv2, no GPU** — the same AC-7 discipline the fakes give us:

* :class:`RFDETRDetector` — the **pure** half: maps raw model class-ids to the project's
  ``player|goalkeeper|referee|ball`` vocabulary, applies the score threshold, and assembles
  the canonical :class:`Detections`. Numpy only; fully unit-tested via an injected backend.
* :class:`RFDETRBackend` — the **heavy** half: decodes the clip's frames and runs the network.
  All torch/cv2/rfdetr imports are lazy (inside methods), so importing this module never pulls
  the heavy stack; it raises an actionable error if the ``cv`` extra is missing.

Swap it in via ``default_ports(detector="rfdetr")`` (wiring) — one fake replaced at a time,
satisfying the very same ``Detector`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Detection, Detections, Detector, FrameDetections
from ...core.scene.provenance import Backend, ModelInfo
from ..io.frames import iter_clip_frames

#: roboflow/sports player-detection class ids → the project's detection vocabulary.
#: Verify against the exact weights in use; override via ``RFDETRDetector(class_map=...)``.
ROBOFLOW_SPORTS_CLASSES: dict[int, str] = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

#: RF-DETR's *base* checkpoint is COCO-pretrained (91-class, 1-indexed): person=1, sports ball=37.
#: COCO has no goalkeeper/referee notion, so every person collapses to "player" — enough to
#: validate the whole pipeline on **freely downloadable** weights (the sports checkpoint is
#: Roboflow-gated). Pass the sports weights via ``--detector-weights`` to split the roles apart.
COCO_BASE_CLASSES: dict[int, str] = {1: "player", 37: "ball"}

#: Named class maps selectable at the composition root (``default_ports(detector_classes=...)``):
#: ``"sports"`` pairs with a fine-tuned sports checkpoint, ``"coco"`` with the free base weights.
DETECTOR_CLASS_MAPS: dict[str, dict[int, str]] = {
    "sports": ROBOFLOW_SPORTS_CLASSES,
    "coco": COCO_BASE_CLASSES,
}


@dataclass
class RawFrameDetections:
    """Backend output for one frame: parallel numpy arrays, pre-mapping/threshold.

    Attributes:
        frame: Source frame index.
        boxes_xyxy: ``(N, 4)`` image-px boxes.
        class_ids: ``(N,)`` raw model class ids.
        scores: ``(N,)`` detection confidences in ``[0, 1]``.
    """

    frame: int
    boxes_xyxy: np.ndarray
    class_ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.boxes_xyxy = np.asarray(self.boxes_xyxy, dtype=float).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=int).reshape(-1)
        self.scores = np.asarray(self.scores, dtype=float).reshape(-1)
        n = self.boxes_xyxy.shape[0]
        if not (self.class_ids.shape[0] == self.scores.shape[0] == n):
            raise ValueError(
                f"ragged raw detections at frame {self.frame}: "
                f"{n} boxes, {self.class_ids.shape[0]} ids, {self.scores.shape[0]} scores"
            )


@runtime_checkable
class DetectionBackend(Protocol):
    """The heavy half: decode the clip's frames and run the detector network.

    Kept behind this protocol so :class:`RFDETRDetector`'s mapping/assembly logic can be
    tested with a stub that returns canned :class:`RawFrameDetections` — no GPU required.
    """

    def detect_raw(self, clip: ClipRef) -> list[RawFrameDetections]:
        """Return one :class:`RawFrameDetections` per processed frame of ``clip``."""
        ...


#: Where the per-clip input squares live. Kept as data, not code, because the best value depends
#: on the clip (its aspect ratio and how far the camera sits), and we have measured only two.
_REPO_ROOT = Path(__file__).resolve().parents[3].parent
RESOLUTION_CONFIG = _REPO_ROOT / "config" / "detector_resolution.yaml"


def resolution_for_clip(clip: ClipRef | None, path: Path | None = None) -> int | None:
    """Input square for this clip: an exact file-name entry, else the config default, else None.

    Returns ``None`` when the config is missing or unreadable, which makes the backend fall back
    to its own default. A missing config must not stop a run.
    """
    cfg_path = path or RESOLUTION_CONFIG
    try:
        import yaml
    except ImportError:
        return None
    try:
        data = yaml.safe_load(cfg_path.read_text()) or {}
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    clips = data.get("clips")
    if clip is not None and clip.uri and isinstance(clips, dict):
        entry = clips.get(Path(str(clip.uri)).name)
        if entry is not None:
            return int(entry)
    default = data.get("default")
    return None if default is None else int(default)


@dataclass
class RFDETRDetector(Detector):
    """RF-DETR player/ball detector (FR-5) — pure mapping over an injected backend.

    Attributes:
        backend: The decode+infer backend. If ``None``, a real :class:`RFDETRBackend` is
            constructed lazily on first use (needs the ``cv`` extra + weights + GPU).
        score_threshold: Authoritative confidence floor; raw boxes below it are dropped here
            (so the threshold is part of the tested logic, not buried in the backend).
        class_map: Raw class-id → vocabulary label. Ids absent from the map are discarded.
        weights: Optional path/identifier for the model weights (forwarded to the backend).
        device: Inference device for the default backend.
    """

    backend: DetectionBackend | None = None
    score_threshold: float = 0.3
    class_map: dict[int, str] = field(default_factory=lambda: dict(ROBOFLOW_SPORTS_CLASSES))
    weights: str | None = None
    device: str = "cuda"
    #: Network input square. ``None`` means "look the clip up in
    #: ``config/detector_resolution.yaml``, else use that file's ``default``". An int forces the
    #: value and skips the lookup. The best square is a property of the clip, not a constant —
    #: see the config file for the measurements and for how to measure a new clip.
    resolution: int | None = None

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="RF-DETR",
            version="sports",
            backend=Backend.LOCAL,
            license="Apache-2.0",
            params={
                "score_threshold": self.score_threshold,
                "device": self.device,
                "resolution": self.resolution,
            },
        )

    def detect(self, clip: ClipRef) -> Detections:
        backend = self.backend or self._default_backend(clip)
        frames: list[FrameDetections] = []
        for raw in backend.detect_raw(clip):
            items: list[Detection] = []
            for bbox, cid, score in zip(raw.boxes_xyxy, raw.class_ids, raw.scores, strict=True):
                cls = self.class_map.get(int(cid))
                if cls is None or score < self.score_threshold:
                    continue
                items.append(Detection(bbox_xyxy=bbox, cls=cls, score=float(score)))
            frames.append(FrameDetections(frame=int(raw.frame), items=items))
        return Detections(frames=frames)

    def _default_backend(self, clip: ClipRef | None = None) -> DetectionBackend:
        res = self.resolution if self.resolution is not None else resolution_for_clip(clip)
        kwargs = {} if res is None else {"resolution": int(res)}
        return RFDETRBackend(weights=self.weights, device=self.device, **kwargs)


@dataclass
class RFDETRBackend:
    """Real RF-DETR inference: lazy torch/cv2/rfdetr, decode frames, run the net.

    Imports the heavy stack only when :meth:`detect_raw` is first called, so this module
    stays import-safe without the ``cv`` extra installed.
    """

    weights: str | None = None
    device: str = "cuda"
    predict_floor: float = 0.05  # permissive backend floor; the adapter does authoritative filtering
    #: Network input square, in px. ``None`` uses RF-DETR's own default of 560.
    #:
    #: RF-DETR resizes the whole frame to ``resolution x resolution`` and does not keep the aspect
    #: ratio. A 1080x1920 portrait clip is squashed to 0.52x across and 0.29x down at 560, so a
    #: 28x72 px player reaches the network as 14x21 px.
    #:
    #: Callers should not set this directly. :class:`RFDETRDetector` resolves it per clip from
    #: ``config/detector_resolution.yaml``, because the best value depends on the clip. The
    #: measurements and the procedure for a new clip are in that file.
    resolution: int | None = None
    _model: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.resolution is not None and int(self.resolution) % 56 != 0:
            raise ValueError(
                f"RF-DETR resolution must be divisible by 56 (its patch stride), got "
                f"{self.resolution}. Nearest valid: {round(int(self.resolution) / 56) * 56}."
            )

    def detect_raw(self, clip: ClipRef) -> list[RawFrameDetections]:
        model = self._load()
        out: list[RawFrameDetections] = []
        for frame_idx, image in _iter_frames(clip):
            det = model.predict(image, threshold=self.predict_floor)  # supervision.Detections
            out.append(
                RawFrameDetections(
                    frame=frame_idx,
                    boxes_xyxy=np.asarray(det.xyxy, dtype=float).reshape(-1, 4),
                    class_ids=np.asarray(det.class_id, dtype=int).reshape(-1),
                    scores=np.asarray(det.confidence, dtype=float).reshape(-1),
                )
            )
        return out

    def _load(self) -> object:
        if self._model is None:
            try:
                from rfdetr import RFDETRBase
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "RF-DETR is not installed. Install the detection extra: "
                    "`pip install 'pitch3d[cv]'` (Apache-2.0), or inject a DetectionBackend."
                ) from exc
            kwargs = {} if self.weights is None else {"pretrain_weights": self.weights}
            if self.resolution is not None:
                kwargs["resolution"] = int(self.resolution)
            self._model = RFDETRBase(**kwargs)
        return self._model


def _iter_frames(clip: ClipRef):  # pragma: no cover - heavy decode path (needs cv2 + media)
    """Yield ``(frame_index, BGR uint8 image)`` for each requested frame of the clip.

    Thin wrapper over the shared :func:`~pitch3d.adapters.io.frames.iter_clip_frames` decoder.
    """
    return iter_clip_frames(clip.uri, clip.frames.tolist())
