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
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Detection, Detections, Detector, FrameDetections
from ...core.scene.provenance import Backend, ModelInfo

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

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="RF-DETR",
            version="sports",
            backend=Backend.LOCAL,
            license="Apache-2.0",
            params={"score_threshold": self.score_threshold, "device": self.device},
        )

    def detect(self, clip: ClipRef) -> Detections:
        backend = self.backend or self._default_backend()
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

    def _default_backend(self) -> DetectionBackend:
        return RFDETRBackend(weights=self.weights, device=self.device)


@dataclass
class RFDETRBackend:
    """Real RF-DETR inference: lazy torch/cv2/rfdetr, decode frames, run the net.

    Imports the heavy stack only when :meth:`detect_raw` is first called, so this module
    stays import-safe without the ``cv`` extra installed.
    """

    weights: str | None = None
    device: str = "cuda"
    predict_floor: float = 0.05  # permissive backend floor; the adapter does authoritative filtering
    _model: object = field(default=None, init=False, repr=False)

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
            self._model = RFDETRBase(**kwargs)
        return self._model


def _iter_frames(clip: ClipRef):  # pragma: no cover - heavy decode path (needs cv2 + media)
    """Yield ``(frame_index, BGR uint8 image)`` for each requested frame of the clip.

    Lazy cv2. ``clip.uri`` may be a video file (seek per index) or a directory of frames.
    """
    import os

    import cv2

    wanted = [int(f) for f in clip.frames.tolist()]
    if os.path.isdir(clip.uri):
        files = sorted(
            f for f in os.listdir(clip.uri) if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        for idx in wanted:
            img = cv2.imread(os.path.join(clip.uri, files[idx]))
            if img is None:
                raise FileNotFoundError(f"frame {idx} unreadable in {clip.uri}")
            yield idx, img
        return

    cap = cv2.VideoCapture(clip.uri)
    try:
        for idx in wanted:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {idx} of {clip.uri}")
            yield idx, img
    finally:
        cap.release()
