"""BoT-SORT association with camera-motion compensation, behind ``TrackingBackend``.

Why a second association backend at all. ByteTrack matches a detection to a track by IoU
against a Kalman prediction made in *image* coordinates, and it has no model of the camera —
Ultralytics' own docs say of it: "There is no appearance model and no camera-motion
compensation." When the camera moves, every box in the frame translates at once, the
predictions all land short, and identities break together rather than one at a time. That is
the measured failure mode of the portrait fan clip: **one whip-pan at f38 breaks six
identities simultaneously** (t1-t5/t9/t16/t17 die at f31-34, t60/t63/t71/t73/t75/t76 are born
at f36-38, all pairable at 2.0-4.5 m and +4..+8 frames — #135 §8).

BoT-SORT's GMC step estimates a per-frame affine warp between consecutive frames and applies
it to every track's prediction before matching, which is exactly the missing term. Whether it
actually helps *here* is an open measurement, not a claim — see ``scripts/bench_association.py``.

This subclasses :class:`~.tracking.ByteTrackBackend` on purpose. Every knob that governs the
association (match threshold, activation score, lost-track buffer, the class set fed to the
tracker) and the whole torso-HSV appearance sampling are inherited unchanged, so swapping this
in changes **the matching step and nothing else**. Anything else would not be an A/B.

Injected by dotted path (ADR-0006), no wiring change:

.. code-block:: bash

    --tracker bytetrack --tracker-backend pitch3d.adapters.models.botsort_backend:make
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Detections
from .tracking import ByteTrackBackend, RawTracklet

#: GMC estimators ultralytics ships. ``sparseOptFlow`` is its default and the cheapest;
#: ``none`` turns the camera model off, which makes this backend a plain ByteTrack again and
#: is the control arm for "did the camera model do it, or did BoT-SORT's other differences?".
GMC_METHODS: frozenset[str] = frozenset({"sparseOptFlow", "orb", "sift", "ecc", "none"})


class _DetView:
    """The three arrays ``BYTETracker.update`` reads off an ultralytics ``Boxes`` object.

    Ultralytics couples detection and tracking in ``model.track()``: it runs its own detector
    and hands the result straight to the tracker. Our port hands the tracker detections that
    were already produced (RF-DETR), so we do not go through ``model.track()`` at all — we
    drive ``BOTSORT.update()`` directly, and this is the minimal object it accepts. Read off
    ``byte_tracker.py``: it touches ``.conf``, ``.xywh`` (or ``.xywhr``) and ``.cls``, nothing
    else. Deliberately *not* an ultralytics ``Boxes``: building one needs torch and an
    ``orig_shape``, and would import the whole inference stack to carry three arrays.
    """

    def __init__(self, boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray):
        xyxy = np.asarray(boxes_xyxy, dtype=float).reshape(-1, 4)
        self.conf = np.asarray(scores, dtype=float).reshape(-1)
        self.cls = np.asarray(class_ids, dtype=float).reshape(-1)
        self.xywh = np.stack(
            [
                (xyxy[:, 0] + xyxy[:, 2]) / 2.0,
                (xyxy[:, 1] + xyxy[:, 3]) / 2.0,
                xyxy[:, 2] - xyxy[:, 0],
                xyxy[:, 3] - xyxy[:, 1],
            ],
            axis=1,
        )

    def __len__(self) -> int:
        return int(self.conf.shape[0])


@dataclass
class BotSortBackend(ByteTrackBackend):
    """BoT-SORT association (ultralytics, AGPL-3.0) — ByteTrack plus a camera-motion model.

    Inherits :class:`~.tracking.ByteTrackBackend`'s thresholds and appearance sampling; only
    :meth:`associate` differs. ``with_reid`` is left off: ultralytics' re-id needs a separate
    appearance model, and #138 measured that at 28-41 px of shirt, identical kits, re-id
    disambiguates rather than identifies (KPR: +7.0 R-1 where multi-person ambiguity exists,
    +0.2 where it does not).
    """

    #: Which GMC estimator runs. ``PITCH3D_GMC_METHOD`` overrides it through :func:`make`.
    gmc_method: str = "sparseOptFlow"
    #: IoU below which a pair is refused outright (ultralytics ``proximity_thresh``).
    proximity_thresh: float = 0.5
    #: Weight ``fuse_score`` gives the detection score inside the IoU cost (ultralytics default).
    fuse_score: bool = True
    #: Detections under this score are dropped before association (ultralytics default).
    low_thresh: float = 0.1

    def __post_init__(self) -> None:
        if self.gmc_method not in GMC_METHODS:
            raise ValueError(
                f"unknown gmc_method {self.gmc_method!r}; expected one of {sorted(GMC_METHODS)}"
            )

    def associate(  # pragma: no cover - heavy path (needs ultralytics + media)
        self, clip: ClipRef, detections: Detections
    ) -> list[RawTracklet]:
        if self.mask_cue is not None:
            # Silently dropping it is the #141 defect class: a capability that is configured,
            # documented, and never reaches the run. The cue works by patching supervision's
            # `matching.iou_distance`, and this backend does not go through supervision.
            raise ValueError(
                "BotSortBackend cannot apply the McByte mask cue (it patches supervision's "
                "matching.iou_distance, which BoT-SORT does not call). Unset PITCH3D_MASK_CUE, "
                "or run --tracker-backend with ByteTrackBackend instead."
            )

        botsort, base_track = self._import_botsort()
        from .detection import _iter_frames

        id_to_cls = {v: k for k, v in self.class_ids.items()}
        per_frame = {int(fd.frame): fd.items for fd in detections.frames}

        # Track ids live on a CLASS-level counter in ultralytics, so a second run in the same
        # process keeps numbering where the first stopped — which makes an A/B in one process
        # produce two different id spaces for the same clip.
        base_track.reset_id()
        tracker = botsort(self._tracker_args(), frame_rate=int(round(clip.fps or 25.0)))

        boxes: dict[int, list[tuple[int, np.ndarray, str]]] = {}
        for frame_idx, image in _iter_frames(clip):
            people = [d for d in per_frame.get(int(frame_idx), []) if d.cls in self.class_ids]
            view = _DetView(
                np.array([d.bbox_xyxy for d in people], dtype=float).reshape(-1, 4),
                np.array([d.score for d in people], dtype=float),
                np.array([self.class_ids[d.cls] for d in people], dtype=float),
            )
            # `image` is what carries the camera model: GMC estimates the frame-to-frame warp
            # from these pixels. Passing None here would silently disable the whole point.
            tracked = np.asarray(tracker.update(view, image), dtype=float).reshape(-1, 8)
            for x0, y0, x1, y1, tid, _score, cid, _idx in tracked:
                boxes.setdefault(int(tid), []).append(
                    (int(frame_idx), np.array([x0, y0, x1, y1], dtype=float), id_to_cls[int(cid)])
                )

        return self._build_tracklets(clip, boxes)

    def _tracker_args(self) -> SimpleNamespace:
        """The namespace ultralytics' trackers read their configuration off.

        Mapped from this class's inherited fields rather than from ``botsort.yaml`` so the two
        backends cannot drift apart: ``match_threshold`` / ``activation_threshold`` /
        ``lost_buffer`` are the same numbers ByteTrack runs with.
        """
        return SimpleNamespace(
            tracker_type="botsort",
            track_high_thresh=self.activation_threshold,
            track_low_thresh=self.low_thresh,
            new_track_thresh=self.activation_threshold,
            track_buffer=self.lost_buffer,
            match_thresh=self.match_threshold,
            fuse_score=self.fuse_score,
            gmc_method=self.gmc_method,
            proximity_thresh=self.proximity_thresh,
            appearance_thresh=0.25,
            with_reid=False,
        )

    def _import_botsort(self):  # pragma: no cover - exercised only without the extra
        try:
            from ultralytics.trackers import BOTSORT
            from ultralytics.trackers.basetrack import BaseTrack
        except ImportError as exc:
            raise RuntimeError(
                "BoT-SORT is not installed. Install the detection extra: "
                "`pip install 'pitch3d[cv]'` (ultralytics, AGPL-3.0), or inject another "
                "TrackingBackend."
            ) from exc
        return BOTSORT, BaseTrack


def make() -> BotSortBackend:
    """Zero-arg factory for ``--tracker-backend`` (ADR-0006 dotted path).

    ``PITCH3D_GMC_METHOD`` selects the camera-motion estimator (``none`` = control arm), the
    manual half of the auto/override chain this repo requires of every estimator.
    """
    return BotSortBackend(
        device=os.environ.get("PITCH3D_DEVICE", "cuda"),
        gmc_method=os.environ.get("PITCH3D_GMC_METHOD", "sparseOptFlow"),
    )
