"""ByteTrack tracker + appearance team classifier — second real adapter (M1, FR-6).

Self-hosted default for `Tracker`: ByteTrack / BoT-SORT association (MIT) plus a team
classifier, behind the optional ``cv`` extra. Split like the detector so the *logic* is
testable with **no supervision, no cv2, no GPU**:

* :class:`ByteTrackTracker` — the **pure** half: takes the backend's associated raw tracklets
  and (a) resolves each track's class by majority vote over its frames (detector labels
  flicker), and (b) clusters the per-track appearance features into ``n_teams`` with a
  deterministic k-means, mapping clusters to stable team ids. Numpy only; unit-tested via an
  injected backend.
* :class:`ByteTrackBackend` — the **heavy** half: runs ByteTrack over the detections and
  samples a jersey-colour feature per track. All ``supervision``/``cv2`` imports are lazy, so
  importing this module never pulls the heavy stack; it raises an actionable error if the
  ``cv`` extra is missing.

Swap it in via ``default_ports(tracker="bytetrack")`` (wiring) — one fake replaced at a time,
satisfying the very same ``Tracker`` port test the fake passes (roadmap M1).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from ...core.ports.io import ClipRef
from ...core.ports.perception import Detections, Tracker, Tracklet, Tracks
from ...core.scene.provenance import Backend, ModelInfo
from ...core.scene.subject import Team

#: Classes that belong to a team (referees are officials → no team).
TEAM_CLASSES: frozenset[str] = frozenset({"player", "goalkeeper"})


@dataclass
class RawTracklet:
    """Backend output for one associated identity: boxes over time + an appearance feature.

    The association (ByteTrack/BoT-SORT) and appearance extraction are the heavy half; this is
    the canned hand-off the pure class-vote + team-clustering logic consumes.

    Attributes:
        track_id: Stable identity assigned by the association step.
        frames: ``(T,)`` source frame indices this identity appears in.
        bboxes_xyxy: ``(T, 4)`` image-px boxes, aligned with ``frames``.
        classes: Per-frame detector label (length ``T``; may flicker frame to frame).
        appearance: ``(D,)`` team-appearance feature (e.g. jersey colour), or ``None`` when the
            backend could not sample one (the track then gets no team).
    """

    track_id: int
    frames: np.ndarray
    bboxes_xyxy: np.ndarray
    classes: list[str]
    appearance: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.frames = np.asarray(self.frames, dtype=int).reshape(-1)
        self.bboxes_xyxy = np.asarray(self.bboxes_xyxy, dtype=float).reshape(-1, 4)
        t = self.frames.shape[0]
        if not (self.bboxes_xyxy.shape[0] == len(self.classes) == t):
            raise ValueError(
                f"ragged raw tracklet {self.track_id}: {t} frames, "
                f"{self.bboxes_xyxy.shape[0]} boxes, {len(self.classes)} classes"
            )
        if self.appearance is not None:
            self.appearance = np.asarray(self.appearance, dtype=float).reshape(-1)


@runtime_checkable
class TrackingBackend(Protocol):
    """The heavy half: associate detections into stable ids and sample appearance features.

    Kept behind this protocol so :class:`ByteTrackTracker`'s class-vote + team-clustering logic
    can be tested with a stub returning canned :class:`RawTracklet`\\ s — no GPU required.
    """

    def associate(self, clip: ClipRef, detections: Detections) -> list[RawTracklet]:
        """Return one :class:`RawTracklet` per stable identity found in ``detections``."""
        ...


def _majority_class(labels: list[str]) -> str:
    """Most frequent per-frame label; ties broken toward the earliest occurrence (stable)."""
    counts = Counter(labels)
    return max(counts, key=lambda c: (counts[c], -labels.index(c)))


def _kmeans(feats: np.ndarray, k: int, *, iters: int = 50) -> np.ndarray:
    """Deterministic k-means (farthest-point seeding, fixed iterations) → ``(n,)`` labels.

    No RNG: centroid 0 is point 0, each next centroid is the point farthest from those chosen.
    That makes team assignment reproducible run to run (the whole reason this lives in the pure
    half). Empty clusters keep their previous centroid rather than collapsing.
    """
    n = feats.shape[0]
    if k <= 1:
        return np.zeros(n, dtype=int)
    if n <= k:
        return np.arange(n, dtype=int)

    centers = [0]
    for _ in range(1, k):
        d = np.min([np.sum((feats - feats[c]) ** 2, axis=1) for c in centers], axis=0)
        centers.append(int(np.argmax(d)))
    cent = feats[centers].astype(float)

    labels = np.full(n, -1, dtype=int)
    for _ in range(iters):
        dists = np.sum((feats[:, None, :] - cent[None, :, :]) ** 2, axis=2)  # (n, k)
        new = np.argmin(dists, axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            members = feats[labels == c]
            if members.shape[0]:
                cent[c] = members.mean(axis=0)
    return labels


@dataclass
class ByteTrackTracker(Tracker):
    """ByteTrack association + appearance team clustering (FR-6) — pure over an injected backend.

    Attributes:
        backend: The associate+appearance backend. If ``None``, a real :class:`ByteTrackBackend`
            is constructed lazily on first use (needs the ``cv`` extra + GPU).
        n_teams: Number of teams to cluster the team-bearing tracks into (2 for a normal match).
        team_ids: Stable labels handed to clusters, ordered by the smallest track id they contain
            (so the first-appearing team is always ``team_ids[0]``).
        min_track_frames: Tracklets shorter than this are dropped as association blips.
        device: Inference device for the default backend.
    """

    backend: TrackingBackend | None = None
    n_teams: int = 2
    team_ids: tuple[str, ...] = ("A", "B")
    min_track_frames: int = 1
    device: str = "cuda"

    def info(self) -> ModelInfo:
        return ModelInfo(
            name="ByteTrack+BoT-SORT",
            backend=Backend.LOCAL,
            license="MIT",
            params={"n_teams": self.n_teams, "device": self.device},
        )

    def track(self, clip: ClipRef, detections: Detections) -> Tracks:
        backend = self.backend or self._default_backend()
        raw = [
            r for r in backend.associate(clip, detections)
            if r.frames.shape[0] >= self.min_track_frames
        ]
        tracklets = [
            Tracklet(
                track_id=r.track_id,
                frames=r.frames,
                bboxes_xyxy=r.bboxes_xyxy,
                cls=_majority_class(r.classes),
            )
            for r in raw
        ]
        teams = self._assign_teams(raw, tracklets)
        return Tracks(tracklets=tracklets, teams=teams)

    def _assign_teams(self, raw: list[RawTracklet], tracklets: list[Tracklet]) -> list[Team]:
        """Cluster team-bearing tracks by appearance and stamp ``team_id`` in place."""
        idx = [
            i for i, t in enumerate(tracklets)
            if t.cls in TEAM_CLASSES and raw[i].appearance is not None
        ]
        if not idx:
            return []
        feats = np.stack([raw[i].appearance for i in idx])
        k = min(self.n_teams, len(self.team_ids), len(idx))
        labels = _kmeans(feats, k)

        # Map raw cluster ids → stable team ids by the smallest track id in each cluster, so the
        # labelling does not depend on k-means' internal ordering.
        def _min_track_id(cluster: int) -> int:
            return min(
                tracklets[idx[j]].track_id for j in range(len(idx)) if labels[j] == cluster
            )

        order = sorted(set(labels.tolist()), key=_min_track_id)
        remap = {c: pos for pos, c in enumerate(order)}

        used: dict[str, Team] = {}
        for j, i in enumerate(idx):
            tid = self.team_ids[remap[labels[j]]]
            tracklets[i].team_id = tid
            used.setdefault(tid, Team(id=tid, name=f"Team {tid}"))
        return [used[t] for t in self.team_ids if t in used]

    def _default_backend(self) -> TrackingBackend:
        return ByteTrackBackend(device=self.device)


@dataclass
class ByteTrackBackend:
    """Real association + jersey-colour sampling: lazy ``supervision``/``cv2``, no import cost.

    Feeds the non-ball detections through ByteTrack frame by frame (the ball has its own
    :class:`~pitch3d.core.ports.perception.BallTracker`), then samples a mean-HSV torso feature
    per track for the team classifier. Imports the heavy stack only when
    :meth:`associate` is first called, so this module stays import-safe without the ``cv`` extra.
    """

    device: str = "cuda"
    #: Stable class→id map fed to ByteTrack (ball is excluded from association on purpose).
    class_ids: dict[str, int] = field(
        default_factory=lambda: {"goalkeeper": 1, "player": 2, "referee": 3}
    )

    def associate(  # pragma: no cover - heavy path
        self, clip: ClipRef, detections: Detections
    ) -> list[RawTracklet]:
        sv = self._import_sv()
        id_to_cls = {v: k for k, v in self.class_ids.items()}
        tracker = sv.ByteTrack(frame_rate=int(round(clip.fps or 25.0)))

        # Accumulate per-track boxes/classes keyed by ByteTrack's tracker id.
        boxes: dict[int, list[tuple[int, np.ndarray, str]]] = {}
        for fd in detections.frames:
            people = [d for d in fd.items if d.cls in self.class_ids]
            det = sv.Detections(
                xyxy=np.array([d.bbox_xyxy for d in people], dtype=float).reshape(-1, 4),
                confidence=np.array([d.score for d in people], dtype=float).reshape(-1),
                class_id=np.array([self.class_ids[d.cls] for d in people], dtype=int).reshape(-1),
            )
            tracked = tracker.update_with_detections(det)
            for xyxy, cid, tid in zip(
                tracked.xyxy, tracked.class_id, tracked.tracker_id, strict=True
            ):
                boxes.setdefault(int(tid), []).append(
                    (int(fd.frame), np.asarray(xyxy, dtype=float), id_to_cls[int(cid)])
                )

        appearance = self._sample_appearance(clip, boxes)
        out: list[RawTracklet] = []
        for tid, seq in sorted(boxes.items()):
            seq.sort(key=lambda r: r[0])
            out.append(
                RawTracklet(
                    track_id=tid,
                    frames=np.array([f for f, _, _ in seq], dtype=int),
                    bboxes_xyxy=np.stack([b for _, b, _ in seq]),
                    classes=[c for _, _, c in seq],
                    appearance=appearance.get(tid),
                )
            )
        return out

    def _sample_appearance(self, clip, boxes):  # pragma: no cover - heavy path (needs cv2 + media)
        """Mean torso HSV per track id, sampled from the first few frames it appears in."""
        from .detection import _iter_frames

        wanted: dict[int, list[tuple[int, np.ndarray]]] = {}
        for tid, seq in boxes.items():
            for frame, box, _ in seq[:5]:
                wanted.setdefault(frame, []).append((tid, box))
        if not wanted:
            return {}

        acc: dict[int, list[np.ndarray]] = {}
        import cv2

        for frame, image in _iter_frames(clip):
            for tid, box in wanted.get(frame, []):
                x0, y0, x1, y1 = (int(round(v)) for v in box)
                cy0, cy1 = y0 + (y1 - y0) // 4, y0 + (y1 - y0) // 2  # upper torso band
                crop = image[max(cy0, 0):max(cy1, cy0 + 1), max(x0, 0):max(x1, x0 + 1)]
                if crop.size:
                    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    acc.setdefault(tid, []).append(hsv.reshape(-1, 3).mean(axis=0))
        return {tid: np.mean(vals, axis=0) for tid, vals in acc.items() if vals}

    def _import_sv(self):  # pragma: no cover - exercised only without the extra
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack is not installed. Install the detection extra: "
                "`pip install 'pitch3d[cv]'` (MIT), or inject a TrackingBackend."
            ) from exc
        return sv
