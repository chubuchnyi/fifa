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
        appearance_series: ``(T, D)`` the same feature sampled *per frame*, or ``None``. One
            vector per track cannot represent a track that changes player mid-way, and #132
            measured 9 of 38 tracks in the target clip doing exactly that; this is what
            :func:`split_on_kit_change` reads to cut them apart.
    """

    track_id: int
    frames: np.ndarray
    bboxes_xyxy: np.ndarray
    classes: list[str]
    appearance: np.ndarray | None = None
    appearance_series: np.ndarray | None = None

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
        if self.appearance_series is not None:
            self.appearance_series = np.asarray(self.appearance_series, dtype=float).reshape(t, -1)


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


def _hsv_to_feature(hsv: np.ndarray) -> np.ndarray:
    """Mean-HSV rows (OpenCV ranges H∈[0,180], S,V∈[0,255]) → a hue-aware, euclidean-safe feature.

    Hue is *circular*, so raw-HSV euclidean k-means is wrong: a single light/shadow (high-V) torso
    becomes the farthest point and seeds a 1-vs-rest split (the observed 19/1 collapse). Encode hue
    as a chroma vector scaled by saturation (a grey / low-sat torso contributes ~no hue, so it can't
    masquerade as a colour) and downweight brightness, so the split is driven by *kit colour*.
    """
    hsv = np.asarray(hsv, dtype=float).reshape(-1, 3)
    hue = hsv[:, 0] * (np.pi / 90.0)  # H∈[0,180] → angle [0, 2π)
    sat = hsv[:, 1] / 255.0
    val = hsv[:, 2] / 255.0
    return np.stack([sat * np.cos(hue), sat * np.sin(hue), 0.25 * val], axis=1)


def split_on_kit_change(
    raw: RawTracklet,
    centroids: np.ndarray,
    min_run: int = 4,
    next_id: int = 0,
) -> list[RawTracklet]:
    """Cut one tracklet wherever its kit colour changes team, and hand each piece its own id.

    A crossing can leave ByteTrack holding the *other* player, and nothing downstream notices: the
    id keeps its avatar, its kit assignment and its motion history while the human under it has
    changed. Measured on the target clip, 9 of 38 tracks in shot 1 do this (`#132`,
    `scripts/track_continuity.py --kit-scan`).

    Splitting is the R-6 answer — the discontinuity is *marked* by becoming two identities rather
    than hidden by averaging two kits into one team label. A track that never changes kit comes
    back unchanged, keeping its original id, so this is a no-op for the other 29.

    Args:
        raw: The tracklet to examine; returned as-is when it carries no ``appearance_series``.
        centroids: ``(k, D)`` team centres in :func:`_hsv_to_feature` space, fitted over *all*
            tracks — a single track is far too small a sample to find the teams from.
        min_run: Frames a kit must hold before it may cut. A crossing briefly puts the other
            player's shirt inside the box, and that must not split a healthy track.
        next_id: First id to hand out for the 2nd and later pieces; the 1st keeps ``track_id``.

    Returns:
        One :class:`RawTracklet` per solid run of a single kit, in frame order.
    """
    if raw.appearance_series is None or raw.frames.shape[0] < 2 * min_run:
        return [raw]
    feats = _hsv_to_feature(raw.appearance_series)
    labels = np.argmin(
        np.sum((feats[:, None, :] - np.asarray(centroids, float)[None, :, :]) ** 2, axis=2), axis=1
    )

    runs: list[list[int]] = []  # [label, start_i, stop_i] over indices into frames
    for i, lab in enumerate(labels.tolist()):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])
    solid = [r for r in runs if r[2] - r[1] + 1 >= min_run]
    merged: list[list[int]] = []
    for r in solid:
        if merged and merged[-1][0] == r[0]:
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    if len(merged) < 2:
        return [raw]

    # Cut midway through the frames between two solid runs -- the boundary itself is the crossing,
    # where the box holds both players and neither label is trustworthy.
    cuts = [0]
    for a, b in zip(merged, merged[1:], strict=False):
        cuts.append((a[2] + b[1]) // 2 + 1)
    cuts.append(raw.frames.shape[0])

    out: list[RawTracklet] = []
    for piece, (lo, hi) in enumerate(zip(cuts, cuts[1:], strict=False)):
        if hi <= lo:
            continue
        series = raw.appearance_series[lo:hi]
        out.append(
            RawTracklet(
                track_id=raw.track_id if piece == 0 else next_id + piece - 1,
                frames=raw.frames[lo:hi],
                bboxes_xyxy=raw.bboxes_xyxy[lo:hi],
                classes=raw.classes[lo:hi],
                appearance=np.median(series, axis=0),
                appearance_series=series,
            )
        )
    return out


def _hsv_to_rgb01(hsv: np.ndarray) -> tuple[float, float, float]:
    """Pure-numpy mean-HSV (OpenCV ranges) → 0..1 RGB for ``Team.color_rgb`` (no cv2 here)."""
    h = (float(hsv[0]) / 180.0) * 6.0  # OpenCV hue [0,180] → sextant [0,6)
    s = min(max(float(hsv[1]) / 255.0, 0.0), 1.0)
    v = min(max(float(hsv[2]) / 255.0, 0.0), 1.0)
    i = int(np.floor(h)) % 6
    f = h - np.floor(h)
    p, q, t = v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f))
    r, g, b = ((v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q))[i]
    return (float(r), float(g), float(b))


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
        kit_split: Cut a track in two where its kit colour changes team (#132). Auto-detect plus
            manual override: set ``False`` to get the pre-fix single-identity behaviour back.
        kit_split_min_run: Boxes a kit must hold before it may cut a track. Swept over shot 1 of
            the target clip: 3 → 0 tracks still changing team, 4 → 1, 5 → 2 (control: 9). 3 wins
            on an asymmetry, not on the count — a missed swap is 100+ frames of an avatar on the
            wrong human, an extra cut is one more id on a track that was already broken — and it
            cuts no track that 4 does not already cut.
        device: Inference device for the default backend.
    """

    backend: TrackingBackend | None = None
    n_teams: int = 2
    team_ids: tuple[str, ...] = ("A", "B")
    min_track_frames: int = 1
    kit_split: bool = True
    kit_split_min_run: int = 3
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
        raw = self._split_swapped(raw)
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

    def _split_swapped(self, raw: list[RawTracklet]) -> list[RawTracklet]:
        """Cut any track that changes team mid-way (#132). No-op without a sampled series.

        The team centres are fitted over every frame of every track at once, because one track is
        far too small a sample to find two kits in — and because a track that *swapped* would
        otherwise define its own two clusters and split on noise.
        """
        if not self.kit_split:
            return raw
        # Referees are not on a team, so their kit is a third colour that would both pollute the
        # centres and let a shadow split a perfectly good official in two. Fit and cut on players
        # and goalkeepers only.
        team_bearing = [
            r for r in raw
            if r.appearance_series is not None and _majority_class(r.classes) in TEAM_CLASSES
        ]
        if len(team_bearing) < 2:
            return raw
        feats = _hsv_to_feature(np.concatenate([r.appearance_series for r in team_bearing]))
        k = min(self.n_teams, feats.shape[0])
        labels = _kmeans(feats, k)
        centroids = np.stack([feats[labels == c].mean(axis=0) for c in range(k)
                              if np.any(labels == c)])
        if centroids.shape[0] < 2:
            return raw

        splittable = {id(r) for r in team_bearing}
        next_id = max(r.track_id for r in raw) + 1
        out: list[RawTracklet] = []
        for r in raw:
            if id(r) not in splittable:
                out.append(r)
                continue
            pieces = split_on_kit_change(r, centroids, self.kit_split_min_run, next_id)
            next_id += max(0, len(pieces) - 1)
            out.extend(pieces)
        return out

    def _assign_teams(self, raw: list[RawTracklet], tracklets: list[Tracklet]) -> list[Team]:
        """Cluster team-bearing tracks by kit colour; stamp ``team_id`` + set ``color_rgb``."""
        idx = [
            i for i, t in enumerate(tracklets)
            if t.cls in TEAM_CLASSES and raw[i].appearance is not None
        ]
        if not idx:
            return []
        hsv = np.stack([a for i in idx if (a := raw[i].appearance) is not None])
        k = min(self.n_teams, len(self.team_ids), len(idx))
        labels = _kmeans(_hsv_to_feature(hsv), k)

        # Map raw cluster ids → stable team ids by the smallest track id in each cluster, so the
        # labelling does not depend on k-means' internal ordering.
        def _min_track_id(cluster: int) -> int:
            return min(
                tracklets[idx[j]].track_id for j in range(len(idx)) if labels[j] == cluster
            )

        order = sorted(set(labels.tolist()), key=_min_track_id)
        remap = {c: pos for pos, c in enumerate(order)}

        team_of = [self.team_ids[remap[int(labels[j])]] for j in range(len(idx))]
        used: dict[str, Team] = {}
        for j, i in enumerate(idx):
            tracklets[i].team_id = team_of[j]
            used.setdefault(team_of[j], Team(id=team_of[j], name=f"Team {team_of[j]}"))
        # Representative kit colour per team = mean HSV of its members → RGB, so the render paints
        # the *measured* colour instead of an arbitrary palette index (v1 recognizability).
        for tid, team in used.items():
            members = hsv[[j for j in range(len(idx)) if team_of[j] == tid]]
            team.color_rgb = _hsv_to_rgb01(members.mean(axis=0))
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

        sampled = self._sample_appearance(clip, boxes)
        out: list[RawTracklet] = []
        for tid, seq in sorted(boxes.items()):
            seq.sort(key=lambda r: r[0])
            series = self._align_series(sampled.get(tid, {}), [f for f, _, _ in seq])
            out.append(
                RawTracklet(
                    track_id=tid,
                    frames=np.array([f for f, _, _ in seq], dtype=int),
                    bboxes_xyxy=np.stack([b for _, b, _ in seq]),
                    classes=[c for _, _, c in seq],
                    appearance=None if series is None else np.median(series, axis=0),
                    appearance_series=series,
                )
            )
        return out

    @staticmethod
    def _align_series(per_frame, frames):  # pragma: no cover - heavy path (needs cv2 + media)
        """``{frame: hsv}`` → ``(T, 3)`` aligned with ``frames``, holding across unsampled frames.

        A crop can come back empty (a box clipped to the frame edge), so the sampler is allowed to
        skip frames. The series must still be one row per frame or it cannot be sliced alongside
        the boxes, so a gap holds the last value — never a fabricated colour.
        """
        if not per_frame:
            return None
        vals, last = [], None
        for f in frames:
            last = per_frame.get(f, last)
            vals.append(last)
        first = next((v for v in vals if v is not None), None)
        if first is None:
            return None
        return np.stack([first if v is None else v for v in vals])

    def _sample_appearance(self, clip, boxes):  # pragma: no cover - heavy path (needs cv2 + media)
        """Robust torso kit-colour (median HSV) per ``{track id: {frame: hsv}}``, over every frame.

        It used to sample only the first 8 frames of each track, which made a mid-track player
        swap undetectable *by construction*: track 97 of the target clip is scored on frames
        112-119 and keeps that team label while wearing the other kit from frame 128 (#132).
        Sampling the whole span costs no extra decode — the same pass just crops more boxes.

        Three robustness measures, because a raw mean over the whole bbox was washing the kit
        colour out (→ the 19/1 mis-cluster): (1) sample only a *central* upper-torso patch (middle
        50% width, 25–50% height) to dodge arms, shorts, the head and the pitch around the
        silhouette; (2) drop grass-green pixels (the dominant background contaminant); (3) take the
        **median**, not the mean, so a stray skin/background pixel can't drag the estimate.
        """
        from .detection import _iter_frames

        wanted: dict[int, list[tuple[int, np.ndarray]]] = {}
        for tid, seq in boxes.items():
            for frame, box, _ in seq:
                wanted.setdefault(frame, []).append((tid, box))
        if not wanted:
            return {}

        acc: dict[int, dict[int, np.ndarray]] = {}
        import cv2

        for frame, image in _iter_frames(clip):
            for tid, box in wanted.get(frame, []):
                x0, y0, x1, y1 = (float(v) for v in box)
                w, h = x1 - x0, y1 - y0
                ix0, iy0 = max(int(round(x0 + 0.25 * w)), 0), max(int(round(y0 + 0.25 * h)), 0)
                ix1 = max(int(round(x0 + 0.75 * w)), ix0 + 1)
                iy1 = max(int(round(y0 + 0.50 * h)), iy0 + 1)
                crop = image[iy0:iy1, ix0:ix1]
                if not crop.size:
                    continue
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
                green = (
                    (hsv[:, 0] >= 35) & (hsv[:, 0] <= 85)
                    & (hsv[:, 1] >= 60) & (hsv[:, 2] >= 40)
                )
                kept = hsv[~green]
                if kept.shape[0] < max(8, hsv.shape[0] // 10):
                    kept = hsv  # almost all grass (bad box) → don't fabricate, use the raw patch
                acc.setdefault(tid, {})[frame] = np.median(kept, axis=0)
        return acc

    def _import_sv(self):  # pragma: no cover - exercised only without the extra
        try:
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError(
                "ByteTrack is not installed. Install the detection extra: "
                "`pip install 'pitch3d[cv]'` (MIT), or inject a TrackingBackend."
            ) from exc
        return sv
