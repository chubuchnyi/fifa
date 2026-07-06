"""Step 3 (measurement) — foot-contact detection + slide magnitude probe.

The user's ongoing complaint: "root moves but pose stays still, foot slides
across the pitch." This module MEASURES the problem — how often are feet
planted, and how far do they drift during those planted windows.

A foot is CONTACTED at frame ``t`` when its Z sits below
``contact_z_threshold_m + floor_z``. A contact RUN is a maximal contiguous
stretch of contact frames of length ≥ ``min_contact_run_frames`` (single-
frame touches are ignored — they're jitter).

Within each contact run we measure the XY displacement of the foot's mean
position. Any run with drift > ``slide_threshold_m`` is a slide event —
the foot physically moved while planted, which is unphysical.

Consumes an injected ``FootPositionProvider`` returning per-frame
``(T, 3)`` world-space foot position (either the lower-of-two-feet, or a
concatenated (T, 2, 3) if the caller distinguishes L/R — this first cut
takes ``(T, 3)`` = whichever foot was lowest per frame). SMPL-X FK adapter
plugs here; a mock provider drives the tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..config.gates import ContactProbeConfig
from ..scene.scene import Scene, Subject

FootPositionProvider = Callable[[Subject], "np.ndarray | None"]


@dataclass
class ContactRun:
    """One contiguous stretch of contact frames + measured foot slide."""

    track_id: int
    row_start: int      # inclusive row index (into subject.pose.frames)
    row_end: int        # inclusive
    frame_start: int
    frame_end: int
    slide_m: float      # XY displacement of foot mean during the run
    mean_z_m: float     # mean foot Z during the run


@dataclass
class SubjectContactReport:
    track_id: int
    n_frames: int = 0
    n_contact_frames: int = 0
    n_runs: int = 0
    n_slides: int = 0                # runs whose slide > threshold
    total_slide_m: float = 0.0       # sum of slide over runs above threshold
    max_slide_m: float = 0.0
    max_run_length: int = 0
    contact_frac: float = 0.0


@dataclass
class ContactProbeReport:
    n_subjects: int = 0
    subjects_with_slides: int = 0
    total_contact_frames: int = 0
    total_runs: int = 0
    total_slides: int = 0
    mean_slide_m: float = 0.0
    max_slide_m: float = 0.0
    subjects: list[SubjectContactReport] = field(default_factory=list)
    runs: list[ContactRun] = field(default_factory=list)


def _find_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Return inclusive ``(start, end)`` row pairs for contiguous True runs ≥ ``min_len``."""
    runs: list[tuple[int, int]] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and mask[j + 1]:
            j += 1
        if j - i + 1 >= min_len:
            runs.append((i, j))
        i = j + 1
    return runs


def contact_probe(
    scene: Scene,
    cfg: ContactProbeConfig | None = None,
    foot_position_provider: FootPositionProvider | None = None,
    floor_z: float = 0.0,
) -> ContactProbeReport:
    """Measure per-subject foot-contact + slide statistics. Never mutates the scene.

    * ``cfg is None`` or ``cfg.enabled is False`` → empty report (measurement
      only makes sense when opted-in).
    * ``foot_position_provider is None`` → returns an empty report; caller must
      wire an FK provider (e.g. ``make_smplx_foot_position_provider``).
    """
    cfg = cfg if cfg is not None else ContactProbeConfig()
    report = ContactProbeReport(n_subjects=len(scene.subjects))
    if not cfg.enabled or foot_position_provider is None:
        return report

    all_slide_m: list[float] = []
    for s in scene.subjects:
        pos = foot_position_provider(s)
        if pos is None:
            report.subjects.append(SubjectContactReport(track_id=int(s.track_id)))
            continue
        pos = np.asarray(pos, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(
                f"foot_position_provider must return (T, 3) for track {s.track_id}, "
                f"got shape {pos.shape}"
            )
        n = pos.shape[0]
        if n == 0:
            report.subjects.append(SubjectContactReport(track_id=int(s.track_id)))
            continue
        r = SubjectContactReport(track_id=int(s.track_id), n_frames=n)
        contact_mask = pos[:, 2] <= floor_z + cfg.contact_z_threshold_m
        r.n_contact_frames = int(contact_mask.sum())
        r.contact_frac = float(r.n_contact_frames / n)
        runs = _find_runs(contact_mask, cfg.min_contact_run_frames)
        r.n_runs = len(runs)
        r.max_run_length = max((e - b + 1 for b, e in runs), default=0)
        frames = np.asarray(s.proposal.pose.frames, dtype=int)
        for b, e in runs:
            xy = pos[b:e + 1, :2]
            # slide = spread of foot XY within the run (max - min per axis)
            diag = float(np.linalg.norm(xy.max(axis=0) - xy.min(axis=0)))
            mean_z = float(pos[b:e + 1, 2].mean())
            frame_b = int(frames[b]) if b < frames.shape[0] else b
            frame_e = int(frames[e]) if e < frames.shape[0] else e
            report.runs.append(ContactRun(
                track_id=int(s.track_id),
                row_start=b, row_end=e,
                frame_start=frame_b, frame_end=frame_e,
                slide_m=diag, mean_z_m=mean_z,
            ))
            if diag > cfg.slide_threshold_m:
                r.n_slides += 1
                r.total_slide_m += diag
                r.max_slide_m = max(r.max_slide_m, diag)
                all_slide_m.append(diag)
        report.total_contact_frames += r.n_contact_frames
        report.total_runs += r.n_runs
        report.total_slides += r.n_slides
        report.max_slide_m = max(report.max_slide_m, r.max_slide_m)
        if r.n_slides:
            report.subjects_with_slides += 1
        report.subjects.append(r)
    if all_slide_m:
        report.mean_slide_m = float(np.mean(all_slide_m))
    return report


__all__ = [
    "ContactRun",
    "ContactProbeReport",
    "FootPositionProvider",
    "SubjectContactReport",
    "contact_probe",
]
