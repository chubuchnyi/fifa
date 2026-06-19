"""Top-level containers: Source → Episode → Scene → Project.

* :class:`Source`  — an imported clip / frame sequence (FR-2).
* :class:`Episode` — a time-range *selection* on a source (FR-3/FR-4).
* :class:`Scene`   — the reconstruction of one episode: world frame, field, camera,
  subjects (proposal motion), ball, the correction stack, confidence, and refs to
  render assets + synthesized views. This is the canonical, serializable unit (FR-5..14).
* :class:`Project` — sources + episodes + scenes + settings (FR-1).
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dcfield
from enum import Enum

from .assets import RenderAssetRef, SynthViewRef
from .camera import CameraTrack
from .field import FieldModel
from .layers import ConfidenceMap, Correction
from .motion import BallTrack
from .provenance import RunLog
from .subject import Subject, Team
from .units import Settings, TimeBase, WorldFrame


class SourceKind(str, Enum):
    VIDEO = "video"
    FRAMES = "frames"


class EpisodeSource(str, Enum):
    """How an episode's time range was chosen (FR-3 vs FR-4)."""

    MANUAL = "manual"
    ACTION_SPOTTING = "action_spotting"


@dataclass
class Source:
    """An imported video or frame sequence and its metadata (FR-2)."""

    id: str
    uri: str
    kind: SourceKind
    time_base: TimeBase
    width: int
    height: int
    frame_count: int


@dataclass
class Episode:
    """A selected time window on a source — the unit chosen for reconstruction."""

    id: str
    source_id: str
    start_frame: int
    end_frame: int
    origin: EpisodeSource = EpisodeSource.MANUAL
    name: str | None = None

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


@dataclass
class Scene:
    """The reconstructed episode — the canonical model (ADR-0005).

    Attributes:
        id: Stable identifier.
        episode_id, source_id: Provenance back to the selection and clip.
        world_frame: Metric frame (Z-up, meters by default).
        field: Pitch model + per-frame homography (world anchor).
        camera: Estimated broadcast camera track.
        subjects: Tracked people, each carrying proposal SMPL-X motion.
        teams: Team definitions referenced by subjects.
        ball: Ball 3D trajectory (with height confidence), or None.
        corrections: The non-destructive edit stack (subjects + ball).
        confidence: Per-frame/joint confidence + reprojection error.
        render_assets: Refs to derived render assets (env/avatars/ball).
        synth_views: Refs to ViewSynthesizer outputs (seams A & B).
        run_log: Which models/params/costs produced this scene (NFR-7).
    """

    id: str
    episode_id: str
    source_id: str
    world_frame: WorldFrame = dcfield(default_factory=WorldFrame)
    field: FieldModel = dcfield(default_factory=FieldModel)
    camera: CameraTrack | None = None
    subjects: list[Subject] = dcfield(default_factory=list)
    teams: list[Team] = dcfield(default_factory=list)
    ball: BallTrack | None = None
    corrections: list[Correction] = dcfield(default_factory=list)
    confidence: ConfidenceMap | None = None
    render_assets: list[RenderAssetRef] = dcfield(default_factory=list)
    synth_views: list[SynthViewRef] = dcfield(default_factory=list)
    run_log: RunLog = dcfield(default_factory=RunLog)

    def subject(self, track_id: int) -> Subject:
        for s in self.subjects:
            if s.track_id == track_id:
                return s
        raise KeyError(f"no subject with track_id {track_id}")

    def corrections_for(self, track_id: int | None) -> list[Correction]:
        """Enabled corrections targeting a subject (or the ball/global if ``None``)."""
        return [
            c
            for c in self.corrections
            if c.enabled and c.target.subject_track_id == track_id
        ]


@dataclass
class Project:
    """A saved project: everything the operator works with (FR-1)."""

    id: str
    name: str
    sources: list[Source] = dcfield(default_factory=list)
    episodes: list[Episode] = dcfield(default_factory=list)
    scenes: list[Scene] = dcfield(default_factory=list)
    settings: Settings = dcfield(default_factory=Settings)
    created_at: str | None = None
