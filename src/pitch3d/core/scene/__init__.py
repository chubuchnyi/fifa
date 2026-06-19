"""Canonical scene model — pure data + (de)serialization, no infrastructure.

The single source of truth for an edited episode. Render representations are derived
from this; they are never stored here as editable state (ADR-0002).
"""

from __future__ import annotations

from .assets import RenderAssetKind, RenderAssetRef, SynthViewRef, SynthViewSeam
from .camera import CameraIntrinsics, CameraTrack
from .field import FieldCalibration, FieldModel
from .layers import (
    ConfidenceMap,
    Correction,
    CorrectionMode,
    CorrectionTarget,
    FrameRange,
    KeyframePayload,
    Layer,
    OffsetPayload,
    RefitPayload,
    SmoothingPayload,
    TargetKind,
)
from .motion import (
    Ball2DTrack,
    BallTrack,
    BodyModel,
    PoseSequence,
    SmplxShape,
    SubjectMotion,
    VectorCurve,
)
from .provenance import Backend, ModelInfo, RunLog, RunRecord
from .review import AttentionItem, attention_list
from .scene import Episode, EpisodeSource, Project, Scene, Source, SourceKind
from .serialization import (
    decode,
    encode,
    from_json,
    load_scene,
    save_scene,
    to_json,
)
from .subject import Role, Subject, Team
from .units import (
    GRAVITY,
    FieldDimensions,
    Handedness,
    Settings,
    TimeBase,
    UpAxis,
    WorldFrame,
)

__all__ = [
    "GRAVITY",
    "AttentionItem",
    "Backend",
    "Ball2DTrack",
    "BallTrack",
    "BodyModel",
    "CameraIntrinsics",
    "CameraTrack",
    "ConfidenceMap",
    "Correction",
    "CorrectionMode",
    "CorrectionTarget",
    "Episode",
    "EpisodeSource",
    "FieldCalibration",
    "FieldDimensions",
    "FieldModel",
    "FrameRange",
    "Handedness",
    "KeyframePayload",
    "Layer",
    "ModelInfo",
    "OffsetPayload",
    "PoseSequence",
    "Project",
    "RefitPayload",
    "RenderAssetKind",
    "RenderAssetRef",
    "Role",
    "RunLog",
    "RunRecord",
    "Scene",
    "Settings",
    "SmoothingPayload",
    "SmplxShape",
    "Source",
    "SourceKind",
    "Subject",
    "SubjectMotion",
    "SynthViewRef",
    "SynthViewSeam",
    "TargetKind",
    "Team",
    "TimeBase",
    "UpAxis",
    "VectorCurve",
    "WorldFrame",
    "attention_list",
    "decode",
    "encode",
    "from_json",
    "load_scene",
    "save_scene",
    "to_json",
]
