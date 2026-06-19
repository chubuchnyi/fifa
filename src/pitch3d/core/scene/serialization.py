"""JSON (de)serialization for the canonical scene model.

This is the **native** save format (ADR-0005): a self-describing, tagged JSON tree that
round-trips dataclasses, enums and numpy arrays losslessly. It is structured to be
USD-mappable, but USD/glTF/FBX are *export* targets handled by ``adapters/export``, not
this codec. In production, very large arrays would move to an ``.npz`` sidecar; for the
scaffold everything is inline so a :class:`Scene` round-trips with stdlib ``json`` only.

Tag scheme (reserved keys):
    ``__ndarray__`` numpy array {dtype, shape, data}
    ``__enum__``    enum {type, value}
    ``__type__``    dataclass {type, fields}
    ``__tuple__``   tuple
    ``__dict__``    dict with arbitrary (incl. non-string) keys
"""

from __future__ import annotations

import dataclasses
import json
from enum import Enum
from typing import Any

import numpy as np

from . import assets, camera, field, layers, motion, provenance, scene, subject, units

# --- Registry of all serializable types ---------------------------------------
_CLASSES: list[type] = [
    # units / provenance
    units.WorldFrame, units.FieldDimensions, units.TimeBase, units.Settings,
    units.UpAxis, units.Handedness,
    provenance.ModelInfo, provenance.RunRecord, provenance.RunLog, provenance.Backend,
    # camera / field
    camera.CameraIntrinsics, camera.CameraTrack,
    field.FieldCalibration, field.FieldModel,
    # motion
    motion.SmplxShape, motion.PoseSequence, motion.SubjectMotion, motion.BallTrack,
    motion.Ball2DTrack, motion.VectorCurve, motion.BodyModel,
    # subject
    subject.Team, subject.Subject, subject.Role,
    # layers
    layers.FrameRange, layers.CorrectionTarget, layers.OffsetPayload,
    layers.KeyframePayload, layers.RefitPayload, layers.SmoothingPayload,
    layers.Correction, layers.ConfidenceMap,
    layers.Layer, layers.TargetKind, layers.CorrectionMode,
    # assets
    assets.RenderAssetRef, assets.SynthViewRef, assets.RenderAssetKind, assets.SynthViewSeam,
    # scene containers
    scene.Source, scene.Episode, scene.Scene, scene.Project,
    scene.SourceKind, scene.EpisodeSource,
]
_REGISTRY: dict[str, type] = {c.__name__: c for c in _CLASSES}


def encode(obj: Any) -> Any:
    """Recursively encode an object into a JSON-safe, self-describing tree."""
    if obj is None:
        return obj
    # Enum BEFORE primitives: our enums subclass ``str`` (``class Role(str, Enum)``),
    # so the primitive check would otherwise swallow them into bare strings.
    if isinstance(obj, Enum):
        return {"__enum__": {"type": type(obj).__name__, "value": obj.value}}
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": {"dtype": str(obj.dtype), "shape": list(obj.shape),
                                "data": obj.tolist()}}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        fields = {f.name: encode(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        return {"__type__": type(obj).__name__, "fields": fields}
    if isinstance(obj, tuple):
        return {"__tuple__": [encode(x) for x in obj]}
    if isinstance(obj, list):
        return [encode(x) for x in obj]
    if isinstance(obj, dict):
        return {"__dict__": [[encode(k), encode(v)] for k, v in obj.items()]}
    raise TypeError(f"cannot encode object of type {type(obj)!r}")


def decode(data: Any) -> Any:
    """Inverse of :func:`encode`."""
    if isinstance(data, dict):
        if "__ndarray__" in data:
            spec = data["__ndarray__"]
            arr = np.array(spec["data"], dtype=spec["dtype"])
            return arr.reshape(spec["shape"])
        if "__enum__" in data:
            spec = data["__enum__"]
            return _REGISTRY[spec["type"]](spec["value"])
        if "__type__" in data:
            cls = _REGISTRY[data["__type__"]]
            kwargs = {k: decode(v) for k, v in data["fields"].items()}
            return cls(**kwargs)
        if "__tuple__" in data:
            return tuple(decode(x) for x in data["__tuple__"])
        if "__dict__" in data:
            return {decode(k): decode(v) for k, v in data["__dict__"]}
        return {k: decode(v) for k, v in data.items()}  # lenient fallback
    if isinstance(data, list):
        return [decode(x) for x in data]
    return data


def to_json(obj: Any, *, indent: int | None = 2) -> str:
    """Serialize any registered scene object to a JSON string."""
    return json.dumps(encode(obj), indent=indent)


def from_json(text: str) -> Any:
    """Deserialize a JSON string produced by :func:`to_json`."""
    return decode(json.loads(text))


def save_scene(obj: Any, path: str) -> None:
    """Write a Scene/Project (or any registered object) to ``path`` as JSON."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_json(obj))


def load_scene(path: str) -> Any:
    """Read back an object written by :func:`save_scene`."""
    with open(path, encoding="utf-8") as fh:
        return from_json(fh.read())
