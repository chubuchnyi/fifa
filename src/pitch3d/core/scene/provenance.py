"""Provenance: which model/version/params produced each artifact (NFR-7).

These types are shared by the scene model (asset refs, run log) and the ports
(``ModelProvider.info``). They are pure data — no behaviour, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Backend(str, Enum):
    """Where a model runs."""

    LOCAL = "local"      # self-hosted (GPU)
    API = "api"          # external / cloud API
    FAKE = "fake"        # deterministic test double
    BUILTIN = "builtin"  # pure-core math (e.g. ball lift)


@dataclass
class ModelInfo:
    """Identity + cost of a model, recorded with everything it produces.

    Attributes:
        name: Model/adapter name (e.g. ``"RF-DETR"``, ``"ReCamMaster"``).
        version: Model or weights version string.
        backend: local / api / fake / builtin.
        license: SPDX-ish license id (e.g. ``"AGPL-3.0"``) — surfaced in UX-7.
        est_cost_usd: Estimated monetary cost of one invocation (API), else 0.
        params: Frozen parameter dict used for this run (feeds the cache key).
    """

    name: str
    version: str = "0"
    backend: Backend = Backend.FAKE
    license: str | None = None
    est_cost_usd: float = 0.0
    params: dict = field(default_factory=dict)

    def identity(self) -> str:
        """Stable one-line identity of what would produce this artifact — the cache key input.

        ``params`` has always been documented as feeding the cache key; until 2026-08-12 nothing
        read it, so two runs that differed only in an injected backend shared a key and the
        second silently returned the first one's result.
        """
        flat = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}|{self.version}|{flat}"


def impl_name(backend: object | None, default: str) -> str:
    """Name the injected backend so ``ModelInfo.params`` can tell two runs apart.

    An adapter that hides a swappable backend (``--tracker-backend``, ``--pose-backend``, …)
    otherwise reports the same identity whichever backend is inside, which defeats both the
    provenance record and the cache key.
    """
    return type(backend).__name__ if backend is not None else f"{default}(default)"


@dataclass
class RunRecord:
    """One executed stage: which model, what it cost, cache hit or miss."""

    stage: str
    model: ModelInfo
    cache_key: str | None = None
    cache_hit: bool = False
    duration_s: float = 0.0
    note: str | None = None


@dataclass
class RunLog:
    """Append-only log of stage executions for one scene (reproducibility, UX-7)."""

    records: list[RunRecord] = field(default_factory=list)

    def add(self, record: RunRecord) -> None:
        self.records.append(record)

    def total_cost_usd(self) -> float:
        return sum(r.model.est_cost_usd for r in self.records)
