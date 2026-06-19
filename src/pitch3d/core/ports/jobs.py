"""JobQueue / Worker — offline, non-blocking execution (UX-8, ADR-0004).

Heavy stages run as jobs so the UI never freezes (R-10). The port is intentionally
minimal: submit a unit of work, poll its state, fetch its result, cancel it. The fake
adapter runs jobs synchronously in-process; a real adapter runs them in a worker
process/pool and can resume after a crash. A unit of work is a zero-arg thunk that closes
over its inputs — distributed adapters that need serializable jobs instead dispatch on
``stage`` + ``meta`` (documented as a future adapter, not core).
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .base import Port


class JobState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobHandle:
    """Opaque handle returned by :meth:`JobQueue.submit`."""

    id: str
    stage: str
    meta: dict = field(default_factory=dict)


class JobQueue(Port):
    """An offline queue of stage jobs."""

    @abstractmethod
    def submit(
        self,
        stage: str,
        thunk: Callable[[], Any],
        *,
        meta: dict | None = None,
    ) -> JobHandle:
        """Enqueue ``thunk`` (a zero-arg unit of work) and return a handle immediately."""
        raise NotImplementedError

    @abstractmethod
    def state(self, job: JobHandle) -> JobState:
        """Return the current state of ``job``."""
        raise NotImplementedError

    @abstractmethod
    def result(self, job: JobHandle) -> Any:
        """Return ``job``'s result, raising if it failed or is not finished."""
        raise NotImplementedError

    @abstractmethod
    def cancel(self, job: JobHandle) -> bool:
        """Attempt to cancel ``job``; return True if it was cancelled before running."""
        raise NotImplementedError


class Worker(Port):
    """Executes a single job's thunk (separated so it can run out-of-process)."""

    @abstractmethod
    def run(self, thunk: Callable[[], Any]) -> Any:
        """Run ``thunk`` to completion and return its result."""
        raise NotImplementedError
