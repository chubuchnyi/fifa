"""In-process JobQueue + Worker — run thunks synchronously, no worker process.

Satisfies :class:`~pitch3d.core.ports.jobs.JobQueue` / :class:`Worker` by executing each
unit of work the moment it is submitted and stashing the result/exception by handle id.
That keeps the orchestration spine identical to a real async queue (submit → result) while
staying dependency-free for tests and the dry-run (ADR-0004). A real adapter runs the same
thunks in a subprocess/pool and can resume after a crash.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pitch3d.core.ports.jobs import JobHandle, JobQueue, JobState, Worker


@dataclass
class InProcessWorker(Worker):
    """Runs a thunk to completion in the calling thread."""

    def run(self, thunk: Callable[[], Any]) -> Any:
        return thunk()


@dataclass
class InProcessJobQueue(JobQueue):
    """Synchronous queue: each ``submit`` runs the thunk now and records the outcome."""

    worker: InProcessWorker = field(default_factory=InProcessWorker)
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(1), repr=False)
    _state: dict[str, JobState] = field(default_factory=dict, repr=False)
    _result: dict[str, Any] = field(default_factory=dict, repr=False)
    _error: dict[str, BaseException] = field(default_factory=dict, repr=False)

    def submit(
        self,
        stage: str,
        thunk: Callable[[], Any],
        *,
        meta: dict | None = None,
    ) -> JobHandle:
        job_id = f"{stage}-{next(self._ids)}"
        handle = JobHandle(id=job_id, stage=stage, meta=dict(meta or {}))
        self._state[job_id] = JobState.RUNNING
        try:
            self._result[job_id] = self.worker.run(thunk)
            self._state[job_id] = JobState.DONE
        except BaseException as exc:  # noqa: BLE001 — recorded, re-raised on result()
            self._error[job_id] = exc
            self._state[job_id] = JobState.FAILED
        return handle

    def state(self, job: JobHandle) -> JobState:
        return self._state.get(job.id, JobState.PENDING)

    def result(self, job: JobHandle) -> Any:
        st = self._state.get(job.id, JobState.PENDING)
        if st is JobState.FAILED:
            raise self._error[job.id]
        if st is not JobState.DONE:
            raise RuntimeError(f"job {job.id} is {st.value}, not done")
        return self._result[job.id]

    def cancel(self, job: JobHandle) -> bool:
        # Work runs synchronously on submit, so by the time a handle exists it has finished.
        return False
