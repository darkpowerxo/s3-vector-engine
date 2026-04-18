"""ProgressActor — fire-and-forget progress tracking for batch jobs.

A Ray actor used as a shared counter. Workers call ``.remote()`` without
blocking, and the API gateway polls for progress updates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Snapshot of a batch job's progress."""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    total: int = 0
    completed: int = 0
    failed: int = 0
    started_at: float = 0.0
    updated_at: float = 0.0
    error: str | None = None
    stage: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def pct(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.completed / self.total * 100, 2)

    @property
    def elapsed_s(self) -> float:
        if self.started_at == 0:
            return 0.0
        end = self.updated_at or time.time()
        return round(end - self.started_at, 2)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "pct": self.pct,
            "elapsed_s": self.elapsed_s,
            "stage": self.stage,
            "error": self.error,
            "metadata": self.metadata,
        }


class ProgressActor:
    """Ray actor for tracking batch job progress.

    Designed as a fire-and-forget pattern: workers call increment/update
    methods via ``.remote()`` without awaiting the result.

    Deploy as a Ray actor::

        import ray
        progress = ProgressActor.options(name="progress").remote()
        progress.start_job.remote("job-123", total=1000)
        # ... workers call ...
        progress.increment.remote("job-123", 10)
        # ... gateway polls ...
        info = ray.get(progress.get_progress.remote("job-123"))

    When Ray is not available, this class works as a plain Python object
    for local testing.
    """

    def __init__(self):
        self._jobs: dict[str, JobProgress] = {}

    def start_job(
        self,
        job_id: str,
        total: int,
        stage: str = "extracting",
        metadata: dict | None = None,
    ) -> None:
        """Register a new job."""
        now = time.time()
        self._jobs[job_id] = JobProgress(
            job_id=job_id,
            status=JobStatus.RUNNING,
            total=total,
            started_at=now,
            updated_at=now,
            stage=stage,
            metadata=metadata or {},
        )

    def increment(self, job_id: str, n: int = 1) -> None:
        """Increment completed count (fire-and-forget from workers)."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.completed += n
        job.updated_at = time.time()
        if job.completed >= job.total:
            job.status = JobStatus.COMPLETED

    def increment_failed(self, job_id: str, n: int = 1) -> None:
        """Increment failed count."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.failed += n
        job.updated_at = time.time()

    def set_stage(self, job_id: str, stage: str) -> None:
        """Update the current processing stage."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.stage = stage
        job.updated_at = time.time()

    def set_total(self, job_id: str, total: int) -> None:
        """Update total count (e.g. after preprocessing discovers item count)."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.total = total
        job.updated_at = time.time()

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.FAILED
        job.error = error
        job.updated_at = time.time()

    def cancel_job(self, job_id: str) -> None:
        """Mark job as cancelled."""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = JobStatus.CANCELLED
        job.updated_at = time.time()

    def get_progress(self, job_id: str) -> dict | None:
        """Get progress snapshot for a job."""
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return job.to_dict()

    def list_jobs(self) -> list[dict]:
        """List all tracked jobs."""
        return [j.to_dict() for j in self._jobs.values()]
