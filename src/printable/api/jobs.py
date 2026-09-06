"""In-memory job store for the web API.

Jobs are purely in-process: no persistence, no DB. A server restart drops
job history and any in-flight job. Generation runs in a single-worker
executor because the GPU backends share one accelerator and are not safe
to run concurrently against it.
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from queue import SimpleQueue
from typing import Any

log = logging.getLogger(__name__)

# Oldest jobs beyond this count are evicted (their directories deleted) to
# keep this bounded across a long-running server session.
MAX_TRACKED_JOBS = 20

# Relative to the CWD `printable serve` was started from, same convention as
# the CLI's own default `-o` (Path("output") / ...) -- both land generated
# files in one gitignored place instead of scattering them under system
# tmp, where they're easy to lose track of and don't survive a reboot.
JOBS_ROOT = Path("output") / "jobs"


@dataclass
class Job:
    id: str
    backend: str
    dir: Path
    # queued | running | awaiting_confirmation | done | error.
    # awaiting_confirmation is spec-sheet-only: extraction finished and the
    # crops are sitting in `dir` for a human to look at before the
    # (expensive, VLM-crop-quality-dependent) generation step runs -- see
    # api/app.py's _extract_spec_views/_generate_from_saved_spec and the
    # POST /api/jobs/{id}/confirm endpoint that resumes from here.
    status: str = "queued"
    events: list[dict] = field(default_factory=list)
    subscribers: list[SimpleQueue] = field(default_factory=list)
    result: Any = None  # PipelineResult, once done
    error: str | None = None
    image_path: Path | None = None
    output_path: Path | None = None
    design_spec: dict | None = None  # set once VLM extraction completes (spec-sheet jobs only)
    classification: dict | None = None  # set once /classify responds (auto jobs only)
    # Generation-phase kwargs stashed at extraction time so /confirm can
    # resume without the caller having to resend them (spec-sheet jobs only).
    pending_kwargs: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def on_stage(self, name: str, status: str) -> None:
        """Passed as run()'s on_stage callback."""
        event = {"stage": name, "status": status}
        with self.lock:
            self.events.append(event)
            subs = list(self.subscribers)
        for q in subs:
            q.put(event)

    def subscribe(self) -> tuple[list[dict], SimpleQueue | None]:
        """Returns (events so far, a live queue if the job isn't finished yet).

        awaiting_confirmation counts as finished here too: no more events
        will arrive until a future /confirm call resumes this same job
        (a fresh _run(), which accepts new subscribers again at that
        point) -- handing back a live queue now would leave an SSE stream
        open forever with nothing ever arriving on it, e.g. a page reload
        while a job is paused for review.
        """
        with self.lock:
            history = list(self.events)
            if self.status in ("done", "error", "awaiting_confirmation"):
                return history, None
            q: SimpleQueue = SimpleQueue()
            self.subscribers.append(q)
            return history, q


class JobStore:
    def __init__(self) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    def create(self, backend: str) -> Job:
        job_id = uuid.uuid4().hex
        job_dir = JOBS_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, backend=backend, dir=job_dir)
        with self._lock:
            self._jobs[job_id] = job
            self._evict_old_locked()
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    def submit(self, job: Job, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._executor.submit(self._run, job, fn, *args, **kwargs)

    def shutdown(self) -> None:
        """Join the worker thread. Not used by `printable serve` itself --
        that process just gets killed -- but a test process that runs to
        completion and calls Py_Finalize benefits from it: an idle executor
        thread left for atexit's own join to clean up can still be holding a
        reference to the last job's torch/HIP tensors on its stack, and in
        general it's safer to release that explicitly, while the
        interpreter is still fully alive, than to leave it to whatever
        order interpreter finalization happens to tear things down in.
        (Investigating a real "corrupted double-linked list" crash at the
        end of this project's own test suite traced it to pymeshlab's
        bundled Qt runtime crashing during interpreter finalization -- not
        to this thread specifically, and not a stray dependency either
        (an earlier pass wrongly concluded that and uninstalled it, which
        broke hy3dshape's postprocessors.py; see tests/conftest.py's
        pytest_unconfigure for the actual fix). Kept anyway as cheap, real
        hygiene: it does no harm and rules out one class of teardown race
        outright rather than hoping it doesn't happen to matter.)"""
        self._executor.shutdown(wait=True)

    def _run(self, job: Job, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        job.status = "running"
        try:
            job.result = fn(*args, **kwargs)
            # A function that wants a non-terminal outcome (spec-sheet
            # extraction pausing for review) sets job.status itself as its
            # last action before returning; only fall back to "done" if it
            # didn't touch it, so every existing job function -- which never
            # sets job.status at all -- keeps working unchanged.
            if job.status == "running":
                job.status = "done"
        except Exception as exc:
            log.exception("job %s failed", job.id)
            job.status = "error"
            job.error = str(exc)
        finally:
            with job.lock:
                subs, job.subscribers = job.subscribers, []
            for q in subs:
                q.put(None)  # sentinel: closes any open SSE stream

    def _evict_old_locked(self) -> None:
        """Keep at most MAX_TRACKED_JOBS; delete older ones' directories.

        Caller holds self._lock. Only evicts finished jobs so a job that's
        still queued/running is never deleted out from under itself.
        """
        while len(self._jobs) > MAX_TRACKED_JOBS:
            oldest_id, oldest = next(iter(self._jobs.items()))
            if oldest.status not in ("done", "error"):
                break  # still in flight; stop evicting rather than delete it
            del self._jobs[oldest_id]
            shutil.rmtree(oldest.dir, ignore_errors=True)


store = JobStore()
