"""Backend interface. Every geometry generator implements this."""

from __future__ import annotations

import abc
import logging
import threading

from printable.types import GenerationRequest, GenerationResult

log = logging.getLogger(__name__)


class GeometryBackend(abc.ABC):
    """Turns a single image into a raw mesh.

    Implementations are free to be slow, to need a GPU, or to hallucinate the
    unseen half of the subject. Everything downstream treats the output as
    untrusted and repairs it before export.
    """

    name: str = "unnamed"
    # Whether this backend needs a working torch + accelerator stack.
    requires_gpu: bool = False
    # Whether run() should run prepare_image() as its own timed/reported stage
    # before calling generate(). False for backends (heightmap, lithophane)
    # that do their own image loading and would be corrupted by a blanket
    # background-strip + square-crop.
    needs_preprocess: bool = False
    # Square size passed to prepare_image() when needs_preprocess is True.
    image_size: int = 518
    # Whether run() should run the geometry-cues stage (depth/normal
    # estimation) before calling generate(). Diagnostic only today -- no
    # backend actually consumes the result yet -- so this stays False until
    # one does. Users can still request the stage ad hoc for any backend via
    # --opt geometry_cues=true.
    wants_geometry_cues: bool = False

    @abc.abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Produce raw geometry. May raise BackendUnavailable."""

    def available(self) -> tuple[bool, str]:
        """Cheap check for whether this backend can run right now.

        Returns (ok, reason). Kept separate from generate() so the CLI can
        report the whole backend matrix without importing heavy deps.
        """
        return True, "ok"

    def warmup(self) -> None:
        """Optional: load weights ahead of the first real request."""
        return


class BackendUnavailable(RuntimeError):
    """Raised when a backend's dependencies or hardware are missing."""


class BackendRegistry:
    """Lazy registry so importing the CLI doesn't drag in torch.

    Instances are created once per name and reused after that. Before this,
    `get()` called the factory fresh every time, which meant every single
    generate() call -- even repeated calls to the *same* backend within one
    CLI invocation's own multi-panel (`--sheet`) or multi-view (`--spec-all-views`,
    spec-mode) loop, not just separate web API requests -- reloaded the
    entire model from scratch. Confirmed traced to a real OOM: three
    consecutive hunyuan3d generations in one long-lived `printable serve`
    process, each a full ~110s model reload, eventually exhausted memory.
    Reuse is safe for every current backend: heightmap/lithophane hold no
    instance state at all, and hunyuan3d/triposr already cache their own
    loaded model on `self` (`self._pipe`/`self._model`) specifically so a
    warm instance skips reloading -- that caching was simply unreachable
    before, since it never survived past one `get()` call.
    """

    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}
        # (factory, instance) so re-registering a name -- register(), or
        # tests monkeypatching `_factories` directly -- naturally invalidates
        # the cache instead of silently serving a stale instance built from
        # the old factory.
        self._instances: dict[str, tuple[callable, GeometryBackend]] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: callable) -> None:
        self._factories[name] = factory

    def get(self, name: str) -> GeometryBackend:
        if name not in self._factories:
            known = ", ".join(sorted(self._factories))
            raise ValueError(f"unknown backend {name!r}; known: {known}")
        factory = self._factories[name]
        with self._lock:
            cached = self._instances.get(name)
            if cached is None or cached[0] is not factory:
                cached = (factory, factory())
                self._instances[name] = cached
            return cached[1]

    def names(self) -> list[str]:
        return sorted(self._factories)

    def clear_instances(self) -> None:
        """Drop cached backend instances, releasing any torch/HIP state.

        `printable serve` never calls this -- that process just gets
        killed, so the cache lives for the process lifetime by design (the
        whole point of caching is to skip reloading). A test process is
        different: it runs to completion and calls Py_Finalize, and it's
        safer for a cached backend still holding real torch tensors/HIP
        resources to be released explicitly here, while everything is
        still fully alive, than to leave that to whatever order interpreter
        finalization's own GC happens to tear things down in. (Investigating
        a real "corrupted double-linked list" crash at the end of this
        project's own test suite eventually traced it to an unrelated stray
        dependency, pymeshlab -- see its removal commit -- not to cached
        backend instances specifically. Kept anyway as cheap, real hygiene:
        it does no harm and rules out one class of teardown race outright.)
        Call this from a session-scoped test fixture, not per-module --
        this registry is a process-wide singleton shared across every test
        module that loads a real GPU backend."""
        with self._lock:
            self._instances.clear()


registry = BackendRegistry()
