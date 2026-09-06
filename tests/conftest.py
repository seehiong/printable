"""Shared pytest fixtures."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    # Stashed on `config` rather than a module global: conftest.py can end
    # up imported under two different module identities (`conftest` vs
    # `tests.conftest`) depending on how pytest resolves rootdir, which
    # silently split a module-global version of this into two unrelated
    # globals and made pytest_unconfigure below always see None -- masking
    # every real test failure's exit code. `config` is the same object
    # instance across both hooks regardless of which module identity ran
    # them, so this doesn't have that problem.
    session.config._hard_exit_status = int(exitstatus)  # type: ignore[attr-defined]


def pytest_unconfigure(config: pytest.Config) -> None:
    """Force-exit instead of letting CPython finalize normally.

    Real, reproducible crash (3/3 via gdb): pymeshlab -- a genuine runtime
    dependency of hy3dshape's postprocessors.py (FaceReducer,
    FloaterRemover, ...), not the stray package an earlier investigation
    mistakenly concluded it was -- bundles a Qt runtime whose
    PluginManager destructor SIGABRTs ("corrupted double-linked list")
    while tearing down a QMap of plugins during interpreter finalization.
    It can't just be uninstalled (hy3dshape needs it), so the only
    tractable fix is skipping the teardown where the crash lives. All test
    results are already reported by this point (pytest_unconfigure is
    pytest's last hook), so os._exit() with the real exit status changes
    nothing CI observes -- it just never reaches the C++ static
    destructor.
    """
    exit_status = getattr(config, "_hard_exit_status", None)
    if exit_status is not None:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_status)


@pytest.fixture(autouse=True, scope="session")
def _release_cached_backends():
    """Release any real GPU backend (triposr/hunyuan3d) the session cached.

    `backends.base.registry` is a process-wide singleton -- test_api.py,
    test_pipeline.py, and test_select.py can each load a real backend into
    it over the course of one pytest run, and it's never torn down on its
    own. See `BackendRegistry.clear_instances()`'s docstring for the
    reasoning (real hygiene, not a confirmed fix for any specific crash
    this project hit) -- session-scoped so this runs once, after every
    test module, regardless of collection order."""
    yield
    from printable.backends.base import registry

    registry.clear_instances()


@pytest.fixture(scope="module")
def sample_image(tmp_path_factory) -> Path:
    """A small gradient image with enough variation to exercise the mesher."""
    n = 96
    y, x = np.mgrid[0:n, 0:n]
    img = np.clip(255 - np.hypot(x - n / 2, y - n / 2) * 4, 0, 255)
    img += 40 * np.sin(x / 6.0)
    path = tmp_path_factory.mktemp("img") / "sample.png"
    Image.fromarray(np.clip(img, 0, 255).astype("uint8")).convert("RGB").save(path)
    return path
