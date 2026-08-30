"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image


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
