"""Tests for the depth-derived normal math. Pure NumPy, no model download."""

from __future__ import annotations

import numpy as np

from printable.backends.depth import normal_from_depth


def test_normal_from_depth_is_unit_length():
    depth = np.random.default_rng(0).uniform(0, 1, size=(16, 16)).astype(np.float32)
    normal = normal_from_depth(depth)
    lengths = np.linalg.norm(normal, axis=-1)
    np.testing.assert_allclose(lengths, 1.0, atol=1e-5)


def test_normal_from_depth_flat_surface_points_at_camera():
    """A constant depth map has zero gradient everywhere: the surface faces
    the camera dead-on, so every normal should be exactly +Z."""
    depth = np.full((8, 8), 5.0, dtype=np.float32)
    normal = normal_from_depth(depth)
    np.testing.assert_allclose(normal, np.dstack(
        (np.zeros((8, 8)), np.zeros((8, 8)), np.ones((8, 8)))
    ), atol=1e-6)
