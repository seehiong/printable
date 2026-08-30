"""Depth and normal estimation: an auxiliary geometry cue.

Unlike the backends in ai.py, this needs no separately vendored research
repo -- transformers ships a depth-estimation pipeline directly. There is no
equivalent transformers-native normal-estimation task, so the normal map
here is derived from the depth map's local gradients rather than predicted
by a learned model; treat it as a coarse cue, not a real normal estimate.

Diagnostic only: no GeometryBackend in ai.py accepts depth or normal as a
conditioning input, so estimate_depth_normal()'s output is written to PNGs
for inspection rather than fed into generation. See GeometryBackend's
wants_geometry_cues flag and pipeline/run.py's "geometry-cues" stage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from printable.types import DepthNormalResult

log = logging.getLogger(__name__)

DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

_pipe = None


def available() -> tuple[bool, str]:
    try:
        import transformers  # noqa: F401
    except ImportError:
        return False, "transformers not installed (pip install printable[depth])"
    return True, "ok"


def _load(model: str):
    global _pipe
    if _pipe is not None:
        return _pipe
    from transformers import pipeline

    log.info("loading depth-estimation model %s (first run downloads weights)", model)
    _pipe = pipeline(task="depth-estimation", model=model)
    return _pipe


def normal_from_depth(depth: np.ndarray) -> np.ndarray:
    """Cross product of the local depth-gradient tangent vectors, unit-normalized.

    Not a learned estimate -- see module docstring.
    """
    gy, gx = np.gradient(depth.astype(np.float32))
    normal = np.dstack((-gx, -gy, np.ones_like(depth, dtype=np.float32)))
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / np.clip(norm, 1e-6, None)


def estimate_depth_normal(
    image: Image.Image, *, model: str = DEFAULT_MODEL
) -> DepthNormalResult:
    """Estimate a depth map and a depth-derived normal map for one image."""
    pipe = _load(model)
    depth = np.asarray(pipe(image.convert("RGB"))["depth"], dtype=np.float32)
    return DepthNormalResult(
        depth=depth, normal=normal_from_depth(depth), source="depth-derived"
    )


def _to_uint8_gray(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def save_geometry_cues(result: DepthNormalResult, output_path: Path) -> list[Path]:
    """Write depth/normal maps as sibling PNGs next to output_path.

    Purely for inspection -- nothing downstream reads these back in yet.
    """
    written = [output_path.with_name(f"{output_path.stem}_depth.png")]
    Image.fromarray(_to_uint8_gray(result.depth)).save(written[0])

    if result.normal is not None:
        normal_path = output_path.with_name(f"{output_path.stem}_normal.png")
        normal_rgb = ((result.normal + 1.0) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(normal_rgb, mode="RGB").save(normal_path)
        written.append(normal_path)

    return written
