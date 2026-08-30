"""Heightmap and lithophane backends. Pure NumPy, no GPU, no model weights.

These treat image brightness as a displacement field. Not true 3D -- the back
of the object is a flat plate -- but deterministic, fast, and reliably
printable, which makes them the reference implementation for the rest of the
pipeline.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh
from PIL import Image, ImageFilter

from printable.backends.base import GeometryBackend, registry
from printable.types import Backend, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)


def _load_mask(path, max_dim: int, key_background: bool, tolerance: float):
    """Return an alpha mask in [0, 1], or None when nothing is keyed out.

    Without this the backdrop becomes geometry: a mid-grey background maps to
    mid-height, so the subject sits in a raised slab instead of standing proud
    of the plate. On this character sheet the backdrop is actually *brighter*
    than the figure, which inverts the relief outright.
    """
    img = Image.open(path)

    if "A" in img.getbands():
        alpha = img.getchannel("A")
    elif key_background:
        from printable.backends.preprocess import flood_background

        alpha = flood_background(img.convert("RGB"), tolerance).getchannel("A")
    else:
        return None

    w, h = alpha.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        alpha = alpha.resize(
            (max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS
        )
    return np.asarray(alpha, dtype=np.float32) / 255.0


def _load_grayscale(path, max_dim: int) -> np.ndarray:
    """Load an image as float32 luminance in [0, 1], capped at max_dim.

    The cap matters: a 4000px photo triangulated naively is ~32M triangles,
    which no slicer will open. Downsampling here is far cheaper than
    decimating the mesh afterwards.
    """
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = min(max_dim / max(w, h), 1.0)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
        log.info("resized %dx%d -> %dx%d", w, h, img.size[0], img.size[1])
    return np.asarray(img, dtype=np.float32) / 255.0


def _grid_mesh(heights: np.ndarray, width_mm: float, base_mm: float) -> trimesh.Trimesh:
    """Build a closed solid from a height grid.

    Produces top surface + flat bottom + four side walls, wound consistently so
    the result is watertight and manifold straight out of the box. Doing this
    correctly here saves the repair stage a lot of work.
    """
    rows, cols = heights.shape
    pitch = width_mm / (cols - 1)
    depth_mm = pitch * (rows - 1)

    xs = np.linspace(0.0, width_mm, cols, dtype=np.float32)
    # Row 0 of the image must land at the far (+Y) edge so the print reads the
    # same way up as the source. Building in this flipped frame reverses the
    # handedness of every face, so windings below are derived for it directly
    # rather than borrowed from a conventional +Y-up grid.
    ys = np.linspace(depth_mm, 0.0, rows, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)

    top = np.stack([gx.ravel(), gy.ravel(), (heights + base_mm).ravel()], axis=1)
    bottom = np.stack([gx.ravel(), gy.ravel(), np.zeros(rows * cols, np.float32)], axis=1)
    vertices = np.concatenate([top, bottom], axis=0)
    n = rows * cols

    def quad(a, b, c, d):
        """Split a quad into two triangles, preserving the given order."""
        return np.stack(
            [np.stack([a, b, c], axis=-1), np.stack([a, c, d], axis=-1)]
        ).reshape(-1, 3)

    idx = np.arange(n, dtype=np.int64).reshape(rows, cols)
    tl, tr = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    bl, br = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()

    faces = [
        quad(tl, bl, br, tr),                       # top surface, normals +Z
        quad(tl + n, tr + n, br + n, bl + n),       # bottom, reversed for -Z
    ]

    # Side walls, one strip per boundary. Each pairs a top-edge run with the
    # matching bottom-edge run; the order is chosen so the resulting normal
    # points away from the solid in this flipped frame.
    left, right = idx[:, 0], idx[:, -1]
    far, near = idx[0, :], idx[-1, :]
    faces.append(quad(left[:-1], left[:-1] + n, left[1:] + n, left[1:]))
    faces.append(quad(right[1:], right[1:] + n, right[:-1] + n, right[:-1]))
    faces.append(quad(far[1:], far[1:] + n, far[:-1] + n, far[:-1]))
    faces.append(quad(near[:-1], near[:-1] + n, near[1:] + n, near[1:]))

    mesh = trimesh.Trimesh(
        vertices=vertices, faces=np.concatenate(faces), process=True
    )
    mesh.merge_vertices()

    # The grid is closed by construction, so a negative volume means the whole
    # solid came out inside-out. Flip rather than leaving it for repair, which
    # would otherwise see a valid-but-inverted mesh and pass it through.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    return mesh


class HeightmapBackend(GeometryBackend):
    """Brightness -> height relief. Bright areas rise."""

    name = "heightmap"
    requires_gpu = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        o = request.options
        max_dim = int(o.get("max_dim", 400))
        width_mm = float(o.get("width_mm", 100.0))
        relief_mm = float(o.get("relief_mm", 5.0))
        base_mm = float(o.get("base_mm", 2.0))
        blur = float(o.get("blur", 1.0))
        invert = bool(o.get("invert", False))
        gamma = float(o.get("gamma", 1.0))

        gray = _load_grayscale(request.image_path, max_dim)
        if blur > 0:
            gray = np.asarray(
                Image.fromarray((gray * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(blur)
                ),
                dtype=np.float32,
            ) / 255.0

        if invert:
            gray = 1.0 - gray
        if gamma != 1.0:
            gray = np.power(np.clip(gray, 0.0, 1.0), gamma)

        mask = _load_mask(
            request.image_path, max_dim,
            bool(o.get("key_background", False)),
            float(o.get("bg_tolerance", 34.0)),
        )
        heights = gray * relief_mm
        if mask is not None:
            # Masked-out pixels fall to the base plate rather than rising.
            heights = heights * mask

        mesh = _grid_mesh(heights, width_mm, base_mm)
        return GenerationResult(
            mesh=mesh,
            backend=Backend.HEIGHTMAP,
            metadata={
                "grid": list(gray.shape),
                "width_mm": width_mm,
                "relief_mm": relief_mm,
            },
        )


class LithophaneBackend(GeometryBackend):
    """Backlit relief. Dark areas print thick so they block more light.

    The inversion is the whole point and the classic first bug: a lithophane
    with bright-equals-thick reads as a photographic negative when lit.
    """

    name = "lithophane"
    requires_gpu = False

    def generate(self, request: GenerationRequest) -> GenerationResult:
        o = request.options
        max_dim = int(o.get("max_dim", 500))
        width_mm = float(o.get("width_mm", 100.0))
        # 0.8mm passes light on most white PLA; 3mm reads as solid black.
        min_mm = float(o.get("min_thickness_mm", 0.8))
        max_mm = float(o.get("max_thickness_mm", 3.0))
        # Linear brightness->thickness looks flat and washed out. This curve
        # pushes midtones toward the thick end, which matches how transmitted
        # light actually falls off.
        gamma = float(o.get("gamma", 2.2))
        blur = float(o.get("blur", 0.5))
        frame_mm = float(o.get("frame_mm", 0.0))

        gray = _load_grayscale(request.image_path, max_dim)
        if blur > 0:
            gray = np.asarray(
                Image.fromarray((gray * 255).astype(np.uint8)).filter(
                    ImageFilter.GaussianBlur(blur)
                ),
                dtype=np.float32,
            ) / 255.0

        # Dark -> thick.
        thickness = 1.0 - np.clip(gray, 0.0, 1.0)
        thickness = np.power(thickness, 1.0 / gamma)
        heights = min_mm + thickness * (max_mm - min_mm)

        if frame_mm > 0:
            pitch = width_mm / (heights.shape[1] - 1)
            border = max(1, round(frame_mm / pitch))
            heights[:border, :] = max_mm
            heights[-border:, :] = max_mm
            heights[:, :border] = max_mm
            heights[:, -border:] = max_mm

        mask = _load_mask(
            request.image_path, max_dim,
            bool(o.get("key_background", False)),
            float(o.get("bg_tolerance", 34.0)),
        )
        if mask is not None:
            # A lithophane must stay continuous, so the background thins to
            # the minimum rather than being cut away entirely.
            heights = min_mm + (heights - min_mm) * mask

        # base_mm is 0: the lithophane *is* the plate, min_mm is baked in.
        mesh = _grid_mesh(heights, width_mm, 0.0)
        return GenerationResult(
            mesh=mesh,
            backend=Backend.LITHOPHANE,
            metadata={
                "grid": list(gray.shape),
                "min_thickness_mm": min_mm,
                "max_thickness_mm": max_mm,
                "gamma": gamma,
            },
        )


registry.register("heightmap", HeightmapBackend)
registry.register("lithophane", LithophaneBackend)
