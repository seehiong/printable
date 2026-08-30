"""Split a multi-view sheet into single-subject panels.

Character turnarounds and reference sheets pack several views of one subject
into a grid. Every image-to-3D model expects a single centred object, so
feeding a sheet in whole makes the model try to reconstruct all the figures as
one mass. Splitting first is what makes such an image usable at all.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def find_seams(image: Image.Image, axis: int, expected: int) -> list[int]:
    """Locate low-variance dividing lines along one axis.

    Panel gutters are near-uniform background, so they show up as minima in
    per-row (or per-column) pixel variance. Searching near the ideal even
    split keeps a busy background from producing spurious seams.
    """
    a = np.asarray(image.convert("RGB"), dtype=np.float32)
    length = a.shape[axis]
    # Collapse every axis but the one of interest.
    var = a.std(axis=tuple(i for i in range(3) if i != axis))

    seams = []
    for k in range(1, expected):
        ideal = length * k // expected
        window = max(4, length // 40)
        lo, hi = max(0, ideal - window), min(length, ideal + window + 1)
        seams.append(int(lo + np.argmin(var[lo:hi])))
    return seams


def split_sheet(
    path: Path,
    rows: int = 2,
    cols: int = 2,
    *,
    trim: int = 4,
    detect_seams: bool = True,
) -> list[Image.Image]:
    """Cut a sheet into rows x cols panels, in reading order.

    `trim` pulls a few pixels off each edge so a panel never carries a sliver
    of the divider, which would otherwise read as geometry.
    """
    image = Image.open(path).convert("RGB")
    w, h = image.size

    if detect_seams:
        ys = [0, *find_seams(image, 0, rows), h]
        xs = [0, *find_seams(image, 1, cols), w]
        log.info("detected seams: rows=%s cols=%s", ys[1:-1], xs[1:-1])
    else:
        ys = [round(h * i / rows) for i in range(rows + 1)]
        xs = [round(w * i / cols) for i in range(cols + 1)]

    panels = []
    for r in range(rows):
        for c in range(cols):
            box = (
                xs[c] + (trim if c else 0),
                ys[r] + (trim if r else 0),
                xs[c + 1] - (trim if c < cols - 1 else 0),
                ys[r + 1] - (trim if r < rows - 1 else 0),
            )
            panels.append(image.crop(box))
    return panels


def crop_views(image: Image.Image, views: dict[str, dict]) -> dict[str, Image.Image]:
    """Crop named views out of a spec sheet using VLM-extracted bounding boxes.

    `views` matches design_spec.json's shape: {name: {"box_2d": [ymin, xmin,
    ymax, xmax]}}, normalized 0-1000 over the full image (see ROADMAP.md's
    Phase 1). Unlike split_sheet's even/seam-detected grid, these boxes come
    from the VLM's own reading of the sheet's actual layout, not an assumed
    grid geometry.
    """
    w, h = image.size
    out = {}
    for name, view in views.items():
        ymin, xmin, ymax, xmax = view["box_2d"]
        box = (
            round(xmin / 1000 * w),
            round(ymin / 1000 * h),
            round(xmax / 1000 * w),
            round(ymax / 1000 * h),
        )
        out[name] = image.crop(box)
    return out


def pick_best_panel(panels: list[Image.Image]) -> int:
    """Heuristic: the panel whose subject is largest and best centred.

    Used when a sheet must be reduced to one view for a single-image model.
    Scores by how much non-background mass sits near the panel centre.
    """
    best, best_score = 0, -np.inf
    for i, panel in enumerate(panels):
        a = np.asarray(panel.convert("RGB"), dtype=np.float32)
        # Treat the modal border colour as background.
        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(border, axis=0)
        mask = np.linalg.norm(a - bg, axis=2) > 30

        if not mask.any():
            continue
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        coverage = mask.mean()
        # Distance of the subject centroid from the panel centre, normalised.
        offset = np.hypot(ys.mean() - h / 2, xs.mean() - w / 2) / np.hypot(h / 2, w / 2)
        score = coverage - offset
        if score > best_score:
            best, best_score = i, score
    return best
