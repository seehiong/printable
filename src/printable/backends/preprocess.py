"""Image preprocessing for the AI backends.

Background removal is the highest-leverage step in the whole pipeline. Every
image-to-3D model is trained on clean object-on-neutral-background renders, so
a busy photo background degrades geometry badly -- often the model tries to
reconstruct the background as part of the object.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


def flood_background(image: Image.Image, tolerance: float = 34.0) -> Image.Image:
    """Key out a flat background by colour-matching from the corners.

    For studio renders and reference sheets on a plain backdrop this beats a
    segmentation model: deterministic, instant, no weights to download. It is
    the wrong tool for a photo with a busy background -- use remove_background
    for those.

    Handles a lighting gradient across the backdrop by fitting a plane to the
    border pixels rather than keying against one flat colour.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w, _ = rgb.shape

    # Studio backdrops are rarely one flat colour -- they usually carry a
    # lighting gradient. Fitting a plane per channel across the border pixels
    # tracks that, where a single median colour keys one end and misses the
    # other.
    band = max(2, min(h, w) // 40)
    yy, xx = np.mgrid[0:h, 0:w]
    border = np.zeros((h, w), dtype=bool)
    border[:band, :] = border[-band:, :] = True
    border[:, :band] = border[:, -band:] = True

    # Robust fit: drop border pixels far from the median so a subject touching
    # the frame edge does not drag the plane toward its own colour.
    med = np.median(rgb[border], axis=0)
    inlier = border & (np.linalg.norm(rgb - med, axis=2) < 60)
    if inlier.sum() < 32:
        inlier = border

    basis = np.stack([xx[inlier], yy[inlier], np.ones(inlier.sum())], axis=1)
    bg = np.empty_like(rgb)
    full = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w)], axis=1)
    for ch in range(3):
        coef, *_ = np.linalg.lstsq(basis, rgb[inlier][:, ch], rcond=None)
        bg[:, :, ch] = (full @ coef).reshape(h, w)

    alpha = (np.linalg.norm(rgb - bg, axis=2) > tolerance).astype(np.uint8) * 255

    # Keep only what connects to the largest region, so background pockets of
    # similar colour inside the subject are not punched through.
    try:
        from scipy import ndimage

        labels, count = ndimage.label(alpha > 0)
        if count > 1:
            sizes = ndimage.sum(alpha > 0, labels, range(1, count + 1))
            alpha = np.where(labels == (np.argmax(sizes) + 1), 255, 0).astype(np.uint8)
        # Soften the cut by one pixel so the mesh edge is not a hard staircase.
        alpha = ndimage.grey_closing(alpha, size=3)
    except ImportError:
        log.debug("scipy unavailable; using raw colour key")

    out = np.dstack([np.asarray(image.convert("RGB")), alpha])
    log.info("flood key removed %.0f%% of the frame", 100.0 * (alpha == 0).mean())
    return Image.fromarray(out, "RGBA")


def remove_background(image: Image.Image, model: str = "u2net") -> Image.Image:
    """Strip the background, returning RGBA with an alpha matte.

    Falls back to the original image if rembg is unavailable -- degraded
    quality, but the pipeline still runs.
    """
    try:
        import rembg
    except ImportError:
        log.warning("rembg not installed; using image as-is (expect worse geometry)")
        return image.convert("RGBA")

    session = rembg.new_session(model)
    out = rembg.remove(image, session=session)
    log.info("background removed with %s", model)
    return out.convert("RGBA")


def center_and_pad(image: Image.Image, size: int = 518, margin: float = 0.85) -> Image.Image:
    """Crop to the subject, centre it, pad to square.

    Models expect the subject to fill a consistent fraction of the frame. An
    off-centre or tightly-cropped subject produces distorted output.
    """
    rgba = image.convert("RGBA")
    alpha = np.asarray(rgba)[:, :, 3]

    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)
    if not rows.any() or not cols.any():
        log.warning("empty alpha channel; skipping crop")
        return rgba.resize((size, size), Image.LANCZOS)

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    subject = rgba.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))

    # Fit the subject into `margin` of the frame, leaving breathing room.
    target = int(size * margin)
    w, h = subject.size
    scale = target / max(w, h)
    subject = subject.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(
        subject,
        ((size - subject.size[0]) // 2, (size - subject.size[1]) // 2),
        subject,
    )
    return canvas


def prepare_image(
    path: Path,
    *,
    size: int = 518,
    strip_background: bool = True,
    bg_model: str = "u2net",
) -> Image.Image:
    """Full preprocessing chain for AI backends."""
    img = Image.open(path).convert("RGB")
    if strip_background:
        img = remove_background(img, bg_model)
    else:
        img = img.convert("RGBA")
    return center_and_pad(img, size)
