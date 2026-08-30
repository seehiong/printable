"""Print preparation: scale to millimetres, orient, seat on the bed, hollow."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from printable.types import PrintSettings

log = logging.getLogger(__name__)


def scale_to_size(mesh: trimesh.Trimesh, target_mm: float) -> trimesh.Trimesh:
    """Scale so the longest axis measures target_mm.

    Generators emit arbitrary units -- usually normalised to a unit cube -- so
    without this every model prints at 1mm tall.
    """
    mesh = mesh.copy()
    longest = float(mesh.extents.max())
    if longest <= 0:
        raise ValueError("degenerate mesh: zero extent")
    mesh.apply_scale(target_mm / longest)
    log.info("scaled to %.1fmm (was %.4f units)", target_mm, longest)
    return mesh


def _support_cost(mesh: trimesh.Trimesh, max_overhang_deg: float) -> tuple[float, float]:
    """Score an orientation: (overhang area, base contact area). Lower/higher is better."""
    normals = mesh.face_normals
    areas = mesh.area_faces

    # A downward-facing steep face needs support. Angle measured from -Z.
    cos_thresh = np.cos(np.radians(90.0 - max_overhang_deg))
    downward = normals[:, 2] < -cos_thresh
    overhang_area = float(areas[downward].sum())

    # Faces within a hair of the bed plane and pointing down give adhesion.
    z_min = mesh.bounds[0][2]
    face_z = mesh.vertices[mesh.faces][:, :, 2].mean(axis=1)
    on_bed = (face_z < z_min + mesh.extents[2] * 0.02) & (normals[:, 2] < -0.9)
    base_area = float(areas[on_bed].sum())

    return overhang_area, base_area


def auto_orient(mesh: trimesh.Trimesh, settings: PrintSettings) -> trimesh.Trimesh:
    """Search candidate rotations for the one that prints most easily.

    Scores each by support area needed minus base contact. Not a full convex
    hull search -- just the axis-aligned faces plus a few tilts, which covers
    the overwhelming majority of real cases at negligible cost.

    Deliberately excludes the 180-degree-about-X/Y candidates (the ones that
    swap top for bottom while keeping the same footprint-vs-height shape).
    For humanoid figures those score *better* than the correct upright pose
    on this metric alone -- a standing figure's arms/head/clothing create
    more real overhang than an inverted one does -- so including them
    reliably flips characters onto their head. Confirmed via a raw TripoSR
    output (character_male.png): upright scored -544 (overhang 745, base
    100) vs -331 flipped (overhang 421, base 45); the flip wins on pure
    overhang-minimization despite being wrong. The remaining 90/270
    candidates still catch the legitimate case (an object reconstructed
    lying on its side that should stand up) without ever swapping which end
    is "up".
    """
    candidates = [np.eye(4)]
    for axis in ([1, 0, 0], [0, 1, 0]):
        for deg in (90, 270):
            candidates.append(trimesh.transformations.rotation_matrix(np.radians(deg), axis))

    best, best_score, best_tf = mesh, -np.inf, np.eye(4)
    for tf in candidates:
        trial = mesh.copy()
        trial.apply_transform(tf)
        overhang, base = _support_cost(trial, settings.max_overhang_deg)
        # Base contact is worth more than avoided support: a part that lifts
        # off the bed fails outright, while supports merely cost material.
        score = base * 2.0 - overhang
        if score > best_score:
            best, best_score, best_tf = trial, score, tf

    if not np.allclose(best_tf, np.eye(4)):
        log.info("auto-oriented (score %.1f)", best_score)
    return best


def apply_rotation(mesh: trimesh.Trimesh, degrees: tuple[float, float, float]) -> trimesh.Trimesh:
    """Rotate by (x, y, z) degrees, in that order, about the mesh's own centroid."""
    mesh = mesh.copy()
    centroid = mesh.centroid
    mesh.apply_translation(-centroid)
    for axis, deg in zip(np.eye(3), degrees):
        if deg:
            mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(deg), axis))
    mesh.apply_translation(centroid)
    log.info("manually rotated %s degrees (x, y, z)", tuple(degrees))
    return mesh


def seat_on_bed(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop the model so its lowest point sits at z=0, centred in X/Y."""
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    return mesh


def add_base(mesh: trimesh.Trimesh, thickness_mm: float) -> trimesh.Trimesh:
    """Union a flat pad under the model for bed adhesion.

    Sized to the model's footprint plus a small margin. Only worth doing when
    the natural contact patch is small.
    """
    if thickness_mm <= 0:
        return mesh

    lo, hi = mesh.bounds
    w, d = (hi[0] - lo[0]) * 1.05, (hi[1] - lo[1]) * 1.05
    pad = trimesh.creation.box(extents=[w, d, thickness_mm])
    pad.apply_translation([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, thickness_mm / 2])

    lifted = mesh.copy()
    lifted.apply_translation([0, 0, thickness_mm * 0.5])

    try:
        out = trimesh.boolean.union([lifted, pad])
        out.fix_normals()
        log.info("added %.1fmm base", thickness_mm)
        return out
    except Exception as exc:  # noqa: BLE001 - boolean backends are flaky
        # Returning the un-unioned parts would export as disconnected bodies
        # and fail validation with a confusing message, so say what actually
        # went wrong and hand back the model unchanged.
        log.warning(
            "base union failed (%s); returning model without a base. "
            "If this persists, check that manifold3d is installed.", exc
        )
        return mesh


def hollow(mesh: trimesh.Trimesh, wall_mm: float) -> trimesh.Trimesh:
    """Hollow the solid, leaving walls of wall_mm.

    Saves material and print time on large parts. Needs a manifold boolean
    backend; falls back to the solid mesh if unavailable.
    """
    if wall_mm <= 0:
        return mesh
    if not mesh.is_watertight:
        log.warning("cannot hollow a non-watertight mesh; skipping")
        return mesh

    try:
        inner = mesh.copy()
        # Shrink about the centroid to approximate an offset surface. Crude
        # versus a true signed-distance offset, but robust and dependency-free.
        centroid = mesh.centroid
        shrink = 1.0 - (2.0 * wall_mm / float(mesh.extents.max()))
        if shrink <= 0.1:
            log.warning("wall too thick to hollow; skipping")
            return mesh
        inner.apply_translation(-centroid)
        inner.apply_scale(shrink)
        inner.apply_translation(centroid)
        inner.invert()

        out = trimesh.boolean.difference([mesh, inner])
        out.fix_normals()
        log.info("hollowed to %.1fmm walls", wall_mm)
        return out
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "hollowing failed (%s); keeping the model solid. "
            "If this persists, check that manifold3d is installed.", exc
        )
        return mesh


def prepare(mesh: trimesh.Trimesh, settings: PrintSettings) -> trimesh.Trimesh:
    """Run the full print-prep chain in the order that matters."""
    mesh = scale_to_size(mesh, settings.target_size_mm)

    if settings.auto_orient:
        mesh = auto_orient(mesh, settings)

    if settings.manual_rotation_deg:
        mesh = apply_rotation(mesh, settings.manual_rotation_deg)

    mesh = seat_on_bed(mesh)

    if settings.hollow:
        mesh = hollow(mesh, settings.hollow_wall_mm)

    if settings.add_base:
        mesh = add_base(mesh, settings.base_thickness_mm)
        mesh = seat_on_bed(mesh)

    return mesh
