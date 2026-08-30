"""Pre-flight checks. Catches unprintable output before it reaches a slicer."""

from __future__ import annotations

import logging

import numpy as np
import trimesh

from printable.types import PrintSettings, ValidationReport

log = logging.getLogger(__name__)


def count_open_edges(mesh: trimesh.Trimesh) -> int:
    """Number of boundary edges: those used by exactly one face.

    trimesh dropped the `edges_open` property in 5.x, and deriving it here
    keeps validation working across versions.
    """
    edges = np.sort(mesh.edges_sorted, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum())


def validate(mesh: trimesh.Trimesh, settings: PrintSettings) -> ValidationReport:
    """Check a prepared mesh against the printer's physical constraints."""
    report = ValidationReport()

    # --- topology -------------------------------------------------------
    if not mesh.is_watertight:
        report.add(
            "error", "not_watertight",
            "Mesh has holes; the slicer cannot tell inside from outside.",
            open_edges=count_open_edges(mesh),
        )

    if not mesh.is_winding_consistent:
        report.add(
            "error", "inconsistent_winding",
            "Face winding is inconsistent; normals point in mixed directions.",
        )

    components = len(mesh.split(only_watertight=False))
    if components > 1:
        report.add(
            "warning", "multiple_bodies",
            f"Mesh contains {components} disconnected bodies.",
            count=components,
        )

    if len(mesh.faces) == 0:
        report.add("error", "empty_mesh", "Mesh has no faces.")
        return report

    # --- size -----------------------------------------------------------
    extents = mesh.extents
    bx, by, bz = settings.build_volume_mm
    if extents[0] > bx or extents[1] > by or extents[2] > bz:
        report.add(
            "error", "exceeds_build_volume",
            f"Model is {extents[0]:.0f}x{extents[1]:.0f}x{extents[2]:.0f}mm, "
            f"build volume is {bx:.0f}x{by:.0f}x{bz:.0f}mm.",
            extents=extents.tolist(),
        )

    if float(extents.min()) < settings.min_wall_thickness_mm:
        report.add(
            "warning", "very_thin_model",
            f"Smallest dimension is {extents.min():.2f}mm, at or below the "
            f"{settings.min_wall_thickness_mm}mm minimum wall.",
        )

    # --- thin features --------------------------------------------------
    # Sample interior points and measure distance to the surface. Anything
    # closer than half the min wall is a feature too thin to print.
    if mesh.is_watertight:
        try:
            pts = trimesh.sample.volume_mesh(mesh, 2000)
            if len(pts) > 50:
                dist = trimesh.proximity.signed_distance(mesh, pts)
                thin = float((dist < settings.min_wall_thickness_mm / 2).mean())
                if thin > 0.05:
                    report.add(
                        "warning", "thin_features",
                        f"{thin * 100:.0f}% of sampled interior is thinner than "
                        f"{settings.min_wall_thickness_mm}mm; fine detail may not print.",
                        fraction=thin,
                    )
        except Exception as exc:  # noqa: BLE001 - sampling is best-effort
            log.debug("thin-feature sampling skipped: %s", exc)

    # --- overhangs ------------------------------------------------------
    # A face pointing straight down is only an overhang if it is *above* the
    # bed. The flat underside of any part points at -Z and rests on the
    # build plate, so counting it here reports every flat-bottomed model as
    # half overhang.
    z_min_all = mesh.bounds[0][2]
    face_z_all = mesh.vertices[mesh.faces][:, :, 2].mean(axis=1)
    bed_tol = max(0.2, mesh.extents[2] * 0.01)
    resting = face_z_all < z_min_all + bed_tol

    cos_thresh = np.cos(np.radians(90.0 - settings.max_overhang_deg))
    steep = (mesh.face_normals[:, 2] < -cos_thresh) & ~resting
    overhang_frac = float(mesh.area_faces[steep].sum() / mesh.area) if mesh.area else 0.0
    if overhang_frac > 0.15:
        report.add(
            "info", "needs_supports",
            f"{overhang_frac * 100:.0f}% of surface overhangs past "
            f"{settings.max_overhang_deg}deg; enable supports.",
            fraction=overhang_frac,
        )

    # --- bed adhesion ---------------------------------------------------
    on_bed = resting & (mesh.face_normals[:, 2] < -0.9)
    base_area = float(mesh.area_faces[on_bed].sum())
    if base_area < 100.0:
        report.add(
            "warning", "small_base",
            f"Only {base_area:.0f}mm^2 touches the bed; risk of the part "
            "detaching mid-print. Consider a brim or a base.",
            area_mm2=base_area,
        )

    # --- stats ----------------------------------------------------------
    report.stats = {
        "faces": len(mesh.faces),
        "vertices": len(mesh.vertices),
        "extents_mm": [round(float(e), 2) for e in extents],
        "volume_cm3": round(float(mesh.volume) / 1000.0, 2) if mesh.is_watertight else None,
        "surface_area_cm2": round(float(mesh.area) / 100.0, 2),
        "watertight": bool(mesh.is_watertight),
        "bodies": components,
        "base_contact_mm2": round(base_area, 1),
        "overhang_fraction": round(overhang_frac, 3),
    }

    # Rough filament estimate at ~15% infill; useful sanity signal.
    if mesh.is_watertight:
        vol_cm3 = float(mesh.volume) / 1000.0
        report.stats["est_filament_g"] = round(vol_cm3 * 0.15 * 1.24, 1)

    return report


def check_dimension_targets(
    report: ValidationReport,
    mesh: trimesh.Trimesh,
    targets_mm: dict[str, float],
    tolerance_pct: float = 15.0,
) -> None:
    """Compare a generated mesh's proportions against spec-sheet target dimensions.

    Appends issues onto an existing report in place, mirroring validate()'s
    own report.add(...) style, so report.printable/.errors reflect the
    combined result without the caller merging two lists.

    Deliberately axis-agnostic: auto_orient (pipeline/prep.py) picks the
    mesh's final orientation by minimizing support cost, not by matching a
    spec sheet's own width/depth/height labels, so there is no reliable
    mesh-axis-to-named-target correspondence to check directly. Comparing
    *sorted* target values against *sorted* mesh extents sidesteps that
    correspondence problem while still catching what actually matters here:
    the AI backend reconstructing the wrong proportions, even when the
    single scaled axis matches target by construction.

    Callers are responsible for selecting which targets_mm keys represent
    overall bounding size (e.g. total_height/width/depth) -- a design spec
    also carries sub-feature dimensions (cap_height, wall thickness, ...)
    that have nothing to do with the mesh's overall bounding box.
    """
    if not targets_mm:
        return
    target_values = sorted(targets_mm.values(), reverse=True)
    actual_values = sorted(mesh.extents.tolist(), reverse=True)
    for i in range(min(len(target_values), len(actual_values))):
        target, actual = target_values[i], actual_values[i]
        if target <= 0:
            continue
        delta_pct = abs(actual - target) / target * 100
        if delta_pct > tolerance_pct:
            report.add(
                "warning", "dimension_mismatch",
                f"Generated mesh's #{i + 1} largest extent is {actual:.1f}mm, "
                f"spec sheet's #{i + 1} largest target dimension is "
                f"{target:.1f}mm ({delta_pct:.0f}% off, tolerance is "
                f"{tolerance_pct:.0f}%).",
                target_mm=target, actual_mm=actual, delta_pct=round(delta_pct, 1),
            )
