"""Tests for the CPU pipeline. No GPU or model weights required."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from printable.backends.base import GeometryBackend, registry
from printable.backends.heightmap import HeightmapBackend, LithophaneBackend
from printable.backends.sheet import crop_views
from printable.pipeline.prep import apply_rotation, auto_orient, prepare, scale_to_size, seat_on_bed
from printable.pipeline.repair import drop_nonmanifold_faces, edge_defects, make_printable
from printable.pipeline.run import run
from printable.pipeline.validate import check_dimension_targets, count_open_edges, validate
from printable.types import (
    Backend,
    GenerationRequest,
    GenerationResult,
    PrintSettings,
    ValidationReport,
)


def _request(path: Path, backend: Backend, **opts) -> GenerationRequest:
    return GenerationRequest(image_path=path, backend=backend, options=opts)


# --- backends -----------------------------------------------------------


def test_lithophane_is_watertight_by_construction(sample_image):
    """The grid mesher must emit a closed solid without any repair."""
    mesh = LithophaneBackend().generate(
        _request(sample_image, Backend.LITHOPHANE, max_dim=80)
    ).mesh

    assert mesh.is_watertight
    assert mesh.is_winding_consistent
    assert len(mesh.split(only_watertight=False)) == 1
    assert mesh.volume > 0, "negative volume means inside-out winding"


def test_lithophane_thickness_within_bounds(sample_image):
    """Thickness must stay inside [min, max]: too thin blows out, too thick reads black."""
    mesh = LithophaneBackend().generate(
        _request(
            sample_image, Backend.LITHOPHANE,
            max_dim=80, min_thickness_mm=0.8, max_thickness_mm=3.0,
        )
    ).mesh

    z = mesh.vertices[:, 2]
    assert z.min() == pytest.approx(0.0, abs=1e-6)
    assert z.max() <= 3.0 + 1e-6
    # The thinnest non-bottom geometry should reach the configured floor.
    assert z[z > 1e-6].min() == pytest.approx(0.8, abs=0.05)


def test_lithophane_inverts_brightness(tmp_path):
    """Dark pixels must print thicker than bright ones, else it reads as a negative."""
    img = np.zeros((32, 32), dtype="uint8")
    img[:, 16:] = 255  # left half black, right half white
    path = tmp_path / "split.png"
    Image.fromarray(img).convert("RGB").save(path)

    mesh = LithophaneBackend().generate(
        _request(path, Backend.LITHOPHANE, max_dim=32, blur=0)
    ).mesh

    v = mesh.vertices
    left = v[(v[:, 0] < v[:, 0].max() * 0.3) & (v[:, 2] > 1e-6), 2]
    right = v[(v[:, 0] > v[:, 0].max() * 0.7) & (v[:, 2] > 1e-6), 2]
    assert left.mean() > right.mean(), "dark side must be thicker"


def test_heightmap_relief_scales_with_option(sample_image):
    thin = HeightmapBackend().generate(
        _request(sample_image, Backend.HEIGHTMAP, max_dim=64, relief_mm=2, base_mm=1)
    ).mesh
    thick = HeightmapBackend().generate(
        _request(sample_image, Backend.HEIGHTMAP, max_dim=64, relief_mm=10, base_mm=1)
    ).mesh
    assert thick.extents[2] > thin.extents[2] * 2


def test_registry_exposes_cpu_backends():
    assert {"heightmap", "lithophane"} <= set(registry.names())


# --- repair -------------------------------------------------------------


def test_edge_defects_detects_open_boundary():
    mesh = trimesh.creation.box()
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)  # punch a hole
    open_edges, nonmanifold = edge_defects(mesh)
    assert open_edges > 0
    assert nonmanifold == 0
    assert count_open_edges(mesh) == open_edges


def test_edge_defects_clean_on_closed_mesh():
    assert edge_defects(trimesh.creation.icosphere()) == (0, 0)


def test_drop_nonmanifold_repairs_extra_face():
    """A face sharing an existing edge makes it non-manifold; repair must clear it."""
    mesh = trimesh.creation.box()
    extra = np.vstack([mesh.faces, mesh.faces[0]])
    broken = trimesh.Trimesh(vertices=mesh.vertices, faces=extra, process=False)
    assert edge_defects(broken)[1] > 0

    assert edge_defects(drop_nonmanifold_faces(broken))[1] == 0


def test_make_printable_removes_floating_islands():
    body = trimesh.creation.icosphere(radius=1.0)
    speck = trimesh.creation.icosphere(radius=0.02)
    speck.apply_translation([5, 0, 0])

    out = make_printable(trimesh.util.concatenate([body, speck]), aggressive=False)
    assert len(out.split(only_watertight=False)) == 1
    assert out.extents[0] < 4, "the distant speck should be gone"


def test_make_printable_keeps_watertight_mesh_closed(sample_image):
    mesh = LithophaneBackend().generate(
        _request(sample_image, Backend.LITHOPHANE, max_dim=80)
    ).mesh
    out = make_printable(mesh, max_faces=len(mesh.faces))
    assert out.is_watertight


def test_decimation_preserves_watertightness(sample_image):
    """Decimation may tear the mesh; the repair ladder must close it again."""
    mesh = LithophaneBackend().generate(
        _request(sample_image, Backend.LITHOPHANE, max_dim=120)
    ).mesh
    target = len(mesh.faces) // 4

    out = make_printable(mesh, max_faces=target)

    assert len(out.faces) <= target * 1.05
    assert out.is_watertight
    assert edge_defects(out) == (0, 0)


def test_decimation_damage_falls_back_to_pre_decimation_mesh(monkeypatch):
    """If decimation breaks an already-watertight mesh and nothing in the
    repair ladder can close it again, make_printable must keep the
    pre-decimation mesh (bigger, but guaranteed watertight) rather than ship
    something broken -- this project treats watertightness as a hard gate
    (see validate.py) and face count as a soft target, not the reverse."""
    from printable.pipeline import repair as repair_mod

    good = trimesh.creation.icosphere(subdivisions=3)
    assert good.is_watertight

    broken = good.copy()
    broken.update_faces(np.arange(len(broken.faces)) != 0)  # punch a hole
    assert not broken.is_watertight

    monkeypatch.setattr(repair_mod, "decimate", lambda mesh, max_faces: broken)
    monkeypatch.setattr(repair_mod, "fill_holes", lambda mesh: mesh)  # doesn't help
    monkeypatch.setattr(repair_mod, "drop_nonmanifold_faces", lambda mesh: mesh)  # doesn't help
    monkeypatch.setattr(repair_mod, "poisson_rebuild", lambda mesh: None)  # gives up

    out = make_printable(good, max_faces=10)

    assert out.is_watertight
    assert len(out.faces) == len(good.faces), "should keep the original, not the broken decimation"


def test_decimation_damage_recovers_via_bounded_retry(monkeypatch):
    """If the requested max_faces decimation breaks the mesh but a gentler
    meet-in-the-middle retry comes out watertight, use that retry instead of
    falling all the way back to the full pre-decimation mesh -- keeps face
    count (and memory) down when a smaller fix is actually available,
    rather than always jumping straight to the raw mesh."""
    from printable.pipeline import repair as repair_mod

    good = trimesh.creation.icosphere(subdivisions=3)
    assert good.is_watertight

    broken = good.copy()
    broken.update_faces(np.arange(len(broken.faces)) != 0)
    assert not broken.is_watertight

    def fake_decimate(mesh, max_faces):
        # The aggressive first attempt (max_faces=10, from the call below)
        # "breaks" the mesh; the gentler meet-in-the-middle retry (a much
        # larger budget) "succeeds" by just handing back the original.
        return broken if max_faces <= 10 else good

    monkeypatch.setattr(repair_mod, "decimate", fake_decimate)
    monkeypatch.setattr(repair_mod, "fill_holes", lambda mesh: mesh)  # doesn't help
    monkeypatch.setattr(repair_mod, "drop_nonmanifold_faces", lambda mesh: mesh)  # doesn't help
    monkeypatch.setattr(repair_mod, "poisson_rebuild", lambda mesh: None)  # doesn't help

    out = make_printable(good, max_faces=10)

    assert out.is_watertight


# --- prep ---------------------------------------------------------------


def test_scale_to_size_sets_longest_axis():
    out = scale_to_size(trimesh.creation.box(extents=[1, 2, 4]), 100.0)
    assert out.extents.max() == pytest.approx(100.0)
    assert out.extents.min() == pytest.approx(25.0)


def test_seat_on_bed_places_model_at_origin():
    mesh = trimesh.creation.box()
    mesh.apply_translation([10, -5, 30])
    out = seat_on_bed(mesh)
    assert out.bounds[0][2] == pytest.approx(0.0)
    assert out.centroid[0] == pytest.approx(0.0, abs=1e-6)


def test_auto_orient_returns_valid_mesh():
    tall = trimesh.creation.box(extents=[1, 1, 8])
    out = auto_orient(tall, PrintSettings())
    assert out.is_watertight
    assert sorted(np.round(out.extents, 5)) == sorted(np.round(tall.extents, 5))


def test_auto_orient_does_not_stand_a_character_on_its_head():
    """Regression test for a real bug: a raw TripoSR character output scores
    worse upright (-534, real overhang from arms/head/clothing) than flipped
    180 degrees (-288, less overhang upside down), so a plain overhang-vs-base
    optimizer picks the wrong one. Fixture is a decimated copy (500 faces) of
    the actual generator output that reproduced this; decimation preserves
    the score ordering that matters here (checked against the un-decimated
    83k-face original before shrinking it for the test)."""
    fixture = Path(__file__).parent / "fixtures" / "character_raw_500f.stl"
    mesh = trimesh.load(fixture)

    out = auto_orient(mesh, PrintSettings())

    # The upright (identity) orientation should win outright -- confirmed
    # above to score best once the top/bottom-swapping candidates are out of
    # the running -- so the mesh should come back untouched.
    assert np.allclose(out.vertices, mesh.vertices, atol=1e-6)


def test_apply_rotation_flips_top_to_bottom():
    """180 degrees about X should swap which end of an asymmetric mesh is up."""
    cone = trimesh.creation.cone(radius=1.0, height=4.0)  # apex at max z
    apex_idx = int(np.argmax(cone.vertices[:, 2]))
    assert cone.vertices[apex_idx, 2] == pytest.approx(cone.bounds[1][2])

    out = apply_rotation(cone, (180.0, 0.0, 0.0))

    # same vertex (rotation doesn't reorder), now at the mesh's minimum z
    assert out.vertices[apex_idx, 2] == pytest.approx(out.bounds[0][2], abs=1e-5)


def test_manual_rotation_applied_before_seat_and_base():
    """Auto-orient has no concept of e.g. 'feet down' for a character mesh --
    manual_rotation_deg is the escape hatch. It must run before seat_on_bed
    and add_base, or the base ends up fused to the pre-rotation bottom
    instead of the corrected one, which defeats the whole point."""
    cone = trimesh.creation.cone(radius=3.0, height=10.0)
    settings = PrintSettings(
        target_size_mm=20.0, auto_orient=False, add_base=False,
        manual_rotation_deg=(180.0, 0.0, 0.0),
    )
    result = prepare(cone.copy(), settings)

    expected = seat_on_bed(apply_rotation(scale_to_size(cone.copy(), 20.0), (180.0, 0.0, 0.0)))
    assert np.allclose(result.vertices, expected.vertices)


# --- validation ---------------------------------------------------------


def test_flat_bottom_is_not_reported_as_overhang():
    """The underside rests on the bed; counting it as overhang is a false alarm."""
    mesh = seat_on_bed(trimesh.creation.box(extents=[50, 50, 5]))
    report = validate(mesh, PrintSettings())
    assert report.stats["overhang_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert report.printable


def test_open_mesh_fails_validation():
    mesh = trimesh.creation.box(extents=[20, 20, 20])
    mesh.update_faces(np.arange(len(mesh.faces)) != 0)
    report = validate(seat_on_bed(mesh), PrintSettings())
    assert not report.printable
    assert any(i.code == "not_watertight" for i in report.errors)


def test_oversized_model_is_rejected():
    mesh = seat_on_bed(trimesh.creation.box(extents=[400, 400, 400]))
    report = validate(mesh, PrintSettings(build_volume_mm=(256, 256, 256)))
    assert any(i.code == "exceeds_build_volume" for i in report.errors)


def test_report_stats_are_populated():
    mesh = seat_on_bed(trimesh.creation.box(extents=[30, 30, 30]))
    stats = validate(mesh, PrintSettings()).stats
    assert stats["watertight"] is True
    assert stats["volume_cm3"] == pytest.approx(27.0, rel=0.01)
    assert stats["bodies"] == 1


# --- end to end ---------------------------------------------------------


def test_run_produces_printable_stl(sample_image, tmp_path):
    out = tmp_path / "out.stl"
    result = run(
        sample_image,
        backend="lithophane",
        settings=PrintSettings(target_size_mm=60.0, add_base=False, auto_orient=False),
        options={"max_dim": 100},
        output_path=out,
    )

    assert out.exists() and out.stat().st_size > 0
    assert result.report.printable, [str(i) for i in result.report.errors]
    assert result.mesh.is_watertight
    assert result.mesh.extents.max() == pytest.approx(60.0, rel=0.02)

    reloaded = trimesh.load(out, force="mesh")
    assert reloaded.is_watertight, "exported STL must survive a round trip"


def test_run_rejects_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "nope.png", backend="lithophane")


def test_run_exports_glb_when_backend_has_color(sample_image, tmp_path, monkeypatch):
    """A backend reporting has_color=True gets a colored preview GLB
    exported alongside the STL, captured right after generate() -- before
    decimate/add_base/hollow, which are confirmed (repair.py/prep.py) to
    silently drop mesh.visual. No real GPU backend needed: monkeypatch the
    registry's "triposr" entry (Backend(str) requires a real enum member,
    so the fake has to reuse one) with a fake that returns a colored box.
    """

    class _ColoredBackend(GeometryBackend):
        name = "triposr"
        requires_gpu = False

        def generate(self, request: GenerationRequest) -> GenerationResult:
            mesh = trimesh.creation.box()
            mesh.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(mesh.vertices), 1))
            return GenerationResult(mesh=mesh, backend=Backend.TRIPOSR, has_color=True)

    monkeypatch.setitem(registry._factories, "triposr", _ColoredBackend)

    out = tmp_path / "colored.stl"
    result = run(sample_image, backend="triposr", output_path=out)

    assert result.glb_path == out.with_suffix(".glb")
    assert result.glb_path.exists()

    reloaded = trimesh.load(result.glb_path)
    if isinstance(reloaded, trimesh.Scene):
        reloaded = trimesh.util.concatenate(list(reloaded.geometry.values()))
    assert isinstance(reloaded.visual, trimesh.visual.color.ColorVisuals)
    assert reloaded.visual.vertex_colors[0][0] > 150, "red channel should survive the round trip"


def test_run_skips_glb_export_without_color(sample_image, tmp_path):
    out = tmp_path / "plain.stl"
    result = run(sample_image, backend="lithophane", options={"max_dim": 40}, output_path=out)
    assert result.glb_path is None
    assert not out.with_suffix(".glb").exists()


# --- sheets and background keying ---------------------------------------


def _sheet(tmp_path: Path, bg=(120, 130, 140)) -> Path:
    """A 2x2 sheet with a distinct blob per panel on a gradient backdrop."""
    n = 200
    a = np.zeros((n, n, 3), dtype=np.uint8)
    a[:, :] = bg
    # Vertical lighting gradient, the thing a flat colour key fails on.
    a = np.clip(a + np.linspace(-40, 40, n)[:, None, None], 0, 255).astype("uint8")
    for r in range(2):
        for c in range(2):
            cy, cx = r * n // 2 + n // 4, c * n // 2 + n // 4
            y, x = np.mgrid[0:n, 0:n]
            a[np.hypot(y - cy, x - cx) < 28] = (220, 60, 60)
    path = tmp_path / "sheet.png"
    Image.fromarray(a).save(path)
    return path


def test_split_sheet_returns_panels_in_reading_order(tmp_path):
    from printable.backends.sheet import split_sheet

    panels = split_sheet(_sheet(tmp_path), 2, 2)
    assert len(panels) == 4
    # Each panel is roughly a quarter of the sheet.
    for p in panels:
        assert 80 < p.size[0] < 120 and 80 < p.size[1] < 120


def test_split_sheet_without_detection_uses_even_cuts(tmp_path):
    from printable.backends.sheet import split_sheet

    panels = split_sheet(_sheet(tmp_path), 2, 2, trim=0, detect_seams=False)
    assert [p.size for p in panels] == [(100, 100)] * 4


def test_flood_background_handles_lighting_gradient(tmp_path):
    """A single flat-colour key fails on a gradient backdrop; the plane fit must not."""
    from printable.backends.preprocess import flood_background

    panels_src = _sheet(tmp_path)
    keyed = flood_background(Image.open(panels_src))
    alpha = np.asarray(keyed)[:, :, 3]

    # Corners are background at both ends of the gradient.
    assert alpha[2, 2] == 0
    assert alpha[-3, -3] == 0
    # The blobs survive.
    assert alpha.max() == 255
    assert 0.02 < (alpha > 0).mean() < 0.5


def test_masked_heightmap_drops_background_to_base(tmp_path):
    """Keyed-out background must sit flat on the plate, not rise with brightness."""
    from printable.backends.heightmap import HeightmapBackend

    path = _sheet(tmp_path)
    base_mm = 2.0
    mesh = HeightmapBackend().generate(
        GenerationRequest(
            image_path=path,
            backend=Backend.HEIGHTMAP,
            options={
                "max_dim": 100, "relief_mm": 6, "base_mm": base_mm,
                "blur": 0, "key_background": True,
            },
        )
    ).mesh

    z = mesh.vertices[:, 2]
    # Something reaches well above the plate, and the plate itself is flat.
    assert z.max() > base_mm + 1.0
    assert np.isclose(z[z > 1e-6].min(), base_mm, atol=0.15)
    assert mesh.is_watertight


def test_add_base_produces_single_body():
    """The base must union into the model.

    Without a boolean backend trimesh returns the parts un-unioned, which
    exports as disconnected bodies and fails validation. Guarding it here
    catches a missing manifold3d in a base install.
    """
    from printable.pipeline.prep import add_base, seat_on_bed

    mesh = seat_on_bed(trimesh.creation.icosphere(radius=20.0))
    out = add_base(mesh, thickness_mm=2.0)

    assert len(out.split(only_watertight=False)) == 1, "base did not union"
    assert out.is_watertight


def test_quickstart_pipeline_is_printable(sample_image, tmp_path):
    """The README quick start must actually produce a printable STL."""
    out = tmp_path / "quickstart.stl"
    result = run(
        sample_image,
        backend="lithophane",
        settings=PrintSettings(target_size_mm=120.0),
        options={"max_dim": 120},
        output_path=out,
    )

    assert result.report.printable, [str(i) for i in result.report.errors]
    assert result.report.stats["bodies"] == 1
    assert result.mesh.is_watertight


# --- spec-sheet mode ------------------------------------------------------


def test_crop_views_converts_normalized_coords():
    """box_2d is [ymin, xmin, ymax, xmax] on a 0-1000 scale over the image."""
    img = Image.new("RGB", (1000, 500), "white")
    views = {"front": {"box_2d": [0, 0, 1000, 500]}}  # full height, half width

    crops = crop_views(img, views)

    assert set(crops) == {"front"}
    assert crops["front"].size == (500, 500)


def test_crop_views_handles_multiple_named_views():
    img = Image.new("RGB", (2000, 1000), "white")
    views = {
        "front": {"box_2d": [0, 0, 500, 500]},
        "back": {"box_2d": [500, 0, 1000, 500]},
    }

    crops = crop_views(img, views)

    assert crops["front"].size == (1000, 500)
    assert crops["back"].size == (1000, 500)


def test_check_dimension_targets_flags_large_mismatch():
    report = ValidationReport()
    mesh = trimesh.creation.box(extents=[10, 10, 10])

    check_dimension_targets(
        report, mesh, {"total_height": 100, "width": 100, "depth": 100}, tolerance_pct=15
    )

    assert any(i.code == "dimension_mismatch" for i in report.issues)


def test_check_dimension_targets_passes_within_tolerance():
    report = ValidationReport()
    mesh = trimesh.creation.box(extents=[10, 10, 10])

    check_dimension_targets(
        report, mesh, {"total_height": 10.5, "width": 10.2, "depth": 9.8}, tolerance_pct=15
    )

    assert not any(i.code == "dimension_mismatch" for i in report.issues)


def test_check_dimension_targets_is_axis_agnostic():
    """Sorted comparison: a rotated/reoriented mesh with the same proportions
    should not be flagged just because axes don't line up with target names."""
    report = ValidationReport()
    mesh = trimesh.creation.box(extents=[26, 30, 60])  # depth/width/height, reordered

    check_dimension_targets(
        report, mesh, {"total_height": 60, "width": 30, "depth": 26}, tolerance_pct=15
    )

    assert not any(i.code == "dimension_mismatch" for i in report.issues)


def test_check_dimension_targets_noop_on_empty_targets():
    report = ValidationReport()
    mesh = trimesh.creation.box(extents=[10, 10, 10])

    check_dimension_targets(report, mesh, {}, tolerance_pct=15)

    assert report.issues == []
