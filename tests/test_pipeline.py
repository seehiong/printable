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

    # Not trimesh.load(): its own GLTF reader drops COLOR_0 back to flat
    # white once a material is present (a real trimesh round-trip gap, not
    # a bug in the file -- see ensure_vertex_color_material's docstring),
    # so read the raw glTF accessors directly instead, same as a spec-
    # compliant external renderer (Godot, Blender, three.js) would.
    gltf, bin_chunk = _read_glb(result.glb_path)
    color0 = _glb_attribute(gltf, bin_chunk, "COLOR_0")
    assert color0 is not None, "expected a COLOR_0 attribute on the exported primitive"
    assert (color0[:, 0] > 150).all(), "red channel should survive the round trip"
    assert gltf["materials"], (
        "a vertex-colored GLB with no material at all loads as flat white in "
        "Godot's glTF importer -- see ensure_vertex_color_material"
    )
    # glTF's default metallicFactor is 1.0; left out, engines render dark chrome.
    assert gltf["materials"][0]["pbrMetallicRoughness"]["metallicFactor"] == 0.0


def _read_glb(path: Path) -> tuple[dict, bytes]:
    """(glTF JSON, BIN chunk), read directly rather than through trimesh:
    its GLTF reader drops COLOR_0 once a material is present, and ignores
    TANGENT entirely -- neither of which a real engine does."""
    import json
    import struct

    data = path.read_bytes()
    length = struct.unpack("<I", data[8:12])[0]
    offset = 12
    chunks = {}
    while offset < length:
        chunk_length, chunk_type = struct.unpack("<II", data[offset : offset + 8])
        chunks[chunk_type] = data[offset + 8 : offset + 8 + chunk_length]
        offset += 8 + chunk_length
    return json.loads(chunks[0x4E4F534A]), chunks.get(0x004E4942)


_GLB_DTYPES = {5121: np.uint8, 5125: np.uint32, 5126: np.float32}
_GLB_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _glb_attribute(gltf: dict, bin_chunk: bytes, name: str) -> np.ndarray | None:
    prim = gltf["meshes"][0]["primitives"][0]
    index = prim["indices"] if name == "indices" else prim["attributes"].get(name)
    if index is None:
        return None
    acc = gltf["accessors"][index]
    view = gltf["bufferViews"][acc["bufferView"]]
    width = _GLB_WIDTH[acc["type"]]
    raw = np.frombuffer(
        bin_chunk,
        dtype=_GLB_DTYPES[acc["componentType"]],
        count=acc["count"] * width,
        offset=view.get("byteOffset", 0) + acc.get("byteOffset", 0),
    )
    return raw.reshape(-1, width) if width > 1 else raw


def _glb_image(gltf: dict, bin_chunk: bytes, texture_index: int) -> np.ndarray:
    import io

    image = gltf["images"][gltf["textures"][texture_index]["source"]]
    view = gltf["bufferViews"][image["bufferView"]]
    start = view.get("byteOffset", 0)
    data = bin_chunk[start : start + view["byteLength"]]
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def _colored_backend(make_mesh):
    """A registry stand-in for "triposr" that returns make_mesh()'s mesh
    with has_color=True (Backend(str) needs a real enum member)."""

    class _Backend(GeometryBackend):
        name = "triposr"
        requires_gpu = False

        def generate(self, request: GenerationRequest) -> GenerationResult:
            return GenerationResult(mesh=make_mesh(), backend=Backend.TRIPOSR, has_color=True)

    return _Backend


def _half_red_half_blue_sphere() -> trimesh.Trimesh:
    mesh = trimesh.creation.icosphere(subdivisions=5)
    mesh.visual.vertex_colors = np.where(
        mesh.vertices[:, :1] > 0, [220, 30, 30, 255], [30, 30, 220, 255]
    ).astype(np.uint8)
    return mesh


def test_game_asset_bakes_color_into_a_textured_low_poly(sample_image, tmp_path, monkeypatch):
    """--game-asset reduces to the budget and bakes color into a texture
    rather than per-vertex color, which smears to blobs at a game budget.
    The GLB must carry what an engine needs to shade it: UVs, normals and
    tangents, a base-color texture, a normal map, and a non-metallic
    material."""
    pytest.importorskip("xatlas")
    monkeypatch.setitem(registry._factories, "triposr", _colored_backend(_half_red_half_blue_sphere))

    out = tmp_path / "sphere.glb"
    result = run(
        sample_image, backend="triposr", game_asset=True, game_asset_max_faces=600,
        game_asset_texture_size=256, output_path=out,
    )

    assert result.report.stats["faces"] <= 600
    assert result.report.stats["raw_faces"] == 20480
    assert len(result.raw_color_mesh.faces) == 20480, "raw mesh must be kept untouched"

    gltf, bin_chunk = _read_glb(out)
    for attr in ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0"):
        assert _glb_attribute(gltf, bin_chunk, attr) is not None, attr
    material = gltf["materials"][0]
    assert material["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    assert "normalTexture" in material
    assert material["doubleSided"] is True  # a leftover hole shows as a cavity, not a gap

    positions = _glb_attribute(gltf, bin_chunk, "POSITION")
    uvs = _glb_attribute(gltf, bin_chunk, "TEXCOORD_0")
    base = _glb_image(gltf, bin_chunk, material["pbrMetallicRoughness"]["baseColorTexture"]["index"])
    px = base[
        np.clip((uvs[:, 1] * 256).astype(int), 0, 255),
        np.clip((uvs[:, 0] * 256).astype(int), 0, 255),
    ].astype(int)
    red_side, blue_side = positions[:, 0] > 0.2, positions[:, 0] < -0.2
    assert (px[red_side, 0] > px[red_side, 2]).all()
    assert (px[blue_side, 2] > px[blue_side, 0]).all()


def test_bake_samples_a_uv_textured_source_the_right_way_up():
    """Hunyuan3D's hy3dpaint output is UV-textured, not vertex-colored.
    A texture split left/right (red/blue) and top/bottom (green/none) must
    land on the matching sides of the baked model -- catches a flipped u
    or v in the source sampler."""
    pytest.importorskip("xatlas")
    from printable.pipeline.bake import make_game_asset

    mesh = trimesh.creation.icosphere(subdivisions=5)
    x, y, z = mesh.vertices.T
    uv = np.stack([np.arctan2(y, x) / (2 * np.pi) + 0.5, 0.5 + 0.5 * z], axis=1)
    texture = np.zeros((64, 64, 3), dtype=np.uint8)
    texture[:, :32, 0] = 220  # left half of the image (u < 0.5, i.e. y < 0): red
    texture[:, 32:, 2] = 220  # right half (y > 0): blue
    texture[:32, :, 1] = 200  # top rows are high v in trimesh's convention (z > 0): green
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=Image.fromarray(texture))

    asset = make_game_asset(mesh, max_faces=800, texture_size=256)
    px = np.asarray(asset.base_color)[
        np.clip((asset.uvs[:, 1] * 256).astype(int), 0, 255),
        np.clip((asset.uvs[:, 0] * 256).astype(int), 0, 255),
    ].astype(int)
    _, vy, vz = asset.vertices.T
    # Keep clear of the seam at u=0/1 and the quadrant borders.
    left = (vy < -0.3) & (np.abs(vz) > 0.3)
    right = (vy > 0.3) & (np.abs(vz) > 0.3)
    assert (px[left, 0] > px[left, 2]).all()
    assert (px[right, 2] > px[right, 0]).all()
    top, bottom = (vz > 0.3) & (np.abs(vy) > 0.3), (vz < -0.3) & (np.abs(vy) > 0.3)
    assert (px[top, 1] > 150).all()
    assert (px[bottom, 1] < 50).all()


def test_baked_normal_map_reconstructs_the_high_poly_surface(tmp_path):
    """Decoding the normal map with the exported tangents -- exactly as the
    glTF spec tells an engine to (bitangent = cross(normal, tangent) * w)
    -- should give back the bumpy high-poly's normals on a smooth low-poly,
    far closer than the low-poly's own normals do. Catches any mismatch
    between the basis the map is baked in and the one it's decoded with."""
    pytest.importorskip("xatlas")
    from printable.pipeline.bake import export_glb, make_game_asset

    source = trimesh.creation.icosphere(subdivisions=6)
    theta = np.arctan2(source.vertices[:, 1], source.vertices[:, 0])
    source.vertices *= (1.0 + 0.04 * np.sin(8 * theta))[:, None]
    source.visual.vertex_colors = np.tile([180, 180, 180, 255], (len(source.vertices), 1))

    asset = make_game_asset(source, max_faces=1500, texture_size=512)
    out = tmp_path / "bumpy.glb"
    export_glb(asset, out)
    gltf, bin_chunk = _read_glb(out)
    faces = _glb_attribute(gltf, bin_chunk, "indices").reshape(-1, 3)
    pos = _glb_attribute(gltf, bin_chunk, "POSITION")
    nrm = _glb_attribute(gltf, bin_chunk, "NORMAL")
    tan = _glb_attribute(gltf, bin_chunk, "TANGENT")
    uv = _glb_attribute(gltf, bin_chunk, "TEXCOORD_0")
    nmap = _glb_image(gltf, bin_chunk, gltf["materials"][0]["normalTexture"]["index"])

    def unit(v):
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    # Sample each exported triangle at its centroid.
    p = pos[faces].mean(axis=1)
    n = unit(nrm[faces].mean(axis=1))
    t = tan[faces][:, :, :3].mean(axis=1)
    t = unit(t - n * (t * n).sum(axis=1, keepdims=True))
    b = np.cross(n, t) * tan[faces[:, 0], 3:4]
    st = uv[faces].mean(axis=1)
    texel = nmap[
        np.clip((st[:, 1] * 512).astype(int), 0, 511), np.clip((st[:, 0] * 512).astype(int), 0, 511)
    ]
    ts = texel / 255.0 * 2.0 - 1.0
    decoded = unit(ts[:, :1] * t + ts[:, 1:2] * b + ts[:, 2:3] * n)

    _, _, tri = trimesh.proximity.closest_point(source, p)
    truth = source.face_normals[tri]

    def mean_angle(a):
        return np.degrees(np.arccos(np.clip((a * truth).sum(axis=1), -1, 1))).mean()

    assert mean_angle(decoded) < 0.6 * mean_angle(n), (mean_angle(decoded), mean_angle(n))
    assert mean_angle(decoded) < 8.0


def test_game_asset_scale_z_squashes_only_that_axis(sample_image, monkeypatch):
    """--opt scale_z=... (also scale_x/scale_y) is the only correction for a
    single-image backend's guessed depth axis, since game_asset runs no
    scale-to-mm/prep at all otherwise (see run.py's game_asset branch). It
    should scale around the mesh's own centroid, changing only the given
    axis's extent -- not translate the mesh, not touch the others.
    """
    pytest.importorskip("xatlas")

    class _ColoredBackend(GeometryBackend):
        name = "triposr"
        requires_gpu = False

        def generate(self, request: GenerationRequest) -> GenerationResult:
            mesh = trimesh.creation.box(extents=[2, 3, 4])
            mesh.apply_translation([5, -5, 5])  # off-origin, to catch a pivot bug
            mesh.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(mesh.vertices), 1))
            return GenerationResult(mesh=mesh, backend=Backend.TRIPOSR, has_color=True)

    monkeypatch.setitem(registry._factories, "triposr", _ColoredBackend)

    baseline = run(sample_image, backend="triposr", game_asset=True)
    squashed = run(sample_image, backend="triposr", game_asset=True, options={"scale_z": 0.5})

    assert np.allclose(squashed.mesh.extents[:2], baseline.mesh.extents[:2])
    assert squashed.mesh.extents[2] == pytest.approx(baseline.mesh.extents[2] * 0.5)
    assert np.allclose(squashed.mesh.centroid, baseline.mesh.centroid, atol=1e-6)


def test_game_asset_closes_holes(sample_image, tmp_path, monkeypatch):
    """game_asset skips the print path's full repair ladder, but still runs
    its cheap part (island removal + hole filling): a raw single-image
    reconstruction routinely isn't watertight, and a hole ships as a
    visible black gap in the engine otherwise."""
    pytest.importorskip("xatlas")

    def holey_box():
        box = trimesh.creation.box()
        mesh = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[1:], process=False)
        mesh.visual.vertex_colors = np.tile([200, 50, 50, 255], (len(mesh.vertices), 1))
        return mesh

    monkeypatch.setitem(registry._factories, "triposr", _colored_backend(holey_box))

    out = tmp_path / "box.glb"
    result = run(sample_image, backend="triposr", game_asset=True, output_path=out)

    assert result.mesh.is_watertight
    assert result.report.stats["watertight"] is True
    gltf, bin_chunk = _read_glb(out)
    base = _glb_image(gltf, bin_chunk, 0)
    assert (base[..., 0] > 150).all(), "the patched face should be baked red too"


def test_art_source_round_trip(sample_image, tmp_path, monkeypatch):
    """--art-source keeps what's needed to reproduce or re-bake a model:
    the reference image with its text-to-image settings still embedded, the
    untouched raw mesh (loadable back as a colored bake source), and a
    prompt.md recording prompt, seed and settings."""
    pytest.importorskip("xatlas")
    from printable.art_source import (
        embed_image_params,
        read_image_params,
        write_for_result,
    )
    from printable.pipeline.bake import make_game_asset

    params = {"prompt": "a bronze bell | chime", "negative_prompt": "floor, shadow", "seed": 1234,
              "steps": 4, "cfg": 1.0, "width": 64, "height": 64, "mirrored": True}
    image_path = tmp_path / "ref.png"
    image_path.write_bytes(embed_image_params(sample_image.read_bytes(), params))
    assert read_image_params(image_path) == params
    assert read_image_params(sample_image) is None

    monkeypatch.setitem(registry._factories, "triposr", _colored_backend(_half_red_half_blue_sphere))
    result = run(image_path, backend="triposr", game_asset=True, game_asset_max_faces=500,
                 game_asset_texture_size=128)

    art = tmp_path / "relic_bell"
    write_for_result(
        art, result, image_path=image_path, image_params=read_image_params(image_path),
        backend="triposr", seed=42, options={"scale_z": 0.5}, max_faces=500, texture_px=128,
        model_name="relic_bell.glb",
    )

    assert read_image_params(art / "reference.png") == params
    raw = trimesh.load(art / "raw.glb", force="mesh")
    assert len(raw.faces) == 20480
    make_game_asset(raw, max_faces=500, texture_size=64)  # re-bakes without error

    md = (art / "prompt.md").read_text()
    assert md.startswith("# relic_bell")
    assert "a bronze bell | chime" in md  # prompts sit in code blocks, unescaped
    assert "| Seed | 1234 |" in md
    assert "left half mirrored onto right" in md
    assert "--max-faces 500 --texture-size 128 --opt scale_z=0.5" in md


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


def test_black_paint_is_rejected_not_shipped(tmp_path, monkeypatch):
    """hy3dpaint can return an all-black texture without raising; that must
    fail the job rather than ship a black model, save its multiview images,
    and say which step lost the color. A dark but real texture passes."""
    from printable.backends.ai import _check_paint

    monkeypatch.chdir(tmp_path)

    def textured(value):
        mesh = trimesh.creation.box()
        pixels = np.full((16, 16, 3), value, dtype=np.uint8)
        mesh.visual = trimesh.visual.TextureVisuals(uv=np.zeros((8, 2)), image=Image.fromarray(pixels))
        return mesh

    black, green = Image.new("RGB", (8, 8), "black"), Image.new("RGB", (8, 8), (40, 160, 60))
    with pytest.raises(RuntimeError, match="diffusion step failed"):
        _check_paint(textured(0), views=[black, black], source=green)
    with pytest.raises(RuntimeError, match="2 of 2 multiview images had color"):
        _check_paint(textured(0), views=[green, green])
    saved = sorted(p.name for p in (tmp_path / "output" / "hy3dpaint_debug").rglob("*.png"))
    assert "source.png" in saved and "view_0.png" in saved
    _check_paint(textured(40))  # dark lacquer is fine
