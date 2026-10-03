"""Bake a high-poly colored mesh onto a game-budget textured low-poly.

`--game-asset`'s output path. Per-vertex color can't survive a reduction
to a game budget: at 3,500 triangles there are ~1,750 vertices left to
carry all the paint, so detail smears into blobs. Instead this keeps the
paint in an image: reduce the geometry, UV-unwrap the result (xatlas),
then for every texel find the matching point on the untouched high-poly
source and copy its color into a base-color texture and its surface
normal into a tangent-space normal map. The low-poly then looks close to
the high-poly at a fraction of the triangles, and can be re-baked at any
other budget later from the same source without regenerating.
"""

from __future__ import annotations

import io
import json
import logging
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from printable.pipeline import repair

log = logging.getLogger(__name__)

# Texel padding between UV islands, in pixels. Bilinear filtering and the
# lower mip levels read across island borders; with too little room they
# pick up a neighbouring island's color as a visible seam.
_ATLAS_PADDING = 4
# Candidate source triangles checked per texel. More is slower but less
# likely to miss the true closest triangle next to a very large one.
_CANDIDATES = 8
_CHUNK = 65_536


@dataclass
class GameAsset:
    # Reduced geometry before UV seams split it: what `watertight` and the
    # face count describe.
    mesh: trimesh.Trimesh
    # The UV-split arrays that are actually exported.
    vertices: np.ndarray  # (n, 3) float32
    faces: np.ndarray  # (m, 3) uint32
    normals: np.ndarray  # (n, 3) float32
    tangents: np.ndarray  # (n, 4) float32, glTF convention
    uvs: np.ndarray  # (n, 2) float32, glTF convention (v down)
    base_color: Image.Image
    normal_map: Image.Image

    @property
    def watertight(self) -> bool:
        return bool(self.mesh.is_watertight)


def make_game_asset(
    source: trimesh.Trimesh,
    *,
    max_faces: int,
    texture_size: int = 1024,
    axis_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> GameAsset:
    """Reduce `source` (vertex-colored or UV-textured) to `max_faces` and
    bake its color and surface detail onto the result. `source` itself is
    not modified."""
    source = _scaled(source, axis_scale)
    low = _reduce(source, max_faces)
    return _bake(source, low, texture_size)


def _scaled(mesh: trimesh.Trimesh, axis_scale: tuple[float, float, float]) -> trimesh.Trimesh:
    """Copy with each axis scaled about the centroid. Neither backend's
    guessed depth is reliable, and nothing downstream rescales a game
    asset, so this is the only correction available. Applied to the bake
    source too, so the low-poly and the surface it's baked from agree."""
    mesh = mesh.copy()
    if tuple(axis_scale) != (1.0, 1.0, 1.0):
        center = mesh.vertices.mean(axis=0)
        mesh.vertices = (mesh.vertices - center) * np.asarray(axis_scale) + center
    return mesh


def _reduce(source: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Geometry-only reduction to at most `max_faces`, holes closed.

    Uses the cheap part of the print path's repair ladder: island removal
    and hole filling before and after decimation, since a quadric pass can
    reopen a closed mesh. Poisson rebuild is left out on purpose; it
    resamples the whole surface and loses the sharp edges a normal map is
    meant to keep.
    """
    clean = trimesh.Trimesh(vertices=source.vertices.copy(), faces=source.faces.copy(), process=False)
    clean = repair.basic_clean(clean)
    clean = repair.keep_largest_component(clean)
    clean = repair.fill_holes(clean)
    low = _decimate_closed(clean, max_faces)
    if len(low.faces) > max_faces:
        # Filling the holes decimation opened added faces back; aim lower
        # by the overshoot so the budget stays a real ceiling.
        low = _decimate_closed(clean, max(4, 2 * max_faces - len(low.faces)))
    return low


def _decimate_closed(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    low = repair.decimate(mesh, max_faces)
    if not low.is_watertight:
        low = repair.fill_holes(low)
    if not low.is_watertight:
        low = repair.drop_nonmanifold_faces(low)
    return repair.keep_largest_component(low)


def _bake(source: trimesh.Trimesh, low: trimesh.Trimesh, size: int) -> GameAsset:
    try:
        import xatlas
    except ImportError as exc:
        raise RuntimeError(
            "texture baking needs xatlas; run: uv pip install xatlas "
            "(or uv sync --extra bake --inexact)"
        ) from exc
    from scipy import ndimage
    from scipy.spatial import cKDTree

    atlas = xatlas.Atlas()
    atlas.add_mesh(low.vertices.astype(np.float32), low.faces.astype(np.uint32))
    pack = xatlas.PackOptions()
    pack.resolution = size
    pack.padding = _ATLAS_PADDING
    pack.bilinear = True
    atlas.generate(xatlas.ChartOptions(), pack)
    vmapping, faces, uvs = atlas[0]

    vertices = low.vertices[vmapping]
    normals = low.vertex_normals[vmapping]
    tangents = _tangents(vertices, faces, uvs, normals)

    texel_xy, face_ids, bary = _rasterize(uvs * size, faces, size)
    if len(face_ids) == 0:
        raise RuntimeError("UV rasterization covered no texels; is the reduced mesh empty?")

    tri = faces[face_ids]
    points = np.einsum("ij,ijk->ik", bary, vertices[tri])
    n_low = _normalize(np.einsum("ij,ijk->ik", bary, normals[tri]))
    t_low = np.einsum("ij,ijk->ik", bary, tangents[tri][:, :, :3])
    t_low = _normalize(t_low - n_low * (t_low * n_low).sum(axis=1, keepdims=True))
    b_low = np.cross(n_low, t_low) * tangents[tri[:, 0], 3:4]

    src_faces, sample_color = _source_sampler(source)
    src_tris = source.vertices[src_faces]
    src_face_normals = _normalize(np.cross(src_tris[:, 1] - src_tris[:, 0], src_tris[:, 2] - src_tris[:, 0]))
    src_vertex_normals = source.vertex_normals
    tree = cKDTree(src_tris.mean(axis=1))
    k = min(_CANDIDATES, len(src_faces))
    # A texel whose nearest source surface faces the other way (the back
    # of a thin disk, the inside of a lip) would pick up the wrong side's
    # color; penalise those candidates by more than any real distance.
    penalty = float(np.linalg.norm(source.extents)) * 10.0

    colors = np.empty((len(points), 3), dtype=np.float32)
    src_normals = np.empty((len(points), 3), dtype=np.float64)
    for start in range(0, len(points), _CHUNK):
        sl = slice(start, start + _CHUNK)
        p, n = points[sl], n_low[sl]
        _, cand = tree.query(p, k=k, workers=-1)
        cand = cand.reshape(len(p), k)
        cp = trimesh.triangles.closest_point(
            src_tris[cand].reshape(-1, 3, 3), np.repeat(p, k, axis=0)
        ).reshape(len(p), k, 3)
        dist = np.linalg.norm(cp - p[:, None, :], axis=2)
        facing = np.einsum("ijk,ik->ij", src_face_normals[cand], n) > 0.0
        best = np.argmin(dist + np.where(facing, 0.0, penalty), axis=1)
        rows = np.arange(len(p))
        best_face = cand[rows, best]
        b = trimesh.triangles.points_to_barycentric(src_tris[best_face], cp[rows, best])
        b = np.clip(np.nan_to_num(b, nan=1.0 / 3.0), 0.0, None)
        b /= np.maximum(b.sum(axis=1, keepdims=True), 1e-12)
        vids = src_faces[best_face]
        colors[sl] = sample_color(vids, b)
        src_normals[sl] = _normalize(np.einsum("ij,ijk->ik", b, src_vertex_normals[vids]))

    ts = np.stack(
        [
            (src_normals * t_low).sum(axis=1),
            (src_normals * b_low).sum(axis=1),
            np.maximum((src_normals * n_low).sum(axis=1), 0.05),
        ],
        axis=1,
    )
    ts = _normalize(ts)

    mask = np.zeros((size, size), dtype=bool)
    base = np.zeros((size, size, 3), dtype=np.float32)
    nmap = np.zeros((size, size, 3), dtype=np.float32)
    ys, xs = texel_xy[:, 1], texel_xy[:, 0]
    mask[ys, xs] = True
    base[ys, xs] = colors
    nmap[ys, xs] = (ts * 0.5 + 0.5) * 255.0

    # Fill every empty texel with its nearest baked neighbour, so island
    # edges and mip levels never sample the black background.
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    base = base[iy, ix]
    nmap = nmap[iy, ix]

    log.info(
        "bake: %d faces, %d texels (%.0f%% of %dpx), from %d source faces",
        len(low.faces), int(mask.sum()), 100.0 * mask.mean(), size, len(src_faces),
    )
    return GameAsset(
        mesh=low,
        vertices=vertices.astype(np.float32),
        faces=faces.astype(np.uint32),
        normals=normals.astype(np.float32),
        tangents=tangents.astype(np.float32),
        uvs=uvs.astype(np.float32),
        base_color=Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB"),
        normal_map=Image.fromarray(np.clip(nmap, 0, 255).astype(np.uint8), "RGB"),
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def _tangents(vertices, faces, uvs, normals) -> np.ndarray:
    """Per-vertex glTF tangents (xyz + handedness w) from UV derivatives.

    Exported alongside the normal map rather than left for the engine to
    derive, so the basis the map was baked in is exactly the one it's
    decoded with. `v` is flipped to point up the image, the direction a
    glTF normal map's +Y (green) means.
    """
    p0, p1, p2 = (vertices[faces[:, i]] for i in range(3))
    w0, w1, w2 = (uvs[faces[:, i]] * np.array([1.0, -1.0]) for i in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-20, 1e-20, r)[:, None]
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r

    tan1 = np.zeros_like(vertices)
    tan2 = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(tan1, faces[:, i], sdir)
        np.add.at(tan2, faces[:, i], tdir)

    t = tan1 - normals * (normals * tan1).sum(axis=1, keepdims=True)
    degenerate = np.linalg.norm(t, axis=1) < 1e-12
    if degenerate.any():
        # No usable UV gradient here; any perpendicular will do.
        helper = np.where(np.abs(normals[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        t[degenerate] = np.cross(normals[degenerate], helper[degenerate])
    t = _normalize(t)
    w = np.where((np.cross(normals, t) * tan2).sum(axis=1) < 0.0, -1.0, 1.0)
    return np.hstack([t, w[:, None]])


def _rasterize(uv_px: np.ndarray, faces: np.ndarray, size: int):
    """Texels whose centres fall inside each UV triangle, with their
    barycentric weights. Returns (texel_xy (n, 2) int, face_ids, bary)."""
    xy_out, face_out, bary_out = [], [], []
    for fi, (a, b, c) in enumerate(faces):
        pa, pb, pc = uv_px[a], uv_px[b], uv_px[c]
        x0 = max(int(np.floor(min(pa[0], pb[0], pc[0]) - 0.5)), 0)
        x1 = min(int(np.ceil(max(pa[0], pb[0], pc[0]) - 0.5)), size - 1)
        y0 = max(int(np.floor(min(pa[1], pb[1], pc[1]) - 0.5)), 0)
        y1 = min(int(np.ceil(max(pa[1], pb[1], pc[1]) - 0.5)), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        px = gx.ravel() + 0.5
        py = gy.ravel() + 0.5
        den = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
        if abs(den) < 1e-12:
            continue
        l0 = ((pb[1] - pc[1]) * (px - pc[0]) + (pc[0] - pb[0]) * (py - pc[1])) / den
        l1 = ((pc[1] - pa[1]) * (px - pc[0]) + (pa[0] - pc[0]) * (py - pc[1])) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        xy_out.append(np.stack([gx.ravel()[inside], gy.ravel()[inside]], axis=1))
        face_out.append(np.full(int(inside.sum()), fi, dtype=np.int64))
        bary_out.append(np.stack([l0[inside], l1[inside], l2[inside]], axis=1))
    if not face_out:
        return np.zeros((0, 2), int), np.zeros(0, int), np.zeros((0, 3))
    return np.concatenate(xy_out), np.concatenate(face_out), np.concatenate(bary_out)


def _source_sampler(
    source: trimesh.Trimesh,
) -> tuple[np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]:
    """(non-degenerate source faces, fn(face vertex ids, barycentrics) -> RGB).

    Handles both kinds of colored backend output: a UV-mapped texture
    (Hunyuan3D's hy3dpaint) is sampled bilinearly at the interpolated UV;
    per-vertex color (TripoSR) is interpolated directly.
    """
    faces = np.asarray(source.faces)
    faces = faces[source.area_faces > 1e-14]
    kind = source.visual.kind

    image = None
    uv = getattr(source.visual, "uv", None) if kind == "texture" else None
    if uv is not None:
        material = source.visual.material
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)

    if uv is not None and image is not None:
        uv = np.asarray(uv, dtype=np.float64)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        h, w = pixels.shape[:2]

        def sample(vids, b):
            st = np.einsum("ij,ijk->ik", b, uv[vids])
            x = np.mod(st[:, 0], 1.0) * w - 0.5
            y = np.mod(1.0 - st[:, 1], 1.0) * h - 0.5
            return _bilinear(pixels, x, y)

        return faces, sample

    if kind in ("vertex", "face"):
        vc = np.asarray(source.visual.vertex_colors[:, :3], dtype=np.float32)
        return faces, lambda vids, b: np.einsum("ij,ijk->ik", b, vc[vids])

    raise ValueError("source mesh carries no color or texture to bake")


def _bilinear(pixels: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = pixels.shape[:2]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    x0c, x1c = np.clip(x0, 0, w - 1), np.clip(x0 + 1, 0, w - 1)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    top = pixels[y0c, x0c] * (1 - fx) + pixels[y0c, x1c] * fx
    bottom = pixels[y1c, x0c] * (1 - fx) + pixels[y1c, x1c] * fx
    return top * (1 - fy) + bottom * fy


def export_glb(
    asset: GameAsset,
    path: Path,
    *,
    name: str | None = None,
    roughness: float = 0.8,
) -> None:
    """Write `asset` as a self-contained GLB.

    Written by hand rather than through trimesh so it carries TANGENT
    (trimesh's exporter doesn't) and an explicit metallicFactor of 0 --
    glTF's default is fully metallic, which an engine renders as dark
    chrome for a ceramic urn or a jade disk. The material is double-sided:
    on a closed mesh back faces never show, so it changes nothing there,
    but a generated mesh can keep a hole too large for hole filling, and
    single-sided it reads as a see-through gap rather than a dark cavity.
    """
    path = Path(path)
    blob = bytearray()
    views: list[dict] = []
    accessors: list[dict] = []

    def add_view(data: bytes, target: int | None = None) -> int:
        while len(blob) % 4:
            blob.append(0)
        view = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        blob.extend(data)
        views.append(view)
        return len(views) - 1

    def add_accessor(array: np.ndarray, component: int, kind: str, target: int, bounds=False) -> int:
        acc = {
            "bufferView": add_view(array.tobytes(), target),
            "componentType": component,
            "count": int(array.shape[0]),
            "type": kind,
        }
        if bounds:
            acc["min"] = array.min(axis=0).tolist()
            acc["max"] = array.max(axis=0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    array_buffer, element_buffer, float32, uint32 = 34962, 34963, 5126, 5125
    attributes = {
        "POSITION": add_accessor(asset.vertices, float32, "VEC3", array_buffer, bounds=True),
        "NORMAL": add_accessor(asset.normals, float32, "VEC3", array_buffer),
        "TANGENT": add_accessor(asset.tangents, float32, "VEC4", array_buffer),
        "TEXCOORD_0": add_accessor(asset.uvs, float32, "VEC2", array_buffer),
    }
    indices = add_accessor(asset.faces.reshape(-1), uint32, "SCALAR", element_buffer)

    def encode(image: Image.Image, fmt: str) -> bytes:
        buf = io.BytesIO()
        # Base color as JPEG keeps the file small; the normal map stays
        # lossless, since block artifacts in it read as surface dents.
        image.save(buf, format=fmt, **({"quality": 92} if fmt == "JPEG" else {}))
        return buf.getvalue()

    images = [
        {"bufferView": add_view(encode(asset.base_color, "JPEG")), "mimeType": "image/jpeg"},
        {"bufferView": add_view(encode(asset.normal_map, "PNG")), "mimeType": "image/png"},
    ]

    gltf = {
        "asset": {"version": "2.0", "generator": "printable"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name or path.stem}],
        "meshes": [
            {
                "name": name or path.stem,
                "primitives": [
                    {"attributes": attributes, "indices": indices, "material": 0, "mode": 4}
                ],
            }
        ],
        "materials": [
            {
                "name": name or path.stem,
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": roughness,
                },
                "normalTexture": {"index": 1},
                "doubleSided": True,
            }
        ],
        "samplers": [{"magFilter": 9729, "minFilter": 9987}],
        "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
        "images": images,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
    }

    while len(blob) % 4:
        blob.append(0)
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(blob)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(blob), 0x004E4942))
        f.write(bytes(blob))
