"""Mesh repair. Turns untrusted generator output into a watertight solid.

Every AI backend produces meshes that look fine in a viewer and are not
printable: non-manifold edges, holes, floating islands, degenerate faces.
This stage is where most of the real reliability lives.
"""

from __future__ import annotations

import logging

import numpy as np
import trimesh

log = logging.getLogger(__name__)


def basic_clean(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cheap fixes that never lose real detail. Always safe to run."""
    mesh = mesh.copy()
    before = len(mesh.faces)

    mesh.remove_infinite_values()
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    # Consistent winding so normals point outward; slicers rely on this to
    # tell inside from outside.
    mesh.fix_normals()

    log.info("basic_clean: %d -> %d faces", before, len(mesh.faces))
    return mesh


def keep_largest_component(mesh: trimesh.Trimesh, min_ratio: float = 0.05) -> trimesh.Trimesh:
    """Drop floating islands.

    Generators routinely emit small disconnected blobs near the subject. They
    slice as unattached debris. Any component whose bounding-box diagonal is
    under min_ratio of the largest component's is removed.
    """
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh

    # Score by bounding-box diagonal: a linear measure that is defined for
    # open meshes too, and that keeps min_ratio intuitive. Volume would scale
    # cubically, so a part half as wide would score at an eighth and a
    # sensible-looking threshold would delete far too much.
    def size(m: trimesh.Trimesh) -> float:
        return float(np.linalg.norm(m.extents))

    parts = sorted(parts, key=size, reverse=True)
    biggest = size(parts[0])
    if biggest <= 0:
        return mesh
    kept = [p for p in parts if size(p) >= biggest * min_ratio]

    if len(kept) < len(parts):
        log.info("dropped %d island(s) of %d", len(parts) - len(kept), len(parts))
    return trimesh.util.concatenate(kept) if len(kept) > 1 else parts[0]


def fill_holes(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Try trimesh's hole filling. Works for small, simple boundaries."""
    if mesh.is_watertight:
        return mesh
    mesh = mesh.copy()
    mesh.fill_holes()
    mesh.fix_normals()
    log.info("fill_holes: watertight=%s", mesh.is_watertight)
    return mesh


def poisson_rebuild(mesh: trimesh.Trimesh, depth: int = 9, samples: int = 200_000):
    """Screened Poisson reconstruction. The sledgehammer.

    Resamples the surface as a point cloud and rebuilds it as a closed
    manifold. Guarantees watertightness by construction, at the cost of
    softening sharp edges. Used only when cheaper repairs fail.

    Requires open3d; returns None if unavailable so the caller can fall back.
    """
    try:
        import open3d as o3d
    except ImportError:
        log.warning("open3d not installed; skipping Poisson rebuild")
        return None

    points, face_idx = trimesh.sample.sample_surface(mesh, samples)
    normals = mesh.face_normals[face_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    rebuilt, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )
    # Poisson inflates a closed hull well beyond the samples; trim the
    # low-confidence regions or you get a bubble around the model.
    densities = np.asarray(densities)
    rebuilt.remove_vertices_by_mask(densities < np.quantile(densities, 0.02))

    out = trimesh.Trimesh(
        vertices=np.asarray(rebuilt.vertices),
        faces=np.asarray(rebuilt.triangles),
        process=True,
    )
    out.fix_normals()
    log.info("poisson_rebuild: %d faces, watertight=%s", len(out.faces), out.is_watertight)
    return out


def edge_defects(mesh: trimesh.Trimesh) -> tuple[int, int]:
    """Count (open, non-manifold) edges.

    An edge used by one face is a boundary; by three or more, non-manifold.
    Distinguishing them matters because they need different repairs.
    """
    edges = np.sort(mesh.edges_sorted, axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts > 2).sum())


def drop_nonmanifold_faces(mesh: trimesh.Trimesh, max_passes: int = 5) -> trimesh.Trimesh:
    """Remove faces on non-manifold edges and re-close, repeating as needed.

    One pass is not enough: deleting a face can leave a neighbouring edge
    non-manifold, and fill_holes can reintroduce the defect while patching.
    Iterate until the mesh is clean or no further progress is made.
    """
    out = mesh.copy()
    for attempt in range(max_passes):
        edges = np.sort(out.edges_sorted, axis=1)
        _, inverse, counts = np.unique(
            edges, axis=0, return_inverse=True, return_counts=True
        )
        bad_edge = counts > 2
        if not bad_edge.any():
            if attempt:
                log.info("non-manifold edges cleared after %d pass(es)", attempt)
            return out

        # edges_sorted runs 3 per face, in face order.
        face_of_edge = np.repeat(np.arange(len(out.faces)), 3)
        bad_faces = np.unique(face_of_edge[bad_edge[inverse.ravel()]])
        log.info(
            "pass %d: dropping %d face(s) on %d non-manifold edge(s)",
            attempt + 1, len(bad_faces), int(bad_edge.sum()),
        )

        keep = np.ones(len(out.faces), dtype=bool)
        keep[bad_faces] = False
        if not keep.any():
            break
        out.update_faces(keep)
        out.remove_unreferenced_vertices()
        out.fill_holes()

    out.fix_normals()
    return out


def decimate(mesh: trimesh.Trimesh, max_faces: int) -> trimesh.Trimesh:
    """Quadric decimation down to max_faces. No-op if already small enough."""
    if len(mesh.faces) <= max_faces:
        return mesh
    before = len(mesh.faces)
    try:
        out = mesh.simplify_quadric_decimation(face_count=max_faces)
        out.fix_normals()
        log.info("decimate: %d -> %d faces", before, len(out.faces))
        return out
    except Exception as exc:  # noqa: BLE001 - optional backend
        # trimesh delegates to fast_simplification; without it we would
        # silently hand a multi-million-face mesh to the slicer.
        log.warning(
            "decimation unavailable (%s); keeping %d faces. "
            "Install fast-simplification to enable --max-faces.", exc, before
        )
        return mesh


def _recover_from_decimation_damage(mesh: trimesh.Trimesh, aggressive: bool) -> trimesh.Trimesh:
    """Best-effort repair chain for a mesh decimation has just damaged.

    Doesn't raise or guarantee success -- caller checks `.is_watertight` on
    the result. Split out so `make_printable` can run it at more than one
    face-count budget without duplicating the escalation logic.
    """
    open_edges, nonmanifold = edge_defects(mesh)
    log.info(
        "decimation damaged the mesh: %d open, %d non-manifold edge(s)",
        open_edges, nonmanifold,
    )
    if open_edges:
        mesh = fill_holes(mesh)
    if not mesh.is_watertight:
        mesh = drop_nonmanifold_faces(mesh)
    if not mesh.is_watertight and aggressive:
        # Next resort: rebuild the surface outright. Costs sharpness but
        # guarantees a closed solid -- when it works.
        log.info("escalating to Poisson after decimation damage")
        rebuilt = poisson_rebuild(mesh)
        mesh = keep_largest_component(rebuilt) if rebuilt is not None else mesh
    return mesh


def make_printable(
    mesh: trimesh.Trimesh,
    *,
    max_faces: int = 300_000,
    aggressive: bool = True,
) -> trimesh.Trimesh:
    """Full repair ladder, cheapest fix first.

    Escalates only as far as needed: basic clean -> island removal -> hole
    fill -> Poisson rebuild. Poisson costs detail, so it is a last resort.
    """
    if len(mesh.faces) == 0:
        # A generative backend can legitimately come back with nothing --
        # confirmed with SPAR3D on an image with no recognizable object.
        # Left unguarded, this reaches poisson_rebuild's sample_surface()
        # and dies on an obscure `IndexError: index -1 is out of bounds for
        # axis 0 with size 0` instead of saying what actually happened.
        raise ValueError(
            "backend produced an empty mesh (0 faces) -- nothing to repair. "
            "Usually means the source image didn't read as a reconstructible "
            "object to the model, not a bug in the repair ladder."
        )

    # Backends that build closed geometry directly (the heightmap family)
    # arrive already watertight. Re-deriving adjacency on a half-million
    # faces to confirm that costs far more than the check.
    already_clean = mesh.is_watertight and mesh.is_winding_consistent
    if already_clean:
        log.info("mesh already watertight and consistently wound; light clean only")
        mesh = mesh.copy()
        mesh.remove_infinite_values()
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    else:
        mesh = basic_clean(mesh)

    # Always run island removal. A model plus a detached blob is watertight
    # as a whole, so gating this on watertightness would ship the debris.
    mesh = keep_largest_component(mesh)

    if not mesh.is_watertight:
        mesh = fill_holes(mesh)

    if not mesh.is_watertight and aggressive:
        log.info("still not watertight; escalating to Poisson")
        rebuilt = poisson_rebuild(mesh)
        if rebuilt is not None:
            mesh = keep_largest_component(rebuilt)

    was_watertight = mesh.is_watertight
    # Kept only when decimation is about to run on an already-good mesh, so
    # there is a fallback if decimation breaks it and repair can't recover
    # it -- see below. A bounded retry at a gentler ratio is tried before
    # falling all the way back to full resolution: jumping straight to the
    # raw mesh can itself be expensive enough to matter on a memory-tight
    # box (a real OOM was traced to exactly that on a ~600k-face generation
    # where the budget was 300k -- the fallback mesh nearly doubled peak
    # memory during validate/export for comparatively little gain here,
    # since 600k was already close to the raw count).
    pre_decimation_mesh = mesh if was_watertight else None
    mesh = decimate(mesh, max_faces)

    # Quadric decimation can damage a previously closed mesh in two distinct
    # ways, and they need different repairs: genuine holes (open edges) call
    # for hole filling, while non-manifold edges -- shared by three or more
    # faces -- do not, and fill_holes silently does nothing for them. See
    # _recover_from_decimation_damage for that chain.
    if was_watertight and not mesh.is_watertight:
        mesh = _recover_from_decimation_damage(mesh, aggressive)

        if not mesh.is_watertight:
            raw_count = len(pre_decimation_mesh.faces)
            retry_target = (max_faces + raw_count) // 2
            if retry_target < raw_count:
                log.info(
                    "decimation to %d faces couldn't be repaired; retrying "
                    "at a gentler %d faces before falling back to the full "
                    "%d-face mesh",
                    max_faces, retry_target, raw_count,
                )
                retry = decimate(pre_decimation_mesh, retry_target)
                retry = _recover_from_decimation_damage(retry, aggressive)
                if retry.is_watertight:
                    mesh = retry

        if not mesh.is_watertight:
            # Nothing recovered it at any face count, but the mesh WAS
            # watertight before decimation touched it -- decimation is
            # provably the cause, not the model's output. Shipping a broken
            # mesh purely to respect a face-count budget is a worse trade
            # than a bigger, correct one: this project treats watertightness
            # as a hard gate (see validate.py) and face count as a soft
            # target, so keep the original.
            log.warning(
                "decimation-damaged mesh could not be repaired at any face "
                "count; keeping the %d-face pre-decimation mesh instead of "
                "the requested max_faces=%d budget (still watertight, just "
                "bigger)",
                len(pre_decimation_mesh.faces), max_faces,
            )
            mesh = pre_decimation_mesh

    # Winding can flip during repair; a negative volume means inside-out.
    if mesh.is_watertight and mesh.volume < 0:
        mesh.invert()

    return mesh
