"""The spine: image in, printable STL out.

Deliberately linear. Every stage takes a mesh and returns a mesh, so backends
stay interchangeable and any stage can be skipped for debugging.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import trimesh

from printable.backends.base import registry
from printable.pipeline import prep, repair, validate
from printable.types import (
    Backend,
    DepthNormalResult,
    GenerationRequest,
    PrintSettings,
    ValidationReport,
)

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    mesh: trimesh.Trimesh
    report: ValidationReport
    output_path: Path | None = None
    # Set when the backend produced real color/texture (--opt texture=true)
    # and a colored preview GLB was written alongside output_path. See the
    # "export-glb" stage below for why this is a separate, lighter-touch
    # mesh than the fully repaired/prepped one in `mesh`.
    glb_path: Path | None = None
    # The cleaned, export-ready colored mesh itself, whenever the backend
    # produced color -- set even when output_path was None (glb_path stays
    # None in that case, since nothing was written to disk here). Exists so
    # a caller that generates several candidates before picking a winner
    # (--sheet, --spec-all-views) can still export the winner's GLB after
    # the fact, without needing to re-run generation just to get color back.
    color_mesh: trimesh.Trimesh | None = None
    # Set whenever --opt geometry_cues=true actually produced an estimate,
    # regardless of output_path -- same reasoning as color_mesh above: a
    # caller that generates several candidates before picking a winner
    # (--sheet, --spec-all-views) needs this to export_winner() the cues
    # for the winner after the fact, since run() only auto-saves them
    # immediately when output_path was given at generation time.
    depth_normal: DepthNormalResult | None = None
    # Paths actually written to disk. Empty when depth_normal is set but
    # output_path was None at generation time (export_winner() below fills
    # this in once a winner is picked and a real path is known).
    geometry_cue_paths: list[Path] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class _Timer:
    """Records wall time per stage so slow steps are obvious.

    Also drives an optional on_stage(name, status) callback so callers (the
    web API's SSE stream) can report live progress without polling logs.
    status is "start", "done", or "error". A broken subscriber must never
    break generation, so callback failures are swallowed and logged.
    """

    def __init__(self, on_stage: Callable[[str, str], None] | None = None) -> None:
        self.timings: dict[str, float] = {}
        self._on_stage = on_stage

    def _notify(self, name: str, status: str) -> None:
        if self._on_stage is None:
            return
        try:
            self._on_stage(name, status)
        except Exception:
            log.exception("on_stage callback failed for %s/%s", name, status)

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t0 = time.perf_counter()
                log.info("--- %s", name)
                timer._notify(name, "start")
                return self

            def __exit__(self, *exc):
                timer.timings[name] = time.perf_counter() - self.t0
                log.info("    %s took %.2fs", name, timer.timings[name])
                timer._notify(name, "error" if exc[0] else "done")
                return False

        return _Ctx()


def run(
    image_path: Path,
    *,
    backend: str = "lithophane",
    settings: PrintSettings | None = None,
    options: dict[str, Any] | None = None,
    output_path: Path | None = None,
    seed: int = 42,
    skip_repair: bool = False,
    max_faces: int = 300_000,
    on_stage: Callable[[str, str], None] | None = None,
) -> PipelineResult:
    """Run the full image-to-STL pipeline."""
    settings = settings or PrintSettings()
    options = options or {}
    image_path = Path(image_path)
    if not image_path.exists():
        msg = f"image file not found: {image_path}"
        # A common trap when a Windows-style path is pasted into a Unix shell:
        # backslash is a literal filename character there, not a separator.
        if "\\" in str(image_path):
            alt = Path(str(image_path).replace("\\", "/"))
            if alt.exists():
                msg += f" (did you mean {alt}? use / as the path separator here)"
        raise FileNotFoundError(msg)

    timer = _Timer(on_stage)

    impl = registry.get(backend)
    ok, reason = impl.available()
    if not ok:
        raise RuntimeError(f"backend {backend!r} unavailable: {reason}")

    preprocessed_image = None
    if impl.needs_preprocess:
        with timer.stage("preprocess"):
            from printable.backends.preprocess import prepare_image

            preprocessed_image = prepare_image(
                image_path,
                size=int(options.get("image_size", impl.image_size)),
                strip_background=bool(options.get("remove_bg", True)),
                bg_model=str(options.get("bg_model", "u2net")),
            )

    depth_normal = None
    geometry_cue_paths: list[Path] = []
    if impl.wants_geometry_cues or bool(options.get("geometry_cues", False)):
        with timer.stage("geometry-cues"):
            from printable.backends import depth as depth_mod

            ok, reason = depth_mod.available()
            if not ok:
                log.warning("geometry-cues requested but unavailable: %s", reason)
            else:
                try:
                    cue_image = preprocessed_image
                    if cue_image is None:
                        from PIL import Image as PILImage

                        cue_image = PILImage.open(image_path).convert("RGB")
                    depth_normal = depth_mod.estimate_depth_normal(cue_image)
                    if output_path is not None:
                        geometry_cue_paths = depth_mod.save_geometry_cues(depth_normal, Path(output_path))
                except Exception:
                    log.exception("geometry-cues estimation failed; continuing without it")

    with timer.stage("generate"):
        result = impl.generate(
            GenerationRequest(
                image_path=image_path,
                backend=Backend(backend),
                seed=seed,
                options=options,
                image=preprocessed_image,
                depth_normal=depth_normal,
            )
        )
        mesh = result.mesh
        log.info("generated %d faces", len(mesh.faces))
        # Captured before repair/prep reassign `mesh` to new objects below.
        # decimate/poisson_rebuild/the add_base+hollow booleans all silently
        # drop mesh.visual (no source-image correspondence for cut/added
        # geometry to interpolate color from), so a colored GLB has to come
        # from here, not from whatever `mesh` becomes downstream.
        # result.color_mesh overrides this when the backend's colored
        # output is a genuinely different mesh from the plain shape output
        # (Hunyuan3D's hy3dpaint remeshes internally); None means the
        # backend attached color directly to `mesh` itself (TripoSR).
        color_mesh = result.color_mesh if result.color_mesh is not None else mesh

    if not skip_repair:
        with timer.stage("repair"):
            # Heightmap-family meshes are built closed and their resolution is
            # already controlled at source via max_dim. Decimating them just
            # trades away the surface detail that is the whole point, so the
            # face budget applies only to generative backends.
            budget = max_faces
            if backend in ("heightmap", "lithophane"):
                budget = max(max_faces, len(mesh.faces))
            mesh = repair.make_printable(mesh, max_faces=budget)

    with timer.stage("prep"):
        mesh = prep.prepare(mesh, settings)

    with timer.stage("validate"):
        report = validate.validate(mesh, settings)

    # Computed whenever the backend produced color, independent of
    # output_path -- callers that pick a winner from several candidates
    # before writing anything to disk (--sheet, --spec-all-views) need
    # this available on the returned PipelineResult even when this run()
    # call itself never writes a GLB (output_path=None here is exactly
    # what those callers pass while scoring candidates).
    color_mesh_out = None
    if result.has_color:
        # Only the non-destructive part of the repair ladder: this is a
        # viewer-only preview mesh, not the print target, so it may carry
        # more fragmentation/defects than `mesh` above -- an accepted
        # tradeoff for keeping the color that decimation, Poisson rebuild,
        # and the boolean ops in the real repair/prep path would otherwise
        # silently strip.
        color_mesh_out = repair.basic_clean(color_mesh)

    written = None
    if output_path is not None:
        with timer.stage("export"):
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(output_path)
            written = output_path
            log.info("wrote %s", output_path)

    glb_written = None
    if written is not None and color_mesh_out is not None:
        with timer.stage("export-glb"):
            glb_path = output_path.with_suffix(".glb")
            color_mesh_out.export(glb_path)
            glb_written = glb_path
            log.info("wrote %s", glb_path)

    return PipelineResult(
        mesh=mesh,
        report=report,
        output_path=written,
        glb_path=glb_written,
        color_mesh=color_mesh_out,
        depth_normal=depth_normal,
        geometry_cue_paths=geometry_cue_paths,
        timings=timer.timings,
        metadata={**result.metadata, "backend": backend},
    )


def export_winner(winner: PipelineResult, output_path: Path) -> None:
    """Write a candidate picked by pipeline.select.pick_best to disk.

    Every caller that generates several candidates before keeping one
    (--sheet, --spec-all-views, and the web API's matching modes) calls
    run() with output_path=None for each candidate -- only the eventual
    winner should land on disk -- which means run()'s own inline export
    logic above never fires for any of them. This is the export step
    those callers run once, after the fact, on whichever candidate won.

    Mutates `winner` in place (output_path, glb_path, geometry_cue_paths),
    matching exactly what run() itself sets when it does the export
    inline, so downstream code (CLI printing, the web API's has_glb/
    has_geometry_cues checks) doesn't need to know which happened. Was
    previously duplicated ad hoc at each call site, and inconsistently --
    the GLB half of it was simply missing in some (a real bug: --opt
    texture=true silently produced no GLB under --sheet/--spec-all-views,
    whichever backend was asked for it). --opt geometry_cues=true had the
    same gap for the same reason (run()'s own save only fires when
    output_path was given at generation time, which it never is for a
    per-candidate call here).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    winner.mesh.export(output_path)
    winner.output_path = output_path
    if winner.color_mesh is not None:
        glb_path = output_path.with_suffix(".glb")
        winner.color_mesh.export(glb_path)
        winner.glb_path = glb_path
    if winner.depth_normal is not None and not winner.geometry_cue_paths:
        from printable.backends import depth as depth_mod

        winner.geometry_cue_paths = depth_mod.save_geometry_cues(winner.depth_normal, output_path)
