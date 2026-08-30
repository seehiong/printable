"""FastAPI app: upload a photo, run the pipeline, stream progress, get an STL."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from printable.api.jobs import Job, store
from printable.api.schemas import BackendInfo, JobStatus, JobSubmitResponse
from printable.backends.base import registry
from printable.options import parse_options
from printable.types import PrintSettings
from printable.vlm_client import DEFAULT_VLM_URL

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _run_spec_sheet(
    job: Job,
    sheet_path: Path,
    backend: str,
    *,
    vlm_url: str,
    size: float | None,
    min_wall: float | None,
    tolerance: float,
    seed: int,
    hollow: bool,
    add_base: bool,
    max_faces: int,
    skip_repair: bool,
) -> Any:
    """Extract a design_spec.json from a sheet, try every crop it named,
    keep whichever is actually most printable. Shared by /api/jobs/spec and
    /api/jobs/auto's design_spec_sheet route -- job.output_path must already
    be set by the caller.
    """
    from PIL import Image

    from printable.backends.sheet import crop_views
    from printable.pipeline.run import run
    from printable.pipeline.select import pick_best
    from printable.pipeline.validate import check_dimension_targets
    from printable.vlm_client import fetch_design_spec

    job.on_stage("extract", "start")
    try:
        spec = fetch_design_spec(sheet_path, vlm_url=vlm_url)
    except Exception:
        job.on_stage("extract", "error")
        raise
    job.design_spec = spec
    job.on_stage("extract", "done")

    dims = spec.get("dimensions_mm", {})
    views = spec.get("views", {})
    if not views:
        raise RuntimeError("VLM extraction returned no views")

    target_size = size if size is not None else dims.get("total_height")
    if target_size is None:
        raise RuntimeError("no 'size' given and spec has no 'total_height'")
    target_min_wall = (
        min_wall
        if min_wall is not None
        else spec.get("print_constraints", {}).get("min_wall_thickness_mm", 0.8)
    )
    settings = PrintSettings(
        target_size_mm=target_size,
        min_wall_thickness_mm=target_min_wall,
        hollow=hollow,
        add_base=add_base,
    )

    sheet_image = Image.open(sheet_path).convert("RGB")
    crops = crop_views(sheet_image, views)
    candidates = []
    for name, crop in crops.items():
        view_path = job.dir / f"{name}.png"
        crop.save(view_path)
        try:
            result = run(
                view_path,
                backend=backend,
                settings=settings,
                output_path=None,
                seed=seed,
                skip_repair=skip_repair,
                max_faces=max_faces,
                on_stage=lambda n, s, _view=name: job.on_stage(f"{_view}: {n}", s),
            )
        except Exception as exc:  # noqa: BLE001 - one bad view shouldn't sink the rest
            log.warning("view %r failed: %s", name, exc)
            continue
        candidates.append((name, result))

    if not candidates:
        raise RuntimeError("no view produced a usable mesh")

    _winner_name, winner = pick_best(candidates)
    targets = {k: dims[k] for k in ("total_height", "width", "depth") if k in dims}
    check_dimension_targets(winner.report, winner.mesh, targets, tolerance_pct=tolerance)
    winner.mesh.export(job.output_path)
    winner.output_path = job.output_path
    return winner


def create_app() -> FastAPI:
    app = FastAPI(title="printable")

    @app.exception_handler(KeyError)
    def _not_found(request, exc: KeyError):
        return JSONResponse({"error": f"job {exc.args[0]!r} not found"}, status_code=404)

    @app.exception_handler(ValueError)
    def _bad_request(request, exc: ValueError):
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.get("/api/backends")
    def list_backends() -> list[BackendInfo]:
        out = []
        for name in registry.names():
            impl = registry.get(name)
            ok, reason = impl.available()
            out.append(
                BackendInfo(
                    name=name, requires_gpu=impl.requires_gpu, available=ok, reason=reason
                )
            )
        return out

    @app.post("/api/jobs", status_code=202)
    async def submit_job(
        image: UploadFile,
        backend: str = Form(...),
        seed: int = Form(42),
        size: float = Form(100.0),
        hollow: bool = Form(False),
        add_base: bool = Form(True),
        max_faces: int = Form(300_000),
        skip_repair: bool = Form(False),
        opt: list[str] = Form(default=[]),  # noqa: B008 - FastAPI's own Form() idiom
    ) -> JobSubmitResponse:
        if backend not in registry.names():
            known = ", ".join(registry.names())
            raise ValueError(f"unknown backend {backend!r}; known: {known}")

        options = parse_options(opt)
        settings = PrintSettings(target_size_mm=size, hollow=hollow, add_base=add_base)

        job = store.create(backend)
        suffix = Path(image.filename or "upload").suffix or ".png"
        image_path = job.dir / f"input{suffix}"
        image_path.write_bytes(await image.read())
        job.image_path = image_path
        job.output_path = job.dir / "result.stl"

        def _generate() -> Any:
            from printable.pipeline.run import run

            return run(
                image_path,
                backend=backend,
                settings=settings,
                options=options,
                output_path=job.output_path,
                seed=seed,
                skip_repair=skip_repair,
                max_faces=max_faces,
                on_stage=job.on_stage,
            )

        store.submit(job, _generate)
        return JobSubmitResponse(job_id=job.id)

    @app.post("/api/jobs/spec", status_code=202)
    async def submit_spec_job(
        image: UploadFile,
        backend: str = Form(...),
        vlm_url: str = Form(DEFAULT_VLM_URL),
        size: float | None = Form(None),
        min_wall: float | None = Form(None),
        tolerance: float = Form(15.0),
        seed: int = Form(42),
        hollow: bool = Form(False),
        add_base: bool = Form(True),
        max_faces: int = Form(300_000),
        skip_repair: bool = Form(False),
    ) -> JobSubmitResponse:
        if backend not in registry.names():
            known = ", ".join(registry.names())
            raise ValueError(f"unknown backend {backend!r}; known: {known}")

        job = store.create(backend)
        suffix = Path(image.filename or "upload.jpg").suffix or ".jpg"
        sheet_path = job.dir / f"input{suffix}"
        sheet_path.write_bytes(await image.read())
        job.image_path = sheet_path
        job.output_path = job.dir / "result.stl"

        def _generate_spec() -> Any:
            return _run_spec_sheet(
                job, sheet_path, backend,
                vlm_url=vlm_url, size=size, min_wall=min_wall, tolerance=tolerance,
                seed=seed, hollow=hollow, add_base=add_base,
                max_faces=max_faces, skip_repair=skip_repair,
            )

        store.submit(job, _generate_spec)
        return JobSubmitResponse(job_id=job.id)

    @app.post("/api/jobs/auto", status_code=202)
    async def submit_auto_job(
        image: UploadFile,
        vlm_url: str = Form(DEFAULT_VLM_URL),
        size: float | None = Form(None),
        min_wall: float | None = Form(None),
        tolerance: float = Form(15.0),
        seed: int = Form(42),
        hollow: bool = Form(False),
        add_base: bool = Form(True),
        max_faces: int = Form(300_000),
        skip_repair: bool = Form(False),
    ) -> JobSubmitResponse:
        """Classify the upload, then route it to whichever existing pipeline
        (spec extraction, sheet split, or a single backend) fits -- see
        pipeline/select.py:route_for_category for the mapping. `backend` is
        deliberately not a request parameter here: that's the whole point.
        """
        job = store.create("auto")
        suffix = Path(image.filename or "upload").suffix or ".png"
        image_path = job.dir / f"input{suffix}"
        image_path.write_bytes(await image.read())
        job.image_path = image_path

        def _generate_auto() -> Any:
            from printable.pipeline.run import run
            from printable.pipeline.select import pick_best, route_for_category
            from printable.vlm_client import classify_image

            job.on_stage("classify", "start")
            try:
                classification = classify_image(image_path, vlm_url=vlm_url)
            except Exception:
                job.on_stage("classify", "error")
                raise
            job.on_stage("classify", "done")

            category = classification.get("category", "unclear")
            available = {n for n in registry.names() if registry.get(n).available()[0]}
            decision = route_for_category(category, available)
            job.classification = {
                **classification,
                "backend": decision.backend,
                "mode": decision.mode,
                "message": decision.message,
            }
            job.backend = decision.backend or "auto"
            job.on_stage("route", "done")

            if decision.mode == "manual":
                return None  # no mesh -- UI falls back to manual backend selection

            job.output_path = job.dir / "result.stl"

            if decision.mode == "spec":
                return _run_spec_sheet(
                    job, image_path, decision.backend,
                    vlm_url=vlm_url, size=size, min_wall=min_wall, tolerance=tolerance,
                    seed=seed, hollow=hollow, add_base=add_base,
                    max_faces=max_faces, skip_repair=skip_repair,
                )

            settings = PrintSettings(
                target_size_mm=size if size is not None else 100.0,
                min_wall_thickness_mm=min_wall if min_wall is not None else 0.8,
                hollow=hollow,
                add_base=add_base,
            )

            if decision.mode == "sheet":
                from printable.backends.sheet import split_sheet

                panels = split_sheet(image_path, 2, 2, trim=4)
                candidates = []
                for i, panel in enumerate(panels):
                    panel_path = job.dir / f"panel_{i}.png"
                    panel.save(panel_path)
                    try:
                        result = run(
                            panel_path,
                            backend=decision.backend,
                            settings=settings,
                            output_path=None,
                            seed=seed,
                            skip_repair=skip_repair,
                            max_faces=max_faces,
                            on_stage=lambda n, s, _i=i: job.on_stage(f"panel{_i}: {n}", s),
                        )
                    except Exception as exc:  # noqa: BLE001 - one bad panel shouldn't sink the rest
                        log.warning("panel %d failed: %s", i, exc)
                        continue
                    candidates.append((str(i), result))

                if not candidates:
                    raise RuntimeError("no panel produced a usable mesh")

                _winner_i, winner = pick_best(candidates)
                winner.mesh.export(job.output_path)
                winner.output_path = job.output_path
                return winner

            # decision.mode == "single"
            return run(
                image_path,
                backend=decision.backend,
                settings=settings,
                output_path=job.output_path,
                seed=seed,
                skip_repair=skip_repair,
                max_faces=max_faces,
                on_stage=job.on_stage,
            )

        store.submit(job, _generate_auto)
        return JobSubmitResponse(job_id=job.id)

    @app.get("/api/jobs/{job_id}/events")
    def job_events(job_id: str):
        job = store.get(job_id)
        history, q = job.subscribe()

        def gen():
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"
            if q is None:
                return
            while True:
                item = q.get()
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> JobStatus:
        job = store.get(job_id)
        printable = None
        stats = None
        issues = None
        has_glb = False
        if job.status == "done" and job.result is not None:
            report = job.result.report
            printable = report.printable
            stats = report.stats
            issues = [str(i) for i in report.issues]
            has_glb = bool(job.result.glb_path)
        return JobStatus(
            id=job.id,
            backend=job.backend,
            status=job.status,
            events=job.events,
            error=job.error,
            printable=printable,
            stats=stats,
            issues=issues,
            design_spec=job.design_spec,
            classification=job.classification,
            has_glb=has_glb,
        )

    @app.get("/api/jobs/{job_id}/stl")
    def job_stl(job_id: str):
        job = store.get(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job is {job.status}, not done")
        if job.output_path is None:
            raise HTTPException(status_code=409, detail="job produced no mesh (see classification)")
        return FileResponse(job.output_path, media_type="model/stl", filename="result.stl")

    @app.get("/api/jobs/{job_id}/glb")
    def job_glb(job_id: str):
        job = store.get(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job is {job.status}, not done")
        glb_path = job.result.glb_path if job.result is not None else None
        if glb_path is None:
            raise HTTPException(status_code=404, detail="this job did not produce a GLB")
        return FileResponse(glb_path, media_type="model/gltf-binary", filename="result.glb")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
