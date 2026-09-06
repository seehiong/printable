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


def _extract_spec_views(job: Job, sheet_path: Path, *, vlm_url: str) -> None:
    """First half of the spec-sheet flow: call the VLM, crop every view it
    named, write each crop to job.dir as "<name>.png". Stops there --
    the caller sets job.status = "awaiting_confirmation" once this
    returns, so a human can look at the actual crops (GET
    /api/jobs/{id}/view/{name}) before the second half runs the
    expensive, VLM-crop-quality-dependent generation step.

    Split out from what used to be one function (_run_spec_sheet) after a
    real, repeated failure mode: the VLM's per-view bounding boxes are
    unreliable enough on some sheets (see ~/vlm-server's own size- and
    clipping-plausibility checks, added after real backdrop/stand and
    cut-in-half meshes) that running straight through to a 10-20 minute
    generation on a bad crop wastes real time on a doomed result. See
    _generate_from_saved_spec below for the second half, and
    POST /api/jobs/{id}/confirm for what triggers it.
    """
    from PIL import Image

    from printable.backends.sheet import crop_views
    from printable.vlm_client import fetch_design_spec

    job.on_stage("extract", "start")
    try:
        spec = fetch_design_spec(sheet_path, vlm_url=vlm_url)
    except Exception:
        job.on_stage("extract", "error")
        raise
    job.design_spec = spec
    job.on_stage("extract", "done")

    views = spec.get("views", {})
    if not views:
        raise RuntimeError("VLM extraction returned no views")

    sheet_image = Image.open(sheet_path).convert("RGB")
    crops = crop_views(sheet_image, views)
    for name, crop in crops.items():
        crop.save(job.dir / f"{name}.png")


def _generate_from_saved_spec(
    job: Job,
    backend: str,
    *,
    size: float | None,
    min_wall: float | None,
    tolerance: float,
    seed: int,
    hollow: bool,
    add_base: bool,
    max_faces: int,
    skip_repair: bool,
) -> Any:
    """Second half of the spec-sheet flow: generate from the view crops
    _extract_spec_views already wrote to job.dir (job.design_spec already
    set too) -- no VLM call here. job.output_path must already be set by
    the caller.
    """
    from printable.pipeline.run import export_winner, run
    from printable.pipeline.select import pick_best
    from printable.pipeline.validate import check_dimension_targets

    spec = job.design_spec or {}
    dims = spec.get("dimensions_mm", {})
    views = spec.get("views", {})

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

    candidates = []
    for name in views:
        view_path = job.dir / f"{name}.png"
        if not view_path.exists():
            continue
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
    export_winner(winner, job.output_path)
    return winner


def _run_sheet(
    job: Job,
    sheet_path: Path,
    backend: str,
    *,
    settings: PrintSettings,
    options: dict[str, Any],
    rows: int,
    cols: int,
    trim: int,
    seed: int,
    max_faces: int,
    skip_repair: bool,
) -> Any:
    """Split a multi-view sheet into rows*cols panels, generate from every
    panel, keep whichever is actually most printable -- the CLI's --sheet
    flag's own approach (`cli/main.py`'s `_cmd_generate_sheet`). Shared by
    /api/jobs/sheet and /api/jobs/auto's "sheet" route (job.output_path
    must already be set by the caller).
    """
    from printable.backends.sheet import split_sheet
    from printable.pipeline.run import export_winner, run
    from printable.pipeline.select import pick_best

    panels = split_sheet(sheet_path, rows, cols, trim=trim)
    candidates = []
    for i, panel in enumerate(panels):
        panel_path = job.dir / f"panel_{i}.png"
        panel.save(panel_path)
        try:
            result = run(
                panel_path,
                backend=backend,
                settings=settings,
                options=options,
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
    export_winner(winner, job.output_path)
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

    @app.post("/api/jobs/sheet", status_code=202)
    async def submit_sheet_job(
        image: UploadFile,
        backend: str = Form(...),
        size: float = Form(100.0),
        rows: int = Form(2),
        cols: int = Form(2),
        trim: int = Form(4),
        hollow: bool = Form(False),
        add_base: bool = Form(True),
        seed: int = Form(42),
        max_faces: int = Form(300_000),
        skip_repair: bool = Form(False),
        opt: list[str] = Form(default=[]),  # noqa: B008 - FastAPI's own Form() idiom
    ) -> JobSubmitResponse:
        """Explicit grid-split mode: split a multi-view sheet into
        rows*cols panels, generate from every one, keep whichever comes
        out most printable -- the CLI's --sheet flag, as a job. Decoupled
        from /api/jobs/auto's own "sheet" route on purpose: that one only
        reaches this same split_sheet()+pick_best() logic if an external
        VLM classifies the upload as "character_turnaround" -- if it
        instead guesses "design_spec_sheet" (an easy mix-up: both are
        multi-view sheets), auto-detect silently takes a completely
        different route (_run_spec_sheet) with an unrelated, VLM-decided
        panel count. This endpoint exists for exactly that case: the
        caller already knows it's a turnaround sheet and wants the
        deterministic grid split without gambling on a VLM guess.
        """
        if backend not in registry.names():
            known = ", ".join(registry.names())
            raise ValueError(f"unknown backend {backend!r}; known: {known}")

        options = parse_options(opt)
        settings = PrintSettings(target_size_mm=size, hollow=hollow, add_base=add_base)

        job = store.create(backend)
        suffix = Path(image.filename or "upload").suffix or ".png"
        sheet_path = job.dir / f"input{suffix}"
        sheet_path.write_bytes(await image.read())
        job.image_path = sheet_path
        job.output_path = job.dir / "result.stl"

        def _generate_sheet() -> Any:
            return _run_sheet(
                job, sheet_path, backend,
                settings=settings, options=options, rows=rows, cols=cols, trim=trim,
                seed=seed, max_faces=max_faces, skip_repair=skip_repair,
            )

        store.submit(job, _generate_sheet)
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
        job.pending_kwargs = {
            "size": size, "min_wall": min_wall, "tolerance": tolerance, "seed": seed,
            "hollow": hollow, "add_base": add_base, "max_faces": max_faces,
            "skip_repair": skip_repair,
        }

        def _extract() -> Any:
            _extract_spec_views(job, sheet_path, vlm_url=vlm_url)
            job.status = "awaiting_confirmation"
            return None

        store.submit(job, _extract)
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
            from printable.pipeline.select import route_for_category
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
                _extract_spec_views(job, image_path, vlm_url=vlm_url)
                job.pending_kwargs = {
                    "size": size, "min_wall": min_wall, "tolerance": tolerance,
                    "seed": seed, "hollow": hollow, "add_base": add_base,
                    "max_faces": max_faces, "skip_repair": skip_repair,
                }
                job.status = "awaiting_confirmation"
                return None

            settings = PrintSettings(
                target_size_mm=size if size is not None else 100.0,
                min_wall_thickness_mm=min_wall if min_wall is not None else 0.8,
                hollow=hollow,
                add_base=add_base,
            )

            if decision.mode == "sheet":
                # Auto-detect doesn't know the backend up front, so it never
                # sends --opt-style options (see /api/jobs/sheet below for
                # the explicit mode, which does).
                return _run_sheet(
                    job, image_path, decision.backend,
                    settings=settings, options={}, rows=2, cols=2, trim=4,
                    seed=seed, max_faces=max_faces, skip_repair=skip_repair,
                )

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

    @app.post("/api/jobs/{job_id}/confirm", status_code=202)
    def confirm_job(job_id: str) -> JobSubmitResponse:
        """Resume a spec-sheet job past its extraction pause -- see
        Job.status's docstring and _extract_spec_views/
        _generate_from_saved_spec above. Takes no body: every setting
        needed for generation was already collected at submit time and
        is sitting on job.pending_kwargs; this only exists to mark "a
        human looked at the crops and wants to proceed", not to let the
        caller change settings mid-flight (a real future improvement,
        not attempted here -- see ROADMAP.md).
        """
        job = store.get(job_id)
        if job.status != "awaiting_confirmation":
            raise ValueError(f"job {job_id!r} is {job.status}, not awaiting confirmation")
        kwargs = job.pending_kwargs or {}

        def _continue() -> Any:
            return _generate_from_saved_spec(job, job.backend, **kwargs)

        store.submit(job, _continue)
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
        has_geometry_cues = False
        if job.status == "done" and job.result is not None:
            report = job.result.report
            printable = report.printable
            stats = report.stats
            issues = [str(i) for i in report.issues]
            has_glb = bool(job.result.glb_path)
            has_geometry_cues = bool(job.result.geometry_cue_paths)
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
            has_geometry_cues=has_geometry_cues,
        )

    @app.get("/api/jobs/{job_id}/view/{name}")
    def job_view(job_id: str, name: str):
        """Serves one of _extract_spec_views' saved crops for review before
        confirming. Whitelisted against job.design_spec's own view names
        (not just any filename job.dir happens to contain) since `name`
        is caller-supplied -- cheap, real protection against path
        shenanigans on a field nothing else validates.
        """
        job = store.get(job_id)
        known_views = (job.design_spec or {}).get("views", {})
        if name not in known_views:
            raise HTTPException(status_code=404, detail=f"no view named {name!r} for this job")
        view_path = job.dir / f"{name}.png"
        if not view_path.exists():
            raise HTTPException(status_code=404, detail=f"view {name!r} crop not found")
        return FileResponse(view_path, media_type="image/png")

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

    def _geometry_cue_path(job_id: str, suffix: str) -> Path:
        job = store.get(job_id)
        if job.status != "done":
            raise HTTPException(status_code=409, detail=f"job is {job.status}, not done")
        paths = job.result.geometry_cue_paths if job.result is not None else []
        match = next((p for p in paths if p.name.endswith(suffix)), None)
        if match is None:
            raise HTTPException(
                status_code=404, detail="this job did not produce that geometry cue"
            )
        return match

    @app.get("/api/jobs/{job_id}/depth")
    def job_depth(job_id: str):
        return FileResponse(_geometry_cue_path(job_id, "_depth.png"), media_type="image/png")

    @app.get("/api/jobs/{job_id}/normal")
    def job_normal(job_id: str):
        return FileResponse(_geometry_cue_path(job_id, "_normal.png"), media_type="image/png")

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app
