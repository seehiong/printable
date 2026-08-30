"""Command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Importing the backend modules is what populates the registry.
from printable.backends import heightmap  # noqa: F401
from printable.backends.base import registry
from printable.options import parse_options as _parse_options
from printable.types import PrintSettings


def _load_ai_backends() -> None:
    """Import the AI backends, tolerating a missing torch stack.

    Kept separate so `printable generate --backend lithophane` works on a
    machine with no ML dependencies installed at all.
    """
    try:
        from printable.backends import ai  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        logging.getLogger(__name__).debug("AI backends unavailable: %s", exc)


def _print_generate_result(result) -> None:
    print()
    print(f"  backend   {result.metadata.get('backend')}")
    print(f"  output    {result.output_path}")
    if result.glb_path:
        print(f"  glb       {result.glb_path}")
    print()
    for key, value in result.report.stats.items():
        print(f"  {key:<20} {value}")
    print()

    for issue in result.report.issues:
        print(f"  {issue}")
    if not result.report.issues:
        print("  no issues found")
    print()

    total = sum(result.timings.values())
    stages = "  ".join(f"{k} {v:.1f}s" for k, v in result.timings.items())
    print(f"  {stages}   (total {total:.1f}s)")
    print()
    print("  PRINTABLE" if result.report.printable else "  NOT PRINTABLE - see errors above")


def _settings_from_args(args: argparse.Namespace) -> PrintSettings:
    return PrintSettings(
        target_size_mm=args.size,
        nozzle_diameter_mm=args.nozzle,
        min_wall_thickness_mm=args.min_wall,
        max_overhang_deg=args.max_overhang,
        build_volume_mm=tuple(args.build_volume),
        hollow=args.hollow,
        hollow_wall_mm=args.hollow_wall,
        add_base=not args.no_base,
        base_thickness_mm=args.base_thickness,
        auto_orient=not args.no_orient,
        manual_rotation_deg=tuple(args.rotate) if args.rotate else None,
    )


def _load_spec(spec_arg: str) -> dict:
    """Load a design_spec.json, given either the file itself or its directory."""
    spec_path = Path(spec_arg)
    if spec_path.is_dir():
        spec_path = spec_path / "design_spec.json"
    return json.loads(spec_path.read_text())


def _apply_spec_defaults(args: argparse.Namespace, spec: dict) -> None:
    """Fill --size/--min-wall from a design_spec.json unless already given."""
    dims = spec.get("dimensions_mm", {})
    constraints = spec.get("print_constraints", {})
    if args.size is None:
        args.size = dims.get("total_height")
    if args.min_wall is None:
        args.min_wall = constraints.get("min_wall_thickness_mm")


def cmd_generate(args: argparse.Namespace) -> int:
    from printable.pipeline.run import run

    spec = None
    if args.spec:
        spec = _load_spec(args.spec)
        _apply_spec_defaults(args, spec)
    if args.size is None:
        args.size = 100.0
    if args.min_wall is None:
        args.min_wall = 0.8

    if args.spec_all_views:
        return _cmd_generate_spec_views(args, spec)

    if not args.image:
        print("error: an image is required unless --spec-all-views is given", file=sys.stderr)
        return 1

    if args.sheet:
        return _cmd_generate_sheet(args)

    settings = _settings_from_args(args)
    out = args.output or Path("output") / f"{Path(args.image).stem}_{args.backend}.stl"

    result = run(
        Path(args.image),
        backend=args.backend,
        settings=settings,
        options=_parse_options(args.opt),
        output_path=Path(out),
        seed=args.seed,
        skip_repair=args.skip_repair,
        max_faces=args.max_faces,
    )

    if spec:
        from printable.pipeline.validate import check_dimension_targets

        dims = spec.get("dimensions_mm", {})
        targets = {k: dims[k] for k in ("total_height", "width", "depth") if k in dims}
        check_dimension_targets(result.report, result.mesh, targets, tolerance_pct=args.tolerance)

    _print_generate_result(result)
    return 0 if result.report.printable else 1


def _cmd_generate_sheet(args: argparse.Namespace) -> int:
    """Split a multi-view sheet, generate from every panel, keep the best.

    "Best" is picked from the actual validation results (printable, then
    fewest disconnected bodies) rather than `pick_best_panel`'s 2D framing
    heuristic — that heuristic scores subject size and centering, which
    says nothing about which *angle* a single-image backend reconstructs
    well from (a well-framed back view is not a good input, and there is
    no way to tell from the 2D crop alone).
    """
    import tempfile

    from printable.backends.sheet import split_sheet
    from printable.pipeline.run import run
    from printable.pipeline.select import pick_best

    settings = _settings_from_args(args)
    src = Path(args.image)
    out = Path(args.output or Path("output") / f"{src.stem}_{args.backend}.stl")

    panels = split_sheet(src, args.rows, args.cols, trim=args.trim)
    options = _parse_options(args.opt)

    candidates = []
    print()
    with tempfile.TemporaryDirectory() as tmp:
        for i, panel in enumerate(panels):
            panel_path = Path(tmp) / f"{src.stem}_v{i}.png"
            panel.save(panel_path)
            try:
                result = run(
                    panel_path,
                    backend=args.backend,
                    settings=settings,
                    options=options,
                    output_path=None,
                    seed=args.seed,
                    skip_repair=args.skip_repair,
                    max_faces=args.max_faces,
                )
            except Exception as exc:  # noqa: BLE001 - one bad panel shouldn't sink the rest
                print(f"  v{i}  failed: {exc}")
                continue
            stats = result.report.stats
            status = "PRINTABLE" if result.report.printable else "NOT PRINTABLE"
            print(
                f"  v{i}  watertight={stats.get('watertight')!s:<5}  "
                f"bodies={stats.get('bodies')!s:<4}  {status}"
            )
            candidates.append((i, result))

        if not candidates:
            print()
            print("error: no panel produced a usable mesh", file=sys.stderr)
            return 1

        winner_i, winner = pick_best(candidates)
        print()
        print(f"  picked v{winner_i}")

        if args.keep_all:
            for i, result in candidates:
                dest = out.with_stem(f"{out.stem}_v{i}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                result.mesh.export(dest)
                print(f"  wrote {dest}")

        out.parent.mkdir(parents=True, exist_ok=True)
        export_start = time.perf_counter()
        winner.mesh.export(out)
        winner.timings["export"] = time.perf_counter() - export_start
        winner.output_path = out

    _print_generate_result(winner)
    return 0 if winner.report.printable else 1


def _cmd_generate_spec_views(args: argparse.Namespace, spec: dict) -> int:
    """Generate from every view a prior `printable generate-spec` extracted,
    keeping whichever result is actually most printable -- the --sheet
    approach, but sourcing candidates from an already-extracted spec's view
    crops instead of a fresh grid split.
    """
    from printable.pipeline.run import run
    from printable.pipeline.select import pick_best
    from printable.pipeline.validate import check_dimension_targets

    views = spec.get("views", {})
    if not views:
        print("error: spec has no 'views' to generate from", file=sys.stderr)
        return 1

    spec_dir = Path(args.spec)
    if not spec_dir.is_dir():
        spec_dir = spec_dir.parent

    settings = _settings_from_args(args)
    options = _parse_options(args.opt)
    out = Path(args.output or Path("output") / f"{spec_dir.name}_{args.backend}.stl")

    candidates = []
    print()
    for name in views:
        view_path = spec_dir / f"{name}.png"
        if not view_path.exists():
            print(f"  {name:<8} failed: {view_path} not found (re-run generate-spec?)")
            continue
        try:
            result = run(
                view_path,
                backend=args.backend,
                settings=settings,
                options=options,
                output_path=None,
                seed=args.seed,
                skip_repair=args.skip_repair,
                max_faces=args.max_faces,
            )
        except Exception as exc:  # noqa: BLE001 - one bad view shouldn't sink the rest
            print(f"  {name:<8} failed: {exc}")
            continue
        stats = result.report.stats
        status = "PRINTABLE" if result.report.printable else "NOT PRINTABLE"
        print(
            f"  {name:<8} watertight={stats.get('watertight')!s:<5}  "
            f"bodies={stats.get('bodies')!s:<4}  {status}"
        )
        candidates.append((name, result))

    if not candidates:
        print()
        print("error: no view produced a usable mesh", file=sys.stderr)
        return 1

    winner_name, winner = pick_best(candidates)
    print()
    print(f"  picked {winner_name}")

    dims = spec.get("dimensions_mm", {})
    targets = {k: dims[k] for k in ("total_height", "width", "depth") if k in dims}
    check_dimension_targets(winner.report, winner.mesh, targets, tolerance_pct=args.tolerance)

    if args.keep_all:
        for name, result in candidates:
            dest = out.with_stem(f"{out.stem}_{name}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            result.mesh.export(dest)
            print(f"  wrote {dest}")

    out.parent.mkdir(parents=True, exist_ok=True)
    export_start = time.perf_counter()
    winner.mesh.export(out)
    winner.timings["export"] = time.perf_counter() - export_start
    winner.output_path = out

    _print_generate_result(winner)
    return 0 if winner.report.printable else 1


def cmd_generate_spec(args: argparse.Namespace) -> int:
    """Design spec sheet -> design_spec.json + reference view crops. Extraction only.

    Calls the external VLM extraction service (see ROADMAP.md's Phase 1) to
    turn a spec-sheet photo into design_spec.json, crops the labeled views
    out of the original sheet, and stops there -- no 3D generation. Feed
    the result into `printable generate --spec <this dir>` (one chosen
    view) or `--spec-all-views` (try every extracted view, keep the best),
    same as `--sheet` does for grid sheets.
    """
    from PIL import Image

    from printable.backends.sheet import crop_views
    from printable.vlm_client import fetch_design_spec

    sheet_path = Path(args.image)
    interim_dir = Path(args.interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)

    print()
    print(f"  extracting spec from {sheet_path} via {args.vlm_url} ...")
    spec = fetch_design_spec(sheet_path, vlm_url=args.vlm_url)
    spec_path = interim_dir / "design_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2))
    print(f"  wrote {spec_path}")

    views = spec.get("views", {})
    if views:
        sheet_image = Image.open(sheet_path).convert("RGB")
        crops = crop_views(sheet_image, views)
        for name, crop in crops.items():
            crop.save(interim_dir / f"{name}.png")
        print(f"  wrote {len(crops)} view(s) to {interim_dir}/")

    print()
    print(f"  title              {spec.get('title', '(untitled)')}")
    for key, value in spec.get("dimensions_mm", {}).items():
        print(f"  {key:<18} {value}")
    for key, value in spec.get("print_constraints", {}).items():
        print(f"  {key:<18} {value}")
    print()
    if not views:
        print("  no views extracted; nothing to generate from")
    print()
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    """Cut a multi-view sheet into single-subject panels."""
    from printable.backends.sheet import pick_best_panel, split_sheet

    src = Path(args.image)
    out_dir = Path(args.output_dir or src.parent / f"{src.stem}_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = split_sheet(src, args.rows, args.cols, trim=args.trim)
    best = pick_best_panel(panels)

    print()
    for i, panel in enumerate(panels):
        dest = out_dir / f"{src.stem}_v{i}.png"
        panel.save(dest)
        mark = "  <- best" if i == best else ""
        print(f"  {dest}  {panel.size[0]}x{panel.size[1]}{mark}")
    print()
    print("  Feed a single panel to an AI backend; a full sheet would be")
    print("  reconstructed as one mass rather than one figure.")
    print()
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    _load_ai_backends()
    print()
    for name in registry.names():
        impl = registry.get(name)
        ok, reason = impl.available()
        mark = "ok " if ok else "-- "
        gpu = "gpu" if impl.requires_gpu else "cpu"
        print(f"  {mark} {name:<12} {gpu:<5} {reason}")
    print()
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Validate an existing mesh file without regenerating it."""
    import trimesh

    from printable.pipeline.validate import validate

    mesh = trimesh.load(args.mesh, force="mesh")
    report = validate(mesh, PrintSettings(target_size_mm=args.size))

    print(json.dumps({"stats": report.stats,
                      "issues": [str(i) for i in report.issues]}, indent=2))
    return 0 if report.printable else 1


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the web API + browser UI."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "server dependencies not installed; run: uv sync --extra api"
        ) from exc

    from printable.api.app import create_app

    _load_ai_backends()
    app = create_app()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except OSError as exc:  # port in use, bind permission, etc.
        raise RuntimeError(str(exc)) from exc
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="printable",
        description="Turn a photo into a 3D-printable STL.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="image -> STL")
    g.add_argument("image", nargs="?", help="input photo (omit only with --spec-all-views)")
    g.add_argument("-b", "--backend", default="lithophane",
                   help="lithophane | heightmap | triposr | hunyuan3d")
    g.add_argument("-o", "--output", help="output STL path")
    g.add_argument("--spec", metavar="PATH",
                   help="a design_spec.json from `printable generate-spec` (or its "
                        "directory) -- fills --size/--min-wall from it unless given "
                        "explicitly, and checks the result's proportions against it")
    g.add_argument("--spec-all-views", action="store_true",
                   help="with --spec: generate from every view it extracted, keep "
                        "whichever result is actually most printable (not just the "
                        "one you'd pick by eye) -- --sheet's approach, applied to a "
                        "spec's views instead of a fresh grid split. No image needed.")
    g.add_argument("--tolerance", type=float, default=15.0,
                   help="--spec: dimension-check tolerance, in percent (default 15)")
    g.add_argument("--size", type=float, default=None,
                   help="longest axis in mm (default 100, or --spec's total_height)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--nozzle", type=float, default=0.4)
    g.add_argument("--min-wall", type=float, default=None,
                   help="default 0.8, or --spec's min_wall_thickness_mm")
    g.add_argument("--max-overhang", type=float, default=45.0)
    g.add_argument("--build-volume", type=float, nargs=3,
                   default=[256.0, 256.0, 256.0], metavar=("X", "Y", "Z"))
    g.add_argument("--hollow", action="store_true")
    g.add_argument("--hollow-wall", type=float, default=2.0)
    g.add_argument("--no-base", action="store_true")
    g.add_argument("--base-thickness", type=float, default=2.0)
    g.add_argument("--no-orient", action="store_true")
    g.add_argument("--rotate", type=float, nargs=3, default=None,
                    metavar=("X", "Y", "Z"),
                    help="manual rotation in degrees, applied after auto-orient "
                         "and before the base -- fixes auto-orient picking a "
                         "semantically wrong pose (e.g. a character lying down)")
    g.add_argument("--skip-repair", action="store_true",
                   help="export raw generator output, for debugging")
    g.add_argument("--max-faces", type=int, default=300_000)
    g.add_argument("--opt", action="append", default=[], metavar="KEY=VALUE",
                   help="backend-specific option, repeatable. Common: "
                        "key_background=true (cut a flat backdrop away), "
                        "relief_mm, max_dim, blur")
    g.add_argument("--sheet", action="store_true",
                   help="image is a multi-view turnaround sheet (e.g. a "
                        "2x2 grid of the same subject): split it, generate "
                        "from every panel, and keep whichever result is "
                        "actually most printable (not just best-framed)")
    g.add_argument("--rows", type=int, default=2, help="--sheet grid rows")
    g.add_argument("--cols", type=int, default=2, help="--sheet grid columns")
    g.add_argument("--trim", type=int, default=4,
                    help="--sheet: pixels to shave off each panel edge (default 4)")
    g.add_argument("--keep-all", action="store_true",
                    help="--sheet: also export every panel's result "
                         "(<output>_v0.stl, _v1.stl, ...), not just the winner")
    g.set_defaults(func=cmd_generate)

    gs = sub.add_parser(
        "generate-spec",
        help="design spec sheet -> design_spec.json + reference view crops",
    )
    gs.add_argument("image", help="design spec sheet photo")
    gs.add_argument("--vlm-url", default="http://localhost:8082",
                     help="VLM extraction service base URL (default http://localhost:8082)")
    gs.add_argument("--interim-dir", default="interim",
                     help="where design_spec.json and cropped views are written")
    gs.set_defaults(func=cmd_generate_spec)

    s_ = sub.add_parser("split", help="cut a multi-view sheet into panels")
    s_.add_argument("image", help="turnaround or reference sheet")
    s_.add_argument("-d", "--output-dir")
    s_.add_argument("--rows", type=int, default=2)
    s_.add_argument("--cols", type=int, default=2)
    s_.add_argument("--trim", type=int, default=4,
                    help="pixels to shave off each panel edge (default 4)")
    s_.set_defaults(func=cmd_split)

    b = sub.add_parser("backends", help="list backends and availability")
    b.set_defaults(func=cmd_backends)

    i = sub.add_parser("inspect", help="validate an existing mesh")
    i.add_argument("mesh")
    i.add_argument("--size", type=float, default=100.0)
    i.set_defaults(func=cmd_inspect)

    sv = sub.add_parser("serve", help="run the web API + browser UI")
    sv.add_argument("--host", default="0.0.0.0",
                     help="bind address (default 0.0.0.0, LAN-reachable)")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if getattr(args, "backend", None) not in (None, "heightmap", "lithophane"):
        _load_ai_backends()
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        if args.verbose:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
