"""Art source: what a game model was made from, kept beside it.

    <dir>/reference.png   the image fed to image-to-3D
    <dir>/raw.glb         the backend's untouched colored output
    <dir>/prompt.md       prompt, negative, seed and every setting used

raw.glb is what `printable bake` reduces from, so a model can be rebuilt
at another triangle budget without regenerating it. reference.png and
prompt.md together reproduce or tweak the whole result.
"""

from __future__ import annotations

import io
import json
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trimesh
from PIL import Image, PngImagePlugin

# PNG text-chunk key holding the text-to-image settings as JSON, so a
# reference image carries its own provenance wherever it's copied.
PNG_KEY = "printable"


def embed_image_params(png: bytes, params: dict[str, Any]) -> bytes:
    image = Image.open(io.BytesIO(png))
    info = PngImagePlugin.PngInfo()
    for key, value in getattr(image, "text", {}).items():
        if key != PNG_KEY:
            info.add_text(key, value)
    info.add_text(PNG_KEY, json.dumps(params))
    out = io.BytesIO()
    image.save(out, format="PNG", pnginfo=info)
    return out.getvalue()


def read_image_params(path: Path) -> dict[str, Any] | None:
    """The settings embedded by `embed_image_params`, or None for any image
    that didn't come from `printable generate-image` / the web UI."""
    try:
        with Image.open(path) as image:
            raw = getattr(image, "text", {}).get(PNG_KEY)
    except OSError:
        return None
    if not raw:
        return None
    try:
        params = json.loads(raw)
    except ValueError:
        return None
    return params if isinstance(params, dict) else None


def write_art_source(
    directory: Path,
    *,
    image_path: Path,
    raw_mesh: trimesh.Trimesh,
    image_params: dict[str, Any] | None,
    generation: dict[str, Any],
    game: dict[str, Any],
    name: str | None = None,
) -> list[Path]:
    """`name` (prompt.md's heading) defaults to the directory's own name,
    e.g. art_source/relics/relic_astrolabe -> relic_astrolabe."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    reference = directory / "reference.png"
    with Image.open(image_path) as image:
        info = PngImagePlugin.PngInfo()
        if image_params:
            info.add_text(PNG_KEY, json.dumps(image_params))
        image.save(reference, format="PNG", pnginfo=info)

    raw = directory / "raw.glb"
    raw_mesh.export(raw)

    prompt = directory / "prompt.md"
    prompt.write_text(render_prompt_md(name or directory.name, image_params, generation, game))
    return [reference, raw, prompt]


def write_for_result(
    directory: Path,
    result: Any,
    *,
    image_path: Path,
    image_params: dict[str, Any] | None,
    backend: str,
    seed: int,
    options: dict[str, Any],
    max_faces: int,
    texture_px: int,
    model_name: str,
    name: str | None = None,
) -> list[Path]:
    """write_art_source() for a game_asset PipelineResult."""
    stats = result.report.stats
    return write_art_source(
        directory,
        name=name,
        image_path=image_path,
        raw_mesh=result.raw_color_mesh,
        image_params=image_params,
        generation={
            "backend": backend,
            "seed": seed,
            "options": {k: v for k, v in options.items() if not k.startswith("scale_")},
            "raw_faces": stats.get("raw_faces"),
        },
        game={
            "max_faces": max_faces,
            "faces": stats.get("faces"),
            "texture_px": texture_px,
            "axis_scale": tuple(float(options.get(f"scale_{a}", 1.0)) for a in "xyz"),
            "watertight": stats.get("watertight"),
            "model_name": model_name,
        },
    )


def render_prompt_md(
    name: str,
    image_params: dict[str, Any] | None,
    generation: dict[str, Any],
    game: dict[str, Any],
) -> str:
    today = datetime.now(timezone.utc).astimezone().date().isoformat()
    lines = [f"# {name}", "", f"Art source written by printable on {today}.", ""]

    lines += ["## Reference image: `reference.png`", ""]
    if image_params:
        lines += _code_block("Prompt", image_params.get("prompt", ""))
        lines += _code_block("Negative prompt", image_params.get("negative_prompt", ""))
        size = ""
        if image_params.get("width") and image_params.get("height"):
            size = f"{image_params['width']} × {image_params['height']}"
        lines += _table(
            [
                ("Seed", image_params.get("seed")),
                ("Steps", image_params.get("steps")),
                ("CFG", image_params.get("cfg")),
                ("Sampler", image_params.get("sampler")),
                ("Size", size),
                ("Checkpoint", _code(image_params.get("checkpoint"))),
                ("Post-processing", "left half mirrored onto right" if image_params.get("mirrored") else None),
            ]
        )
    else:
        lines += ["Supplied directly; no text-to-image settings were recorded.", ""]

    options = generation.get("options") or {}
    lines += ["## Image to 3D: `raw.glb`", ""]
    lines += _table(
        [
            ("Backend", generation.get("backend")),
            ("Seed", generation.get("seed")),
            ("Options", _code(" ".join(f"{k}={v}" for k, v in sorted(options.items())))),
            ("Raw faces", _count(generation.get("raw_faces"))),
        ]
    )

    scale = game.get("axis_scale") or (1.0, 1.0, 1.0)
    lines += ["## Game model", ""]
    lines += _table(
        [
            ("Triangle budget", _count(game.get("max_faces"))),
            ("Faces", _count(game.get("faces"))),
            ("Texture", f"{game.get('texture_px')} px base color + normal map"),
            ("Axis scale", " × ".join(f"{s:g}" for s in scale)),
            ("Watertight", {True: "yes", False: "no"}.get(game.get("watertight"))),
        ]
    )

    command = [
        "printable", "bake", "raw.glb", "-o", game.get("model_name", "model.glb"),
        "--max-faces", str(game.get("max_faces")),
        "--texture-size", str(game.get("texture_px")),
    ]
    for axis, value in zip("xyz", scale):
        if value != 1.0:
            command += ["--opt", f"scale_{axis}={value:g}"]
    lines += [
        "Re-bake from `raw.glb` at another budget without regenerating:",
        "",
        "```bash",
        shlex.join(command),
        "```",
        "",
    ]
    return "\n".join(lines)


def _code_block(title: str, text: str) -> list[str]:
    return [f"**{title}**", "", "```", text, "```", ""]


def _table(rows: list[tuple[str, Any]]) -> list[str]:
    out = ["| Setting | Value |", "|---|---|"]
    for key, value in rows:
        if value in (None, ""):
            continue
        cell = str(value).replace("|", r"\|")
        out.append(f"| {key} | {cell} |")
    return out + [""]


def _code(value: Any) -> str | None:
    return f"`{value}`" if value else None


def _count(value: Any) -> str | None:
    return f"{value:,}" if isinstance(value, int) else None
