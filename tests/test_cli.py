"""Tests for the CLI layer: spec-driven generation and generate-spec's
extraction-only behaviour. CPU-only (lithophane), no GPU/model weights,
no live VLM server -- fetch_design_spec is monkeypatched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from printable.cli.main import _apply_spec_defaults, _load_spec, main
from printable.pipeline.select import pick_best


def _write_spec(path: Path, **overrides) -> dict:
    spec = {
        "title": "Test Widget",
        "dimensions_mm": {"total_height": 42.0, "width": 20.0, "depth": 15.0},
        "print_constraints": {"min_wall_thickness_mm": 1.1},
        "views": {
            "front": {"box_2d": [0, 0, 1000, 1000]},
        },
    }
    spec.update(overrides)
    path.write_text(json.dumps(spec))
    return spec


class _FakeResult:
    """Minimal stand-in for GenerationResult, just enough for pick_best's key."""

    def __init__(self, printable: bool, bodies: int):
        self.report = argparse.Namespace(
            printable=printable, stats={"bodies": bodies}
        )


def test_load_spec_from_file(tmp_path):
    spec_path = tmp_path / "design_spec.json"
    written = _write_spec(spec_path)
    assert _load_spec(str(spec_path)) == written


def test_load_spec_from_directory(tmp_path):
    _write_spec(tmp_path / "design_spec.json")
    loaded = _load_spec(str(tmp_path))
    assert loaded["title"] == "Test Widget"


def test_apply_spec_defaults_fills_only_unset_values():
    spec = {
        "dimensions_mm": {"total_height": 42.0},
        "print_constraints": {"min_wall_thickness_mm": 1.1},
    }

    unset = argparse.Namespace(size=None, min_wall=None)
    _apply_spec_defaults(unset, spec)
    assert unset.size == 42.0
    assert unset.min_wall == 1.1

    explicit = argparse.Namespace(size=99.0, min_wall=2.0)
    _apply_spec_defaults(explicit, spec)
    assert explicit.size == 99.0
    assert explicit.min_wall == 2.0


def test_pick_best_prefers_printable_then_fewest_bodies():
    candidates = [
        ("a", _FakeResult(printable=False, bodies=1)),
        ("b", _FakeResult(printable=True, bodies=3)),
        ("c", _FakeResult(printable=True, bodies=1)),
    ]
    name, _ = pick_best(candidates)
    assert name == "c"


def test_generate_with_spec_fills_size_from_spec(tmp_path, sample_image, monkeypatch):
    """--spec should set --size from total_height without it being passed explicitly."""
    monkeypatch.chdir(tmp_path)
    spec_path = tmp_path / "design_spec.json"
    _write_spec(spec_path)
    out = tmp_path / "out.stl"

    rc = main([
        "generate", str(sample_image), "-b", "lithophane",
        "--spec", str(spec_path), "-o", str(out),
    ])

    assert rc == 0
    assert out.exists()


def test_generate_spec_only_extracts_no_generation(tmp_path, monkeypatch):
    """generate-spec must write design_spec.json + view crops and stop there --
    no STL, no run() call, no backend needed at all.
    """
    # monkeypatch.setattr below imports printable.vlm_client to resolve the
    # dotted path, and that module does a bare `import httpx` at module
    # level -- httpx lives behind the optional `spec` extra, so this needs
    # its own guard rather than erroring out on a bare `uv sync` install
    # (confirmed: this was a real CI gap, not hypothetical -- a bare install
    # hit an ImportError here, not a skip).
    pytest.importorskip("httpx")
    from PIL import Image

    sheet_path = tmp_path / "sheet.png"
    Image.new("RGB", (200, 200), "white").save(sheet_path)
    interim_dir = tmp_path / "interim"

    fake_spec = {
        "title": "Widget",
        "dimensions_mm": {"total_height": 42.0},
        "print_constraints": {},
        "views": {"front": {"box_2d": [0, 0, 1000, 1000]}},
    }
    monkeypatch.setattr(
        "printable.vlm_client.fetch_design_spec", lambda *a, **k: fake_spec
    )

    rc = main([
        "generate-spec", str(sheet_path),
        "--interim-dir", str(interim_dir),
    ])

    assert rc == 0
    assert (interim_dir / "design_spec.json").exists()
    assert (interim_dir / "front.png").exists()
    # The whole point of the split: no STL should appear anywhere.
    assert not list(tmp_path.rglob("*.stl"))
