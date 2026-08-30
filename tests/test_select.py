"""Tests for pipeline/select.py: pure, no VLM/HTTP involved."""

from __future__ import annotations

from printable.pipeline.select import route_for_category


def test_design_spec_sheet_prefers_hunyuan3d():
    decision = route_for_category("design_spec_sheet", {"hunyuan3d", "triposr", "lithophane"})
    assert decision.mode == "spec"
    assert decision.backend == "hunyuan3d"


def test_design_spec_sheet_falls_back_to_triposr():
    decision = route_for_category("design_spec_sheet", {"triposr", "lithophane"})
    assert decision.mode == "spec"
    assert decision.backend == "triposr"


def test_design_spec_sheet_with_no_3d_backend_goes_manual():
    decision = route_for_category("design_spec_sheet", {"lithophane"})
    assert decision.mode == "manual"
    assert decision.backend is None


def test_character_turnaround_routes_to_sheet_mode():
    decision = route_for_category("character_turnaround", {"hunyuan3d"})
    assert decision.mode == "sheet"
    assert decision.backend == "hunyuan3d"


def test_portrait_photo_routes_to_lithophane():
    decision = route_for_category("portrait_photo", {"lithophane", "hunyuan3d"})
    assert decision.mode == "single"
    assert decision.backend == "lithophane"


def test_portrait_photo_without_lithophane_goes_manual():
    decision = route_for_category("portrait_photo", {"hunyuan3d"})
    assert decision.mode == "manual"
    assert decision.backend is None


def test_object_photo_routes_to_object_backend():
    decision = route_for_category("object_photo", {"triposr"})
    assert decision.mode == "single"
    assert decision.backend == "triposr"


def test_unclear_goes_manual():
    decision = route_for_category("unclear", {"hunyuan3d", "lithophane", "triposr"})
    assert decision.mode == "manual"
    assert decision.backend is None
    assert decision.category == "unclear"


def test_unrecognized_category_falls_back_to_manual():
    decision = route_for_category("something-the-model-made-up", {"hunyuan3d"})
    assert decision.mode == "manual"
    assert decision.backend is None
    assert decision.category == "unclear"


def test_every_decision_has_a_nonempty_message():
    for category in (
        "design_spec_sheet", "character_turnaround", "portrait_photo", "object_photo", "unclear",
    ):
        decision = route_for_category(category, set())
        assert decision.message
