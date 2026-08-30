"""Picking a winner among generation candidates, and picking a pipeline for
an unclassified upload.

Both are pure, dependency-free functions on purpose: `pick_best` used to be
duplicated once (cli/main.py's `_pick_best`, api/app.py's inline `min(...)`
for spec mode) and would have been duplicated a third time for the
auto-routing sheet branch. `route_for_category` is new, and kept pure so the
category->backend mapping is unit-testable without a VLM or HTTP call.
"""

from __future__ import annotations

from dataclasses import dataclass

# Preferred backend, in order, for a full single-object 3D reconstruction --
# same list the web UI already hardcodes client-side for its spec-mode
# checkbox (heightmap/lithophane don't make sense for a dimensioned or
# multi-view 3D part; see static/index.html's SPEC_MODE_PREFERRED_BACKENDS).
OBJECT_BACKEND_PREFERENCE = ("hunyuan3d", "triposr")


def pick_best(candidates: list[tuple[str, object]]) -> tuple[str, object]:
    """Pick the winner among named GenerationResults: printable first, then fewest bodies.

    Each candidate is (name, result) where result has `.report.printable`
    (bool) and `.report.stats` (dict, may have a "bodies" count).
    """

    def score(item: tuple[str, object]) -> tuple[int, int]:
        _, result = item
        return (0 if result.report.printable else 1, result.report.stats.get("bodies", 999))

    return min(candidates, key=score)


@dataclass
class RouteDecision:
    """What to do with an upload, chosen from its VLM classification."""

    category: str
    mode: str  # "single" | "sheet" | "spec" | "manual"
    backend: str | None
    message: str


def _first_available(preference: tuple[str, ...], available_backends: set[str]) -> str | None:
    for name in preference:
        if name in available_backends:
            return name
    return None


def route_for_category(category: str, available_backends: set[str]) -> RouteDecision:
    """Map a /classify category onto one of printable's existing pipelines.

    Table-driven on purpose: every branch below routes to a pipeline that
    already exists (spec extraction, --sheet's grid-split-and-pick-best,
    single-image generation) -- nothing new is invented here for the model
    to route into, see ROADMAP.md's classification writeup for why.
    """
    if category == "design_spec_sheet":
        backend = _first_available(OBJECT_BACKEND_PREFERENCE, available_backends)
        if backend is None:
            return RouteDecision(
                category, "manual", None, "This looks like a design spec sheet, but no 3D "
                "reconstruction backend is available on this server -- pick one manually."
            )
        return RouteDecision(
            category, "spec", backend,
            "This looks like a design spec sheet, so I'm reading its dimensions and "
            f"print settings and reconstructing it with {backend}.",
        )

    if category == "character_turnaround":
        backend = _first_available(OBJECT_BACKEND_PREFERENCE, available_backends)
        if backend is None:
            return RouteDecision(
                category, "manual", None, "This looks like a character turnaround sheet, but "
                "no 3D reconstruction backend is available on this server -- pick one manually."
            )
        return RouteDecision(
            category, "sheet", backend,
            "This looks like a multi-view character sheet, so I'm splitting it into panels "
            f"and reconstructing the best one with {backend}.",
        )

    if category == "portrait_photo":
        if "lithophane" not in available_backends:
            return RouteDecision(
                category, "manual", None,
                "This looks like a portrait photo, but the lithophane backend isn't "
                "available on this server -- pick a backend manually.",
            )
        return RouteDecision(
            category, "single", "lithophane",
            "This looks like a portrait photo, so I'm generating a lithophane from it.",
        )

    if category == "object_photo":
        backend = _first_available(OBJECT_BACKEND_PREFERENCE, available_backends)
        if backend is None:
            return RouteDecision(
                category, "manual", None, "This looks like a photo of an object, but no 3D "
                "reconstruction backend is available on this server -- pick one manually."
            )
        return RouteDecision(
            category, "single", backend,
            f"This looks like a photo of a single object, so I'm reconstructing it with {backend}.",
        )

    return RouteDecision(
        "unclear", "manual", None,
        "I couldn't confidently tell what kind of image this is -- pick a backend manually.",
    )
