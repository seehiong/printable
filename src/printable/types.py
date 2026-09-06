"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image


class Backend(str, Enum):
    """Available geometry-generation backends."""

    HEIGHTMAP = "heightmap"
    LITHOPHANE = "lithophane"
    TRIPOSR = "triposr"
    HUNYUAN3D = "hunyuan3d"


@dataclass
class DepthNormalResult:
    """Auxiliary geometry cues estimated from one image.

    Diagnostic only for now: no backend accepts these as a conditioning
    input, so this exists for inspection (written to interim/ PNGs) and
    future use, not to change today's generated mesh.
    """

    depth: np.ndarray
    # None when even the depth-derived estimate couldn't be computed.
    normal: np.ndarray | None
    # "depth-derived" -- there is no transformers-native normal-estimation
    # task, so this is a gradient-based approximation, not a learned normal
    # map. Kept as a field rather than assumed, so callers don't mistake it
    # for one if a real estimator is added later.
    source: str


@dataclass
class GenerationRequest:
    """Everything a backend needs to turn an image into raw geometry."""

    image_path: Path
    backend: Backend
    seed: int = 42
    # Backend-specific knobs. Kept loose on purpose: each backend documents
    # its own keys rather than forcing a lowest-common-denominator schema.
    options: dict[str, Any] = field(default_factory=dict)
    # Pre-processed image, set by run() for backends with needs_preprocess.
    # A backend called directly (bypassing run()) gets None here and falls
    # back to preprocessing image_path itself.
    image: Image.Image | None = None
    # Set by run() when the geometry-cues stage ran. See DepthNormalResult.
    depth_normal: DepthNormalResult | None = None


@dataclass
class GenerationResult:
    """Raw geometry straight out of a backend, before any print prep."""

    mesh: trimesh.Trimesh
    backend: Backend
    # True when a colored/textured preview is available (a backend that was
    # asked for it via --opt texture=true, and actually produced it) -- set
    # explicitly by the backend rather than re-derived from mesh.visual's
    # type at every call site. Gates the pipeline's GLB export branch.
    has_color: bool = False
    # The mesh the GLB preview export should use, when it differs from
    # `mesh` above. None means "use `mesh` itself" (TripoSR's per-vertex
    # color path: same geometry, just with real color on mesh.visual).
    # Set this instead when a backend's colored output has its own
    # different topology from the plain shape mesh -- e.g. Hunyuan3D's
    # hy3dpaint, which remeshes and UV-unwraps internally, so its textured
    # result is a genuinely different mesh object, not `mesh` with color
    # attached.
    color_mesh: trimesh.Trimesh | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrintSettings:
    """Physical constraints of the target printer and desired output."""

    # Longest axis of the finished object, in millimetres.
    target_size_mm: float = 100.0
    # Nozzle diameter drives the minimum feature that can actually print.
    nozzle_diameter_mm: float = 0.4
    # Walls thinner than this get flagged; roughly 2 perimeters.
    min_wall_thickness_mm: float = 0.8
    # Overhangs steeper than this need supports.
    max_overhang_deg: float = 45.0
    # Build volume, used to reject oversized output early.
    build_volume_mm: tuple[float, float, float] = (256.0, 256.0, 256.0)

    hollow: bool = False
    hollow_wall_mm: float = 2.0
    # Drain holes so resin/powder can escape; harmless on FDM.
    drain_holes: int = 0

    add_base: bool = True
    base_thickness_mm: float = 2.0
    auto_orient: bool = True
    # Manual (x, y, z) rotation in degrees, applied after auto_orient and
    # before the base is added -- auto_orient optimizes purely for print
    # success and has no concept of e.g. "feet down" for a character mesh,
    # so this is the escape hatch when it picks a semantically wrong pose.
    manual_rotation_deg: tuple[float, float, float] | None = None


@dataclass
class ValidationIssue:
    """A single problem found during pre-flight checks."""

    severity: str  # "error" | "warning" | "info"
    code: str
    message: str
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


@dataclass
class ValidationReport:
    """Aggregate verdict on whether a mesh is fit to slice."""

    issues: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def printable(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str, **detail: Any) -> None:
        self.issues.append(ValidationIssue(severity, code, message, detail))
