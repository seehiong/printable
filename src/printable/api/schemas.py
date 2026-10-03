"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from printable.comfyui_client import (
    DEFAULT_CHECKPOINT,
    DEFAULT_COMFYUI_URL,
    DEFAULT_NEGATIVE_PROMPT,
)


class Txt2ImgRequest(BaseModel):
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    steps: int = 4
    cfg: float = 1.0
    width: int = 1024
    height: int = 1024
    seed: int | None = None
    checkpoint: str = DEFAULT_CHECKPOINT
    comfyui_url: str = DEFAULT_COMFYUI_URL


class BackendInfo(BaseModel):
    name: str
    requires_gpu: bool
    available: bool
    reason: str


class JobSubmitResponse(BaseModel):
    job_id: str


class JobEvent(BaseModel):
    stage: str
    status: str  # start | done | error
    elapsed: float | None = None


class JobStatus(BaseModel):
    id: str
    backend: str
    status: str  # queued | running | done | error
    events: list[JobEvent]
    error: str | None = None
    printable: bool | None = None
    stats: dict[str, Any] | None = None
    issues: list[str] | None = None
    design_spec: dict[str, Any] | None = None
    classification: dict[str, Any] | None = None
    has_glb: bool = False
    has_geometry_cues: bool = False
    # True for a --game-asset job: output_path is itself a textured
    # low-poly .glb (no STL at all), not print-checked -- see /api/jobs'
    # game_asset param and README's "Game assets" section.
    game_asset: bool = False
