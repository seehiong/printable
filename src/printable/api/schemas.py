"""Pydantic request/response models for the web API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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
