"""Tests for the web API. CPU-only (lithophane/heightmap), no GPU needed.

Skipped entirely if the `api` extra isn't installed, so a bare `uv sync`
(no extras) still runs the base 26-test CPU-only suite untouched.
"""

from __future__ import annotations

import io
import json
import shutil
import time

import pytest
import trimesh

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from printable.api import jobs
from printable.api.app import create_app
from printable.backends import heightmap  # noqa: F401 - populates the registry


@pytest.fixture(autouse=True, scope="module")
def _cleanup_job_dirs():
    """jobs.JOBS_ROOT is repo-local (output/jobs/), not system tmp -- clean up
    after this module so repeated test runs don't leave job directories
    behind (JobStore's eviction is in-memory only, per server process, so it
    never sees directories from an earlier pytest invocation)."""
    yield
    # jobs.store is a process-wide singleton (this is the only test module
    # that touches it), so this module teardown is the one place safe to
    # fully join its worker thread -- do that before anything else, so it's
    # not still idling with a reference to the last job's tensors on its
    # stack when Python starts interpreter finalization later. See
    # JobStore.shutdown()'s docstring for why that race matters. This also
    # subsumes the older "sleep and hope the straggler job finished writing"
    # mitigation it replaced -- shutdown(wait=True) blocks until the worker
    # thread has actually exited, a strictly stronger guarantee than a fixed
    # delay.
    jobs.store.shutdown()
    shutil.rmtree(jobs.JOBS_ROOT, ignore_errors=True)


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


def _submit(client: TestClient, sample_image, backend: str = "lithophane", **form) -> str:
    with open(sample_image, "rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"image": ("sample.png", f, "image/png")},
            data={"backend": backend, "size": "40", **form},
        )
    assert resp.status_code == 202, resp.text
    return resp.json()["job_id"]


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            return status
        time.sleep(0.05)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def test_backends_endpoint_lists_cpu_backends(client):
    backends = {b["name"]: b for b in client.get("/api/backends").json()}
    assert backends["lithophane"]["available"] is True
    assert backends["heightmap"]["available"] is True
    assert backends["lithophane"]["requires_gpu"] is False


def test_submit_job_lithophane_end_to_end(client, sample_image):
    job_id = _submit(client, sample_image, max_faces="20000")
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "done", status
    assert status["printable"] is True
    assert status["stats"]["watertight"] is True


def test_job_stream_emits_expected_stages(client, sample_image):
    job_id = _submit(client, sample_image)
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        stages = {}
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            stages.setdefault(event["stage"], []).append(event["status"])
            if len(stages.get("export", [])) >= 2 or "error" in stages.get(event["stage"], []):
                break

    # lithophane never sets needs_preprocess, so no "preprocess" stage.
    assert "preprocess" not in stages
    for stage in ("generate", "repair", "prep", "validate", "export"):
        assert stages.get(stage) == ["start", "done"], (stage, stages)


def test_job_stream_emits_geometry_cues_stage_when_requested(client, sample_image):
    """geometry_cues=true adds a "geometry-cues" stage for any backend, even
    with the `depth` extra not installed -- unavailability is a warning
    inside the stage, not a failure of it (see pipeline/run.py)."""
    job_id = _submit(client, sample_image, opt="geometry_cues=true")
    with client.stream("GET", f"/api/jobs/{job_id}/events") as resp:
        stages = {}
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: ") :])
            stages.setdefault(event["stage"], []).append(event["status"])
            if len(stages.get("export", [])) >= 2 or "error" in stages.get(event["stage"], []):
                break

    assert stages.get("geometry-cues") == ["start", "done"], stages


def test_download_stl_after_completion(client, sample_image):
    job_id = _submit(client, sample_image)
    _wait_for_terminal(client, job_id)
    resp = client.get(f"/api/jobs/{job_id}/stl")
    assert resp.status_code == 200
    mesh = trimesh.load(io.BytesIO(resp.content), file_type="stl")
    assert len(mesh.faces) > 0


def test_download_stl_before_completion_is_conflict(client, sample_image):
    job_id = _submit(client, sample_image)
    resp = client.get(f"/api/jobs/{job_id}/stl")
    assert resp.status_code in (404, 409)


def test_unknown_job_id_is_404(client):
    for path in ("/api/jobs/does-not-exist", "/api/jobs/does-not-exist/stl"):
        assert client.get(path).status_code == 404


def test_bad_backend_name_returns_useful_error(client, sample_image):
    with open(sample_image, "rb") as f:
        resp = client.post(
            "/api/jobs",
            files={"image": ("sample.png", f, "image/png")},
            data={"backend": "not-a-real-backend", "size": "40"},
        )
    assert resp.status_code == 400
    assert "unknown backend" in resp.json()["error"]


# --- colored GLB export -----------------------------------------------------
# Needs a real GPU backend, unlike the rest of this file. Skips cleanly, not
# fails, when unavailable -- same convention as the spec-sheet tests below.


def _backend_available(client: TestClient, name: str) -> bool:
    from printable.backends import ai  # noqa: F401 - populates the registry with GPU backends

    backends = {b["name"]: b for b in client.get("/api/backends").json()}
    return backends.get(name, {}).get("available", False)


def test_job_stream_emits_export_glb_stage_for_colored_backend(client, sample_image):
    """--opt texture=true on triposr produces a colored preview GLB
    alongside the STL (see pipeline/run.py's "export-glb" stage) -- run
    against real hardware, not mocked, since this session already verified
    the actual GLB byte layout by hand against this exact backend."""
    if not _backend_available(client, "triposr"):
        pytest.skip("triposr backend not available")

    job_id = _submit(client, sample_image, backend="triposr", opt="texture=true")
    status = _wait_for_terminal(client, job_id, timeout=120.0)
    assert status["status"] == "done", status
    assert status["has_glb"] is True, status

    resp = client.get(f"/api/jobs/{job_id}/glb")
    assert resp.status_code == 200
    assert resp.content[:4] == b"glTF"


# Deliberately only one real-GPU-backend test in this file. Loading a
# second real GPU backend (a different one, e.g. spar3d while triposr's
# above) in the same process hung this entire machine hard enough to need
# a physical reboot -- not a slow test, an OS-level lockup, apparently from
# switching GPU backends within one process on this ROCm nightly build. Do
# not add a second real-GPU-backend test to this file without reading that
# warning in docs/SETUP.md's Troubleshooting section first.


# --- auto mode ---------------------------------------------------------------
# The classify call itself is monkeypatched for these (CPU-only, no live VLM
# server) -- route_for_category's own mapping already has direct unit tests
# in test_select.py, so these confirm the *endpoint* wires a classification
# result through to a real generation job end-to-end.


def _submit_auto(client: TestClient, sample_image, **form) -> str:
    with open(sample_image, "rb") as f:
        resp = client.post(
            "/api/jobs/auto",
            files={"image": ("sample.png", f, "image/png")},
            data={"size": "40", **form},
        )
    assert resp.status_code == 202, resp.text
    return resp.json()["job_id"]


def test_auto_job_routes_portrait_to_lithophane(client, sample_image, monkeypatch):
    monkeypatch.setattr(
        "printable.vlm_client.classify_image",
        lambda *a, **k: {
            "category": "portrait_photo", "confidence": 0.9, "rationale": "a face"
        },
    )
    job_id = _submit_auto(client, sample_image, max_faces="20000")
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "done", status
    assert status["backend"] == "lithophane"
    assert status["classification"]["category"] == "portrait_photo"
    assert status["classification"]["backend"] == "lithophane"

    resp = client.get(f"/api/jobs/{job_id}/stl")
    assert resp.status_code == 200
    mesh = trimesh.load(io.BytesIO(resp.content), file_type="stl")
    assert len(mesh.faces) > 0


def test_auto_job_unclear_category_produces_no_mesh(client, sample_image, monkeypatch):
    monkeypatch.setattr(
        "printable.vlm_client.classify_image",
        lambda *a, **k: {"category": "unclear", "confidence": 0.2, "rationale": "not sure"},
    )
    job_id = _submit_auto(client, sample_image)
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "done", status
    assert status["classification"]["mode"] == "manual"
    assert status["classification"]["backend"] is None

    resp = client.get(f"/api/jobs/{job_id}/stl")
    assert resp.status_code == 409


def test_auto_job_bubbles_up_classify_errors(client, sample_image, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("VLM server at http://x timed out after 1.0s")

    monkeypatch.setattr("printable.vlm_client.classify_image", _boom)
    job_id = _submit_auto(client, sample_image)
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "error", status
    assert "timed out" in status["error"]


# --- spec-sheet mode --------------------------------------------------------
# Needs a live VLM server (see ROADMAP.md's Phase 1) and a real GPU backend --
# unlike the rest of this file, not CPU-only. Skips cleanly, not fails, when
# either isn't available, since this is real infrastructure this session
# already stood up and verified by hand, not something CI can assume exists.


def _vlm_server_reachable() -> bool:
    import httpx

    try:
        return httpx.get("http://localhost:8082/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def test_submit_spec_job_end_to_end(client):
    if not _vlm_server_reachable():
        pytest.skip("vlm-server not running at localhost:8082")
    if not _backend_available(client, "triposr"):
        pytest.skip("triposr backend not available")

    spec_image = "assets/examples/keychain_boba_spec.jpg"
    with open(spec_image, "rb") as f:
        resp = client.post(
            "/api/jobs/spec",
            files={"image": ("keychain_boba_spec.jpg", f, "image/jpeg")},
            data={"backend": "triposr"},
        )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    # 300s, not a round guess: a real run of this exact job measured 188s
    # end to end, and nearly all of it (~161s) is the VLM "extract" stage's
    # own inference latency against a 27B model, not mesh generation --
    # 180s was cutting it too close. Letting the job actually finish also
    # avoids returning while its background ThreadPoolExecutor worker
    # (jobs.py) is still mid-generate; racing that against interpreter
    # teardown at the end of the test session risks corrupting the process
    # (seen as a "corrupted double-linked list" glibc abort after a run
    # that timed out here).
    deadline = time.time() + 300.0
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in ("done", "error"):
            break
        time.sleep(0.5)
    assert status is not None and status["status"] == "done", status

    spec = status["design_spec"]
    assert spec is not None
    assert {"dimensions_mm", "print_constraints", "views"} <= spec.keys()

    stl_resp = client.get(f"/api/jobs/{job_id}/stl")
    assert stl_resp.status_code == 200
    mesh = trimesh.load(io.BytesIO(stl_resp.content), file_type="stl")
    assert len(mesh.faces) > 0


# No live end-to-end test for /api/jobs/auto against a real design spec
# sheet: route_for_category prefers hunyuan3d over triposr for that category
# (see pipeline/select.py's OBJECT_BACKEND_PREFERENCE), and this file already
# has exactly one real-GPU-backend test (triposr, above) for the documented
# reason that a second real GPU backend loading in the same process has hung
# this whole machine before. Confirmed by hand instead: a real browser
# session against a live `printable serve` + vlm-server correctly showed
# "design spec sheet" classified and routed into the same _run_spec_sheet
# path test_submit_spec_job_end_to_end exercises directly -- the CPU-only
# monkeypatched auto-mode tests above cover the endpoint's wiring.
