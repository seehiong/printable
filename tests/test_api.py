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
from PIL import Image

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


def test_geometry_cues_downloadable_after_completion(client, sample_image, monkeypatch):
    """has_geometry_cues + /depth + /normal: the actual estimator needs the
    `depth` extra (transformers>=4.49), which conflicts with TripoSR's
    transformers==4.35.0 pin (see docs/SETUP.md) and so is never installed
    alongside a GPU backend -- monkeypatch estimate_depth_normal itself so
    this test exercises the real save/serve path deterministically,
    independent of whether a real depth model is importable here."""
    import numpy as np

    from printable.types import DepthNormalResult

    fake = DepthNormalResult(
        depth=np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16),
        normal=np.zeros((16, 16, 3), dtype=np.float32),
        source="fake-for-test",
    )
    monkeypatch.setattr("printable.backends.depth.available", lambda: (True, "ok"))
    monkeypatch.setattr("printable.backends.depth.estimate_depth_normal", lambda image: fake)

    job_id = _submit(client, sample_image, opt="geometry_cues=true")
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "done", status
    assert status["has_geometry_cues"] is True, status

    depth_resp = client.get(f"/api/jobs/{job_id}/depth")
    assert depth_resp.status_code == 200
    assert depth_resp.headers["content-type"] == "image/png"
    depth_img = Image.open(io.BytesIO(depth_resp.content))
    assert depth_img.size == (16, 16)

    normal_resp = client.get(f"/api/jobs/{job_id}/normal")
    assert normal_resp.status_code == 200
    assert normal_resp.headers["content-type"] == "image/png"


def test_geometry_cues_endpoints_404_when_not_requested(client, sample_image):
    job_id = _submit(client, sample_image)
    status = _wait_for_terminal(client, job_id)
    assert status["status"] == "done", status
    assert status["has_geometry_cues"] is False

    assert client.get(f"/api/jobs/{job_id}/depth").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/normal").status_code == 404


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
# second real GPU backend (a since-dropped one, while triposr's above was
# already loaded) in the same process hung this entire machine hard enough
# to need a physical reboot -- not a slow test, an OS-level lockup from
# switching GPU backends within one process. Confirmed with that backend,
# but the underlying finding still applies to today's TripoSR/Hunyuan3D
# pairing -- see docs/SETUP.md's Troubleshooting section. Do not add a
# second real-GPU-backend test to this file without reading that warning
# first.


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


def _wait_for_status(client: TestClient, job_id: str, want: tuple[str, ...], timeout: float) -> dict:
    deadline = time.time() + timeout
    status = None
    while time.time() < deadline:
        status = client.get(f"/api/jobs/{job_id}").json()
        if status["status"] in want:
            return status
        time.sleep(0.5)
    raise TimeoutError(f"job {job_id} did not reach {want} within {timeout}s (last: {status})")


def test_submit_spec_job_end_to_end(client):
    """Two-phase since 2026-09-02: extraction pauses at
    "awaiting_confirmation" for a human to look at the actual crops
    (GET .../view/{name}) before the expensive generation step runs --
    see api/jobs.py's Job.status docstring for why. This test plays both
    halves: confirm immediately, like a UI that shows the crops and the
    user clicks straight through.
    """
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

    # This sheet's extraction time is genuinely bimodal, not just slow: one
    # real run succeeded in 161s, another spent the client's entire 1500s
    # timeout (vlm_client.DEFAULT_TIMEOUT) retrying box checks before
    # raising -- both are real VLM behavior on a hard sheet, not a bug. The
    # wait here has to clear that ceiling, or it fails on its own polling
    # timeout before the job ever reaches a real terminal state, which
    # reads as "broken" when it's actually just "still extracting."
    status = _wait_for_status(client, job_id, ("awaiting_confirmation", "error"), timeout=1560.0)
    assert status["status"] == "awaiting_confirmation", status

    spec = status["design_spec"]
    assert spec is not None
    assert {"dimensions_mm", "print_constraints", "views"} <= spec.keys()
    assert spec["views"], "expected at least one named view to review"

    # The actual point of pausing here: the crop is a real, fetchable
    # image, not just a name in the spec -- a UI can show it before the
    # user commits to generation.
    one_view = next(iter(spec["views"]))
    view_resp = client.get(f"/api/jobs/{job_id}/view/{one_view}")
    assert view_resp.status_code == 200
    assert view_resp.headers["content-type"] == "image/png"
    crop = Image.open(io.BytesIO(view_resp.content))
    assert crop.size[0] > 0 and crop.size[1] > 0

    confirm_resp = client.post(f"/api/jobs/{job_id}/confirm")
    assert confirm_resp.status_code == 202, confirm_resp.text

    # Letting the job actually finish avoids returning while its background
    # ThreadPoolExecutor worker (jobs.py) is still mid-generate; racing
    # that against interpreter teardown at the end of the test session
    # risks corrupting the process (seen as a "corrupted double-linked
    # list" glibc abort after a run that timed out here).
    status = _wait_for_status(client, job_id, ("done", "error"), timeout=120.0)
    assert status["status"] == "done", status

    stl_resp = client.get(f"/api/jobs/{job_id}/stl")
    assert stl_resp.status_code == 200
    mesh = trimesh.load(io.BytesIO(stl_resp.content), file_type="stl")
    assert len(mesh.faces) > 0


def test_confirm_rejected_when_not_awaiting_confirmation(client, sample_image):
    """/confirm only makes sense for a spec-sheet job paused for review --
    calling it on an ordinary job (never reaches "awaiting_confirmation"
    at all) should fail clearly, not silently do something."""
    job_id = _submit(client, sample_image)
    _wait_for_terminal(client, job_id)
    resp = client.post(f"/api/jobs/{job_id}/confirm")
    assert resp.status_code == 400, resp.text


def test_view_endpoint_rejects_unknown_view_name(client, sample_image):
    """Whitelisted against the job's own design_spec views, not an open
    filename read -- a job with no design_spec at all (never went through
    spec extraction) must reject every name, not serve whatever happens
    to be sitting in its directory (input image, result.stl, ...)."""
    job_id = _submit(client, sample_image)
    _wait_for_terminal(client, job_id)
    resp = client.get(f"/api/jobs/{job_id}/view/front")
    assert resp.status_code == 404
    # Confirm it's really the whitelist rejecting this, not a missing-file
    # 404 for an unrelated reason -- "input" matches this job's own actual
    # saved image filename stem, so this specifically checks the endpoint
    # doesn't just serve anything under job.dir by name.
    resp2 = client.get(f"/api/jobs/{job_id}/view/input")
    assert resp2.status_code == 404


def test_confirm_resumes_and_completes_generation(client, sample_image, monkeypatch):
    """Fast, deterministic counterpart to test_submit_spec_job_end_to_end
    above: mocks the VLM call itself so this exercises the actual
    extract -> awaiting_confirmation -> view -> confirm -> generate wiring
    on every run, CPU-only, independent of a live VLM server's latency or
    its (real, documented above) occasional total extraction failure on a
    hard sheet -- that test proves the VLM integration works when it's up;
    this one proves the job-state-machine plumbing around it always does.
    """
    img = Image.open(sample_image)
    w, h = img.size
    fake_spec = {
        "title": "Test Widget",
        "dimensions_mm": {"total_height": 40.0},
        "print_constraints": {"min_wall_thickness_mm": 0.8},
        "views": {"front": {"box_2d": [0, 0, 1000, 1000]}},
    }
    monkeypatch.setattr("printable.vlm_client.fetch_design_spec", lambda *a, **k: fake_spec)

    with open(sample_image, "rb") as f:
        resp = client.post(
            "/api/jobs/spec",
            files={"image": ("sample.png", f, "image/png")},
            data={"backend": "lithophane"},
        )
    assert resp.status_code == 202, resp.text
    job_id = resp.json()["job_id"]

    status = _wait_for_status(client, job_id, ("awaiting_confirmation", "error"), timeout=30.0)
    assert status["status"] == "awaiting_confirmation", status
    assert status["design_spec"]["views"] == fake_spec["views"]

    # box_2d [0, 0, 1000, 1000] covers the whole normalized 0-1000 range,
    # so the crop should be the entire sheet unchanged.
    view_resp = client.get(f"/api/jobs/{job_id}/view/front")
    assert view_resp.status_code == 200
    assert view_resp.headers["content-type"] == "image/png"
    crop = Image.open(io.BytesIO(view_resp.content))
    assert crop.size == (w, h)

    confirm_resp = client.post(f"/api/jobs/{job_id}/confirm")
    assert confirm_resp.status_code == 202, confirm_resp.text

    status = _wait_for_status(client, job_id, ("done", "error"), timeout=30.0)
    assert status["status"] == "done", status

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
