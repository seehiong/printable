"""HTTP client for the external VLM spec-extraction service (see ROADMAP.md)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

DEFAULT_VLM_URL = "http://localhost:8082"
# The server's /extract does one metadata call plus one call per view (each
# with up to two possible repair retries as of 2026-09-02, up from one --
# see ~/vlm-server/server.py's MAX_REPAIR_ATTEMPTS), and the resident model
# as of this writing (Qwen3.8-27B) reasons heavily before answering --
# single calls have run past 1800 completion tokens at ~20 tok/s, so
# four-plus sequential calls can comfortably exceed two minutes even
# without a retry. Repair calls specifically now allow up to 9000 tokens
# (~450s at that rate), also up from the first pass's 6000 -- a real
# degraded run (a hard sheet burning every attempt across the metadata
# call, two named views, and the primary-view fallback) can now chain
# through around 10 model calls in the worst case, several of them at the
# larger repair budget. /classify is a single call and usually much
# faster, but shares the same generous timeout since the same model's
# reasoning length is unpredictable per-call.
DEFAULT_TIMEOUT = 1500.0


def _post_image(endpoint: str, image_path: Path, vlm_url: str, timeout: float) -> dict[str, Any]:
    image_path = Path(image_path)
    try:
        with httpx.Client(timeout=timeout) as client, open(image_path, "rb") as f:
            resp = client.post(
                f"{vlm_url}/{endpoint}", files={"image": (image_path.name, f, "image/*")}
            )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"could not reach VLM server at {vlm_url} ({exc}); is it running? "
            "see ROADMAP.md's Phase 1 for how to start it"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"VLM server at {vlm_url} timed out after {timeout}s") from exc

    if resp.status_code != 200:
        raise RuntimeError(f"VLM server returned {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def fetch_design_spec(
    image_path: Path,
    vlm_url: str = DEFAULT_VLM_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST image_path to <vlm_url>/extract, return the parsed design_spec.json."""
    return _post_image("extract", image_path, vlm_url, timeout)


def classify_image(
    image_path: Path,
    vlm_url: str = DEFAULT_VLM_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """POST image_path to <vlm_url>/classify, return {category, confidence, rationale}."""
    return _post_image("classify", image_path, vlm_url, timeout)
