"""HTTP client for a local ComfyUI server: text description -> PNG bytes,
feeding printable's own generate pipeline (see README's "Text to image"
section)."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from printable.art_source import embed_image_params

DEFAULT_COMFYUI_URL = "http://localhost:8188"
# ComfyUI's own bundled "SDXL Turbo" example workflow (checkpoints/
# sd_xl_turbo_1.0_fp16.safetensors) -- 1-4 steps is enough for that
# checkpoint, and a plain product-photo-style prompt (single object, flat
# background) is exactly the profile printable's own generate pipeline
# wants as input. Pass a different checkpoint= if some other SDXL-family
# model is installed instead.
DEFAULT_CHECKPOINT = "sd_xl_turbo_1.0_fp16.safetensors"
DEFAULT_NEGATIVE_PROMPT = "text, watermark, blurry, low quality, deformed"
DEFAULT_TIMEOUT = 120.0
_POLL_INTERVAL = 0.5


@dataclass
class GeneratedImage:
    # PNG bytes with `params` embedded as a text chunk (art_source.PNG_KEY).
    png: bytes
    params: dict[str, Any]


def _workflow(
    prompt: str,
    negative_prompt: str,
    *,
    checkpoint: str,
    steps: int,
    cfg: float,
    sampler: str,
    width: int,
    height: int,
    seed: int,
) -> dict[str, Any]:
    """API-format graph equivalent to ComfyUI's bundled
    sdxlturbo_example.json workflow (CheckpointLoaderSimple ->
    SDTurboScheduler + SamplerCustom -> VAEDecode -> SaveImage), built by
    hand against the node definitions rather than round-tripped through
    that UI-format JSON, which ComfyUI's HTTP API doesn't accept directly."""
    return {
        "20": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["20", 1]}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["20", 1]},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": sampler}},
        "22": {
            "class_type": "SDTurboScheduler",
            "inputs": {"model": ["20", 0], "steps": steps, "denoise": 1.0},
        },
        "13": {
            "class_type": "SamplerCustom",
            "inputs": {
                "model": ["20", 0],
                "add_noise": True,
                "noise_seed": seed,
                "cfg": cfg,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "sampler": ["14", 0],
                "sigmas": ["22", 0],
                "latent_image": ["5", 0],
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["20", 2]}},
        "27": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "printable"},
        },
    }


def generate_image(
    prompt: str,
    *,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    steps: int = 4,
    cfg: float = 1.0,
    sampler: str = "euler_ancestral",
    width: int = 1024,
    height: int = 1024,
    seed: int | None = None,
    checkpoint: str = DEFAULT_CHECKPOINT,
    comfyui_url: str = DEFAULT_COMFYUI_URL,
    timeout: float = DEFAULT_TIMEOUT,
) -> GeneratedImage:
    """Queue a text-to-image generation on a running ComfyUI server, wait
    for it, and return the PNG plus the settings that made it.

    seed=None (the default) picks a random seed so repeated calls with the
    same prompt produce different images; the one actually used is in the
    result's params (and embedded in the PNG), so any result can be
    reproduced exactly.

    Raises RuntimeError if ComfyUI isn't reachable, rejects the workflow
    (e.g. checkpoint isn't the filename of an installed model), or
    generation itself fails.
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler": sampler,
        "width": width,
        "height": height,
        "checkpoint": checkpoint,
    }
    workflow = _workflow(
        prompt,
        negative_prompt,
        checkpoint=checkpoint,
        steps=steps,
        cfg=cfg,
        sampler=sampler,
        width=width,
        height=height,
        seed=seed,
    )

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{comfyui_url}/prompt", json={"prompt": workflow})
            data = resp.json()
            if resp.status_code != 200:
                raise RuntimeError(
                    f"ComfyUI rejected the workflow: {data.get('error')} "
                    f"(node_errors: {data.get('node_errors')})"
                )
            prompt_id = data["prompt_id"]

            deadline = time.monotonic() + timeout
            record: dict[str, Any] = {}
            while time.monotonic() < deadline:
                history = client.get(f"{comfyui_url}/history/{prompt_id}").json()
                if prompt_id in history:
                    record = history[prompt_id]
                    break
                time.sleep(_POLL_INTERVAL)
            else:
                raise RuntimeError(f"ComfyUI generation timed out after {timeout}s")

            status = record.get("status") or {}
            if status.get("status_str") == "error":
                raise RuntimeError(f"ComfyUI generation failed: {status.get('messages')}")

            images = []
            for node_output in record.get("outputs", {}).values():
                images.extend(node_output.get("images", []))
            if not images:
                raise RuntimeError("ComfyUI finished but produced no image output")
            image_info = images[0]

            img_resp = client.get(
                f"{comfyui_url}/view",
                params={
                    "filename": image_info["filename"],
                    "subfolder": image_info.get("subfolder", ""),
                    "type": image_info.get("type", "output"),
                },
            )
    except httpx.ConnectError as exc:
        raise RuntimeError(
            f"could not reach ComfyUI at {comfyui_url} ({exc}); is it running? "
            "see README's Text to image section"
        ) from exc
    except httpx.TimeoutException as exc:
        raise RuntimeError(f"ComfyUI at {comfyui_url} timed out after {timeout}s") from exc

    return GeneratedImage(png=embed_image_params(img_resp.content, params), params=params)
