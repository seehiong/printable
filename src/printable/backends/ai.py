"""AI image-to-3D backends: Hunyuan3D, TripoSR.

Each wraps an upstream repo that must be installed separately. These are
research codebases with heavy, conflicting dependencies, so adapting to them
beats vendoring them. See docs/SETUP.md for install steps.

TRELLIS and SPAR3D used to live here too. Dropped: both consistently
fragmented on real character images in practice (TRELLIS's known
disconnected-shell problem; SPAR3D breaking into multiple bodies on every
real test image tried), while Hunyuan3D was the one backend that reliably
came out watertight and single-body. Not worth the install-matrix cost
(each was its own multi-step native-extension saga) for output that needed
explaining away every time. See docs/SETUP.md's history if reviving either
is ever worth revisiting.

Design note: every adapter converts to a plain trimesh.Trimesh as early as
possible. Downstream stages never see model-specific types.
"""

from __future__ import annotations

import contextlib
import io
import logging
import warnings

import numpy as np
import trimesh

from printable.backends.base import BackendUnavailable, GeometryBackend, registry
from printable.backends.preprocess import prepare_image
from printable.types import Backend, GenerationRequest, GenerationResult

log = logging.getLogger(__name__)


@contextlib.contextmanager
def _quiet_import():
    """Suppress third-party import-time chatter.

    transformers/torch emit FutureWarnings about deprecated pytree
    registration on every import of these heavy packages. None of this is
    actionable for printable's users -- it just spams `printable backends`
    and the start of every generate call.
    """
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", FutureWarning)
        yield


def _torch_device() -> str:
    """Pick an accelerator. ROCm reports itself as 'cuda' to torch."""
    try:
        import torch
    except ImportError as exc:
        raise BackendUnavailable("torch is not installed") from exc

    if torch.cuda.is_available():
        log.info("using accelerator: %s", torch.cuda.get_device_name(0))
        return "cuda"
    log.warning("no accelerator found; falling back to CPU (very slow)")
    return "cpu"


def _to_trimesh(obj) -> trimesh.Trimesh:
    """Coerce whatever a model returns into a trimesh.Trimesh."""
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        return trimesh.util.concatenate(list(obj.geometry.values()))

    # Most repos return an object exposing .vertices / .faces as tensors.
    verts = getattr(obj, "vertices", None)
    faces = getattr(obj, "faces", None)
    if verts is None or faces is None:
        raise TypeError(f"cannot convert {type(obj)!r} to a mesh")

    def arr(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    return trimesh.Trimesh(vertices=arr(verts), faces=arr(faces), process=False)


class Hunyuan3DBackend(GeometryBackend):
    """Tencent Hunyuan3D 2.1. Splits shape generation from texture synthesis.

    Only the shape model matters for printing: texture is discarded on STL
    export, so the paint stage is skipped and the time saved. (2.1's paint
    pipeline is also blocked on this hardware -- its custom_rasterizer
    native extension faults the GPU on real render calls; see docs/SETUP.md.
    Shape generation doesn't touch that extension at all, so it's unaffected.)

    2.1's shape model (hunyuan3d-dit-v2-1, 7.37GB) is a distinct, larger
    checkpoint from 2.0's (hunyuan3d-dit-v2-0, 4.6GB) -- confirmed by
    comparing actual downloaded file sizes, not documentation claims. This
    isn't the same model under a new version number.
    """

    name = "hunyuan3d"
    requires_gpu = True
    needs_preprocess = True
    image_size = 518
    DEFAULT_MODEL = "tencent/Hunyuan3D-2.1"

    def __init__(self) -> None:
        self._pipe = None

    def available(self) -> tuple[bool, str]:
        with _quiet_import():
            try:
                import torch  # noqa: F401
            except ImportError:
                return False, "torch not installed"
            try:
                import hy3dshape  # noqa: F401
            except ImportError:
                return False, "Hunyuan3D 2.1 not installed (see docs/SETUP.md)"
        return True, "ok"

    def _load(self, model_id: str, subfolder: str | None):
        if self._pipe is not None:
            return self._pipe
        ok, reason = self.available()
        if not ok:
            raise BackendUnavailable(f"hunyuan3d: {reason}")

        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

        log.info("loading Hunyuan3D %s (first run downloads weights)", model_id)
        kwargs = {"subfolder": subfolder} if subfolder else {}
        pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_id, **kwargs)
        try:
            pipe.to(_torch_device())
        except AttributeError:
            pass  # some pipeline versions place themselves at load time
        self._pipe = pipe
        return pipe

    def warmup(self) -> None:
        self._load(self.DEFAULT_MODEL, None)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        o = request.options
        # subfolder selects a variant, e.g. "hunyuan3d-dit-v2-mini" for the
        # small model.
        pipe = self._load(o.get("model", self.DEFAULT_MODEL), o.get("subfolder"))

        image = request.image
        if image is None:  # direct call, bypassing run()'s preprocess stage
            image = prepare_image(
                request.image_path,
                size=int(o.get("image_size", self.image_size)),
                strip_background=bool(o.get("remove_bg", True)),
            )

        try:
            import torch

            generator = torch.Generator().manual_seed(request.seed)
        except ImportError:
            generator = None

        result = pipe(
            image=image,
            num_inference_steps=int(o.get("steps", 30)),
            guidance_scale=float(o.get("guidance", 5.5)),
            octree_resolution=int(o.get("octree_resolution", 256)),
            generator=generator,
        )

        # 2.1's pipeline returns List[List[trimesh.Trimesh]] (one inner list
        # per input image); 2.0 returned a flat list. Unwrap either shape.
        mesh_out = result[0] if isinstance(result, (list, tuple)) else result
        mesh_out = mesh_out[0] if isinstance(mesh_out, (list, tuple)) else mesh_out
        mesh = _to_trimesh(mesh_out)

        if o.get("clean", True):
            # Hunyuan ships cleaners tuned to its own extractor artefacts.
            try:
                from hy3dshape.postprocessors import DegenerateFaceRemover, FloaterRemover

                mesh = _to_trimesh(FloaterRemover()(mesh))
                mesh = _to_trimesh(DegenerateFaceRemover()(mesh))
            except Exception as exc:  # noqa: BLE001 - optional helpers
                log.debug("hunyuan cleaners unavailable: %s", exc)

        return GenerationResult(
            mesh=mesh,
            backend=Backend.HUNYUAN3D,
            metadata={"model": o.get("model", self.DEFAULT_MODEL), "seed": request.seed},
        )


class TripoSRBackend(GeometryBackend):
    """TripoSR. Fast feed-forward reconstruction, the draft-mode option.

    Lower quality than Hunyuan3D, but seconds instead of minutes and far
    fewer exotic dependencies, which makes it the right first spike when
    proving out a new accelerator stack.
    """

    name = "triposr"
    requires_gpu = True
    needs_preprocess = True
    image_size = 512
    DEFAULT_MODEL = "stabilityai/TripoSR"

    def __init__(self) -> None:
        self._model = None

    def available(self) -> tuple[bool, str]:
        with _quiet_import():
            try:
                import torch  # noqa: F401
            except ImportError:
                return False, "torch not installed"
            try:
                import tsr  # noqa: F401
            except ImportError:
                return False, "TripoSR not installed (see docs/SETUP.md)"
        return True, "ok"

    def _load(self, model_id: str):
        if self._model is not None:
            return self._model
        ok, reason = self.available()
        if not ok:
            raise BackendUnavailable(f"triposr: {reason}")

        from tsr.system import TSR

        log.info("loading TripoSR %s", model_id)
        model = TSR.from_pretrained(
            model_id, config_name="config.yaml", weight_name="model.ckpt"
        )
        model.renderer.set_chunk_size(8192)
        model.to(_torch_device())
        self._model = model
        return model

    def warmup(self) -> None:
        self._load(self.DEFAULT_MODEL)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        o = request.options
        model = self._load(o.get("model", self.DEFAULT_MODEL))

        image = request.image
        if image is None:  # direct call, bypassing run()'s preprocess stage
            image = prepare_image(
                request.image_path,
                size=int(o.get("image_size", self.image_size)),
                strip_background=bool(o.get("remove_bg", True)),
            )

        # TripoSR wants RGB composited on grey, not RGBA: transparent pixels
        # read as black and leave a dark halo in the reconstruction.
        rgba = np.asarray(image).astype(np.float32) / 255.0
        alpha = rgba[:, :, 3:4]
        composited = rgba[:, :, :3] * alpha + 0.5 * (1.0 - alpha)

        import torch
        from PIL import Image as PILImage

        pil = PILImage.fromarray((composited * 255).astype(np.uint8))
        device = _torch_device()

        # has_vertex_color runs one extra, cheap query against the
        # already-loaded triplane decoder at just the mesh's surface
        # vertices -- no full re-render, no extra model. trimesh.Trimesh's
        # own constructor turns the returned per-vertex RGB into a real
        # ColorVisuals, which _to_trimesh's isinstance(Trimesh) fast path
        # then preserves untouched.
        want_texture = bool(o.get("texture", False))
        with torch.no_grad():
            codes = model([pil], device=device)
            meshes = model.extract_mesh(
                codes, has_vertex_color=want_texture, resolution=int(o.get("mc_resolution", 256))
            )

        return GenerationResult(
            mesh=_to_trimesh(meshes[0]),
            backend=Backend.TRIPOSR,
            has_color=want_texture,
            metadata={"model": o.get("model", self.DEFAULT_MODEL)},
        )


registry.register("hunyuan3d", Hunyuan3DBackend)
registry.register("triposr", TripoSRBackend)
