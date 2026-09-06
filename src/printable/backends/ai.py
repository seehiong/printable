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
import os
import sys
import sysconfig
import tempfile
import warnings
from pathlib import Path

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


def _ensure_rocm_sdk_ld_library_path() -> None:
    """Add TheRock's scattered rocm-sdk lib dirs to LD_LIBRARY_PATH.

    Only `hy3dpaint`'s native extensions (custom_rasterizer,
    mesh_inpaint_processor) need this -- torch and hy3dshape's own compiled
    bits resolve fine without it. Confirmed that mutating os.environ here,
    after the process has already started and before the first `import` of
    those extensions, is sufficient: unlike the initial executable's own
    DT_NEEDED libraries (resolved once at exec()), dlopen() -- what
    Python's import machinery uses for compiled extension modules --
    re-reads LD_LIBRARY_PATH from the current environment on each call, so
    no subprocess/re-exec is needed. Verified directly (empty
    LD_LIBRARY_PATH at process start, set here, extension import succeeds).

    No-op on any stack that doesn't lay out packages this way (e.g. a
    non-TheRock ROCm install, or a future TheRock layout change) --
    `hy3dpaint` will then fail with a clear ImportError instead, same as
    running the commands in docs/SETUP.md by hand would.
    """
    site_packages = Path(sysconfig.get_paths()["purelib"])
    rocm_core = site_packages / "_rocm_sdk_core"
    if not rocm_core.is_dir():
        return
    candidates = [
        site_packages / "torch" / "lib",
        rocm_core / "lib",
        rocm_core / "lib" / "host-math" / "lib",
        rocm_core / "lib" / "rocm_sysdeps" / "lib",
        rocm_core / "lib" / "llvm" / "lib",
        site_packages / "_rocm_sdk_libraries" / "lib",
    ]
    new_dirs = [str(p) for p in candidates if p.is_dir()]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    combined = ":".join(new_dirs + ([existing] if existing else []))
    os.environ["LD_LIBRARY_PATH"] = combined


def _shim_basicsr_functional_tensor() -> None:
    """`basicsr` (a `realesrgan` dependency, pulled in by hy3dpaint's
    super-resolution step) imports a private torchvision module removed in
    newer torchvision releases -- `ModuleNotFoundError:
    torchvision.transforms.functional_tensor`. docs/SETUP.md documents
    patching basicsr's own site-packages file directly, but that edit
    doesn't survive a fresh venv or a `basicsr` reinstall. Inject a
    compatibility shim into sys.modules instead, so the fix lives in
    printable's own code.
    """
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    try:
        import types

        import torchvision.transforms.functional as F
    except ImportError:
        return

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = F.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


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

    By default, only the shape model runs: texture is discarded on STL
    export anyway, so skipping the paint stage saves real time (the paint
    pipeline itself is another ~1.5-2 min). `--opt texture=true` opts into
    running it too, via `hy3dpaint` (a separate on-disk checkout, not a
    pip package -- see `_load_paint()`); when it does, the STL is still
    built from the plain shape mesh, but a real PBR-textured preview GLB
    (`output.glb` alongside `output.stl`) gets written from `hy3dpaint`'s
    own remeshed+UV-unwrapped output, the same "export-glb" mechanism
    TripoSR's cheap vertex-color path already uses -- see
    `GenerationResult.color_mesh`.

    2.1's paint pipeline used to be additionally blocked on this hardware
    by a custom_rasterizer GPU page-fault; no longer reproducible as of
    the ROCm 10.0 migration (see docs/SETUP.md's Texture painting
    section), which is what made wiring this in worthwhile.

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
        self._paint_pipe = None

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

    def _hunyuan3d_root(self) -> Path:
        """Locate the on-disk Hunyuan3D-2.1 checkout from the installed
        `hy3dshape` package, rather than hardcoding a path -- `hy3dshape`
        is installed editable (`pip install -e hy3dshape`, see
        docs/SETUP.md), so `hy3dshape.__file__` is
        `<checkout>/hy3dshape/hy3dshape/__init__.py`; three parents up is
        the checkout root, where the sibling `hy3dpaint/` directory lives.
        """
        import hy3dshape

        return Path(hy3dshape.__file__).resolve().parent.parent.parent

    def _load_paint(self, max_num_view: int, resolution: int):
        if self._paint_pipe is not None:
            return self._paint_pipe
        ok, reason = self.available()
        if not ok:
            raise BackendUnavailable(f"hunyuan3d: {reason}")

        hy3dpaint_dir = self._hunyuan3d_root() / "hy3dpaint"
        if not hy3dpaint_dir.is_dir():
            raise BackendUnavailable(
                "hunyuan3d: hy3dpaint/ not found next to hy3dshape -- "
                "texture painting needs the full Hunyuan3D-2.1 checkout "
                "with custom_rasterizer and DifferentiableRenderer built "
                "(see docs/SETUP.md's Texture painting section)"
            )
        if str(hy3dpaint_dir) not in sys.path:
            sys.path.insert(0, str(hy3dpaint_dir))

        _ensure_rocm_sdk_ld_library_path()
        _shim_basicsr_functional_tensor()

        try:
            from textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        except ImportError as exc:
            raise BackendUnavailable(
                "hunyuan3d: hy3dpaint not fully set up -- custom_rasterizer/"
                "DifferentiableRenderer likely not built (see docs/SETUP.md's "
                "Texture painting section)"
            ) from exc

        config = Hunyuan3DPaintConfig(max_num_view=max_num_view, resolution=resolution)
        # Both default to paths that are relative to a CWD the caller
        # doesn't control (one repo-root-relative, one hy3dpaint-relative
        # -- a real upstream inconsistency, see docs/SETUP.md). Pin both
        # to absolute paths derived from the checkout root so this works
        # regardless of printable's own working directory.
        config.multiview_cfg_path = str(hy3dpaint_dir / "cfgs" / "hunyuan-paint-pbr.yaml")
        config.realesrgan_ckpt_path = str(hy3dpaint_dir / "ckpt" / "RealESRGAN_x4plus.pth")

        log.info("loading Hunyuan3D paint pipeline (first run downloads weights)")
        self._paint_pipe = Hunyuan3DPaintPipeline(config)
        return self._paint_pipe

    def _paint(self, mesh: trimesh.Trimesh, image, o: dict) -> trimesh.Trimesh:
        """Run hy3dpaint on `mesh`, return a self-contained textured mesh.

        `mesh` here is the plain shape output, before printable's own
        repair/prep -- this is a preview mesh with its own remeshed
        topology (hy3dpaint's own quadric decimation + UV unwrap), not the
        mesh that becomes the STL. Raises on failure rather than degrading
        silently: `--opt texture=true` was asked for explicitly, so a
        missing GLB should be loud, not a quiet no-op (this was exactly
        the confusion hit before this backend read the option at all).
        """
        pipe = self._load_paint(
            max_num_view=int(o.get("paint_max_views", 6)),
            resolution=int(o.get("paint_resolution", 512)),
        )

        with tempfile.TemporaryDirectory(prefix="printable_hunyuan3d_paint_") as tmp:
            tmp_dir = Path(tmp)
            mesh_path = tmp_dir / "shape.obj"
            mesh.export(mesh_path)

            image_path = tmp_dir / "source.png"
            image.convert("RGB").save(image_path)

            output_path = tmp_dir / "textured.obj"
            pipe(
                mesh_path=str(mesh_path),
                image_path=str(image_path),
                output_mesh_path=str(output_path),
                # Blender's bpy has no wheel for this venv's Python (see
                # docs/SETUP.md), so hy3dpaint's own OBJ->GLB step always
                # silently no-ops -- trimesh below does that conversion
                # for real instead, so don't bother asking hy3dpaint for it.
                save_glb=False,
            )
            return trimesh.load(output_path, force="mesh")

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

        color_mesh = None
        if o.get("texture", False):
            # Real PBR texture, not TripoSR's per-vertex trick -- has its
            # own remeshed topology (see _paint()'s docstring), so it's a
            # separate preview mesh from `mesh` above, not the same object
            # with color attached. `mesh`/the STL are unaffected either
            # way: texture painting only ever feeds the GLB preview.
            color_mesh = self._paint(mesh, image, o)

        return GenerationResult(
            mesh=mesh,
            backend=Backend.HUNYUAN3D,
            has_color=color_mesh is not None,
            color_mesh=color_mesh,
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
