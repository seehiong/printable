# Setup

The CPU pipeline (lithophane / heightmap) needs nothing beyond the base install. The AI backends each need their upstream repo installed separately, because they are research codebases with heavy, mutually incompatible pins.

## Base install

```bash
uv sync
source .venv/bin/activate
# Windows: .venv\Scripts\activate
```

Verify:

```bash
printable backends
printable generate assets/examples/character.png --backend lithophane --size 100
```

Optional extras — get `preprocess` before touching the AI backends, it's
what strips photo backgrounds (`rembg`), which measurably improves geometry:

```bash
uv sync --extra repair --extra preprocess   # or --all-extras for both
```

---

## ROCm on Strix Halo (Ryzen AI Max+ 395)

Do this before installing any model. It is the step most likely to consume
a day. **Verified on Ubuntu 26.04 LTS**, real hardware — on a different
release, check AMD's ROCm support matrix first, a newer LTS can lag ROCm
packaging by months.

### 1. Allocate GPU memory in BIOS

**Set `UMA Frame Buffer Size` to 512 MB** — not a large static allocation.
Confirmed by hand on this hardware; see
[write-up](https://seehiong.github.io/posts/2026/08/running-llama.cpp-on-amd-strix-halo/).

<details>
<summary>Why 512 MB, not 48–96 GB</summary>

Strix Halo's unified memory has three layers. **VRAM** (this BIOS setting)
is display framebuffer only — not what GPU compute draws from. **GTT**
(~115 GB on a 128 GB system) is what the amdgpu driver actually exposes to
the GPU for compute, dynamically, out of system LPDDR5X. **TTM** sits
between the two. A large static BIOS allocation permanently partitions
memory away from the OS and isn't what Linux GPU compute uses — GTT already
provides far more. If a ceiling is ever hit, `amdgpu.gttsize` is the kernel
parameter to check, not a bigger BIOS number.

</details>

### 2. Install ROCm and PyTorch

**Pick ONE of the two commands below, not both.**

**→ On `gfx1151` (Strix Halo), Ubuntu 26.04 LTS:**

```bash
uv pip install --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ torch torchvision
```

Verify:

```bash
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True <your GPU name>`. If `False`, see Troubleshooting.

**Never run bare `uv sync` after this** — torch lives outside `printable`'s
lockfile and `uv sync` will silently remove it. Use `uv run --no-sync ...`
for everything from here on.

<details>
<summary>Why this exact command, not the generic PyTorch ROCm wheels</summary>

The generic multi-arch wheels from `download.pytorch.org` segfault on
`gfx1151` on any GPU memory access — a real, confirmed upstream bug
([ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853), filed against
this exact chip). The command above is
[TheRock](https://github.com/ROCm/TheRock)'s gfx1151-native nightly build,
which ships its own self-contained ROCm kernels and sidesteps the bug.

**This is a nightly index — it changes daily, and this URL may not last.**
Verified working as of 2026-08-29 — it's actually frozen at build
`7.13.0a20260513` regardless of when you install (not truly rolling), which
is fine for now but means it won't self-update either.

**Tried moving to TheRock's newer index, reverted (2026-08-29).**
`nightly.repo.amd.com/rocm/whl-next/` with `torch[device-gfx1151]` syntax —
which TheRock's own repo points at as the current replacement — installs
cleanly and `torch.cuda.is_available()` works. But building any native
extension from source against it (`torchmcubes`, required for TripoSR)
fails inside torch's own `LoadHIP.cmake`: `HIP_VERSION_MAJOR`/`MINOR` come
back empty from TheRock's new split `hip-lang`/`hip` CMake packages,
producing `math cannot parse the expression: "( * 100) + "`. Real,
reproducible, upstream — not a local misconfiguration (also needed
`rocm-sdk-devel` + `rocm-sdk init` just to get `hip-lang-config.cmake` to
exist at all, a separate missing-package issue from the version-parsing
one). Consistent with this being a same-day nightly (`10.1.0a20260829`) —
worth retrying in a few weeks once gfx1151 support on the new index has
had time to stabilize, not on day 3 of a major version-scheme change. Real
verified upside if it does end up working: the
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` flag this doc requires
throughout appeared to become unnecessary on the new build (flash
attention worked unflagged, and faster than the old build even *with* the
flag set) — checked via an isolated torch-only smoke test before hitting
the `torchmcubes` build failure above, so `hunyuan3d`/`triposr` real-model
behavior was never actually confirmed either way.

If the command above 404s, breaks, or stops resolving, that's expected for
a nightly build going stale, not a sign something else is wrong. Check
[TheRock's releases page](https://github.com/ROCm/TheRock/releases) or
[RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for
the current index, and re-attempt the newer index above once there's
reason to believe the CMake issue has been fixed upstream.

</details>

**→ Any other GPU (not `gfx1151`):**

```bash
hipconfig --version   # e.g. 7.1.52801 -> use the rocm7.1 index below
uv pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch torchvision
```

<details>
<summary>Why match the version instead of picking one</summary>

Torch bundles its own `libamdhip64.so` matching whatever wheel index you
installed from. If that disagrees with the ROCm your system otherwise has
installed, native extensions built against system headers (TripoSR's
`torchmcubes`, below) fail at import with `undefined symbol` — a different,
less obvious failure than "no GPU found."

</details>

### 3. Prove the stack with TripoSR first

TripoSR has the fewest exotic dependencies of the GPU backends — one mesh
out of it proves the accelerator works before touching Hunyuan3D's heavier
install. Commands are in the **## TripoSR** section below (next major
heading after step 4); come back here first only if you hit a segfault.

<details>
<summary>If TripoSR segfaults on any GPU op</summary>

That's [ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853), the
generic-wheels bug step 2 already routes you around — confirmed here on two
different generic-wheel pairings before switching. If you're on the
TheRock URL from step 2, you shouldn't hit this. `HIP_VISIBLE_DEVICES=""`
forces CPU as a way to confirm the rest of the pipeline (install, build,
model download) is fine while you sort out the GPU side.

</details>

### 4. Build the ROCm CMake shim — skip if you used AMD's official `/opt/rocm` installer

**Check which one you have:**

```bash
test -d /opt/rocm && echo "official installer -- skip this step" || echo "apt (or TheRock) -- run this step"
```

CMake needs ROCm's files laid out at `/opt/rocm`; `apt` scatters them
elsewhere instead. One-time fix, copy-paste as one block — it defines
`build_rocm_shim` and calls it on the last line:

```bash
# extra -dev packages CMake needs that the base ROCm packages don't pull in
sudo apt install -y librocrand-dev libhiprand-dev libmiopen-dev libhipfft-dev \
  libhipsparse-dev librocprim-dev libhipcub-dev librocthrust-dev \
  libhipsolver-dev librocm-core-dev librocm-smi-dev

build_rocm_shim() {
  local shim="$1" clang_ver
  clang_ver=$(ls /usr/lib/rocm/llvm/lib/clang)
  mkdir -p "$shim/lib/cmake"
  ln -sfn /usr/include "$shim/include"
  ln -sfn /usr/bin "$shim/bin"
  ln -sfn /usr/lib/rocm/llvm "$shim/lib/llvm"
  ln -sfn /usr/lib/rocm/llvm/lib/clang "$shim/lib/clang"
  ln -sfn "/usr/lib/rocm/llvm/lib/clang/$clang_ver/amdgcn" "$shim/amdgcn"
  for f in /usr/lib/x86_64-linux-gnu/*.so*; do
    ln -sfn "$f" "$shim/lib/$(basename "$f")"
  done
  for d in /usr/lib/x86_64-linux-gnu/cmake/*/ /usr/share/cmake/*/; do
    ln -sfn "${d%/}" "$shim/lib/cmake/$(basename "$d")"
  done
  # Caffe2Targets.cmake references roc::hipsparselt unconditionally, but
  # Ubuntu doesn't package hipsparselt at all yet. Extensions that never
  # call into it (torchmcubes included) just need the target to exist.
  mkdir -p "$shim/lib/cmake/hipsparselt"
  cat > "$shim/lib/cmake/hipsparselt/hipsparselt-config.cmake" <<'EOF'
if(NOT TARGET roc::hipsparselt)
  add_library(roc::hipsparselt INTERFACE IMPORTED)
endif()
set(hipsparselt_FOUND TRUE)
EOF
  # amd_comgr-config.cmake derives its prefix by textually stripping path
  # components instead of resolving symlinks first, so it needs this one
  # link at the shim's *parent* directory, not inside the shim itself.
  mkdir -p "$(dirname "$shim")/lib"
  ln -sfn /usr/lib/x86_64-linux-gnu "$(dirname "$shim")/lib/x86_64-linux-gnu"
}
build_rocm_shim ~/.cache/rocm-shim/root
```

The shim directory itself is one-time — it stays on disk, never rebuild it.
Every native-extension build command further below in this doc has
`ROCM_PATH=~/.cache/rocm-shim/root` already prefixed on it, like:
`ROCM_PATH=~/.cache/rocm-shim/root uv pip install -e . ...` — that's a
per-command prefix (nothing persists in your shell), already written into
each command for you. No `.bashrc` edit needed. If you skipped this step
(official installer), just drop that one variable from each command you
copy.

---

**Before any command below, activate `printable`'s venv** (these clones
live outside the project tree; a fresh terminal won't have it active):

```bash
source /path/to/printable/.venv/bin/activate
```

## TripoSR

TripoSR ships no `pyproject.toml` or `setup.py` — upstream's own instructions
are `pip install -r requirements.txt` and run scripts from inside the
checkout. `uv pip install -e .` (or plain `pip install -e .`) fails with
*"does not appear to be a Python project"* regardless of `pip`/`uv`; that's
not a packaging regression, TripoSR has just never had install metadata.

```bash
git clone https://github.com/VAST-AI-Research/TripoSR.git
cd TripoSR
uv pip install cmake ninja scikit-build-core pybind11
```

One requirement, `torchmcubes`, compiles a HIP/CUDA extension from source via
CMake, which needs `torch` already importable — it also needs
`--no-build-isolation-package torchmcubes` so the build sees your installed
torch instead of an isolated one. Run both lines below as one command (the
`\` continues the line); on AMD's official installer, drop the
`ROCM_PATH=~/.cache/rocm-shim/root \` line entirely:

```bash
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
ROCM_PATH=~/.cache/rocm-shim/root \
PYTORCH_ROCM_ARCH=gfx1151 \
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$SITE_PACKAGES -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1" \
uv pip install --no-build-isolation-package torchmcubes -r requirements.txt
```

`PYTORCH_ROCM_ARCH` is Strix Halo's `gfx1151`; use your own GPU's arch string.
The `_GLIBCXX_USE_CXX11_ABI=1` flag works around some torch builds leaving
`TORCH_CXX_FLAGS` empty, which otherwise silently links `torchmcubes` against
the wrong C++ `std::string` ABI — that surfaces later as an `undefined symbol:
...torchCheckFail...` `ImportError`, not a build failure, so it's easy to miss
if you only check that the build succeeded.

Finally, since TripoSR has no packaging metadata, make `tsr` importable with a
`.pth` file instead of an actual install:

```bash
echo "$(pwd)" > "$SITE_PACKAGES/triposr.pth"
```

**Now restore `printable`'s own pinned versions** — the `torchmcubes` install
above just downgraded them to whatever TripoSR's `requirements.txt` wants
(confirmed happens every time, not just-in-case):

```bash
uv pip install numpy==2.5.2 pandas==3.0.5 pillow==12.3.0 rembg==2.0.81 scipy==1.18.1 tifffile==2026.8.23 trimesh==5.0.0 markupsafe==3.0.3
```

**From here on: never run bare `uv sync` or bare `uv run <anything>`.**
Both silently remove `torch`/`torchmcubes`/everything just installed, since
none of it is in `printable`'s own lockfile. Use `uv run --no-sync
<command>` instead, or just call the venv's binaries directly (`pytest`,
`python`, `printable`, no `uv run` at all — safest, can't trigger a sync).

<details>
<summary>Why <code>uv run</code> alone is just as dangerous as <code>uv sync</code></summary>

`uv run` does the identical reconciliation by default before running
whatever command you gave it — `uv run pytest`, `uv run python -c ...`, all
of them, not just literal `uv sync`. `uv sync --help` documents `--inexact`
("do not remove extraneous packages") as the *opt-out*, meaning removal is
the default. If you do get bitten (`printable backends` reports "torch not
installed" with no `uv sync` in sight), the shim and every checkout survive
on disk — reinstall torch from TheRock's index, rebuild `torchmcubes`, then
re-run the version-restore command above.

</details>

```bash
cd /path/to/printable   # back to the printable checkout

# character.png is a 4-panel turnaround sheet -- --sheet splits it, generates
# from every panel, and keeps whichever one comes out actually printable
# (not just best-framed; see the README's Usage section for why)
printable generate assets/examples/character.png --backend triposr --size 80 --sheet

# Force CPU instead (e.g. not on gfx1151, or not using TheRock's build):
HIP_VISIBLE_DEVICES="" printable generate assets/examples/character.png --backend triposr --size 80 --sheet
```

Options: `mc_resolution` (marching cubes grid, default 256), `image_size`,
`remove_bg`, `texture` (`true` adds per-vertex color and a `.glb` preview
alongside the STL — a second, cheap query against the already-loaded
triplane decoder at the mesh's surface vertices, no extra model load; low
fidelity, capped by vertex density, not a real UV texture).

## Hunyuan3D 2.1

**What `ai.py` uses. Shape generation works; texture painting is broken on
this GPU** (see Texture painting subsection below — shape-only is fine, it's
what matters for STL output).

```bash
source /path/to/printable/.venv/bin/activate
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git
cd Hunyuan3D-2.1
```

Skip upstream's CUDA-specific torch install (`--index-url .../cu124`) — the
ROCm/TheRock build from step 2 is already installed.

<details>
<summary>2.1's shape model isn't the same as 2.0's, if you're wondering</summary>

Confirmed by comparing actual downloaded checkpoint files: 2.0's was
`hunyuan3d-dit-v2-0/model.fp16.ckpt` at 4.6GB, 2.1's is
`hunyuan3d-dit-v2-1/model.fp16.ckpt` at 7.37GB — genuinely different,
larger model, not the same weights renamed. Upstream's README also has
zero mention of ROCm/AMD anywhere.

</details>

**Do not run `uv pip install -r requirements.txt` directly — it fails
outright as written**: `requirements.txt` self-contradicts (`numpy==1.24.4`
alongside `pandas==2.2.2`, which needs `numpy>=1.26.0`), and several of its
other pins would break TripoSR or pull CUDA-only/unused packages. Run this
filtered version instead — it skips those and keeps whatever's already
installed for them:

```bash
INSTALLED=$(uv pip list | tail -n +3 | awk '{print tolower($1)}')
python3 - "$INSTALLED" << 'EOF'
import re, sys
installed = set(sys.argv[1].split())
skip = {"numpy", "pandas", "cupy-cuda12x", "transformers", "huggingface-hub",
        "bpy", "deepspeed", "gradio", "fastapi", "uvicorn"}
# needed here, but the exact pin has no cp312 wheel -- resolve fresh instead
unpin = {"pymeshlab"}
out = []
with open("requirements.txt") as f:
    for line in f:
        raw = line.rstrip("\n")
        s = raw.strip()
        if not s or s.startswith("#") or s.startswith("--"):
            continue  # also drops the China-region mirror index lines
        m = re.match(r"^([A-Za-z0-9_.\-\[\]]+)==", s)
        name = (m.group(1) if m else s).lower()
        if name in skip:
            continue
        out.append(name if name in installed or name in unpin else raw)
open("requirements-rocm.txt", "w").write("\n".join(out) + "\n")
EOF
uv pip install -r requirements-rocm.txt
uv pip install huggingface-hub==0.36.2
```

The second line matters: `huggingface-hub` is skipped above (left
unconstrained), so `diffusers` and friends resolve it to whatever's newest —
which `transformers==4.35.0` then hard-rejects at import (`huggingface-hub
<1.0` required). Must be a separate command, not folded into the block
above: `tokenizers==0.14.1` (already installed via TripoSR) declares its
own `huggingface-hub<0.18` bound, so a joint resolve of everything together
fails on that conflict — a single-package upgrade afterward doesn't
re-validate already-installed transitive bounds the same way.

<details>
<summary>Why each package in <code>skip</code> is there</summary>

`transformers==4.46.0`/`huggingface-hub==0.30.2` would break TripoSR, which
needs exactly `transformers==4.35.0` (its checkpoint's internal ViT layer
names changed in later versions — a `state_dict` mismatch, not an import
error). `cupy-cuda12x` is CUDA-only, meaningless on ROCm. `bpy`/`deepspeed`
are heavy and not needed for inference. `gradio`/`fastapi`/`uvicorn` are for
Hunyuan3D-2.1's own demo app, which `printable` doesn't use — and `gradio`
specifically drags `huggingface-hub` to `1.28.0` via an unpinned transitive
dependency if you let it resolve, even though nothing asked for it directly.

</details>

`hy3dshape` has no `setup.py` of its own, so give it one (proper editable
install instead of upstream's `sys.path.insert` approach):

```bash
cat > hy3dshape/setup.py << 'EOF'
from setuptools import setup, find_packages

setup(
    name="hy3dshape",
    version="2.1.0",
    url="https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1",
    packages=find_packages(),
)
EOF
uv pip install -e hy3dshape --no-deps
```

Verify it imports alongside everything else:

```bash
python -c "from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline; from tsr.system import TSR; print('both OK')"
```

Shape-only smoke test, run from inside the `Hunyuan3D-2.1` checkout
(`assets/demo.png` ships with it):

```bash
mkdir -p output
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 python3 <<'EOF'
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh = pipeline(image='assets/demo.png')
mesh = mesh[0] if isinstance(mesh, list) else mesh  # bare Trimesh for one image, list for several
mesh.export('output/test.glb')
print('exported output/test.glb')
EOF
```

**Confirmed working**: 188,589 verts/377,274 faces in ~236s standalone, and
end-to-end through `printable`'s own CLI (146,000 faces, watertight,
`PRINTABLE`, ~110s).

VRAM, per upstream: ~10GB shape-only, ~21GB texture-only, ~29GB combined —
comfortable headroom on a unified-memory box once GTT is set up per step 1
above (the BIOS framebuffer setting itself doesn't bound this).

---

## Setup is complete here

**You're done. Nothing past this point is required.** Shape generation is
confirmed working end-to-end through `printable`'s own CLI — that's the
actual goal, and it's what `ai.py` uses for every `hunyuan3d` generation.

Everything below is either optional (Depth/normal geometry cues) or a
documented dead end (texture painting doesn't work on this GPU, kept only
for reference). Stop here, or skip to whichever of those you actually want.

---

<details>
<summary>Texture painting: blocked by a real GPU crash, not a config issue (not needed for setup)</summary>

Everything below was chased down and fixed, and texture painting still
doesn't work here — the remaining blocker is a genuine bug, not something
`printable` can configure around. Documented in full because the individual
fixes are still correct and may matter again (e.g. if upstream patches the
actual blocker and painting becomes viable).

Two native-extension builds, both new relative to 2.0's single
`custom_rasterizer`:

```bash
cd hy3dpaint/custom_rasterizer
PYTORCH_ROCM_ARCH=gfx1151 uv pip install -e . --no-build-isolation
cd ../..
cd hy3dpaint/DifferentiableRenderer
bash compile_mesh_painter.sh
cd ../..
```

**Both need the same runtime fix to actually import**: `ImportError:
libc10.so`/`librocm-openblas.so.0: cannot open shared object file`. TheRock's
ROCm SDK ships scattered across several nested `site-packages`
subdirectories, none on the default library search path:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/host-math/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib"
```

**`compile_mesh_painter.sh` can build against the wrong Python** if your
shell's bare `python3-config` resolves to the system Python instead of the
venv's — it did here (built for `cpython-314`, venv is `cpython-312`). Find
the venv's own `python3.12-config` (e.g.
`~/.local/share/uv/python/cpython-3.12.13-*/bin/python3.12-config`) and
rebuild targeting that explicitly if the extension fails to import after a
clean build.

Checkpoint download unrelated to Hugging Face:

```bash
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
```

Constructing the paint pipeline (`sys.path.insert(0, './hy3dshape')` and
`'./hy3dpaint'`, run from the `Hunyuan3D-2.1` repo root — the paint config's
own path defaults are inconsistently repo-root-relative vs
`hy3dpaint`-relative, see the checkpoint-path override below) then hits a
chain of real, fixable bugs, in the order they surface:

1. **`cannot import name 'get_cached_repo_tree' from 'huggingface_hub'`** —
   `diffusers` drifts to a version (`0.40.0` seen here) whose
   `pipeline_utils.py` hard-imports a function that doesn't exist in
   TripoSR's protected `huggingface-hub==0.36.2` pin. `huggingface-hub>=1.0`
   is not a fix — `transformers` itself enforces `<1.0` at import time, a
   hard wall, confirmed by trying it. The real fix: `hy3dpaint`'s own
   `requirements.txt` (which the filtered install above skips `diffusers`
   from) pins `diffusers==0.30.0`, which predates this import entirely.
   ```bash
   uv pip install "diffusers==0.30.0"
   ```
2. **`ModuleNotFoundError: No module named 'bpy'`** — Blender's Python
   module has no wheel for this venv's Python at all (checked: nothing for
   `cp312`, nearest are `cp311`/`cp313`). It's only used by one function,
   `convert_obj_to_glb()` in `DifferentiableRenderer/mesh_utils.py`, which
   `printable` doesn't need (it has its own trimesh-based OBJ→GLB export
   already). Patched the import to fail soft instead of hard, in the
   `Hunyuan3D-2.1` checkout itself:
   ```python
   # DifferentiableRenderer/mesh_utils.py, top of file
   try:
       import bpy
   except ImportError:
       bpy = None  # only convert_obj_to_glb() needs it; that function
                    # already reports failure via its own try/except
   ```
3. **`ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`** —
   `basicsr` (a `realesrgan` dependency) imports a
   private torchvision module removed in newer torchvision. Well-known,
   standard fix — the function moved to the public module:
   ```bash
   # in .venv/lib/python3.12/site-packages/basicsr/data/degradations.py:
   # from torchvision.transforms.functional_tensor import rgb_to_grayscale
   # -> from torchvision.transforms.functional import rgb_to_grayscale
   ```
4. **`FileNotFoundError: 'ckpt/RealESRGAN_x4plus.pth'`** — upstream's own
   `Hunyuan3DPaintConfig` defaults are inconsistent: `multiview_cfg_path`
   defaults to a repo-root-relative path, `realesrgan_ckpt_path` defaults to
   a `hy3dpaint`-relative one. No single CWD satisfies both. Override the
   checkpoint path with an absolute one after construction:
   ```python
   config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
   config.realesrgan_ckpt_path = "/path/to/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
   ```
5. **`pymeshlab.pmeshlab.PyMeshLabException: Unknown format for load: obj`**
   — pymeshlab's OBJ-loading plugin silently fails to load
   (`libOpenGL.so.0: cannot open shared object file`), and pymeshlab doesn't
   surface that as an error until you actually try to use the format. Real
   missing system package, not a Python-level issue:
   ```bash
   sudo apt install -y libopengl0
   ```
6. **`ValueError: target_reduction must be between 0 and 1`** — not a fresh
   trimesh bug upstream never saw: `requirements.txt` pins `trimesh==4.4.7`
   exactly, where `simplify_quadric_decimation`'s first positional argument
   meant a target face count. The filtered-install step above deliberately
   drops that pin in favor of whatever's already installed when the package
   is already present — and `printable` itself already has a newer trimesh
   (`pyproject.toml` pins `trimesh>=4.0` and relies on 5.x-specific repair
   behavior), so this installs `trimesh` 5.0.0 instead. In 5.0.0 that same
   first positional argument means `percent` (a 0–1 fraction) — a real
   signature change between those two trimesh versions. Hunyuan3D-2.1's own
   `hy3dpaint/utils/simplify_mesh_utils.py` calls it positionally with a
   face-count integer, which silently binds to the wrong parameter under the
   newer trimesh. Downgrading trimesh to match upstream's pin isn't a good
   option here (would risk breaking `printable`'s own repair pipeline);
   fixing the call site to use the keyword explicitly is version-proof
   either way:
   ```python
   # simplify_mesh_utils.py:
   # courent.simplify_quadric_decimation(target_count)
   # -> courent.simplify_quadric_decimation(face_count=target_count)
   ```

**After all six fixes, the pipeline gets past every Python-level bug and
crashes the GPU itself.** A real `[gfxhub] page fault` inside
`custom_rasterizer`'s hipified kernels, confirmed reproducible three times in
a row: `dmesg`/`journalctl -k` shows `amdgpu 0000:c6:00.0: ring gfx_0.0.0
timeout` followed by a forced ring reset, right as rendering setup begins
(before diffusion sampling even starts — reducing `max_num_view` doesn't
help, since the crash isn't in the diffusion step). The GPU driver
self-recovers each time (`Ring gfx_0.0.0 reset succeeded... device wedged,
but recovered through reset` — no reboot needed, `uptime` stays continuous),
but because this is an integrated APU, the reset briefly stalls **every**
GPU client sharing the chip, including the desktop compositor
(`gnome-shell`) and VS Code's own GPU process — visible as the whole desktop
session appearing to lock up or restart.

This is a real, unported CUDA-specific assumption in `custom_rasterizer`'s
kernels faulting on ROCm's memory model, not a timeout/contention issue —
`MeshRender.py` has no CPU or alternate-GPU rasterizer fallback (`raster_mode
= "cr"` is the only implemented mode; every other branch just raises). Not
something `printable` can configure around. Options if revisiting this:
read the actual CUDA kernel source in `custom_rasterizer` for the specific
pointer/memory-layout assumption that doesn't hold on ROCm, watch for an
upstream fix, or try a different ROCm/kernel driver version.

**Current state: shape generation only.** `printable`'s `hunyuan3d` backend
uses 2.1 for shape generation (proven working, see status callout above).
Texture-derived color on the exported GLB currently only comes from
TripoSR's vertex-color path (`--opt texture=true`), not from Hunyuan3D.

</details>

## Depth/normal geometry cues (transformers-native, optional)

**Optional — not required for setup.** A diagnostic feature (extra geometry
hints alongside generation), not something `printable generate` needs to
work. Skip this whole section unless you specifically want it.

**Breaks TripoSR if you enable it — confirmed here, not just in theory.**
`pyproject.toml`'s `depth` extra needs `transformers>=4.49`; TripoSR needs
exactly `transformers==4.35.0`. These cannot coexist in the same venv,
under any sync flag — `--inexact` only stops *removal* of packages outside
the lockfile (torch/torchmcubes/hy3dshape correctly survive), it does
**not** protect the *version* of lockfile-tracked packages like
`transformers` from being bumped to satisfy a real declared constraint.
`uv sync --extra depth` (even `--inexact`) will bump `transformers` to
5.x and `huggingface-hub` to 1.x, which then breaks TripoSR outright.

**It also rewrites `uv.lock` itself**, not just the venv — confirmed here.
If you run this, check `git diff uv.lock` afterward and revert it
(`git checkout -- uv.lock`) before committing anything, unless you actually
intend to change what `printable`'s base resolution locks to for everyone
who pulls this repo:

```bash
cd /path/to/printable   # back to the printable checkout, not a backend clone
uv sync --extra depth --inexact
```

**To switch back to TripoSR/Hunyuan3D afterward** (run from anywhere, venv
active — `uv pip install`, unlike `uv sync` above, doesn't care about cwd).
**Two separate commands, not one** — `tokenizers==0.14.1` (pulled in by
`transformers==4.35.0`) declares its own `huggingface-hub<0.18` bound, so a
joint resolve of both together fails outright on that conflict, the same
trap as everywhere else in this doc:

```bash
uv pip install transformers==4.35.0
uv pip install huggingface-hub==0.36.2
```

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 printable generate assets/examples/character.png --backend hunyuan3d --size 100 \
  --opt geometry_cues=true
```

The env var is the same one required for the shape-only smoke test earlier
— still required here, `printable` doesn't set it for you.

Writes `<output>_depth.png` and `<output>_normal.png` alongside the STL for
inspection. Diagnostic only today — no backend accepts either map as a
conditioning input yet, so this doesn't change the generated mesh. The
normal map is derived from the depth map's gradients, not a learned
estimate: `transformers` has no `normal-estimation` pipeline task to call.

If TripoSR/Hunyuan3D's `transformers==4.35.0` is active (the normal state,
per the switch-back command above), this stage fails with `KeyError:
'depth_anything'` and no PNGs get written — expected, not a bug. The stage
logs a warning and generation continues without it; see the conflict
explained above.

---

## Troubleshooting

> **⚠ Loading two different real GPU backends' models in one process has
> hung this entire machine** — not a slow test, an OS-level lockup needing a
> physical power cycle. Confirmed here (with a since-dropped third backend;
> the underlying finding — switching GPU backends within one process — still
> applies to today's TripoSR/Hunyuan3D pairing). Each backend works fine
> alone in a fresh process.
>
> **Live risk in `printable serve`, not just tests**: one long-running
> process serves every job regardless of backend (`api/jobs.py`'s
> single-worker executor). Restart `printable serve` between jobs using a
> different GPU backend than the previous job. For testing, always use
> separate CLI invocations (separate processes), never two real GPU
> backends' models loaded in one Python process.

**`torch.cuda.is_available()` is False** — ROCm not installed, or the GPU
architecture is unrecognised. Try `HSA_OVERRIDE_GFX_VERSION=11.5.1`.

**`ImportError: undefined symbol` mentioning `amdhip64` or `torchCheckFail`**
— an ABI/version mismatch, not a missing file. The former means torch's
*bundled* `libamdhip64.so` (matching its wheel's ROCm version) disagrees with
the system ROCm used to build some other extension against it — reinstall
torch from the index matching `hipconfig --version`. The latter means that
extension was linked against the wrong C++ `_GLIBCXX_USE_CXX11_ABI` value;
see the TripoSR section above.

**GPU tensor ops segfault even though `is_available()` is True** — on Strix
Halo (`gfx1151`), this is fixed by switching to TheRock's gfx1151-native
build (step 2 above); see the callout in step 3 for the full story. Until
you switch, or on other hardware hitting something similar,
`HIP_VISIBLE_DEVICES=""` forces CPU (there's no `--opt` for this — the
backends always prefer the accelerator when `is_available()` is True) to
confirm the rest of the pipeline works.

**Out of memory** — lower `octree_resolution` or `mc_resolution`. On Strix
Halo, check `amdgpu.gttsize` rather than raising the BIOS UMA framebuffer —
GTT (not the framebuffer) is what actually bounds GPU compute memory here;
see step 1 above.

**Mesh generates but will not slice** — run `printable inspect model.stl` to see
which check fails. Non-watertight output usually means the repair ladder could
not close it; install the `repair` extra so Poisson rebuild is available.

**Decimation does nothing** — `fast-simplification` is missing. It is a base
dependency, so reinstall with `uv sync` — but only if you haven't installed
a GPU backend per this doc yet; `uv sync` (and bare `uv run`) removes `torch`
and everything built on top of it (see the warning in the TripoSR section).
If you have, reinstall the specific package instead: `uv pip install
fast-simplification`.

**Base plate ends up on the head instead of the feet (or any other
semantically wrong face)** — `auto_orient()` optimizes purely for print
success, with no concept of "feet down" for a character mesh; a standing
figure can lose to lying flat on its back, which gives more bed contact.
Confirmed real on at least one run (TripoSR, `auto-oriented (score
-1411.9)`, base fused to the head) — but **not consistent**: the same
`auto_orient()` gets it right plenty of the time too, including a later
`character.png` + `hunyuan3d` + `--sheet` run that came out feet-down with
no help at all. Don't assume every generation needs correcting.

`--rotate X Y Z` (degrees) runs after auto-orient and before the base is
added, so the base correctly follows the new orientation *if you actually
need it* — rotating the finished STL in a slicer instead leaves the base
fused to the old, wrong face. Not fixable by tuning auto-orient itself;
see `ROADMAP.md`'s Known Limitations for why.

**`--rotate` is a blind delta — applying it speculatively can flip an
already-correct mesh upside down instead of fixing one.** Confirmed here:
guessing `--rotate 180 0 0` on a `--sheet` run produced a head-stand
result, but re-running the *same* command without `--rotate` showed the
picked panel was already standing correctly on its own. The rotation
almost certainly flipped a mesh that didn't need touching. **Always check
the unrotated result first**, then only rotate if it's actually wrong,
using whatever direction you actually observe (not a guessed `180 0 0`):

```bash
# 1. look first, no --rotate -- --sheet: see all four raw candidates
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 printable generate assets/examples/character.png \
  -b hunyuan3d --size 100 --sheet --keep-all
```

Open `output/character_hunyuan3d_v0.stl` through `_v3.stl`. If the picked
one (named in the command's own output) is already feet-down, you're
done — no `--rotate` needed. If it genuinely isn't, cut that panel's
source image out and regenerate it alone with the rotation *that specific
mesh* needs (figure out the right X/Y/Z from what you actually see, not
by guessing):

```bash
printable split assets/examples/character.png -d output/panels/
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 printable generate output/panels/character_v0.png \
  -b hunyuan3d --size 100 --rotate <X> <Y> <Z>
```
