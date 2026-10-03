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
uv sync --extra repair --extra preprocess --inexact   # or --all-extras for both
```

### Make `uv sync` stop being a trap

**Do this now, before installing any GPU backend below.** Every `uv sync`
in this doc uses `--inexact` for a reason: torch and every GPU-backend
package intentionally live outside `printable`'s lockfile (step 2 below
explains why), so a plain `uv sync` — run for any reason, including just
grabbing an extra you forgot, months from now — silently deletes all of
them. `--inexact` means "add what I asked for, don't remove anything
else"; there's no downside to it here. The only failure mode is
forgetting to type it, which is exactly what just happened if that's how
you got here.

Make it impossible to forget instead of relying on remembering it every
time — scoped to this repo only, so it can't surprise you in some other
project that actually wants plain `uv sync` semantics:

```bash
# Add to ~/.bashrc / ~/.zshrc once. Only overrides `sync`/`run` while your
# shell's cwd is inside a repo whose pyproject.toml says `name = "printable"`
# (walks up from cwd, so it applies in subdirectories too) -- everywhere
# else, and every other `uv` subcommand even inside this repo, is untouched.
# `command uv sync ...` still works if you ever genuinely want strict removal.
uv() {
  local dir="$PWD"
  while [ "$dir" != "/" ]; do
    if [ -f "$dir/pyproject.toml" ] && grep -q '^name = "printable"' "$dir/pyproject.toml" 2>/dev/null; then
      case "$1" in
        sync) command uv sync --inexact "${@:2}"; return ;;
        run)  command uv run --no-sync "${@:2}"; return ;;
      esac
      break
    fi
    dir="$(dirname "$dir")"
  done
  command uv "$@"
}
```

<details>
<summary>Why both `sync` and `run` need covering, and why this is safe even before any GPU backend exists</summary>

`uv run <anything>` — `uv run pytest`, `uv run python -c ...`, all of it —
reconciles the environment to the lockfile first, by default, same as
bare `uv sync`. `--no-sync` skips that step. Wrapping both here closes the
same trap two different commands fall into.

Harmless before you've installed anything extra, too: with nothing
extraneous in the venv yet, `--inexact`/`--no-sync` behave identically to
the plain command. There's no scenario where adding this function changes
behavior for the worse — worst case, outside this repo or on a subcommand
other than `sync`/`run`, it's a no-op passthrough to the real `uv`.

</details>

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
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1151]" torchvision
```

Verify:

```bash
uv run --no-sync python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expect `True <your GPU name>`. If `False`, see Troubleshooting.

**Torch now lives outside `printable`'s lockfile — bare `uv sync` (or bare
`uv run <anything>`) will silently remove it from here on.** If you added
the shell function in "Make `uv sync` stop being a trap" above, you're
already covered. Otherwise, use `uv run --no-sync ...` for everything
below, or call the venv's binaries directly with no `uv run` prefix at all.

<details>
<summary>Why this exact command, not the generic PyTorch ROCm wheels</summary>

The generic multi-arch wheels from `download.pytorch.org` segfault on
`gfx1151` on any GPU memory access — a real, confirmed upstream bug
([ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853), filed against
this exact chip). The command above is
[TheRock](https://github.com/ROCm/TheRock)'s gfx1151-native **stable**
release channel, which ships its own self-contained ROCm kernels and
sidesteps the bug. Current pin: `torch==2.13.0+rocm10.0.0`, confirmed
end-to-end (full test suite, real TripoSR/Hunyuan3D generation) —
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` is no longer required on it.

Full migration history — an earlier nightly-channel attempt that was tried
and reverted, and a real `uv` build-cache bug found along the way
(rebuilding a native extension like `torchmcubes` after switching torch
versions needs `--no-cache`, or it silently reuses an ABI-incompatible
`.so` — the TripoSR section below calls this out where it's actually
needed) — is in `ROADMAP.md`'s "ROCm 10.0" entry, not repeated here.

If the command above 404s, breaks, or stops resolving, check
[TheRock's releases page](https://github.com/ROCm/TheRock/releases) or
[RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) for
the current stable index.

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

That's [ROCm/ROCm#5853](https://github.com/ROCm/ROCm/issues/5853), which
step 2's TheRock command already routes you around — if you followed step
2, you shouldn't hit this. `HIP_VISIBLE_DEVICES=""` forces CPU as a way to
confirm the rest of the pipeline (install, build, model download) is fine
while you sort out the GPU side.

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
torch instead of an isolated one. On the stable TheRock channel above,
CMake's `find_package(HIP)` needs `rocm-sdk-devel`'s expanded devel tree
(one-time per venv):

```bash
uv pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ rocm-sdk-devel
rocm-sdk init   # expands the devel tree; re-run after installing/removing a device wheel
```

Run both lines below as one command (the `\` continues the line):

```bash
SITE_PACKAGES=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
DEVEL_ROOT=$(rocm-sdk path --root)
ROCM_PATH="$DEVEL_ROOT" \
PYTORCH_ROCM_ARCH=gfx1151 \
HSA_OVERRIDE_GFX_VERSION=11.5.1 \
CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$SITE_PACKAGES -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=1" \
uv pip install --no-cache --no-build-isolation-package torchmcubes -r requirements.txt
```

**Always include `--no-cache` here** — a stale cached build from a
previous torch version is silently ABI-incompatible and produces a
confusing runtime error, not a build failure (full symptom in
`ROADMAP.md`'s "ROCm 10.0" entry if curious). If you're on the
apt-installed official ROCm 10.0 (`/opt/rocm`,
step 4 below) instead of this stable TheRock channel, use
`ROCM_PATH=~/.cache/rocm-shim/root` in place of `$DEVEL_ROOT` — untested
against this stable torch build specifically, but that's what the
shim exists for.

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
above always downgrades them to whatever TripoSR's `requirements.txt` wants,
not just sometimes:

```bash
uv pip install numpy==2.5.2 pandas==3.0.5 pillow==12.3.0 rembg==2.0.81 scipy==1.18.1 tifffile==2026.8.23 trimesh==5.0.0 markupsafe==3.0.3
```

**From here on: never run bare `uv sync` or bare `uv run <anything>`** —
same trap as step 2 above, now also covering `torchmcubes`. The shell
function from "Make `uv sync` stop being a trap" handles both; without
it, use `uv run --no-sync <command>`, or just call the venv's binaries
directly (`pytest`, `python`, `printable`, no `uv run` prefix at all).

If you do get bitten (`printable backends` reports "torch not installed"
with no `uv sync` in sight), nothing is actually lost — the shim and every
checkout survive on disk. Reinstall torch from TheRock's index, rebuild
`torchmcubes`, then re-run the version-restore command above.

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

**What `ai.py` uses. Shape generation and texture painting (`--opt
texture=true`) both work** — see the Texture painting subsection below for
its setup steps; shape-only is all that's required for plain STL output.

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
python3 <<'EOF'
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
mesh = pipeline(image='assets/demo.png')
mesh = mesh[0] if isinstance(mesh, list) else mesh  # bare Trimesh for one image, list for several
mesh.export('output/test.glb')
print('exported output/test.glb')
EOF
```

Expect roughly 188,589 verts / 377,274 faces in ~236s standalone (~110s
and 146,000 faces through `printable`'s own CLI, which decimates further).

VRAM, per upstream: ~10GB shape-only, ~21GB texture-only, ~29GB combined —
comfortable headroom on a unified-memory box once GTT is set up per step 1
above (the BIOS framebuffer setting itself doesn't bound this).

---

## Setup is complete here

**You're done. Nothing past this point is required.** Shape generation is
confirmed working end-to-end through `printable`'s own CLI — that's the
actual goal, and it's what `ai.py` uses for every `hunyuan3d` generation.

Everything below is optional: Depth/normal geometry cues, or texture
painting — the latter **now works and is wired into `ai.py`**
(`--opt texture=true`; see the details block below for setup steps, or
`ROADMAP.md`'s "ROCm 10.0" entry for the full fix history) — but nothing
past this point is required just to reproduce today's `hunyuan3d`
shape-only generation.

---

<details>
<summary>Texture painting: setup steps and gotchas (works since the ROCm 10.0 migration; full incident history in ROADMAP.md)</summary>

`printable`'s `hunyuan3d` backend already reads `--opt texture=true` for
real — `Hunyuan3DBackend._paint()` in `src/printable/backends/ai.py` runs
`hy3dpaint` on the shape mesh and writes a PBR-textured preview GLB
alongside the STL (the STL itself is always the plain shape mesh, which
has no texture slot regardless of backend). The steps below are what it
took to get there, needed once per fresh venv — kept in full because
they're still genuinely required, not because they're interesting history.

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
subdirectories, none on the default library search path (path below is for
the stable channel's package layout — `_rocm_sdk_libraries`, no `_gfx1151`
suffix; the old nightly build used `_rocm_sdk_libraries_gfx1151` instead):

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/host-math/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/lib:$VIRTUAL_ENV/lib/python3.12/site-packages/_rocm_sdk_libraries/lib"
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

**Watch out — one more environment-drift fix, still a required manual
step**: `ModuleNotFoundError: No module named 'pkg_resources'`. Newer
`setuptools` no longer bundles it, but `pytorch_lightning`'s
`lightning_fabric` still hard-imports it, and (unlike the `basicsr` import
in fix #3, which `ai.py`'s `_shim_basicsr_functional_tensor()` now patches
at runtime automatically) this one can't be shimmed in-process — it's a
real removed module other packages import directly:

```bash
uv pip install "setuptools==80.9.0"
```

Triggers a harmless `pkg_resources is deprecated` warning on import. Not
in `pyproject.toml` since it's a transitive dependency outside
`printable`'s own lockfile — same "reinstall after `uv sync`" category as
everything else in this doc. `ai.py`'s `_ensure_rocm_sdk_ld_library_path()`
also sets the `LD_LIBRARY_PATH` above automatically at runtime, so that
export is only needed for a standalone reproduction against the
`Hunyuan3D-2.1` checkout directly, not for `printable`'s own CLI.

**Watch out — used to crash the GPU here on the old ROCm 7.13 build**, even
after all six fixes above: a real `[gfxhub] page fault` and forced ring
reset, confirmed reproducible three times, briefly stalling every GPU
client on the machine (not just this process) each time. No longer
reproducible since migrating to the ROCm 10.0 stable channel — full
incident writeup, including whether it's actually root-caused or just not
currently triggered, is in `ROADMAP.md`'s "ROCm 10.0" entry. If it
resurfaces on a future ROCm/torch update, that's the reference reproduction
to start from.

**Watch out — topology**: the painted mesh is *not* the same object as the
shape mesh used for the STL. `hy3dpaint` remeshes and UV-unwraps
internally (`GenerationResult.color_mesh`, distinct from
`GenerationResult.mesh`), so don't assume vertex/face correspondence
between the STL and the GLB preview the way you safely can for TripoSR's
`--opt texture=true` path.

</details>

## Depth/normal geometry cues (transformers-native, optional)

**Optional — not required for setup.** A diagnostic feature (extra geometry
hints alongside generation), not something `printable generate` needs to
work. Skip this whole section unless you specifically want it.

**Works alongside TripoSR/Hunyuan3D — no separate install, no version
conflict.** `backends/depth.py`'s `DEFAULT_MODEL` is
`Intel/dpt-hybrid-midas` (DPT), chosen specifically because it's been
registered in `transformers` since well before this project's actual
floor, `transformers==4.35.0` (TripoSR/Hunyuan3D's own pin) — unlike
`depth-anything/Depth-Anything-V2-Small-hf`, which needs
`transformers>=4.49` and would conflict. If you already have TripoSR or
Hunyuan3D installed per this doc, `transformers` is already present and
this just works:

```bash
printable generate assets/examples/character.png --backend hunyuan3d --size 100 \
  --opt geometry_cues=true
```

Writes `<output>_depth.png` and `<output>_normal.png` alongside the STL for
inspection. Diagnostic only today — no backend accepts either map as a
conditioning input yet, so this doesn't change the generated mesh. The
normal map is derived from the depth map's gradients, not a learned
estimate: `transformers` has no `normal-estimation` pipeline task to call.
`printable serve`'s web UI has its own "Geometry cues (diagnostic)"
checkbox for this same option, with thumbnails + download links for
whichever of the two maps a job actually produced — see README.md's Web
API section.

<details>
<summary>If you're on a CPU-only setup (no TripoSR/Hunyuan3D), `transformers` isn't installed yet</summary>

`heightmap`/`lithophane` don't pull in `transformers` themselves (this
diagnostic is the only thing that needs it on a CPU-only install) — get it
via the `depth` extra:

```bash
uv sync --extra depth --inexact
```

No version conflict to worry about here either way — `pyproject.toml`'s
`depth` extra declares plain `transformers`, no floor, precisely because
DPT doesn't need one.

</details>

---

## Troubleshooting

> **⚠ Loading two different real GPU backends' models in one process has
> hung this entire machine** — not a slow test, an OS-level lockup needing a
> physical power cycle. The finding is about switching GPU backends within
> one process generally, so it applies to any two of `printable`'s GPU
> backends, not just a specific pairing. Each backend works fine alone in
> a fresh process.
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
printable generate assets/examples/character.png \
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
printable generate output/panels/character_v0.png \
  -b hunyuan3d --size 100 --rotate <X> <Y> <Z>
```
