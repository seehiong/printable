# Example images

This directory provides reference inputs, benchmark assets, and stress-test cases for testing each of the generation pipelines in `printable`.

---

## `portrait.jpg`

A high-contrast monochrome portrait photograph with dramatic chiaroscuro studio lighting.

- **Primary pipeline**: Lithophane generation (`-b lithophane`)
- **Key options**: `--opt frame_mm=3` (adds structural border frame), `--opt max_thickness_mm=3.2`
- **Example command**:
  ```bash
  printable generate assets/examples/portrait.jpg -b lithophane --size 120 \
    --opt frame_mm=3 --opt max_thickness_mm=3.2
  ```
- **Why it fits**: Lithophanes map tonal brightness directly into physical shell thickness (dark regions become thicker, light regions become thinner). High contrast and smooth tonal gradients produce crisp backlit detail when 3D printed with translucent or white filament.

---

## `character_male.png`

A single stylized cartoon adventurer character in a clean full-body front view on a flat, neutral grey studio backdrop.

- **Primary pipelines**: Bas-relief heightmaps (`-b heightmap`) or single-view 3D preview (`-b triposr`)
- **Key options**: `--opt key_background=true` (removes solid backdrop before mesh projection), `--opt relief_mm=7`, `--opt texture=true` (writes `.glb` preview), `--rotate 180 0 0`
- **Example commands**:
  ```bash
  # Bas-relief heightmap with background keyed out
  printable generate assets/examples/character_male.png -b heightmap --size 90 \
    --opt key_background=true --opt relief_mm=7

  # True 3D fast preview with colored .glb output
  printable generate assets/examples/character_male.png -b triposr --size 90 \
    --opt texture=true --rotate 180 0 0
  ```
- **Why it fits**: The flat grey studio background is effortlessly keyed out with `key_background=true`, leaving only the embossed character geometry without background artefacts. In `triposr`, it provides a fast ~10s test for 3D reconstruction and vertex coloring.

---

## `figure.jpg`

A single full-body collectible tabletop miniature figurine (paladin knight with shield and sword) on a neutral grey background.

- **Primary pipeline**: True 3D AI reconstruction with wall hollowing (`-b hunyuan3d`)
- **Key options**: `--hollow --hollow-wall 1.6` (hollows interior to save filament and resin), `--max-faces 500000`, `--no-base`
- **Example command**:
  ```bash
  printable generate assets/examples/figure.jpg -b hunyuan3d --size 80 \
    --hollow --hollow-wall 1.6 --max-faces 500000 --no-base
  ```
- **Why it fits**: High structural detail, distinct silhouette, and balanced lighting make it an ideal input for AI 3D mesh reconstruction and interior cavity hollowing. Keeping `--max-faces 500000` avoids decimation artifacts on dense raw outputs.

---

## `character_female_turnaround.png`

A 4-angle character turnaround model sheet in a 2x2 grid showing the same female champion adventurer from front, back, left profile, and right profile views with matching proportions, lowered cowl/hood, and braided hair.

- **Primary pipeline**: Multi-view sheet splitting and selective generation (`--sheet`)
- **Example commands**:
  ```bash
  # Split, generate from every panel, and keep the best watertight result
  printable generate assets/examples/character_female_turnaround.png -b hunyuan3d --size 80 --sheet

  # Or manually split into individual panels
  printable split assets/examples/character_female_turnaround.png -d panels/
  ```
- **Why it fits**: Exercises `printable generate --sheet`, which automatically detects seams, crops individual panels, attempts generation across each angle, and picks the most printable, watertight result.

---

## `character.png`

The original character turnaround sheet: four views of the same figure in a 2x2 grid.

Generated with Stable Diffusion 1.5 (`runwayml/stable-diffusion-v1-5`) using the `charturnerv2` textual inversion:

```python
pipe.load_textual_inversion("./charturnerv2.pt", token="charturnerv2")
prompt = ("charturnerv2, multiple views of the same character in the same "
          "outfit, a fit character for a RPG game in best quality, "
          "intricate details.")
```

- **Primary pipelines**: Quickstart lithophane (`-b lithophane`) or multi-view sheet processing (`--sheet`)
- **Example commands**:
  ```bash
  # Quickstart lithophane (CPU only, seconds)
  printable generate assets/examples/character.png --backend lithophane --size 120

  # Multi-view sheet splitting with TripoSR
  printable generate assets/examples/character.png -b triposr --size 80 --sheet
  ```
- **Source & Provenance**: [Text-to-Image with StableDiffusionPipeline](https://seehiong.github.io/posts/2024/02/text-to-image-with-stablediffusionpipeline/) (10 Feb 2024); background in [Stable Diffusion: Text-to-Image Modeling Journey](https://seehiong.github.io/posts/2024/02/stable-diffusion-text-to-image-modeling-journey/).
- **Why it fits**: Useful for testing `key_background=true` against backdrops with lighting gradients as well as turnaround panel cropping.

---

## `figure_medusa_scale_statue.jpg` & `figure_xuner_scale_statue.jpg`

High-detail collectible scale statues featuring complex silhouettes: flowing translucent drapery, floating ribbon auras, crystal/water/flame visual effects, and ornate circular display bases.

- **Primary pipeline**: 3D mesh reconstruction stress testing (`-b hunyuan3d` / `-b triposr`)
- **Example command**:
  ```bash
  printable generate assets/examples/figure_medusa_scale_statue.jpg -b hunyuan3d --size 120
  ```
- **Why they fit (Benchmark & Stress-Test)**:
  - **Translucent / Sheer Material Edge Case**: As documented in the main [README.md](../../README.md#backends), translucent fabrics, wings, and crystal formations pose challenges for background removal models (`u2net`) and single-image depth/reconstruction models, often collapsing fine lace or lattice structures into flattened slabs.
  - **Complex Silhouettes & Thin Geometry**: Excellent benchmark inputs for testing the limits of Poisson surface reconstruction, thin-wall validation warnings, and support requirement estimation.

---

## `keychain_boba_spec.jpg`

A complete 3D printable design specification sheet for a Matcha Boba Cup Keychain.
Contains multi-view renders (Front, Back, Side), dimension callouts (60mm height, 30mm width, 26mm depth, 4.0mm keyring hole), printing recommendations, and keychain mockup.

- **Primary pipeline**: VLM-guided spec extraction and automated generation
- **Example commands**:
  ```bash
  # 1. Extract dimensions, print settings, and view crops into interim/design_spec.json
  printable generate-spec assets/examples/keychain_boba_spec.jpg

  # 2. Generate from extracted view crops and validate dimensions against the spec
  printable generate --spec interim/design_spec.json --spec-all-views -b hunyuan3d
  ```
- **Why it fits**: Serves as the benchmark test case for the VLM-guided spec extraction and interim processing pipeline documented in [ROADMAP.md](../../ROADMAP.md).

---

## `test.png`

A 256×256 synthetic grayscale benchmark pattern featuring smooth gradients, radial brightness, and sharp hard-edged step transitions.

- **Primary pipeline**: Automated smoke tests, depth/heightmap calibration, and edge-handling verification
- **Example command**:
  ```bash
  printable generate assets/examples/test.png -b heightmap --size 50
  ```
- **Why it fits**: Fully deterministic, lightweight, and requires no model downloads. It exercises contrast inversion, step-edge preservation, and mesh manifold checks without photographic noise.
