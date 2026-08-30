"""Generate the printable logo.

The isometric block is derived from a real mesh rather than hand-plotted
polygons: build the stepped solid with trimesh, project its faces, and emit
them back-to-front. Hand-deriving staircase geometry in screen space produces
self-intersecting outlines that are tedious to debug; letting the mesh library
own the geometry removes that class of error entirely.
"""

from pathlib import Path

import numpy as np
import trimesh

# Column heights, mirroring the silhouette drawn in the flat frame at left.
HEIGHTS = [1, 2, 3, 2]

TOP = "#5eead4"
LEFT = "#14b8a6"
RIGHT = "#0f766e"

BLOCK_W = 155.0        # target on-screen width of the projected block
OX, OY = 208.0, 88.0   # where the block is centred on the canvas


def stepped_solid() -> trimesh.Trimesh:
    """One box per column, unioned into a single stepped block."""
    parts = []
    for i, h in enumerate(HEIGHTS):
        box = trimesh.creation.box(extents=[1.0, 2.4, h])
        box.apply_translation([i - len(HEIGHTS) / 2.0 + 0.5, 0.0, h / 2.0])
        parts.append(box)

    solid = trimesh.boolean.union(parts)
    solid.fix_normals()
    return solid


def iso_matrix() -> np.ndarray:
    """A true isometric camera: yaw 45 degrees, then pitch to the magic angle.

    arctan(1/sqrt(2)) ~= 54.736 degrees is what makes the three axes
    foreshorten equally. Rolling a projection by hand instead -- inventing
    x/y/z ratios -- gives a camera that is not self-consistent, and the solid
    renders as a hollow ribbon.
    """
    yaw = trimesh.transformations.rotation_matrix(np.radians(45), [0, 0, 1])
    pitch = trimesh.transformations.rotation_matrix(np.radians(-54.736), [1, 0, 0])
    return pitch @ yaw


def face_svg(mesh: trimesh.Trimesh) -> list[str]:
    """Emit visible faces back-to-front, shaded by their world orientation."""
    M = iso_matrix()
    view = trimesh.transform_points(mesh.vertices, M)
    view_normals = trimesh.transform_points(mesh.face_normals, M, translate=False)

    # After the transform the camera looks straight down -Z, so a face is
    # visible exactly when its projected normal points back at us.
    visible = np.nonzero(view_normals[:, 2] > 1e-6)[0]
    # Painter's algorithm on projected depth.
    order = visible[np.argsort(view[mesh.faces[visible]][:, :, 2].mean(axis=1))]

    # Fit the projected block to the canvas.
    lo, hi = view[:, :2].min(axis=0), view[:, :2].max(axis=0)
    scale = BLOCK_W / (hi[0] - lo[0])
    centre = (lo + hi) / 2.0

    out = []
    for fi in order:
        n = mesh.face_normals[fi]        # shade by world normal, not view
        if n[2] > 0.5:
            fill = TOP                   # lit upper surface
        elif n[1] > 0.5:
            fill = LEFT                  # front wall, toward the viewer
        else:
            fill = RIGHT                 # end cap, in shadow

        pts = view[mesh.faces[fi]][:, :2]
        sx = OX + (pts[:, 0] - centre[0]) * scale
        sy = OY - (pts[:, 1] - centre[1]) * scale
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(sx, sy))
        # A matching stroke closes the hairline gaps between adjacent
        # triangles that anti-aliasing would otherwise leave visible.
        out.append(
            f'    <polygon points="{d}" fill="{fill}" '
            f'stroke="{fill}" stroke-width="1.2"/>'
        )
    return out


def build() -> str:
    body = face_svg(stepped_solid())

    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 170" width="520" height="170" role="img" aria-label="printable">
  <title>printable</title>

  <defs>
    <linearGradient id="pr-flat" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#cbd5e1"/>
      <stop offset="1" stop-color="#94a3b8"/>
    </linearGradient>
  </defs>

  <style>
    /* The deep teal wordmark disappears against a dark README background,
       so lift it when the viewer prefers dark. */
    .pr-word {{ fill: #0f766e; }}
    .pr-sub  {{ fill: #14b8a6; }}
    @media (prefers-color-scheme: dark) {{
      .pr-word {{ fill: #5eead4; }}
      .pr-sub  {{ fill: #2dd4bf; }}
    }}
  </style>

  <!-- Left: the flat source. The same stepped profile as the solid, so the
       mark reads as one shape being transformed, not two unrelated icons. -->
  <g>
    <rect x="14" y="46" width="78" height="84" rx="8"
          fill="url(#pr-flat)" stroke="#64748b" stroke-width="2.5"/>
    <path d="M26 118 L26 100 L39 100 L39 84 L53 84 L53 70 L66 70 L66 88 L80 88 L80 118 Z"
          fill="#475569" opacity=".9"/>
    <circle cx="74" cy="62" r="7" fill="#475569" opacity=".45"/>
  </g>

  <!-- Transform arrow -->
  <g stroke="#2dd4bf" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="M104 88 L128 88"/>
    <path d="M120 79 L129 88 L120 97"/>
  </g>

  <!-- Right: the extruded solid, projected from a real mesh. -->
  <g>
{chr(10).join(body)}
  </g>

  <!-- Wordmark -->
  <text x="316" y="94"
        font-family="ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
        font-size="40" font-weight="600" letter-spacing="-1.5" class="pr-word">printable</text>
  <text x="318" y="116"
        font-family="ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, monospace"
        font-size="12" letter-spacing="2.6" class="pr-sub">PHOTO &#8594; STL</text>
</svg>
"""


def build_icon() -> str:
    """Square mark: the solid alone, for favicons and avatars.

    The horizontal lockup stops being readable below about 140px, so anything
    smaller needs the block on its own.
    """
    global BLOCK_W, OX, OY
    # Fit inside the tile with real margin: the projected block is taller than
    # it is wide once the steps are included, so sizing to width alone
    # overflows the rounded corners.
    BLOCK_W, OX, OY = 74.0, 64.0, 66.0
    body = face_svg(stepped_solid())
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128" width="128" height="128" role="img" aria-label="printable">
  <title>printable</title>
  <!-- Deep slate tile rather than teal: a teal block on a teal ground has
       too little contrast to read at 16px. -->
  <rect width="128" height="128" rx="27" fill="#0f172a"/>
  <g>
{chr(10).join(body)}
  </g>
</svg>
"""


if __name__ == "__main__":
    # Resolve relative to this file so the script works from any cwd.
    assets = Path(__file__).resolve().parent.parent / "assets"
    (assets / "logo.svg").write_text(build(), encoding="utf-8")
    (assets / "icon.svg").write_text(build_icon(), encoding="utf-8")
    print(f"wrote {assets / 'logo.svg'}")
    print(f"wrote {assets / 'icon.svg'}")
