"""Quick software renders of Wickwood models for geometry QA (back-face culled, flat shaded).

Culling hides any triangle whose winding faces away from the camera, so inverted faces appear as holes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from wk_mesh import MATERIALS


def texture_means(textures_dir: Path):
    prov = json.loads((Path(textures_dir) / "texture-provenance.json").read_text(encoding="utf-8"))
    return {t["name"]: t["meanRGB"] for t in prov["textures"]}


def albedo(material, means):
    mat = MATERIALS[material]
    c = mat["color"]
    if mat.get("glow"):
        return tuple(int(255 * v) for v in c), True
    if mat.get("sign"):
        return (210, 196, 150), True
    t = means.get(mat.get("texture") or "", (200, 200, 200))
    return tuple(int(min(255, c[i] * t[i] * 1.25)) for i in range(3)), False


def render(mesh, means, size=240, azimuth=35.0, elevation=24.0, background=(28, 34, 36)):
    az, el = math.radians(azimuth), math.radians(elevation)
    # camera looks toward -forward; forward points from the model to the camera (front-right-above)
    fwd = (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))
    right = (math.cos(az), 0.0, -math.sin(az))
    up = (-math.sin(az) * math.sin(el), math.cos(el), -math.cos(az) * math.sin(el))
    light = (0.4, 0.8, 0.45)
    ln = math.sqrt(sum(v * v for v in light))
    light = tuple(v / ln for v in light)
    tris = []
    pts_all = []
    for material, part in mesh.parts.items():
        col, unlit = albedo(material, means)
        for a, b, c in part.triangles:
            pa, pb, pc = part.vertices[a], part.vertices[b], part.vertices[c]
            u = [pb[i] - pa[i] for i in range(3)]
            v = [pc[i] - pa[i] for i in range(3)]
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            nl = math.sqrt(sum(x * x for x in n)) or 1.0
            n = tuple(x / nl for x in n)
            if sum(n[i] * fwd[i] for i in range(3)) <= 0:
                continue  # back face
            shade = 1.0 if unlit else 0.35 + 0.65 * max(0.0, sum(n[i] * light[i] for i in range(3)))
            proj = [(sum(p[i] * right[i] for i in range(3)), sum(p[i] * up[i] for i in range(3)), sum(p[i] * fwd[i] for i in range(3))) for p in (pa, pb, pc)]
            depth = sum(q[2] for q in proj) / 3
            tris.append((depth, proj, tuple(int(ch * shade) for ch in col)))
            pts_all.extend(proj)
    img = Image.new("RGB", (size, size), background)
    if not pts_all:
        return img
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 0.5)
    scale = (size - 20) / span
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    draw = ImageDraw.Draw(img)
    for depth, proj, col in sorted(tris, key=lambda t: t[0]):
        draw.polygon([((p[0] - cx) * scale + size / 2, size / 2 - (p[1] - cy) * scale) for p in proj], fill=col)
    return img


def sheet(meshes, means, out_path: Path, cols=8, size=240):
    names = list(meshes)
    rows = (len(names) + cols - 1) // cols
    img = Image.new("RGB", (cols * size, rows * (size + 18)), (16, 18, 20))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    for i, name in enumerate(names):
        tile = render(meshes[name], means, size)
        x, y = (i % cols) * size, (i // cols) * (size + 18)
        img.paste(tile, (x, y))
        draw.text((x + 4, y + size + 3), name, fill=(220, 220, 210), font=font)
    img.save(out_path)
    return out_path
