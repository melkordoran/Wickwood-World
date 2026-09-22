"""Original, seamlessly tiling Wickwood textures generated from the world seed (numpy + Pillow)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

SIZE = 256


def spectral(rng, size=SIZE, beta=1.6, aniso=(1.0, 1.0)):
    """Periodic (tileable) fractal noise in [0, 1]; aniso stretches features along an axis."""
    white = rng.standard_normal((size, size))
    fy = np.fft.fftfreq(size)[:, None] * aniso[1]
    fx = np.fft.fftfreq(size)[None, :] * aniso[0]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1.0
    spectrum = np.fft.fft2(white) / f ** beta
    spectrum[0, 0] = 0
    n = np.real(np.fft.ifft2(spectrum))
    n -= n.min()
    return n / max(n.max(), 1e-9)


def lerp(a, b, t):
    t = np.clip(t, 0, 1)[..., None]
    return np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def speckle(rng, density, size=SIZE, radius=1):
    """Tileable random dots of the given coverage fraction."""
    base = rng.random((size, size)) < density
    out = base.astype(float)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                out = np.maximum(out, np.roll(np.roll(base, dy, 0), dx, 1))
    return out


def grain(rng, amount=0.06, size=SIZE):
    return (rng.random((size, size)) - 0.5) * amount


def finish(img, rng, variation=0.05):
    img = img * (1 + grain(rng, variation)[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def forest_floor(rng):
    mottle = spectral(rng, beta=1.9)
    detail = spectral(rng, beta=1.1)
    img = lerp((64, 68, 42), (84, 64, 42), smoothstep(0.35, 0.65, mottle))
    img = lerp(img, (48, 40, 30), smoothstep(0.55, 0.9, detail) * 0.6)
    leaves = speckle(rng, 0.004, radius=2) * (rng.random((SIZE, SIZE)) < 0.7)
    img = lerp(img, (122, 86, 44), leaves * 0.8)
    twigs = spectral(rng, beta=0.9, aniso=(1, 7))
    img = lerp(img, (38, 32, 24), smoothstep(0.82, 0.95, twigs))
    return finish(img, rng)


def clearing_moss(rng):
    mottle = spectral(rng, beta=1.7)
    blades = spectral(rng, beta=0.8, aniso=(1, 5))
    img = lerp((66, 88, 50), (92, 112, 62), smoothstep(0.3, 0.7, mottle))
    img = lerp(img, (120, 136, 78), smoothstep(0.7, 0.92, blades) * 0.7)
    img = lerp(img, (70, 60, 40), smoothstep(0.75, 0.95, spectral(rng, beta=1.5)) * 0.5)
    return finish(img, rng)


def rock(rng, base=(112, 114, 110)):
    body = spectral(rng, beta=1.8)
    ridges = 1 - np.abs(spectral(rng, beta=1.3) - 0.5) * 2
    img = lerp(np.array(base) * 0.8, np.array(base) * 1.08, body)
    img = lerp(img, (52, 52, 52), smoothstep(0.93, 0.99, ridges))
    lichen = smoothstep(0.7, 0.85, spectral(rng, beta=1.6))
    img = lerp(img, (104, 116, 74), lichen * 0.55)
    return finish(img, rng, 0.08)


def shore_mud(rng):
    body = spectral(rng, beta=1.6)
    img = lerp((56, 52, 44), (86, 76, 62), body)
    pebbles = speckle(rng, 0.006, radius=2)
    img = lerp(img, (128, 124, 114), pebbles * 0.7)
    return finish(img, rng)


def dell_moss(rng):
    body = spectral(rng, beta=1.7)
    img = lerp((40, 76, 66), (62, 108, 88), body)
    flecks = speckle(rng, 0.0025, radius=1)
    img = lerp(img, (132, 204, 170), flecks * 0.8)
    return finish(img, rng)


def bark(rng):
    fibers = spectral(rng, beta=1.2, aniso=(1.0, 9.0))
    cracks = spectral(rng, beta=1.0, aniso=(1.0, 12.0))
    img = lerp((70, 54, 42), (104, 82, 62), fibers)
    img = lerp(img, (34, 26, 20), smoothstep(0.78, 0.92, cracks))
    img = lerp(img, (72, 88, 58), smoothstep(0.75, 0.9, spectral(rng, beta=1.8)) * 0.35)
    return finish(img, rng, 0.08)


def birch_bark(rng):
    body = spectral(rng, beta=1.5, aniso=(1, 3))
    img = lerp((188, 186, 178), (222, 220, 212), body)
    marks = spectral(rng, beta=0.8, aniso=(9.0, 1.0))
    img = lerp(img, (52, 48, 44), smoothstep(0.72, 0.82, marks))
    patches = spectral(rng, beta=1.6)
    img = lerp(img, (120, 112, 100), smoothstep(0.8, 0.92, patches) * 0.6)
    return finish(img, rng)


def needles(rng):
    body = spectral(rng, beta=1.4)
    strands = spectral(rng, beta=0.7, aniso=(3.0, 1.0))
    img = lerp((30, 52, 44), (52, 80, 60), body)
    img = lerp(img, (84, 110, 78), smoothstep(0.72, 0.9, strands) * 0.8)
    img = lerp(img, (18, 30, 26), smoothstep(0.1, 0.0, strands))
    return finish(img, rng, 0.1)


def leaves(rng):
    clumps = spectral(rng, beta=1.5)
    detail = spectral(rng, beta=0.6)
    img = lerp((40, 62, 36), (78, 106, 56), smoothstep(0.3, 0.75, clumps))
    img = lerp(img, (112, 136, 70), smoothstep(0.8, 0.95, detail) * 0.7)
    img = lerp(img, (22, 34, 22), smoothstep(0.25, 0.05, clumps))
    return finish(img, rng, 0.1)


def moss(rng):
    body = spectral(rng, beta=1.4)
    img = lerp((62, 90, 48), (98, 124, 62), body)
    return finish(img, rng, 0.12)


def masonry(rng):
    rows, cols = 8, 4
    img = np.zeros((SIZE, SIZE, 3))
    tone = spectral(rng, beta=1.7)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    row = yy * rows // SIZE
    offset = (row % 2) * (SIZE // cols // 2)
    col = ((xx + offset) % SIZE) * cols // SIZE
    block_tone = rng.random((rows, cols))[row, col]
    img = lerp((96, 96, 92), (132, 130, 122), 0.6 * block_tone + 0.4 * tone)
    ry = (yy % (SIZE // rows)) < 2
    rx = ((xx + offset) % (SIZE // cols)) < 2
    img = lerp(img, (54, 52, 48), (ry | rx).astype(float))
    img = lerp(img, (86, 102, 70), smoothstep(0.75, 0.9, spectral(rng, beta=1.6)) * 0.4)
    return finish(img, rng, 0.08)


def timber(rng):
    boards = 6
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    board = yy * boards // SIZE
    tone = rng.random(boards)[board]
    grain_n = spectral(rng, beta=1.0, aniso=(12.0, 1.0))
    img = lerp((96, 72, 50), (136, 104, 72), 0.5 * tone + 0.5 * grain_n)
    gap = (yy % (SIZE // boards)) < 2
    img = lerp(img, (40, 30, 22), gap.astype(float))
    return finish(img, rng, 0.07)


def plaster(rng):
    body = spectral(rng, beta=1.8)
    img = lerp((178, 168, 148), (212, 202, 180), body)
    # Periodic weathering stains (a vertical gradient would leave a seam when the texture tiles).
    img = lerp(img, (128, 116, 98), smoothstep(0.62, 0.9, spectral(rng, beta=1.4, aniso=(1.0, 3.0))) * 0.45)
    return finish(img, rng, 0.05)


def shingles(rng):
    rows, cols = 10, 8
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    row = yy * rows // SIZE
    offset = (row % 2) * (SIZE // cols // 2)
    col = ((xx + offset) % SIZE) * cols // SIZE
    tone = rng.random((rows, cols))[row, col]
    frac = (yy % (SIZE // rows)) / (SIZE // rows)
    img = lerp((62, 56, 52), (98, 88, 80), 0.7 * tone + 0.3 * spectral(rng, beta=1.4))
    img = img * (0.72 + 0.28 * frac)[..., None]
    edge = ((xx + offset) % (SIZE // cols)) < 2
    img = lerp(img, (30, 26, 24), edge.astype(float) * 0.8)
    img = lerp(img, (70, 86, 56), smoothstep(0.78, 0.92, spectral(rng, beta=1.7)) * 0.45)
    return finish(img, rng, 0.06)


def iron(rng):
    body = spectral(rng, beta=1.5)
    img = lerp((44, 44, 46), (72, 70, 70), body)
    rust = smoothstep(0.72, 0.9, spectral(rng, beta=1.6))
    img = lerp(img, (108, 62, 34), rust * 0.55)
    return finish(img, rng, 0.08)


def path_earth(rng):
    body = spectral(rng, beta=1.7)
    img = lerp((96, 78, 60), (132, 108, 82), body)
    pebbles = speckle(rng, 0.005, radius=2)
    img = lerp(img, (150, 140, 124), pebbles * 0.6)
    leaves_n = speckle(rng, 0.0015, radius=2)
    img = lerp(img, (126, 84, 40), leaves_n * 0.8)
    ruts = spectral(rng, beta=1.0, aniso=(8.0, 1.0))
    img = lerp(img, (70, 56, 44), smoothstep(0.8, 0.95, ruts) * 0.5)
    return finish(img, rng)


def water(rng):
    ripples = spectral(rng, beta=1.3)
    img = lerp((22, 46, 50), (46, 78, 82), ripples)
    return finish(img, rng, 0.04)


RECIPES = {
    "terrain0": ("forest floor: moss, leaf litter and twigs", forest_floor),
    "terrain1": ("clearing moss and grass", clearing_moss),
    "terrain2": ("grey rock with lichen", rock),
    "terrain3": ("tarn shore mud and pebbles", shore_mud),
    "terrain4": ("teal dell moss with pale flecks", dell_moss),
    "wk_bark": ("dark fibrous bark", bark),
    "wk_birchbark": ("pale birch bark with lenticels", birch_bark),
    "wk_needles": ("conifer needles", needles),
    "wk_leaves": ("broadleaf clusters", leaves),
    "wk_moss": ("moss", moss),
    "wk_rock": ("rock for boulders and standing stones", rock),
    "wk_stone": ("coursed masonry blocks", masonry),
    "wk_timber": ("weathered planks", timber),
    "wk_plaster": ("limewash daub", plaster),
    "wk_shingle": ("mossy shingles", shingles),
    "wk_iron": ("dark iron with rust", iron),
    "wk_path": ("packed trail earth", path_earth),
    "wk_water": ("dark tarn water", water),
}


def build(out_dir: Path, seed: int):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, (name, (description, fn)) in enumerate(RECIPES.items()):
        rng = np.random.default_rng([seed, index, 7331])
        pixels = fn(rng)
        path = out_dir / f"{name}.jpg"
        Image.fromarray(pixels, "RGB").save(path, "JPEG", quality=90, optimize=True, progressive=False, subsampling=0)
        data = path.read_bytes()
        records.append({"name": name, "file": path.name, "description": description, "generator": fn.__name__,
                        "seed": [seed, index, 7331], "size": [SIZE, SIZE], "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(), "tileable": True,
                        "meanRGB": [round(float(c), 1) for c in pixels.reshape(-1, 3).mean(0)]})
    (out_dir / "texture-provenance.json").write_text(json.dumps({
        "source": "Generated by pipeline/wk_textures.py from the Wickwood seed; original project images, no photographs or external images.",
        "textures": records}, indent=2) + "\n", encoding="utf-8")
    return records
