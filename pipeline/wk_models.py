"""Procedural Wickwood models. Every model's origin is its base centre (or attachment point) and its
front faces local +z. Units are metres; see wk_mesh for the native conventions."""
from __future__ import annotations

import math
import random

from wk_mesh import Mesh

TAU = math.tau


def rng_for(seed, name):
    return random.Random(f"{seed}:{name}")


# ---------------------------------------------------------------------------------------------- trees
def pine(name, seed, height, base_r, tiers, material="needles", spread=0.36):
    r = rng_for(seed, name)
    m = Mesh(name)
    m.cylinder(0, -0.3, 0, base_r, height * 0.92 + 0.3, "bark", n=8, r1=base_r * 0.25)
    first = height * 0.24
    for i in range(tiers):
        f = i / max(1, tiers - 1)
        y0 = first + (height - first) * f * 0.84
        radius = height * spread * (1 - 0.72 * f) * r.uniform(0.9, 1.08)
        h = height * (0.34 - 0.12 * f)
        m.cone(r.uniform(-0.15, 0.15), y0, r.uniform(-0.15, 0.15), radius, h, material, n=9, twist=r.uniform(0, TAU))
    return m


def oak(name, seed, height, base_r):
    r = rng_for(seed, name)
    m = Mesh(name)
    fork = height * 0.42
    m.cylinder(0, -0.3, 0, base_r, fork + 0.3, "bark", n=9, r1=base_r * 0.8)
    limbs = 4
    for i in range(limbs):
        a = i * TAU / limbs + r.uniform(-0.4, 0.4)
        reach = height * r.uniform(0.22, 0.3)
        top = (math.cos(a) * reach, fork + height * r.uniform(0.22, 0.32), math.sin(a) * reach)
        m.tube((0, fork - 0.4, 0), top, base_r * 0.55, base_r * 0.22, "bark", n=6)
        jitter = lambda i2, j2, a=a: 1 + 0.12 * math.sin(3 * i2 + 2 * j2 + a * 5)
        m.ellipsoid((top[0], top[1] + height * 0.06, top[2]), (height * 0.2, height * 0.14, height * 0.2), "leaves_oak", n=8, rings=5, jitter=jitter)
    m.ellipsoid((0, height * 0.82, 0), (height * 0.24, height * 0.16, height * 0.24), "leaves_oak", n=9, rings=5,
                jitter=lambda i2, j2: 1 + 0.1 * math.cos(2 * i2 + j2))
    return m


def birch(name, seed, height, base_r):
    r = rng_for(seed, name)
    m = Mesh(name)
    lean = (r.uniform(-0.4, 0.4), r.uniform(-0.4, 0.4))
    m.tube((0, -0.3, 0), (lean[0], height * 0.95, lean[1]), base_r, base_r * 0.35, "birch", n=7)
    for k in range(4):
        a = k * TAU / 4 + r.uniform(0, 1)
        y = height * (0.55 + 0.1 * k)
        c = (lean[0] * y / height + math.cos(a) * height * 0.08, y, lean[1] * y / height + math.sin(a) * height * 0.08)
        m.ellipsoid(c, (height * 0.13, height * 0.16, height * 0.13), "leaves_birch", n=7, rings=4,
                    jitter=lambda i2, j2, k=k: 1 + 0.12 * math.sin(i2 * 2.3 + j2 + k))
    return m


def snag(name, seed, height, base_r):
    r = rng_for(seed, name)
    m = Mesh(name)
    m.cylinder(0, -0.3, 0, base_r, height + 0.3, "deadwood", n=8, r1=base_r * 0.5)
    for k in range(3):
        a = r.uniform(0, TAU)
        y = height * r.uniform(0.45, 0.85)
        m.tube((0, y, 0), (math.cos(a) * height * 0.22, y + height * 0.12, math.sin(a) * height * 0.22), base_r * 0.3, base_r * 0.08, "deadwood", n=5)
    return m


# ---------------------------------------------------------------------------------------------- understory
def fern(name, seed, size):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(7):
        a = k * TAU / 7 + r.uniform(-0.2, 0.2)
        ca, sa = math.cos(a), math.sin(a)
        pts = []
        for s, (reach, lift, width) in enumerate([(0.0, 0.05, 0.02), (0.35, 0.45, 0.16), (0.7, 0.6, 0.14), (1.0, 0.35, 0.02)]):
            pts.append((ca * reach * size, lift * size, sa * reach * size, width * size))
        for (x0, y0, z0, w0), (x1, y1, z1, w1) in zip(pts, pts[1:]):
            nx, nz = -sa, ca
            quad = [(x0 - nx * w0, y0, z0 - nz * w0), (x1 - nx * w1, y1, z1 - nz * w1), (x1 + nx * w1, y1, z1 + nz * w1), (x0 + nx * w0, y0, z0 + nz * w0)]
            m.face(quad, "fern", double=True)
    return m


def shrub(name, seed, size):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(3):
        a = k * TAU / 3 + r.uniform(0, 1)
        m.ellipsoid((math.cos(a) * size * 0.3, size * 0.45, math.sin(a) * size * 0.3), (size * 0.5, size * 0.42, size * 0.5), "shrub", n=7, rings=4,
                    jitter=lambda i, j, k=k: 1 + 0.15 * math.sin(i * 1.7 + j * 2.1 + k))
    return m


def log(name, seed, length, radius):
    m = Mesh(name)
    m.tube((0, radius * 0.8, -length / 2), (0.1, radius * 0.75, length / 2), radius, radius * 0.9, "deadwood", n=8)
    m.box(0, radius * 1.5, 0, radius * 0.9, 0.04, length * 0.7, "moss")
    return m


def stump(name, seed):
    m = Mesh(name)
    m.cylinder(0, -0.2, 0, 0.45, 0.75, "bark", n=9, r1=0.38)
    m.disc(0, 0.56, 0, 0.36, "timber", n=9)
    m.box(0.1, 0.55, -0.1, 0.3, 0.05, 0.25, "moss")
    return m


def rock(name, seed, size, flat=0.6):
    r = rng_for(seed, name)
    m = Mesh(name)
    phase = [r.uniform(0, TAU) for _ in range(3)]
    m.ellipsoid((0, size * flat * 0.25, 0), (size * 0.6, size * flat * 0.6, size * 0.5), "rock", n=8, rings=5,
                jitter=lambda i, j: 1 + 0.16 * math.sin(i * 1.9 + phase[0]) * math.cos(j * 1.3 + phase[1]) + 0.06 * math.sin(i * 4 + phase[2]))
    m.box(0, size * flat * 0.62, 0, size * 0.5, 0.04, size * 0.4, "moss")
    return m


def toadstools(name, seed):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(4):
        x, z = r.uniform(-0.5, 0.5), r.uniform(-0.5, 0.5)
        h = r.uniform(0.12, 0.3)
        m.cylinder(x, 0, z, 0.03, h, "stem", n=6)
        m.cone(x, h, z, r.uniform(0.08, 0.14), 0.07, "cap_red", n=8)
    return m


def glowcaps(name, seed, material, count=6, scale=1.0):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(count):
        a, d = r.uniform(0, TAU), r.uniform(0, 0.8) * scale
        x, z = math.cos(a) * d, math.sin(a) * d
        h = r.uniform(0.15, 0.55) * scale
        cap = r.uniform(0.1, 0.24) * scale
        m.cylinder(x, 0, z, 0.03 * scale + 0.01, h, "stem", n=6)
        m.ellipsoid((x, h, z), (cap, cap * 0.45, cap), material, n=8, rings=3)
    return m


def glowmoss(name, seed, size):
    r = rng_for(seed, name)
    m = Mesh(name)
    n = 12
    pts = [(math.cos(-k * TAU / n) * size * r.uniform(0.6, 1.0), 0.03, math.sin(-k * TAU / n) * size * r.uniform(0.6, 1.0)) for k in range(n)]
    m.face(pts, "moss_glow")
    return m


def fireflies(name, seed, count, spread):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(count):
        c = (r.uniform(-spread, spread), r.uniform(0.4, 2.8), r.uniform(-spread, spread))
        s = 0.035
        top, bot = (c[0], c[1] + s, c[2]), (c[0], c[1] - s, c[2])
        ring = [(c[0] + s, c[1], c[2]), (c[0], c[1], c[2] + s), (c[0] - s, c[1], c[2]), (c[0], c[1], c[2] - s)]
        for i in range(4):
            j = (i + 1) % 4
            m.face([ring[i], top, ring[j]], "firefly")
            m.face([ring[j], bot, ring[i]], "firefly")
    return m


def reeds(name, seed):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(14):
        x, z = r.uniform(-0.6, 0.6), r.uniform(-0.6, 0.6)
        h = r.uniform(0.9, 1.7)
        a = r.uniform(0, TAU)
        w = 0.03
        m.face([(x - w * math.cos(a), 0, z - w * math.sin(a)), (x + w * math.cos(a), 0, z + w * math.sin(a)),
                (x + 0.1 * math.sin(a), h, z - 0.1 * math.cos(a))], "fern", double=True)
    return m


# ---------------------------------------------------------------------------------------------- lanterns
def hanging_lantern(m, x, y, z, chain=0.25, scale=1.0):
    """Adds a lantern that hangs from (x, y, z)."""
    s = scale
    if chain > 0:
        m.tube((x, y, z), (x, y - chain, z), 0.012, 0.012, "iron", n=4)
    top = y - chain
    m.cone(x, top - 0.12 * s, z, 0.17 * s, 0.12 * s, "iron", n=6)
    m.box(x, top - 0.44 * s, z, 0.22 * s, 0.32 * s, 0.22 * s, "lanternglass")
    for dx in (-1, 1):
        for dz in (-1, 1):
            m.box(x + dx * 0.115 * s, top - 0.44 * s, z + dz * 0.115 * s, 0.025 * s, 0.32 * s, 0.025 * s, "iron")
    m.box(x, top - 0.48 * s, z, 0.28 * s, 0.04 * s, 0.28 * s, "iron")


def lantern_post(name):
    m = Mesh(name)
    m.box(0, -0.25, 0, 0.5, 0.4, 0.5, "stone")
    m.box(0, 0.15, 0, 0.14, 2.35, 0.14, "darktimber")
    m.tube((0, 2.35, -0.02), (0, 2.5, 0.25), 0.045, 0.04, "darktimber", n=5)
    m.tube((0, 2.5, 0.25), (0, 2.45, 0.55), 0.04, 0.035, "darktimber", n=5)
    hanging_lantern(m, 0, 2.45, 0.55, chain=0.18)
    return m


def lantern_hang(name):
    m = Mesh(name)
    hanging_lantern(m, 0, 0, 0, chain=0.3)
    return m


def lantern_stone(name):
    m = Mesh(name)
    m.box(0, 0, 0, 0.7, 0.2, 0.7, "stone")
    m.cylinder(0, 0.2, 0, 0.14, 0.7, "stone", n=8)
    m.box(0, 0.9, 0, 0.62, 0.08, 0.62, "stone")
    m.box(0, 0.98, 0, 0.42, 0.34, 0.42, "lanternglass")
    for dx in (-1, 1):
        for dz in (-1, 1):
            m.box(dx * 0.19, 0.98, dz * 0.19, 0.07, 0.34, 0.07, "stone")
    m.box(0, 1.32, 0, 0.8, 0.08, 0.8, "stone", top_scale=0.6)
    m.cone(0, 1.4, 0, 0.36, 0.28, "stone", n=4, twist=TAU / 8)
    return m


def lantern_float(name):
    m = Mesh(name)
    m.cylinder(0, -0.05, 0, 0.28, 0.1, "timber", n=8)
    m.box(0, 0.05, 0, 0.26, 0.34, 0.26, "lanternglass")
    m.box(0, 0.39, 0, 0.3, 0.03, 0.3, "darktimber")
    return m


def lantern_tree(name):
    m = Mesh(name)
    oct8 = [(math.cos(k * TAU / 8 + TAU / 16) * 2.8, math.sin(k * TAU / 8 + TAU / 16) * 2.8) for k in range(8)]
    m.slab(oct8, -0.3, 0.35, "stone")
    oct6 = [(math.cos(k * TAU / 8 + TAU / 16) * 1.3, math.sin(k * TAU / 8 + TAU / 16) * 1.3) for k in range(8)]
    m.slab(oct6, 0.35, 0.7, "stone")
    m.cylinder(0, 0.7, 0, 0.16, 7.0, "iron", n=8, r1=0.09)
    for tier, (y, reach) in enumerate([(4.6, 1.5), (6.4, 1.0)]):
        for k in range(4):
            a = k * TAU / 4 + tier * TAU / 8
            ca, sa = math.cos(a), math.sin(a)
            mid = (ca * reach * 0.55, y + 0.35, sa * reach * 0.55)
            tip = (ca * reach, y + 0.2, sa * reach)
            m.tube((0, y - 0.2, 0), mid, 0.05, 0.04, "iron", n=5)
            m.tube(mid, tip, 0.04, 0.03, "iron", n=5)
            hanging_lantern(m, tip[0], tip[1], tip[2], chain=0.25, scale=1.3)
    hanging_lantern(m, 0, 7.75, 0, chain=0.0, scale=1.5)
    m.cone(0, 7.7, 0, 0.25, 0.5, "iron", n=6)
    return m


def candles(name, seed, count=7, spread=0.35):
    r = rng_for(seed, name)
    m = Mesh(name)
    for k in range(count):
        a, d = r.uniform(0, TAU), r.uniform(0, spread)
        x, z = math.cos(a) * d, math.sin(a) * d
        h = r.uniform(0.12, 0.42)
        m.cylinder(x, 0, z, r.uniform(0.03, 0.05), h, "candle", n=6)
        m.cone(x, h + 0.01, z, 0.02, 0.07, "flame", n=4)
    return m


def brazier(name):
    m = Mesh(name)
    for k in range(3):
        a = k * TAU / 3
        m.tube((math.cos(a) * 0.9, 0, math.sin(a) * 0.9), (math.cos(a) * 0.35, 1.2, math.sin(a) * 0.35), 0.06, 0.05, "iron", n=5)
    m.cylinder(0, 1.1, 0, 0.35, 0.3, "iron", n=10, r1=0.85)
    m.disc(0, 1.38, 0, 0.8, "ember", n=10)
    for k in range(5):
        a = k * TAU / 5
        m.cone(math.cos(a) * 0.35, 1.38, math.sin(a) * 0.35, 0.22, 0.9, "flame", n=5)
    m.cone(0, 1.38, 0, 0.35, 1.5, "flame", n=6)
    return m


def wayshrine(name):
    m = Mesh(name)
    m.box(0, -0.2, 0, 0.18, 1.7, 0.18, "darktimber")
    m.box(0, 1.5, 0, 0.6, 0.06, 0.5, "darktimber")
    m.box(0, 1.56, -0.2, 0.56, 0.55, 0.06, "timber")  # ends tuck inside the side panels (no shared side faces)
    for dx in (-0.27, 0.27):
        m.box(dx, 1.56, 0.02, 0.06, 0.55, 0.42, "timber")
    hanging_lantern(m, 0, 2.08, 0.02, chain=0.08, scale=0.9)
    m.box(0, 2.11, 0.0625, 0.8, 0.06, 0.775, "shingle")  # one board: two overlapping boards flickered
    return m


def gate(name):
    m = Mesh(name)
    for x in (-1.9, 1.9):
        m.box(x, -0.3, 0, 0.3, 3.9, 0.3, "darktimber")
        m.box(x, -0.3, 0, 0.5, 0.5, 0.5, "stone")
    m.box(0, 3.6, 0, 4.6, 0.26, 0.34, "darktimber")
    m.box(0, 3.86, 0, 5.0, 0.12, 0.5, "shingle")
    for x in (-1.2, 1.2):
        hanging_lantern(m, x, 3.6, 0, chain=0.25)
    hanging_lantern(m, 0, 3.6, 0, chain=0.1, scale=1.2)
    return m


# ---------------------------------------------------------------------------------------------- paths
PATH_WIDTH = 3.2


def path_segment(name, grade):
    """10 m trail ribbon rising `grade` along local +z; top surface 7 cm above the design line."""
    m = Mesh(name)
    start = m._begin_closed("path")
    w = PATH_WIDTH / 2
    rise = grade * 10.0
    top0, top1, bot = 0.07, 0.07 + rise, -0.06
    uv_top = [(0, 0), (0, 10 / 3.2), (1, 10 / 3.2), (1, 0)]
    # top: seen from above, (-x,0) -> (-x,10) -> (+x,10) -> (+x,0) faces up
    m.face([(-w, top0, 0), (-w, top1, 10), (w, top1, 10), (w, top0, 0)], "path", uv=uv_top)
    m.face([(-w, bot, 0), (w, bot, 0), (w, bot + rise, 10), (-w, bot + rise, 10)], "path")
    m.face([(-w, bot, 0), (-w, bot + rise, 10), (-w, top1, 10), (-w, top0, 0)], "path")
    m.face([(w, bot, 0), (w, top0, 0), (w, top1, 10), (w, bot + rise, 10)], "path")
    m.face([(-w, bot, 0), (-w, top0, 0), (w, top0, 0), (w, bot, 0)], "path")
    m.face([(-w, bot + rise, 10), (w, bot + rise, 10), (w, top1, 10), (-w, top1, 10)], "path")
    m._end_closed("path", start)
    return m


def path_joint(name):
    m = Mesh(name)
    ring = [(math.cos(k * TAU / 10) * 1.62, math.sin(k * TAU / 10) * 1.62) for k in range(10)]
    m.slab(ring, -0.06, 0.078, "path")
    return m


def paving_ring(name, inner, outer, n=24):
    m = Mesh(name)
    for k in range(n):
        a0, a1 = k * TAU / n, (k + 1) * TAU / n
        quad = [(math.cos(a0) * inner, math.sin(a0) * inner), (math.cos(a0) * outer, math.sin(a0) * outer),
                (math.cos(a1) * outer, math.sin(a1) * outer), (math.cos(a1) * inner, math.sin(a1) * inner)]
        m.slab(quad, -0.08, 0.08, "stone")
    return m


# ---------------------------------------------------------------------------------------------- buildings
def wall_with_openings(m, x0, x1, z, thickness, height, openings, material, y0=0.0, axis="x"):
    """A straight wall from x0 to x1 (along local x at depth z, or along z when axis='z') with rectangular
    openings [(centre, width, bottom, top)] cut as a set of closed boxes."""
    cuts = sorted(openings)
    spans = []
    cursor = x0
    for c, w, b, t in cuts:
        spans.append((cursor, c - w / 2, y0, y0 + height))
        spans.append((c - w / 2, c + w / 2, y0, y0 + b))
        spans.append((c - w / 2, c + w / 2, y0 + t, y0 + height))
        cursor = c + w / 2
    spans.append((cursor, x1, y0, y0 + height))
    for a, b, lo, hi in spans:
        if b - a < 1e-3 or hi - lo < 1e-3:
            continue
        if axis == "x":
            m.box((a + b) / 2, lo, z, b - a, hi - lo, thickness, material)
        else:
            m.box(z, lo, (a + b) / 2, thickness, hi - lo, b - a, material)


def window_pane(m, cx, cz, y0, y1, w, axis="x"):
    if axis == "x":
        m.face([(cx - w / 2, y0, cz), (cx + w / 2, y0, cz), (cx + w / 2, y1, cz), (cx - w / 2, y1, cz)], "window", double=True)
    else:
        m.face([(cx, y0, cz - w / 2), (cx, y0, cz + w / 2), (cx, y1, cz + w / 2), (cx, y1, cz - w / 2)], "window", double=True)


def prism_x(m, section, xa, xb, material):
    """Closed prism: polygon `section` of (z, y) points extruded along x from xa to xb."""
    q = [(y, z) for z, y in section]
    area = sum(q[i][0] * q[(i + 1) % len(q)][1] - q[(i + 1) % len(q)][0] * q[i][1] for i in range(len(q)))
    if area < 0:
        q = q[::-1]
    start = m._begin_closed(material)
    m.face([(xb, y, z) for y, z in q], material)
    m.face([(xa, y, z) for y, z in q[::-1]], material)
    for i in range(len(q)):
        j = (i + 1) % len(q)
        m.face([(xa, q[i][0], q[i][1]), (xa, q[j][0], q[j][1]), (xb, q[j][0], q[j][1]), (xb, q[i][0], q[i][1])], material)
    m._end_closed(material, start)


def gable_roof(m, half_w, half_d, eave, ridge, overhang=0.45, thickness=0.14, material="shingle"):
    """Ridge along local x: two sloped boards whose undersides pass through the wall tops at z = +/-half_d."""
    ze = half_d + overhang
    drop = (ridge - eave) * overhang / half_d
    for side in (1, -1):
        section = [(side * ze, eave - drop), (0.0, ridge), (0.0, ridge + thickness), (side * ze, eave - drop + thickness)]
        prism_x(m, section, -half_w - overhang, half_w + overhang, material)
    m.box(0, ridge + thickness - 0.04, 0, 2 * half_w + 2 * overhang, 0.14, 0.3, "darktimber")


def gable_ends(m, half_w, half_d, eave, ridge, thickness=0.25, material="plaster"):
    """Closed triangular gable walls at x = +/-half_w from the wall tops up to the roof underside."""
    for sx in (-1, 1):
        section = [(-half_d - thickness / 2, eave), (half_d + thickness / 2, eave), (0.0, ridge + 0.02)]
        x0, x1 = sx * half_w - thickness / 2, sx * half_w + thickness / 2
        prism_x(m, section, x0, x1, material)


def cottage(name, seed, width, depth, door_x=0.0, windows=((-2.1,), (2.1,)), chimney=True):
    """Door on the front (+z) long wall, ridge along x, closed gables at +/-x, walkable timber floor."""
    r = rng_for(seed, name)
    m = Mesh(name)
    hw, hd, t = width / 2, depth / 2, 0.25
    eave, ridge = 3.0, 3.0 + depth * 0.42
    m.box(0, -0.4, 0, width + 0.5, 0.4, depth + 0.5, "stone")
    m.box(0, 0.0, 0, width - t, 0.15, depth - t, "timber")
    front = [(door_x, 1.3, 0.0, 2.35)] + [(wx, 0.9, 1.15, 2.05) for (wx,) in windows]
    wall_with_openings(m, -hw, hw, hd, t, eave, front, "plaster")
    wall_with_openings(m, -hw, hw, -hd, t, eave, [(wx, 0.9, 1.15, 2.05) for (wx,) in windows], "plaster")
    wall_with_openings(m, -hd + t / 2, hd - t / 2, hw, t, eave, [(0.0, 0.8, 1.2, 2.0)], "plaster", axis="z")
    wall_with_openings(m, -hd + t / 2, hd - t / 2, -hw, t, eave, [(0.0, 0.8, 1.2, 2.0)], "plaster", axis="z")
    for (wx,) in windows:
        window_pane(m, wx, hd, 1.15, 2.05, 0.9)
        window_pane(m, wx, -hd, 1.15, 2.05, 0.9)
    window_pane(m, hw, 0.0, 1.2, 2.0, 0.8, axis="z")
    window_pane(m, -hw, 0.0, 1.2, 2.0, 0.8, axis="z")
    for sx in (-1, 1):
        for sz in (-1, 1):
            m.box(sx * (hw + 0.02), 0, sz * (hd + 0.02), 0.3, eave, 0.3, "darktimber")
    m.box(0, eave - 0.2, hd + 0.16, width + 0.3, 0.2, 0.08, "darktimber")
    m.box(0, eave - 0.2, -hd - 0.16, width + 0.3, 0.2, 0.08, "darktimber")
    m.box(door_x, 2.35, hd + 0.16, 1.6, 0.16, 0.08, "darktimber")
    gable_ends(m, hw, hd, eave, ridge)
    gable_roof(m, hw, hd, eave, ridge)
    if chimney:
        cx = hw * 0.55 * (1 if r.random() < 0.5 else -1)
        m.box(cx, ridge - 1.6, -hd * 0.35, 0.7, 2.9, 0.7, "stone")
        m.disc(cx, ridge + 1.31, -hd * 0.35, 0.22, "ember", n=6)
    hanging_lantern(m, door_x + 0.95, 2.5, hd + 0.35, chain=0.2)
    m.tube((door_x + 0.95, 2.5, hd + 0.02), (door_x + 0.95, 2.5, hd + 0.4), 0.03, 0.03, "iron", n=4)
    hanging_lantern(m, 0, eave - 0.1, 0, chain=0.3, scale=1.1)
    m.box(-hw * 0.4, 0.15, -hd * 0.3, 1.4, 0.8, 0.8, "timber")
    return m


def workshop(name):
    m = Mesh(name)
    w, d, t = 10.0, 6.0, 0.25
    hw, hd = w / 2, d / 2
    eave, ridge = 3.2, 5.6
    m.box(0, -0.4, 0, w + 0.4, 0.4, d + 0.4, "stone")
    m.box(0, 0, 0, w - t, 0.15, d - t, "timber")
    wall_with_openings(m, -hw, hw, -hd, t, eave, [(-2.5, 1.0, 1.2, 2.1), (2.5, 1.0, 1.2, 2.1)], "timber")
    wall_with_openings(m, -hd + t / 2, hd - t / 2, hw, t, eave, [], "timber", axis="z")
    wall_with_openings(m, -hd + t / 2, hd - t / 2, -hw, t, eave, [], "timber", axis="z")
    window_pane(m, -2.5, -hd, 1.2, 2.1, 1.0)
    window_pane(m, 2.5, -hd, 1.2, 2.1, 1.0)
    for x in (-hw + 0.15, -1.7, 1.7, hw - 0.15):
        m.box(x, 0.15, hd - 0.15, 0.25, eave - 0.15, 0.25, "darktimber")
    m.box(0, eave - 0.25, hd - 0.15, w, 0.25, 0.3, "darktimber")
    gable_ends(m, hw, hd, eave, ridge, material="timber")
    gable_roof(m, hw, hd, eave, ridge)
    m.box(0, 0.15, -hd + 1.0, 6.0, 0.9, 0.9, "timber")
    for k in range(5):
        hanging_lantern(m, -3 + k * 1.5, eave - 0.05, 0.3, chain=0.35 + 0.1 * (k % 2))
    for k in range(6):  # display lanterns spaced along the 6 m workbench top (x -3.0 to 3.0)
        m.box(-2.5 + k * 1.0, 1.05, -hd + 0.9, 0.2, 0.28, 0.2, "lanternglass")
    return m


def well(name):
    m = Mesh(name)
    n = 12
    for k in range(n):
        a0, a1 = k * TAU / n, (k + 1) * TAU / n
        quad = [(math.cos(a0) * 0.8, math.sin(a0) * 0.8), (math.cos(a0) * 1.15, math.sin(a0) * 1.15),
                (math.cos(a1) * 1.15, math.sin(a1) * 1.15), (math.cos(a1) * 0.8, math.sin(a1) * 0.8)]
        m.slab(quad, -0.2, 0.9, "stone")
    m.disc(0, 0.2, 0, 0.8, "water", n=12)
    for x in (-1.0, 1.0):
        m.box(x, 0.9, 0, 0.16, 1.7, 0.16, "darktimber")
    m.box(0, 2.6, 0, 2.4, 0.14, 0.14, "darktimber")
    m.box(0, 2.74, 0, 2.6, 0.08, 1.9, "shingle")  # one board: two overlapping halves flickered along the ridge
    hanging_lantern(m, 0.5, 2.6, 0, chain=0.2)
    m.cylinder(-0.3, 1.1, 0, 0.16, 0.25, "timber", n=8, r1=0.18)
    # rope from the crossbeam to the bucket; its ends sit 2 cm inside the beam and the bucket's lid
    m.tube((-0.3, 2.62, 0), (-0.3, 1.33, 0), 0.018, 0.018, "rope", n=6)
    return m


def fence(name):
    m = Mesh(name)
    for x in (-3.0, -1.0, 1.0, 3.0):
        m.box(x, -0.2, 0, 0.12, 1.3, 0.12, "darktimber")
    for y in (0.45, 0.9):
        m.box(0, y, 0, 6.2, 0.08, 0.06, "timber")
    return m


def bench(name):
    m = Mesh(name)
    m.box(0, 0.42, 0, 1.8, 0.1, 0.45, "timber")
    for x in (-0.7, 0.7):
        m.box(x, 0, 0, 0.12, 0.42, 0.4, "darktimber")
        # back post in line with the leg: rises from inside the seat, frames the backrest and caps 5 cm above it
        m.box(x, 0.47, -0.2, 0.1, 0.73, 0.08, "darktimber")
    m.box(0, 0.8, -0.2, 1.8, 0.35, 0.06, "timber")
    return m


def lantern_rack(name):
    m = Mesh(name)
    for x in (-1.2, 1.2):
        m.box(x, 0, 0, 0.12, 2.26, 0.12, "darktimber")  # posts cap 6 cm above the top bar
    m.box(0, 2.1, 0, 2.6, 0.1, 0.1, "darktimber")  # 2 cm shallower than the posts, so no faces share a plane
    m.box(0, 1.2, 0, 2.6, 0.08, 0.1, "darktimber")
    for k in range(4):
        hanging_lantern(m, -0.9 + k * 0.6, 2.1, 0, chain=0.15 + 0.05 * (k % 2), scale=0.8)
        hanging_lantern(m, -0.9 + k * 0.6, 1.2, 0, chain=0.1, scale=0.7)
    return m


def jetty(name, length=32.0, deck=1.7):
    m = Mesh(name)
    w = 1.6
    m.box(0, deck - 0.12, length / 2, 2 * w, 0.12, length, "timber")
    for z in [k * 4.0 for k in range(int(length // 4) + 1)]:
        for x in (-w + 0.12, w - 0.12):
            m.box(x, -2.5, min(z, length - 0.12), 0.2, deck - 0.12 + 2.5, 0.2, "darktimber")
    for x in (-w + 0.06, w - 0.06):
        m.box(x, deck + 0.9, length / 2 + 2, 0.08, 0.08, length - 4, "timber")
        for z in [4.0 + k * 4.0 for k in range(int((length - 4) // 4) + 1)]:
            m.box(x, deck, z, 0.1, 1.0, 0.1, "darktimber")
    for z in (8.0, 16.0, 24.0, length - 0.5):
        m.box(w - 0.06, deck, z, 0.12, 2.2, 0.12, "darktimber")
        m.tube((w - 0.06, deck + 2.15, z), (w - 0.5, deck + 2.25, z), 0.035, 0.03, "darktimber", n=4)
        hanging_lantern(m, w - 0.5, deck + 2.25, z, chain=0.15)
    return m


def rowboat(name):
    m = Mesh(name)
    hull = [(0, 2.1), (0.7, 1.4), (0.8, 0), (0.7, -1.4), (0, -1.9), (-0.7, -1.4), (-0.8, 0), (-0.7, 1.4)]
    m.slab(hull, -0.25, 0.35, "darktimber")
    m.box(0, 0.35, 0, 1.4, 0.06, 0.3, "timber")
    m.box(0, 0.35, 0.9, 1.2, 0.06, 0.3, "timber")
    hanging_lantern(m, 0, 1.1, -1.5, chain=0.1, scale=0.8)
    m.box(0, 0.35, -1.5, 0.06, 0.8, 0.06, "darktimber")
    return m


def beacon(name):
    m = Mesh(name)
    for k in range(6):
        size = 9.0 - k * 1.1
        y0 = -0.5 if k == 0 else 0.25 * k
        m.box(0, y0, 0, size, 0.25 * (k + 1) - y0, size, "stone")
    top = 1.5
    m.merge(brazier("wk_tmp_brazier"), 0, top, 0)
    return m


def spire(name):
    m = Mesh(name)
    m.box(0, -0.3, 0, 1.6, 1.0, 1.6, "stone")
    m.box(0, 0.7, 0, 1.0, 12.0, 1.0, "stone", top_scale=0.55)
    m.box(0, 12.7, 0, 0.75, 0.12, 0.75, "stone")
    hanging_lantern(m, 0, 13.6, 0, chain=0.0, scale=1.8)
    m.cone(0, 13.55, 0, 0.5, 0.9, "iron", n=6)
    return m


def standing_stone(name, seed, height, width):
    r = rng_for(seed, name)
    m = Mesh(name)
    phase = r.uniform(0, TAU)
    m.ellipsoid((0, height * 0.42, 0), (width * 0.5, height * 0.55, width * 0.3), "rock", n=7, rings=6,
                jitter=lambda i, j: 1 + 0.08 * math.sin(i * 2.1 + phase) + 0.05 * math.cos(j * 3 + phase))
    return m


def altar(name, seed):
    m = Mesh(name)
    for x in (-0.9, 0.9):
        m.box(x, -0.2, 0, 0.5, 1.0, 0.8, "rock")
    m.box(0, 0.8, 0, 2.6, 0.22, 1.2, "rock")
    m.merge(candles("wk_tmp_altar_candles", seed, count=9, spread=0.45), 0, 1.02, 0)
    return m


def signboard(name, width=2.4, height=1.2, board_y=1.25):
    """Post-mounted board; the tag-100 face on the front (+z) carries the native sign text."""
    m = Mesh(name)
    for x in (-width / 2 + 0.2, width / 2 - 0.2):
        m.box(x, -0.3, -0.12, 0.12, board_y + height + 0.35, 0.1, "darktimber")
    m.box(0, board_y - 0.06, 0, width + 0.16, height + 0.12, 0.14, "timber")
    y0, y1 = board_y, board_y + height
    face_z = 0.09  # 2 cm in front of the board (as in the tested Lumenwood sign) to avoid depth fighting
    m.face([(-width / 2, y0, face_z), (width / 2, y0, face_z), (width / 2, y1, face_z), (-width / 2, y1, face_z)], "sign",
           uv=[(0, 1), (1, 1), (1, 0), (0, 0)])
    m.box(0, y1 + 0.06, 0, width + 0.4, 0.08, 0.34, "shingle")
    return m


def milestone(name):
    m = Mesh(name)
    m.box(0, -0.2, 0, 0.4, 0.9, 0.25, "stone", top_scale=0.8)
    m.box(0, 0.62, 0.13, 0.22, 0.04, 0.02, "moss")
    return m


def giant_oak(name, seed):
    """Hollow Oak: a walk-through chamber (front and back openings) under a solid trunk and limbs."""
    r = rng_for(seed, name)
    m = Mesh(name)
    r_out, r_in, chamber = 4.2, 2.9, 7.5
    gap = math.radians(24)
    segs = 26
    for k in range(segs):
        a0, a1 = k * TAU / segs, (k + 1) * TAU / segs
        mid = (a0 + a1) / 2
        # openings face local +z (front) and -z (back): directions where cos(angle - pi/2) ~ +/-1
        if abs(math.sin(mid)) > math.cos(gap) and abs(math.cos(mid)) < math.sin(gap) * 1.6:
            continue
        quad = [(math.cos(a0) * r_in, math.sin(a0) * r_in), (math.cos(a0) * r_out, math.sin(a0) * r_out),
                (math.cos(a1) * r_out, math.sin(a1) * r_out), (math.cos(a1) * r_in, math.sin(a1) * r_in)]
        m.slab(quad, -0.4, chamber, "bark")
    m.cylinder(0, chamber, 0, 4.0, 9.5, "bark", n=14, r1=2.5)
    for k in range(3):
        hanging_lantern(m, math.cos(k * TAU / 3) * 1.1, chamber, math.sin(k * TAU / 3) * 1.1, chain=0.9 + 0.3 * k, scale=1.2)
    for k in range(8):
        a = k * TAU / 8 + TAU / 16
        if abs(math.cos(a)) < 0.45:
            continue
        m.tube((math.cos(a) * 3.6, 1.6, math.sin(a) * 3.6), (math.cos(a) * 6.8, -0.3, math.sin(a) * 6.8), 0.9, 0.35, "bark", n=6)
    limb_tips = []
    for k in range(5):
        a = k * TAU / 5 + r.uniform(-0.3, 0.3)
        base = (math.cos(a) * 1.2, chamber + 8.0, math.sin(a) * 1.2)
        tip = (math.cos(a) * r.uniform(9, 12), chamber + r.uniform(15, 19), math.sin(a) * r.uniform(9, 12))
        m.tube(base, tip, 1.1, 0.35, "bark", n=7)
        limb_tips.append(tip)
        low = (math.cos(a) * 6.0, chamber + 10.5, math.sin(a) * 6.0)
        hanging_lantern(m, low[0], low[1], low[2], chain=1.4, scale=1.3)
    for k, tip in enumerate(limb_tips):
        m.ellipsoid((tip[0], tip[1] + 2.0, tip[2]), (8.5, 5.5, 8.5), "leaves_oak", n=9, rings=5,
                    jitter=lambda i, j, k=k: 1 + 0.12 * math.sin(i * 1.7 + j + k))
    m.ellipsoid((0, chamber + 22.0, 0), (10.0, 6.0, 10.0), "leaves_oak", n=10, rings=5, jitter=lambda i, j: 1 + 0.1 * math.cos(i * 2 + j))
    return m


# ---------------------------------------------------------------------------------------------- avatars
def avatar_lamplighter(name):
    m = Mesh(name)
    m.cylinder(0, 0.0, 0, 0.36, 1.32, "cloth", n=10, r1=0.24)
    m.ellipsoid((0, 1.5, 0), (0.24, 0.12, 0.2), "cloth", n=8, rings=4)
    m.ellipsoid((0, 1.62, 0.02), (0.11, 0.13, 0.11), "stem", n=8, rings=5)
    m.cone(0, 1.55, -0.02, 0.17, 0.3, "cloth", n=8)
    m.tube((0.22, 1.42, 0.0), (0.3, 1.05, 0.2), 0.06, 0.05, "cloth", n=5)
    m.tube((-0.22, 1.42, 0.0), (-0.28, 1.0, 0.08), 0.06, 0.05, "cloth", n=5)
    m.tube((0.32, 0.02, 0.24), (0.32, 1.95, 0.24), 0.02, 0.02, "darktimber", n=5)
    hanging_lantern(m, 0.32, 1.96, 0.3, chain=0.1, scale=0.55)
    return m


def avatar_wanderer(name):
    m = Mesh(name)
    m.cylinder(0, 0.0, 0, 0.34, 1.3, "darktimber", n=10, r1=0.23)
    m.ellipsoid((0, 1.48, 0), (0.23, 0.12, 0.19), "cloth", n=8, rings=4)
    m.ellipsoid((0, 1.6, 0.02), (0.11, 0.13, 0.11), "stem", n=8, rings=5)
    m.cylinder(0, 1.68, 0.02, 0.36, 0.03, "cloth", n=12)
    m.cone(0, 1.71, 0.02, 0.15, 0.22, "cloth", n=8)
    m.box(0, 0.95, -0.24, 0.36, 0.45, 0.16, "timber")
    m.tube((-0.3, 0.0, 0.18), (-0.3, 1.45, 0.14), 0.02, 0.02, "darktimber", n=5)
    m.tube((0.2, 1.4, 0.0), (0.26, 1.02, 0.12), 0.055, 0.05, "cloth", n=5)
    m.tube((-0.21, 1.4, 0.0), (-0.3, 1.1, 0.14), 0.055, 0.05, "cloth", n=5)
    return m


# ---------------------------------------------------------------------------------------------- catalogue
def catalogue(seed):
    """All world models (name -> Mesh). Temporary helper meshes are merged, never exported."""
    models = []
    for size, (h, r, t) in {"s": (9, 0.2, 4), "m": (14, 0.3, 5), "l": (20, 0.42, 6)}.items():
        models.append(pine(f"wk_pine_{size}", seed, h, r, t))
    models.append(pine("wk_spruce_m", seed, 16, 0.3, 7, "needles_dark", spread=0.26))
    models.append(pine("wk_spruce_l", seed, 22, 0.4, 8, "needles_dark", spread=0.24))
    for size, (h, r) in {"s": (10, 0.35), "m": (14, 0.5), "l": (18, 0.7)}.items():
        models.append(oak(f"wk_oak_{size}", seed, h, r))
    for size, (h, r) in {"s": (9, 0.14), "m": (12, 0.18), "l": (15, 0.22)}.items():
        models.append(birch(f"wk_birch_{size}", seed, h, r))
    models.append(snag("wk_snag_a", seed, 8, 0.3))
    models.append(snag("wk_snag_b", seed, 12, 0.4))
    models.append(giant_oak("wk_giant_oak", seed))
    models += [fern("wk_fern_a", seed, 1.1), fern("wk_fern_b", seed, 1.6), shrub("wk_shrub_a", seed, 1.4), shrub("wk_shrub_b", seed, 2.0),
               log("wk_log_a", seed, 4.5, 0.35), log("wk_log_b", seed, 6.5, 0.5), stump("wk_stump", seed),
               rock("wk_rock_a", seed, 0.9), rock("wk_rock_b", seed, 1.7), rock("wk_boulder", seed, 3.2, 0.7),
               toadstools("wk_toadstools", seed), glowcaps("wk_glowcap_teal", seed, "glow_teal"),
               glowcaps("wk_glowcap_amber", seed, "glow_amber"), glowcaps("wk_glowcap_violet", seed, "glow_violet"),
               glowcaps("wk_glowcap_giant", seed, "glow_teal", count=5, scale=2.4),
               glowmoss("wk_glowmoss", seed, 2.2), fireflies("wk_fireflies_a", seed, 26, 2.2), fireflies("wk_fireflies_b", seed, 40, 3.5),
               reeds("wk_reeds", seed)]
    models += [lantern_post("wk_lantern_post"), lantern_hang("wk_lantern_hang"), lantern_stone("wk_lantern_stone"),
               lantern_float("wk_lantern_float"), lantern_tree("wk_lanterntree"), candles("wk_candles", seed),
               brazier("wk_brazier"), wayshrine("wk_wayshrine"), gate("wk_gate")]
    for g in range(0, 16, 2):
        models.append(path_segment(f"wk_path_g{g:02d}", g / 100))
    models += [path_joint("wk_path_joint"), paving_ring("wk_paving_ring", 3.0, 9.5)]
    models += [cottage("wk_cottage_a", seed, 7.0, 6.0), cottage("wk_cottage_b", seed, 8.4, 6.4, door_x=-1.4, windows=((1.0,), (2.9,))),
               workshop("wk_workshop"), well("wk_well"), fence("wk_fence"), bench("wk_bench"), lantern_rack("wk_lantern_rack"),
               jetty("wk_jetty"), rowboat("wk_rowboat"), beacon("wk_beacon"), spire("wk_spire"),
               standing_stone("wk_stone_a", seed, 4.2, 1.3), standing_stone("wk_stone_b", seed, 3.4, 1.6), standing_stone("wk_stone_c", seed, 5.0, 1.2),
               altar("wk_altar", seed), signboard("wk_sign"), signboard("wk_board", width=3.6, height=1.8, board_y=1.2), milestone("wk_milestone")]
    return {m.name: m for m in models}


def avatars():
    return {"wk_av_lamplighter": ("Wickwood Lamplighter", avatar_lamplighter("wk_av_lamplighter")),
            "wk_av_wanderer": ("Wickwood Wanderer", avatar_wanderer("wk_av_wanderer"))}
