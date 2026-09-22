"""Deterministic Wickwood native terrain, trail profiles and a height sampler.

Grid: 512 x 512 samples at 10 m spacing; sample (i, j) sits at x = -2560 + 10 i, z = -2560 + 10 j
(the minimum corner of cell (i - 256, j - 256)). Buffers are Z-major with X varying fastest:
int32 little-endian centimetre heights and uint16 texture ids, as the native TerrainSet rows expect.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

SIZE = 512
MIN_CELL = -256
SPACING = 10.0
MIN_M = MIN_CELL * SPACING
XS = MIN_M + SPACING * np.arange(SIZE)
ZS = MIN_M + SPACING * np.arange(SIZE)
GX, GZ = np.meshgrid(XS, ZS)  # [j, i]

TARN_PLAZA = (455.0, 298.0, 18.0, 1.6)  # x, z, radius, height of the tarn arrival shore
ISLAND = (612.0, 352.0, 9.0, 1.3)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def field_noise(rng, beta):
    white = rng.standard_normal((SIZE, SIZE))
    f = np.sqrt(np.fft.fftfreq(SIZE)[:, None] ** 2 + np.fft.fftfreq(SIZE)[None, :] ** 2)
    f[0, 0] = 1.0
    spec = np.fft.fft2(white) / f ** beta
    spec[0, 0] = 0
    n = np.real(np.fft.ifft2(spec))
    return n / (np.abs(n).max() + 1e-12)


class Sampler:
    """Heights of the quantized native grid. Both triangle diagonals are evaluated because the
    client's split is not documented; ground() returns the lower of the candidates."""

    def __init__(self, heights_m: np.ndarray):
        self.h = heights_m

    def _corners(self, x, z):
        gx = (np.asarray(x, float) - MIN_M) / SPACING
        gz = (np.asarray(z, float) - MIN_M) / SPACING
        ix = np.clip(np.floor(gx).astype(int), 0, SIZE - 2)
        iz = np.clip(np.floor(gz).astype(int), 0, SIZE - 2)
        u, v = gx - ix, gz - iz
        a, b = self.h[iz, ix], self.h[iz, ix + 1]
        c, d = self.h[iz + 1, ix], self.h[iz + 1, ix + 1]
        return a, b, c, d, u, v

    def candidates(self, x, z):
        a, b, c, d, u, v = self._corners(x, z)
        bil = a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v
        diag_a = np.where(v <= u, a + (b - a) * u + (d - b) * v, a + (d - c) * u + (c - a) * v)
        diag_b = np.where(u + v <= 1, a + (b - a) * u + (c - a) * v, d + (c - d) * (1 - u) + (b - d) * (1 - v))
        return bil, diag_a, diag_b

    def height(self, x, z):
        return self.candidates(x, z)[0]

    def ground(self, x, z):
        return np.minimum.reduce(self.candidates(x, z))

    def spread(self, x, z):
        cands = self.candidates(x, z)
        return np.maximum.reduce(cands) - np.minimum.reduce(cands)


def catmull_rom(points, step=0.5):
    """Centripetal Catmull-Rom spline through (x, z) points, sampled finely."""
    p = [np.array(points[0]) * 2 - np.array(points[1])] + [np.array(q, float) for q in points] + [np.array(points[-1]) * 2 - np.array(points[-2])]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        t0 = 0.0
        t1 = t0 + max(np.linalg.norm(p1 - p0), 1e-6) ** 0.5
        t2 = t1 + max(np.linalg.norm(p2 - p1), 1e-6) ** 0.5
        t3 = t2 + max(np.linalg.norm(p3 - p2), 1e-6) ** 0.5
        n = max(2, int(np.linalg.norm(p2 - p1) / step))
        for t in np.linspace(t1, t2, n, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            out.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    out.append(np.array(points[-1], float))
    return np.array(out)


def resample(curve, spacing):
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    total = s[-1]
    stations = list(np.arange(0, total, spacing))
    if total - stations[-1] > 0.5:
        stations.append(total)
    else:
        stations[-1] = total
    xs = np.interp(stations, s, curve[:, 0])
    zs = np.interp(stations, s, curve[:, 1])
    return np.stack([xs, zs], axis=1)


def profile(points, natural, start_h, end_h, grade_step, max_grade, flat_start=1, flat_end=1):
    """Heights at 10 m stations. A smooth, slope-limited target is built first (level where the trail
    touches a clearing core), then segment grades are quantized to multiples of grade_step by error
    diffusion, so consecutive grades rarely differ by more than one step."""
    L = np.linalg.norm(np.diff(points, axis=0), axis=1)
    n = len(L)
    flat_start = max(1, min(flat_start, n))
    flat_end = max(1, min(flat_end, n - flat_start))
    last_free = n - flat_end
    idx = np.arange(n + 1)
    pin = (idx <= flat_start) | (idx >= last_free)
    pinned = np.where(idx <= flat_start, start_h, end_h)
    p = np.asarray(natural, float).copy()
    limit = min(max_grade * 0.9, 0.09) * L  # gentle target; quantized grades may still reach max_grade
    for _ in range(900):
        p[pin] = pinned[pin]
        for k in range(n):
            p[k + 1] = np.clip(p[k + 1], p[k] - limit[k], p[k] + limit[k])
        for k in range(n - 1, -1, -1):
            p[k] = np.clip(p[k], p[k + 1] - limit[k], p[k + 1] + limit[k])
        p[pin] = pinned[pin]
        q = p.copy()
        q[1:-1] = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
        p = np.where(pin, p, q)
    p[pin] = pinned[pin]
    # Smooth the grades themselves, taper them into the level ends and rescale to the required drop,
    # so every vertical curve is gentle before quantization.
    free = np.zeros(n, bool)
    free[flat_start:last_free] = True
    gd = np.where(free, np.diff(p) / L, 0.0)
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], float)
    kernel /= kernel.sum()
    for _ in range(3):
        gd = np.where(free, np.convolve(gd, kernel, mode="same"), 0.0)
    taper = np.ones(n)
    ramp = 4
    for k in range(n):
        if free[k]:
            a = k - flat_start + 1
            b = last_free - k
            taper[k] = min(1.0, a / (ramp + 1), b / (ramp + 1))
    gd *= taper
    need = end_h - start_h
    have = float((gd * L).sum())
    if abs(have) > 1e-6 and np.sign(have) == np.sign(need):
        gd *= need / have
    else:
        gd = np.where(free, taper, 0.0) * need / float((np.where(free, taper, 0.0) * L).sum())
    over = np.abs(gd) > max_grade * 0.95
    if over.any():  # spread any excess over the remaining free segments
        gd = np.clip(gd, -max_grade * 0.95, max_grade * 0.95)
        spare = free & ~over
        rest = need - float((gd * L).sum())
        if spare.any():
            gd[spare] += rest / float(L[spare].sum())
    p = np.concatenate([[start_h], start_h + np.cumsum(gd * L)])
    grades = np.zeros(n)
    h = np.zeros(n + 1)
    h[0] = start_h
    max_change = 2 * grade_step
    for k in range(n):
        if flat_start <= k < last_free:
            g = round((p[k + 1] - h[k]) / L[k] / grade_step) * grade_step
            prev = grades[k - 1] if k > 0 else 0.0
            g = float(np.clip(g, prev - max_change, prev + max_change))
            grades[k] = float(np.clip(g, -max_grade, max_grade))
        h[k + 1] = h[k] + grades[k] * L[k]
    # Close any remaining end error with single-step changes (at most one per segment) where they help most.
    adjusted = set()
    for _ in range(400):
        err = h[-1] - end_h
        if abs(err) <= grade_step * 10 * 0.55:
            break
        direction = -np.sign(err)
        best, best_gain = None, -1e18
        for k in range(flat_start, last_free):
            if k in adjusted or abs(grades[k] + direction * grade_step) > max_grade + 1e-9:
                continue
            delta = direction * grade_step * L[k]
            downstream = slice(k + 1, n + 1)
            gain = np.abs(h[downstream] - p[downstream]).sum() - np.abs(h[downstream] + delta - p[downstream]).sum() + 1e-3 * k
            if gain > best_gain:
                best, best_gain = k, gain
        if best is None:
            break
        grades[best] += direction * grade_step
        adjusted.add(best)
        h = np.concatenate([[start_h], start_h + np.cumsum(grades * L)])
    return h, grades, L


def segment_planes(points, h, grades):
    return [{"a": points[s].tolist(), "b": points[s + 1].tolist(), "h": float(h[s]), "grade": float(grades[s]),
             "length": float(np.linalg.norm(points[s + 1] - points[s]))} for s in range(len(grades))]


def distance_to_segments(px, pz, planes):
    """For arrays of points, the nearest segment index, distance and plane height."""
    best_d = np.full(px.shape, np.inf)
    best_h = np.zeros(px.shape)
    best_i = np.full(px.shape, -1)
    for i, s in enumerate(planes):
        ax, az = s["a"]
        bx, bz = s["b"]
        dx, dz = bx - ax, bz - az
        L2 = dx * dx + dz * dz
        t = np.clip(((px - ax) * dx + (pz - az) * dz) / L2, 0.0, 1.0)
        qx, qz = ax + t * dx, az + t * dz
        d = np.hypot(px - qx, pz - qz)
        hh = s["h"] + s["grade"] * t * math.sqrt(L2)
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_h = np.where(closer, hh, best_h)
        best_i = np.where(closer, i, best_i)
    return best_i, best_d, best_h


def build(design: dict, out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = design["seed"]
    rng = np.random.default_rng([seed, 11])
    T = design["terrain"]
    places = {p["id"]: p for p in design["places"]}

    H = T["base"] + 7.5 * field_noise(rng, 2.3) + 2.2 * field_noise(rng, 1.7) + 0.5 * field_noise(rng, 1.2)
    for f in T["features"]:
        d = np.hypot(GX - f["x"], GZ - f["z"])
        if f["kind"] == "hill":
            H += f["height"] * np.exp(-(d / f["radius"]) ** 2)
        elif f["kind"] == "hollow":
            H -= f["depth"] * (1 - smoothstep(0, f["radius"], d))
        elif f["kind"] == "lake":
            H -= 17.0 * (1 - smoothstep(0, 360, d))  # broad basin so trails descend gently to the shore
    # Lake with an irregular shoreline, shelving bed and a small island.
    lake = next(f for f in T["features"] if f["kind"] == "lake")
    ang = np.arctan2(GZ - lake["z"], GX - lake["x"])
    shore_r = lake["radius"] + 9 * np.sin(3 * ang + 0.7) + 5 * np.sin(7 * ang + 2.1)
    r = np.hypot(GX - lake["x"], GZ - lake["z"])
    lake_h = np.where(r < shore_r, 0.4 - lake["depth"] * smoothstep(0, 32, shore_r - r) - 0.4 * smoothstep(0, 4, shore_r - r),
                      0.4 + (r - shore_r) * 0.17)
    H = np.where(r < shore_r + 60, np.minimum(H, lake_h), H)
    H = np.where((r >= shore_r + 3) & (r < shore_r + 80), np.maximum(H, 0.9), H)
    ix, iz, ir, ih = ISLAND
    di = np.hypot(GX - ix, GZ - iz)
    H = np.maximum(H, ih - 1.7 * (di / ir) ** 2)

    # Flatten clearings to the natural height at their centre (tarn uses its shore plaza).
    clearings = []
    for p in design["places"]:
        if p["id"] == "tarn":
            cx, cz, cr, ch = TARN_PLAZA
        else:
            cx, cz, cr = p["x"], p["z"], p["radius"]
            ch = float(Sampler(H).height(cx, cz))
        clearings.append({"id": p["id"], "x": cx, "z": cz, "radius": cr, "height": round(ch, 2)})
    def flatten_clearings(H, blend=50.0):
        for c in clearings:
            b = 10.0 if c["id"] == "tarn" else blend  # the shore plaza must not fill the lake
            w = 1 - smoothstep(c["radius"], c["radius"] + b, np.hypot(GX - c["x"], GZ - c["z"]))
            H = H * (1 - w) + c["height"] * w
        return H
    H = flatten_clearings(H)
    pre_path = H.copy()

    # Trails: quantized-grade profiles; the ground under each trail becomes the exact segment plane.
    P = design["paths"]
    routes = []
    for route in design["routes"]:
        curve = catmull_rom(route["points"])
        pts = resample(curve, P["segmentMetres"])
        natural = Sampler(pre_path).height(pts[:, 0], pts[:, 1])
        c0 = next(c for c in clearings if c["id"] == route["from"])
        c1 = next(c for c in clearings if c["id"] == route["to"])
        # Segments touching a clearing core stay level at the clearing height.
        d0 = np.hypot(pts[:, 0] - c0["x"], pts[:, 1] - c0["z"])
        d1 = np.hypot(pts[:, 0] - c1["x"], pts[:, 1] - c1["z"])
        flat_start = int(np.argmax(d0 > c0["radius"] + 6)) if np.any(d0 > c0["radius"] + 6) else 1
        flat_end = int(np.argmax(d1[::-1] > c1["radius"] + 6)) if np.any(d1 > c1["radius"] + 6) else 1
        h, grades, L = profile(pts, natural, c0["height"], c1["height"], P["gradeStepPercent"] / 100,
                               P["maxGradePercent"] / 100, flat_start, flat_end)
        if abs(h[-1] - c1["height"]) > 0.3:
            raise ValueError(f"route {route['id']}: profile misses the destination by {h[-1] - c1['height']:+.2f} m")
        planes = segment_planes(pts, h, grades)
        routes.append({"id": route["id"], "name": route["name"], "from": route["from"], "to": route["to"],
                       "stations": [[round(float(x), 3), round(float(z), 3), round(float(y), 3)] for (x, z), y in zip(pts, h)],
                       "planes": planes, "length": round(float(L.sum()), 1),
                       "endError": round(float(h[-1] - c1["height"]), 3),
                       "maxGradePercent": round(float(np.abs(grades).max() * 100), 1),
                       "cutFillMax": round(float(np.abs(h - natural).max()), 2)})
    flat, blend = P["flattenMetres"], P["blendMetres"]
    for rt in routes:
        xs = [s["a"][0] for s in rt["planes"]] + [rt["planes"][-1]["b"][0]]
        zs = [s["a"][1] for s in rt["planes"]] + [rt["planes"][-1]["b"][1]]
        i0 = max(0, int((min(xs) - blend - MIN_M) // SPACING) - 1)
        i1 = min(SIZE, int((max(xs) + blend - MIN_M) // SPACING) + 2)
        j0 = max(0, int((min(zs) - blend - MIN_M) // SPACING) - 1)
        j1 = min(SIZE, int((max(zs) + blend - MIN_M) // SPACING) + 2)
        sx, sz = GX[j0:j1, i0:i1], GZ[j0:j1, i0:i1]
        _, d, hp = distance_to_segments(sx, sz, rt["planes"])
        w = 1 - smoothstep(flat, blend, d)
        H[j0:j1, i0:i1] = H[j0:j1, i0:i1] * (1 - w) + hp * w
    # Re-assert exact clearing cores (trail ends are flat at the same heights).
    for c in clearings:
        core = np.hypot(GX - c["x"], GZ - c["z"]) < c["radius"]
        H = np.where(core, c["height"], H)

    # Texture ids.
    tex_ids = T["textures"]
    tex = np.full(H.shape, tex_ids["forest"], dtype=np.uint16)
    gy, gx = np.gradient(H, SPACING)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    dell = places["dell"]
    tex[np.hypot(GX - dell["x"], GZ - dell["z"]) < 125] = tex_ids["dellmoss"]
    for c in clearings:
        if c["id"] in ("dell",):
            continue
        tex[np.hypot(GX - c["x"], GZ - c["z"]) < c["radius"] - 3] = tex_ids["clearing"]
    tex[(r < shore_r + 14) & (H < 2.2)] = tex_ids["shore"]
    beacon = places["beacon"]
    tex[(slope > 27) | (np.hypot(GX - beacon["x"], GZ - beacon["z"]) < 44)] = tex_ids["rock"]

    heights_cm = np.round(H * 100).astype("<i4")
    textures = tex.astype("<u2")
    hp_path = out_dir / "terrain-heights-cm.i32le"
    tx_path = out_dir / "terrain-textures.u16le"
    hp_path.write_bytes(heights_cm.tobytes(order="C"))
    tx_path.write_bytes(textures.tobytes(order="C"))
    sampler = Sampler(heights_cm.astype(float) / 100.0)
    meta = {
        "grid": SIZE, "minCell": MIN_CELL, "spacingMetres": SPACING, "order": "z-major, x fastest",
        "heights": {"file": hp_path.name, "format": "int32 little-endian centimetres", "sha256": hashlib.sha256(hp_path.read_bytes()).hexdigest(),
                    "minMetres": float(heights_cm.min()) / 100, "maxMetres": float(heights_cm.max()) / 100},
        "textures": {"file": tx_path.name, "format": "uint16 little-endian texture id", "sha256": hashlib.sha256(tx_path.read_bytes()).hexdigest(),
                     "counts": {str(k): int(v) for k, v in zip(*np.unique(textures, return_counts=True))}},
        "waterLevelMetres": T["waterLevelMetres"],
        "clearings": clearings,
        "tarnPlaza": {"x": TARN_PLAZA[0], "z": TARN_PLAZA[1], "radius": TARN_PLAZA[2], "height": TARN_PLAZA[3]},
        "island": {"x": ISLAND[0], "z": ISLAND[1], "radius": ISLAND[2], "height": ISLAND[3]},
        "routes": routes,
        "maxSlopeDegrees": round(float(slope.max()), 1),
    }
    (out_dir / "landscape.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return meta, sampler


def load(out_dir: Path):
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "landscape.json").read_text(encoding="utf-8"))
    hs = np.fromfile(out_dir / "terrain-heights-cm.i32le", dtype="<i4").reshape(SIZE, SIZE)
    return meta, Sampler(hs.astype(float) / 100.0)
