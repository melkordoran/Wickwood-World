"""Wickwood mesh builder and native RWX/ZIP exporter.

Geometry is authored in metres in AW model space: +x west, +y up, +z north (right-handed).
Model fronts face local +z. A triangle's front side is where (b-a)x(c-a) points, which is
counter-clockwise when seen from the front; closed parts therefore have positive signed volume.

RWX conventions match the exporter that was tested on the reference Axis host:
coordinates are written in RWX units (1 unit = 10 m), one clump per material, vertex indices
restart at 1 in each clump, glowing parts use Surface 0 0 0 with per-vertex prelight, sign
faces use Surface 1 0 0 / Texture Null / TextureModes Foreshorten with Tag 100 and no prelight.
"""
from __future__ import annotations

import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

ZIP_TIME = (2026, 9, 22, 0, 0, 0)

# name: color, optional texture, uv (metres per texture repeat), surface (ambient, diffuse, specular),
# glow (prelight colour, unlit), opacity, collision (default True), sign (tag 100 face)
MATERIALS: dict[str, dict] = {
    "bark": {"color": (0.62, 0.52, 0.42), "texture": "wk_bark", "uv": 2.0},
    "birch": {"color": (0.86, 0.85, 0.80), "texture": "wk_birchbark", "uv": 1.6},
    "deadwood": {"color": (0.50, 0.47, 0.42), "texture": "wk_bark", "uv": 2.0},
    "needles": {"color": (0.44, 0.58, 0.50), "texture": "wk_needles", "uv": 3.0, "collision": False},
    "needles_dark": {"color": (0.30, 0.42, 0.38), "texture": "wk_needles", "uv": 3.0, "collision": False},
    "leaves": {"color": (0.50, 0.62, 0.44), "texture": "wk_leaves", "uv": 3.5, "collision": False},
    "leaves_birch": {"color": (0.62, 0.70, 0.46), "texture": "wk_leaves", "uv": 3.0, "collision": False},
    "leaves_oak": {"color": (0.44, 0.54, 0.38), "texture": "wk_leaves", "uv": 4.5, "collision": False},
    "fern": {"color": (0.36, 0.56, 0.34), "texture": None, "collision": False},
    "shrub": {"color": (0.24, 0.36, 0.26), "texture": "wk_leaves", "uv": 2.0, "collision": False},
    "moss": {"color": (0.55, 0.70, 0.50), "texture": "wk_moss", "uv": 2.0},
    "rock": {"color": (0.72, 0.74, 0.72), "texture": "wk_rock", "uv": 3.0},
    "stone": {"color": (0.78, 0.78, 0.74), "texture": "wk_stone", "uv": 3.0},
    "timber": {"color": (0.70, 0.58, 0.46), "texture": "wk_timber", "uv": 2.5},
    "darktimber": {"color": (0.42, 0.33, 0.26), "texture": "wk_timber", "uv": 2.5},
    "plaster": {"color": (0.86, 0.80, 0.70), "texture": "wk_plaster", "uv": 3.0},
    "shingle": {"color": (0.56, 0.52, 0.50), "texture": "wk_shingle", "uv": 3.0},
    "iron": {"color": (0.30, 0.30, 0.31), "texture": "wk_iron", "uv": 1.5},
    "path": {"color": (0.80, 0.72, 0.62), "texture": "wk_path", "uv": 3.2},
    "soil": {"color": (0.20, 0.17, 0.14), "texture": None},
    "water": {"color": (0.08, 0.16, 0.18), "texture": None, "opacity": 0.85, "collision": False},
    "cap_red": {"color": (0.62, 0.18, 0.12), "texture": None, "collision": False},
    "stem": {"color": (0.80, 0.76, 0.66), "texture": None, "collision": False},
    "cloth": {"color": (0.36, 0.30, 0.26), "texture": None},
    "rope": {"color": (0.52, 0.44, 0.32), "texture": None, "collision": False},
    # Unlit (prelit) materials: rendered at their prelight colour regardless of scene lighting.
    "flame": {"color": (1.0, 0.72, 0.34), "glow": True, "collision": False},
    "lanternglass": {"color": (1.0, 0.64, 0.28), "glow": True, "collision": False},
    "window": {"color": (0.98, 0.66, 0.30), "glow": True, "collision": False},
    "ember": {"color": (1.0, 0.46, 0.16), "glow": True, "collision": False},
    "candle": {"color": (0.96, 0.90, 0.74), "glow": True, "collision": False},
    "glow_teal": {"color": (0.30, 0.95, 0.80), "glow": True, "collision": False},
    "glow_green": {"color": (0.56, 1.0, 0.46), "glow": True, "collision": False},
    "glow_violet": {"color": (0.72, 0.52, 1.0), "glow": True, "collision": False},
    "glow_amber": {"color": (1.0, 0.70, 0.30), "glow": True, "collision": False},
    "firefly": {"color": (0.92, 1.0, 0.56), "glow": True, "collision": False},
    "moss_glow": {"color": (0.20, 0.62, 0.50), "glow": True, "collision": False},
    "sign": {"color": (1.0, 1.0, 1.0), "sign": True},
}

Vec = tuple[float, float, float]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def rot_y(p: Vec, yaw_deg: float) -> Vec:
    """Rotate a local point by an AW yaw (0 north, 90 west): local +z maps to (sin, cos)."""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return (p[0] * c + p[2] * s, p[1], -p[0] * s + p[2] * c)


@dataclass
class Part:
    vertices: list = field(default_factory=list)
    uvs: list = field(default_factory=list)
    triangles: list = field(default_factory=list)


class Mesh:
    def __init__(self, name: str):
        if not re.fullmatch(r"wk_[a-z0-9_]+", name):
            raise ValueError(f"unsafe model name {name!r}")
        self.name = name
        self.parts: dict[str, Part] = {}
        self.closed: list[tuple[str, int, int]] = []  # (material, first triangle, last triangle) of closed solids

    # ---------------------------------------------------------------- primitives
    def face(self, points, material, uv=None, double=False):
        if material not in MATERIALS:
            raise KeyError(material)
        if len(points) < 3:
            return self
        part = self.parts.setdefault(material, Part())
        base = len(part.vertices)
        if uv is None:
            scale = MATERIALS[material].get("uv", 4.0)
            n = _cross(_sub(points[1], points[0]), _sub(points[2], points[0]))
            drop = max(range(3), key=lambda i: abs(n[i]))
            axes = [i for i in range(3) if i != drop]
            uv = [(p[axes[0]] / scale, p[axes[1]] / scale) for p in points]
        part.vertices.extend(tuple(float(c) for c in p) for p in points)
        part.uvs.extend(tuple(float(c) for c in t) for t in uv)
        for i in range(1, len(points) - 1):
            a, b, c = points[0], points[i], points[i + 1]
            if _norm(_cross(_sub(b, a), _sub(c, a))) < 1e-9:
                continue
            tri = (base, base + i, base + i + 1)
            part.triangles.append(tri)
            if double:
                part.triangles.append((tri[0], tri[2], tri[1]))
        return self

    def _begin_closed(self, material):
        part = self.parts.setdefault(material, Part())
        return len(part.triangles)

    def _end_closed(self, material, start):
        self.closed.append((material, start, len(self.parts[material].triangles)))

    def box(self, x, y, z, w, h, d, material, yaw=0.0, top_scale=1.0):
        """Closed box: centre x,z; base y; width along local x, depth along local z; rotated by yaw."""
        start = self._begin_closed(material)
        lo = [(-w / 2, 0, -d / 2), (w / 2, 0, -d / 2), (w / 2, 0, d / 2), (-w / 2, 0, d / 2)]
        hi = [(p[0] * top_scale, h, p[2] * top_scale) for p in lo]
        pl = [self._at(p, x, y, z, yaw) for p in lo]
        ph = [self._at(p, x, y, z, yaw) for p in hi]
        self.face(pl, material)
        self.face(ph[::-1], material)
        for i in range(4):
            j = (i + 1) % 4
            self.face([pl[i], ph[i], ph[j], pl[j]], material)
        self._end_closed(material, start)
        return self

    def slab(self, corners, y0, y1, material):
        """Closed prism from a CCW-from-above footprint polygon [(x,z),...] between y0 and y1."""
        start = self._begin_closed(material)
        area = sum(corners[i][0] * corners[(i + 1) % len(corners)][1] - corners[(i + 1) % len(corners)][0] * corners[i][1] for i in range(len(corners)))
        pts = corners if area > 0 else corners[::-1]  # positive shoelace area in (x, z) faces down
        lo = [(x, y0, z) for x, z in pts]
        hi = [(x, y1, z) for x, z in pts]
        self.face(lo, material)
        self.face(hi[::-1], material)
        for i in range(len(lo)):
            j = (i + 1) % len(lo)
            self.face([lo[i], hi[i], hi[j], lo[j]], material)
        self._end_closed(material, start)
        return self

    def cylinder(self, x, y, z, r0, h, material, n=10, r1=None, cap=True, twist=0.0, uv_wrap=True):
        """Closed frustum/cylinder around a vertical axis."""
        r1 = r0 if r1 is None else r1
        start = self._begin_closed(material)
        scale = MATERIALS[material].get("uv", 4.0)
        circ = math.tau * max(r0, r1, 0.05)
        lo = [(x + r0 * math.cos(i * math.tau / n + twist), y, z + r0 * math.sin(i * math.tau / n + twist)) for i in range(n)]
        hi = [(x + r1 * math.cos(i * math.tau / n + twist), y + h, z + r1 * math.sin(i * math.tau / n + twist)) for i in range(n)]
        for i in range(n):
            j = (i + 1) % n
            u0, u1 = i / n * circ / scale, (i + 1) / n * circ / scale
            quad = [lo[i], hi[i], hi[j], lo[j]]
            uv = [(u0, 0), (u0, h / scale), (u1, h / scale), (u1, 0)] if uv_wrap else None
            if r1 < 1e-6:
                self.face([lo[i], hi[i], lo[j]], material, uv=[(u0, 0), ((u0 + u1) / 2, h / scale), (u1, 0)] if uv_wrap else None)
            else:
                self.face(quad, material, uv=uv)
        if cap:
            self.face(lo, material)
            if r1 >= 1e-6:
                self.face(hi[::-1], material)
        self._end_closed(material, start)
        return self

    def cone(self, x, y, z, r, h, material, n=10, twist=0.0):
        return self.cylinder(x, y, z, r, h, material, n=n, r1=0.0, twist=twist)

    def tube(self, p0, p1, r0, r1, material, n=7, cap=True):
        """Closed frustum between two arbitrary points (branches, rails, posts)."""
        axis = _sub(p1, p0)
        length = _norm(axis)
        if length < 1e-6:
            return self
        w = _unit(axis)
        ref = (0.0, 1.0, 0.0) if abs(w[1]) < 0.9 else (1.0, 0.0, 0.0)
        u = _unit(_cross(ref, w))
        v = _cross(u, w)  # this basis makes (a[i], b[i], b[j], a[j]) face outward
        start = self._begin_closed(material)
        scale = MATERIALS[material].get("uv", 4.0)
        ring = lambda c, r, t: [
            (c[0] + r * (math.cos(i * math.tau / n + t) * u[0] + math.sin(i * math.tau / n + t) * v[0]),
             c[1] + r * (math.cos(i * math.tau / n + t) * u[1] + math.sin(i * math.tau / n + t) * v[1]),
             c[2] + r * (math.cos(i * math.tau / n + t) * u[2] + math.sin(i * math.tau / n + t) * v[2]))
            for i in range(n)]
        a, b = ring(p0, r0, 0.0), ring(p1, r1, 0.0)
        circ = math.tau * max(r0, r1)
        for i in range(n):
            j = (i + 1) % n
            u0, u1 = i / n * circ / scale, (i + 1) / n * circ / scale
            if r1 < 1e-6:
                self.face([a[i], b[i], a[j]], material, uv=[(u0, 0), (u0, length / scale), (u1, 0)])
            else:
                self.face([a[i], b[i], b[j], a[j]], material, uv=[(u0, 0), (u0, length / scale), (u1, length / scale), (u1, 0)])
        if cap:
            self.face(a, material)
            if r1 >= 1e-6:
                self.face(b[::-1], material)
        self._end_closed(material, start)
        return self

    def ellipsoid(self, c, radii, material, n=8, rings=5, jitter=None):
        """Closed ellipsoid; optional deterministic jitter(i, j) -> factor for lumpy foliage and rocks."""
        start = self._begin_closed(material)
        rx, ry, rz = radii
        grid = []
        for j in range(rings + 1):
            t = -math.pi / 2 + j * math.pi / rings
            row = []
            for i in range(n):
                p = i * math.tau / n
                k = jitter(i, j) if (jitter and 0 < j < rings) else 1.0
                row.append((c[0] + k * rx * math.cos(t) * math.cos(p), c[1] + k * ry * math.sin(t), c[2] + k * rz * math.cos(t) * math.sin(p)))
            grid.append(row)
        for j in range(rings):
            for i in range(n):
                i2 = (i + 1) % n
                a, b, cc, d = grid[j][i], grid[j + 1][i], grid[j + 1][i2], grid[j][i2]
                if j == 0:
                    self.face([a, b, cc], material)
                elif j == rings - 1:
                    self.face([a, b, d], material)
                else:
                    self.face([a, b, cc, d], material)
        self._end_closed(material, start)
        return self

    def disc(self, x, y, z, r, material, n=12, double=False):
        pts = [(x + r * math.cos(-i * math.tau / n), y, z + r * math.sin(-i * math.tau / n)) for i in range(n)]
        return self.face(pts, material, double=double)

    def quad(self, a, b, c, d, material, uv=None, double=False):
        return self.face([a, b, c, d], material, uv=uv, double=double)

    def merge(self, other: "Mesh", dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
        for material, part in other.parts.items():
            dst = self.parts.setdefault(material, Part())
            base = len(dst.vertices)
            first = len(dst.triangles)
            for p in part.vertices:
                q = rot_y(p, yaw)
                dst.vertices.append((q[0] + dx, q[1] + dy, q[2] + dz))
            dst.uvs.extend(part.uvs)
            dst.triangles.extend(tuple(base + i for i in t) for t in part.triangles)
            for m, s, e in other.closed:
                if m == material:
                    self.closed.append((material, first + s, first + e))
        return self

    @staticmethod
    def _at(p, x, y, z, yaw):
        q = rot_y(p, yaw)
        return (q[0] + x, q[1] + y, q[2] + z)

    # ---------------------------------------------------------------- statistics
    def bounds(self):
        vs = [v for part in self.parts.values() for v in part.vertices]
        if not vs:
            return {"min": [0, 0, 0], "max": [0, 0, 0]}
        return {"min": [min(v[i] for v in vs) for i in range(3)], "max": [max(v[i] for v in vs) for i in range(3)]}

    def triangle_count(self):
        return sum(len(p.triangles) for p in self.parts.values())

    def closed_volumes(self):
        """Signed volume of every closed solid; positive means outward-facing (front-facing) winding."""
        out = []
        for material, s, e in self.closed:
            part = self.parts[material]
            vol = 0.0
            for a, b, c in part.triangles[s:e]:
                pa, pb, pc = part.vertices[a], part.vertices[b], part.vertices[c]
                vol += _dot(pa, _cross(pb, pc)) / 6.0
            out.append((material, vol))
        return out

    def textures(self):
        return sorted({MATERIALS[m]["texture"] for m in self.parts if MATERIALS[m].get("texture")})

    # ---------------------------------------------------------------- export
    def rwx_text(self, comment="Original Wickwood geometry generated by the Wickwood pipeline"):
        lines = [f"# {comment}; coordinates are RWX units (10 metres).", "ModelBegin", "ClumpBegin"]
        for material, part in self.parts.items():
            if not part.triangles:
                continue
            mat = MATERIALS[material]
            color = " ".join(f"{c:.3g}" for c in mat["color"])
            lines += ["ClumpBegin", "Surface 0.45 0.7 0", "LightSampling Facet", "GeometrySampling Solid", f"Color {color}"]
            glow = bool(mat.get("glow"))
            if mat.get("sign"):
                lines += ["Surface 1 0 0", "Texture Null", "TextureModes Foreshorten"]
            elif glow:
                lines += ["Surface 0 0 0", "Color 1 1 1"]
            if mat.get("opacity") is not None:
                lines.append(f"Opacity {mat['opacity']:.3g}")
            if mat.get("collision") is False:
                lines.append("Collision Off")
            if mat.get("texture"):
                lines += [f"Texture {mat['texture']}", "TextureModes Lit Foreshorten"]
            prelight = " prelight " + color if glow else ""
            for (x, y, z), (u, v) in zip(part.vertices, part.uvs):
                lines.append(f"Vertex {x / 10:.6f} {y / 10:.6f} {z / 10:.6f} UV {u:.5f} {v:.5f}{prelight}")
            tag = " Tag 100" if mat.get("sign") else ""
            lines += [f"Triangle {a + 1} {b + 1} {c + 1}{tag}" for a, b, c in part.triangles]
            lines.append("ClumpEnd")
        lines += ["ClumpEnd", "ModelEnd"]
        return "\n".join(lines) + "\n"

    def export(self, folder: Path, crlf=False):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        text = self.rwx_text()
        if crlf:
            text = text.replace("\n", "\r\n")
        data = text.encode("ascii")
        (folder / f"{self.name}.rwx").write_bytes(data)
        write_zip(folder / f"{self.name}.zip", f"{self.name}.rwx", data)
        return {
            "name": self.name,
            "triangles": self.triangle_count(),
            "bounds": self.bounds(),
            "textures": self.textures(),
            "materials": sorted(m for m, p in self.parts.items() if p.triangles),
            "rwxBytes": len(data),
            "zipBytes": (folder / f"{self.name}.zip").stat().st_size,
        }


def write_zip(path: Path, member: str, data: bytes):
    info = zipfile.ZipInfo(member, date_time=ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, data)
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None or archive.read(member) != data:
            raise IOError(f"zip verification failed for {path}")
