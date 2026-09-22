"""Geometry QA for Wickwood models: find parts that float free of the rest of a model.

Every closed solid the mesh builder records, and every group of open faces joined by shared corners, becomes an
item with an axis-aligned box. Items whose boxes touch (within a tolerance) are joined. A group of items that
neither reaches the model's base (y <= 0.03 m) nor touches anything that does is reported as floating.
Boxes over-approximate slanted parts, so this can miss a gap but never invents one between parts that touch.
"""
from __future__ import annotations

INTENTIONAL = {"wk_fireflies_a", "wk_fireflies_b"}  # drifting lights, meant to hang in the air
# Models reviewed and deliberately kept with a floating part. The well (bucket now on a rope) and the workshop
# (display lanterns now on the workbench) were fixed on 2026-09-22, so nothing is currently exempt this way.
ACCEPTED: set[str] = set()
MIN_CLEARANCE = 0.10  # smaller gaps (flames over wicks, sign faces, chimney glow, plant bases) are deliberate or invisible


def _items(mesh):
    items = []
    for material, part in mesh.parts.items():
        covered = set()
        for m, s, e in mesh.closed:
            if m != material:
                continue
            covered.update(range(s, e))
            pts = [part.vertices[i] for t in part.triangles[s:e] for i in t]
            items.append((material, [min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)]))
        # open faces: union triangles that share a corner position
        open_tris = [t for i, t in enumerate(part.triangles) if i not in covered]
        parent = {}

        def find(a):
            while parent.setdefault(a, a) != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        key = lambda i: tuple(round(c, 4) for c in part.vertices[i])
        for t in open_tris:
            ks = [key(i) for i in t]
            for k in ks[1:]:
                parent[find(ks[0])] = find(k)
        groups = {}
        for t in open_tris:
            groups.setdefault(find(key(t[0])), []).extend(part.vertices[i] for i in t)
        for pts in groups.values():
            items.append((material, [min(p[k] for p in pts) for k in range(3)], [max(p[k] for p in pts) for k in range(3)]))
    return items


def floating_parts(mesh, tol=0.005, ground=0.03):
    items = _items(mesh)
    n = len(items)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            (_, lo1, hi1), (_, lo2, hi2) = items[i], items[j]
            if all(lo1[k] <= hi2[k] + tol and lo2[k] <= hi1[k] + tol for k in range(3)):
                parent[find(i)] = find(j)
    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    out = []
    for members in clusters.values():
        if any(items[i][1][1] <= ground for i in members):
            continue
        lo = [min(items[i][1][k] for i in members) for k in range(3)]
        hi = [max(items[i][2][k] for i in members) for k in range(3)]
        others = [j for j in range(n) if find(j) != find(members[0])]
        gap = min((_box_gap(items[i], items[j]) for i in members for j in others), default=float("inf"))
        clearance = min(gap, lo[1])  # distance to the nearest other part or down to the model base
        out.append({"materials": sorted({items[i][0] for i in members}), "items": len(members),
                    "min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi], "clearance": round(clearance, 3)})
    return out


def _box_gap(a, b):
    d = [max(0.0, b[1][k] - a[2][k], a[1][k] - b[2][k]) for k in range(3)]
    return sum(v * v for v in d) ** 0.5


def scan(meshes, min_clearance=MIN_CLEARANCE, exempt=INTENTIONAL | ACCEPTED):
    """Models whose floating parts stand more than min_clearance clear of everything else."""
    out = {}
    for name, mesh in meshes.items():
        if name in exempt:
            continue
        found = [c for c in floating_parts(mesh) if c["clearance"] > min_clearance]
        if found:
            out[name] = found
    return out


# ---------------------------------------------------------------- coplanar overlaps (z-fighting)
def _tris(mesh):
    for material, part in mesh.parts.items():
        for t in part.triangles:
            yield material, [part.vertices[i] for i in t]


def _plane(p):
    u = [p[1][k] - p[0][k] for k in range(3)]
    v = [p[2][k] - p[0][k] for k in range(3)]
    n = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]
    ln = sum(c * c for c in n) ** 0.5
    if ln < 1e-12:
        return None
    n = [c / ln for c in n]
    return n, sum(n[k] * p[0][k] for k in range(3))


def _clip(poly, a, b):
    """Keep the part of a 2D polygon on the left of the directed edge a->b (Sutherland-Hodgman)."""
    out = []
    side = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    for i, cur in enumerate(poly):
        prev = poly[i - 1]
        sc, sp = side(cur), side(prev)
        if sc >= 0:
            if sp < 0:
                t = sp / (sp - sc)
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            out.append(cur)
        elif sp >= 0:
            t = sp / (sp - sc)
            out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
    return out


def _area(poly):
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def coplanar_overlaps(mesh, plane_tol=0.0005, min_area=1e-4):
    """Pairs of same-facing triangles in one plane that overlap by more than min_area (square metres)."""
    items = []
    for material, p in _tris(mesh):
        pl = _plane(p)
        if pl:
            items.append((material, p, pl[0], pl[1]))
    groups = {}
    for it in items:
        groups.setdefault(tuple(round(c, 3) for c in it[2]), []).append(it)
    found = []
    for n, group in groups.items():
        group.sort(key=lambda it: it[3])
        # 2D basis in the plane
        ref = (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0)
        e1 = [n[1] * ref[2] - n[2] * ref[1], n[2] * ref[0] - n[0] * ref[2], n[0] * ref[1] - n[1] * ref[0]]
        l1 = sum(c * c for c in e1) ** 0.5
        e1 = [c / l1 for c in e1]
        e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]]
        to2d = lambda q: (sum(q[k] * e1[k] for k in range(3)), sum(q[k] * e2[k] for k in range(3)))
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[j][3] - group[i][3] > plane_tol:
                    break
                a = [to2d(q) for q in group[i][1]]
                b = [to2d(q) for q in group[j][1]]
                if _area(a) < 0:
                    a = a[::-1]
                if _area(b) < 0:
                    b = b[::-1]
                poly = a
                for k in range(3):
                    poly = _clip(poly, b[k], b[(k + 1) % 3])
                    if len(poly) < 3:
                        break
                if len(poly) >= 3 and abs(_area(poly)) > min_area:
                    cu = sum(q[0] for q in poly) / len(poly)
                    cv = sum(q[1] for q in poly) / len(poly)
                    at = [n[k] * group[i][3] + e1[k] * cu + e2[k] * cv for k in range(3)]
                    found.append({"materials": sorted({group[i][0], group[j][0]}), "normal": [round(c, 3) for c in n],
                                  "offset": round(group[i][3], 4), "area": round(abs(_area(poly)), 5),
                                  "at": [round(c, 2) for c in at]})
    return found
