"""Geometry QA for Wickwood models: find parts that float free of the rest of a model.

Every closed solid the mesh builder records, and every group of open faces joined by shared corners, becomes an
item with an axis-aligned box. Items whose boxes touch (within a tolerance) are joined. A group of items that
neither reaches the model's base (y <= 0.03 m) nor touches anything that does is reported as floating.
Boxes over-approximate slanted parts, so this can miss a gap but never invents one between parts that touch.
"""
from __future__ import annotations

INTENTIONAL = {"wk_fireflies_a", "wk_fireflies_b"}  # drifting lights, meant to hang in the air
# Reviewed on 2026-09-22 and kept as they are at the user's request: the well's bucket hangs over the opening
# without a rope, and the workshop's two end display lanterns stand past the workbench ends.
ACCEPTED = {"wk_well", "wk_workshop"}
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
