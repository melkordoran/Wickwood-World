"""Deterministic Wickwood object layout: trails, lanterns, places, signs and forest.

Positions are metres in AW axes (+x west, +z north); yaw uses the AW convention (0 north, 90 west),
so a model's local +z front points along (sin yaw, cos yaw).
"""
from __future__ import annotations

import math
import random

import numpy as np

LIGHT = "create light color=ffae52 brightness=0.7 radius=14"
LIGHT_FLICKER = "create light color=ffa040 brightness=0.75 radius=14 fx=flicker"
SIGN_COLORS = "color=ffd98a bcolor=22170f"


def yaw_of(dx, dz):
    return math.degrees(math.atan2(dx, dz)) % 360.0


def teleport(x, z, y, yaw):
    ns = f"{abs(z) / 10:.2f}{'N' if z >= 0 else 'S'}"
    we = f"{abs(x) / 10:.2f}{'W' if x >= 0 else 'E'}"
    return f"{ns} {we} {y / 10:.2f}a {int(round(yaw)) % 360}"


class Hash:
    def __init__(self, cell=12.0):
        self.cell = cell
        self.grid: dict[tuple[int, int], list] = {}

    def add(self, x, z, radius, kind):
        self.grid.setdefault((int(x // self.cell), int(z // self.cell)), []).append((x, z, radius, kind))

    def near(self, x, z, reach):
        c = self.cell
        r = int(math.ceil(reach / c))
        ix, iz = int(x // c), int(z // c)
        for gx in range(ix - r, ix + r + 1):
            for gz in range(iz - r, iz + r + 1):
                yield from self.grid.get((gx, gz), ())

    def clear(self, x, z, radius, reach=16.0, spacing=None):
        """True if (x, z) with `radius` does not intrude on anything registered nearby."""
        for ox, oz, orad, kind in self.near(x, z, reach):
            need = radius + orad
            if spacing is not None and kind == "tree":
                need = max(need, spacing)
            if (ox - x) ** 2 + (oz - z) ** 2 < need * need:
                return False
        return True


class Layout:
    def __init__(self, design, meta, sampler, seed):
        self.D = design
        self.meta = meta
        self.S = sampler
        self.rng = random.Random(f"{seed}:layout")
        self.placements: list[dict] = []
        self.hash = Hash()
        self.places = {p["id"]: p for p in design["places"]}
        self.clearings = {c["id"]: c for c in meta["clearings"]}
        self.destinations: dict[str, dict] = {}
        self._segments = np.array([[*p["a"], *p["b"]] for r in meta["routes"] for p in r["planes"]])
        self._distance_field()

    # ------------------------------------------------------------------ helpers
    def add(self, model, x, z, y=None, yaw=0.0, role="scenery", description="", action="", radius=0.0, kind=None, sink=0.0, **extra):
        if y is None:
            y = float(self.S.ground(x, z)) - sink
        p = {"model": model, "x": float(x), "y": float(y), "z": float(z), "yaw": float(yaw) % 360.0, "role": role,
             "description": description, "action": action}
        p.update(extra)
        self.placements.append(p)
        if radius > 0:
            self.hash.add(x, z, radius, kind or role)
        return p

    def ground(self, x, z):
        return float(self.S.ground(x, z))

    def _distance_field(self, res=4.0, reach=380.0):
        """Distance (m) from every 4 m grid point to the nearest trail centreline, within `reach`."""
        xs = np.arange(-1500, 1500 + res, res)
        zs = np.arange(-1200, 1300 + res, res)
        self._fx, self._fz, self._res = xs, zs, res
        field = np.full((len(zs), len(xs)), np.inf)
        for ax, az, bx, bz in self._segments:
            i0 = max(0, int((min(ax, bx) - reach - xs[0]) // res))
            i1 = min(len(xs), int((max(ax, bx) + reach - xs[0]) // res) + 2)
            j0 = max(0, int((min(az, bz) - reach - zs[0]) // res))
            j1 = min(len(zs), int((max(az, bz) + reach - zs[0]) // res) + 2)
            gx, gz = np.meshgrid(xs[i0:i1], zs[j0:j1])
            dx, dz = bx - ax, bz - az
            t = np.clip(((gx - ax) * dx + (gz - az) * dz) / (dx * dx + dz * dz), 0, 1)
            d = np.hypot(gx - ax - t * dx, gz - az - t * dz)
            field[j0:j1, i0:i1] = np.minimum(field[j0:j1, i0:i1], d)
        self._field = field

    def path_distance(self, x, z):
        """Exact distance to the nearest trail centreline (uses the field to prune)."""
        i = int(round((x - self._fx[0]) / self._res))
        j = int(round((z - self._fz[0]) / self._res))
        if 0 <= j < self._field.shape[0] and 0 <= i < self._field.shape[1]:
            coarse = self._field[j, i]
            if coarse > 30:
                return float(coarse)
        segs = self._segments
        dx, dz = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        t = np.clip(((x - segs[:, 0]) * dx + (z - segs[:, 1]) * dz) / (dx * dx + dz * dz), 0, 1)
        return float(np.hypot(x - segs[:, 0] - t * dx, z - segs[:, 1] - t * dz).min())

    def in_clearing(self, x, z, margin=0.0):
        for c in self.clearings.values():
            if math.hypot(x - c["x"], z - c["z"]) < c["radius"] + margin:
                return c["id"]
        return None

    def lake_distance(self, x, z):
        lake = next(f for f in self.D["terrain"]["features"] if f["kind"] == "lake")
        return math.hypot(x - lake["x"], z - lake["z"])

    # ------------------------------------------------------------------ trails
    def trails(self):
        seg_count = 0
        for rt in self.meta["routes"]:
            planes = rt["planes"]
            for k, p in enumerate(planes):
                (ax, az), (bx, bz) = p["a"], p["b"]
                g = p["grade"]
                steps = int(round(abs(g) * 100))
                model = f"wk_path_g{steps:02d}"
                if g >= 0:
                    self.add(model, ax, az, y=p["h"], yaw=yaw_of(bx - ax, bz - az), role="trail", route=rt["id"])
                else:
                    self.add(model, bx, bz, y=p["h"] + g * p["length"], yaw=yaw_of(ax - bx, az - bz), role="trail", route=rt["id"])
                seg_count += 1
                if 0 < k:
                    self.add("wk_path_joint", ax, az, y=p["h"], role="trail", route=rt["id"])
        return seg_count

    def trail_lanterns(self, spacing=20.0, offset=2.6):
        count = 0
        for rt in self.meta["routes"]:
            st = rt["stations"]
            side = 1
            lit = 0
            for k in range(1, len(st) - 1, int(spacing // 10)):
                x, z, y = st[k]
                xa, za, _ = st[k - 1]
                xb, zb, _ = st[k + 1]
                dx, dz = xb - xa, zb - za
                n = math.hypot(dx, dz) or 1.0
                nx, nz = -dz / n, dx / n
                px, pz = x + nx * offset * side, z + nz * offset * side
                if self.in_clearing(px, pz, margin=4.0):
                    side = -side
                    continue
                if not self.hash.clear(px, pz, 0.7, reach=8):
                    side = -side
                    continue
                action = ""
                if count % 2 == 0:
                    action = LIGHT_FLICKER if lit % 5 == 4 else LIGHT
                    lit += 1
                self.add("wk_lantern_post", px, pz, yaw=yaw_of(-nx * side, -nz * side), role="lantern", action=action, radius=0.7, route=rt["id"])
                count += 1
                side = -side
            # wayside shrines on long trails, opposite the lantern side
            if rt["length"] > 450:
                for k in range(25, len(st) - 10, 25):
                    x, z, y = st[k]
                    xa, za, _ = st[k - 1]
                    xb, zb, _ = st[k + 1]
                    dx, dz = xb - xa, zb - za
                    n = math.hypot(dx, dz) or 1.0
                    nx, nz = -dz / n, dx / n
                    s = -1 if (k // 2) % 2 == 0 else 1
                    px, pz = x + nx * 3.4 * s, z + nz * 3.4 * s
                    if self.in_clearing(px, pz, 4.0) or not self.hash.clear(px, pz, 0.8, reach=8):
                        continue
                    self.add("wk_wayshrine", px, pz, yaw=yaw_of(-nx * s, -nz * s), role="lantern", radius=0.8,
                             description=f"Wayside shrine on {rt['name']}", route=rt["id"])
        return count

    def trailheads(self):
        for rt in self.meta["routes"]:
            st = rt["stations"]
            for end, cid in ((0, rt["from"]), (-1, rt["to"])):
                c = self.clearings[cid]
                order = range(len(st)) if end == 0 else range(len(st) - 1, -1, -1)
                for k in order:
                    x, z, y = st[k]
                    if math.hypot(x - c["x"], z - c["z"]) >= c["radius"] - 6:
                        nk = k + 1 if end == 0 else k - 1
                        nk = max(0, min(len(st) - 1, nk))
                        dx, dz = st[nk][0] - x, st[nk][1] - z
                        self.add("wk_gate", x, z, y=c["height"] if math.hypot(x - c["x"], z - c["z"]) < c["radius"] else y,
                                 yaw=yaw_of(dx, dz), role="gate", description=f"{rt['name']}", radius=0.0)
                        for s in (-1, 1):
                            n = math.hypot(dx, dz) or 1.0
                            self.hash.add(x + (-dz / n) * 1.9 * s, z + (dx / n) * 1.9 * s, 0.6, "gate")
                        break

    # ------------------------------------------------------------------ places
    def arrival_on_route(self, route_id, place_id, inset=10.0):
        """Point on the trail `inset` metres inside the clearing edge, facing the clearing centre."""
        rt = next(r for r in self.meta["routes"] if r["id"] == route_id)
        c = self.clearings[place_id]
        st = rt["stations"] if rt["to"] == place_id else rt["stations"][::-1]
        best = min(st, key=lambda s: abs(math.hypot(s[0] - c["x"], s[1] - c["z"]) - (c["radius"] - inset)))
        x, z = best[0], best[1]
        return x, z, c["height"], yaw_of(c["x"] - x, c["z"] - z)

    def set_destination(self, pid, x, z, y, yaw):
        self.destinations[pid] = {"id": pid, "name": self.places[pid]["name"], "x": round(x, 2), "z": round(z, 2), "y": round(y + 0.1, 2),
                                  "yaw": round(yaw, 1), "teleport": teleport(x, z, y + 0.1, yaw), "description": self.places[pid]["description"]}

    def signs(self, pid, targets, anchor, facing_yaw, spacing=3.4):
        """A row of signs centred on `anchor`, fronts facing `facing_yaw`."""
        ax, az = anchor
        right = (math.cos(math.radians(facing_yaw)), -math.sin(math.radians(facing_yaw)))
        front = (math.sin(math.radians(facing_yaw)), math.cos(math.radians(facing_yaw)))
        for _ in range(8):  # step the row toward the arrival until it is clear of other objects
            offsets = [(i - (len(targets) - 1) / 2) * spacing for i in range(len(targets))]
            if all(self.hash.clear(ax + right[0] * o, az + right[1] * o, 1.5, reach=10) for o in offsets):
                break
            ax, az = ax + front[0], az + front[1]
        for i, t in enumerate(targets):
            off = (i - (len(targets) - 1) / 2) * spacing
            x, z = ax + right[0] * off, az + right[1] * off
            d = self.destinations[t]
            action = f'create sign "{d["name"]}" {SIGN_COLORS}; activate teleport {d["teleport"]}'
            self.add("wk_sign", x, z, y=self.clearings[pid]["height"], yaw=facing_yaw, role="sign", description=f"Teleport: {d['name']}",
                     action=action, radius=1.5, target=t)

    def glade(self):
        c = self.clearings["glade"]
        h = c["height"]
        self.add("wk_paving_ring", 0, 0, y=h, role="feature")
        self.add("wk_lanterntree", 0, 0, y=h, role="landmark", description="The Lantern Tree of Wickgate Glade",
                 action="create light color=ffb65c brightness=1 radius=30", radius=3.2)
        for k in range(10):
            a = math.radians(18 + k * 36)
            x, z = 17 * math.sin(a), 17 * math.cos(a)
            self.add("wk_lantern_stone", x, z, y=h, yaw=yaw_of(-x, -z), role="lantern", action="create light color=ffb060 brightness=0.6 radius=12", radius=0.6)
        for theta in (0, 60, 90, 270, 300):  # on the paving, clear of the southern arc of signs
            a = math.radians(theta)
            x, z = 7.0 * math.sin(a), 7.0 * math.cos(a)
            self.add("wk_bench", x, z, y=h, yaw=yaw_of(-x, -z), role="furniture", radius=1.0)
        self.add("wk_board", 0, -21, y=h, yaw=180, role="sign", description="Welcome to Wickwood",
                 action=f'create sign "Welcome to Wickwood" {SIGN_COLORS}', radius=2.0)

    def glade_signs(self):
        c = self.clearings["glade"]
        order = ["hollow", "tarn", "beacon", "oak", "circle", "dell"]
        for i, t in enumerate(order):
            theta = 180 + (i - 2.5) * 24
            a = math.radians(theta)
            x, z = 13 * math.sin(a), 13 * math.cos(a)
            d = self.destinations[t]
            action = f'create sign "{d["name"]}" {SIGN_COLORS}; activate teleport {d["teleport"]}'
            self.add("wk_sign", x, z, y=c["height"], yaw=theta, role="sign", description=f"Teleport: {d['name']}", action=action, radius=1.5, target=t)

    def hollow(self):
        c = self.clearings["hollow"]
        cx, cz, h = c["x"], c["z"], c["height"]
        plan = [(50, "wk_cottage_a"), (85, "wk_cottage_b"), (160, "wk_cottage_a"), (200, "wk_workshop"), (290, "wk_cottage_b"), (330, "wk_cottage_a")]
        names = ["The Wickmaker's Cottage", "Tallow House", "The Glassblower's Cottage", "The Lantern Workshop", "Ember Cottage", "The Lamplighter's Rest"]
        for (theta, model), title in zip(plan, names):
            a = math.radians(theta)
            x, z = cx + 32 * math.sin(a), cz + 32 * math.cos(a)
            self.add(model, x, z, y=h, yaw=theta + 180, role="building", description=title, radius=6.5, kind="building")
        self.add("wk_well", cx, cz, y=h, role="feature", description="The hamlet well", radius=1.6)
        for k in range(4):
            a = math.radians(45 + k * 90)
            x, z = cx + 5 * math.sin(a), cz + 5 * math.cos(a)
            self.add("wk_bench", x, z, y=h, yaw=yaw_of(cx - x, cz - z), role="furniture", radius=1.0)
        for k in range(8):
            a = math.radians(k * 45 + 22.5)
            x, z = cx + 15 * math.sin(a), cz + 15 * math.cos(a)
            self.add("wk_lantern_post", x, z, y=h, yaw=yaw_of(cx - x, cz - z), role="lantern", action=LIGHT if k % 2 == 0 else "", radius=0.7)
        wa = math.radians(200)
        wx, wz = cx + 32 * math.sin(wa), cz + 32 * math.cos(wa)
        for s in (-1, 1):
            a = math.radians(200 + s * 14)
            x, z = cx + 25 * math.sin(a), cz + 25 * math.cos(a)
            self.add("wk_lantern_rack", x, z, y=h, yaw=yaw_of(cx - x, cz - z), role="feature", radius=1.4, description="Lanterns drying on a rack")

    def tarn(self):
        c = self.clearings["tarn"]
        px, pz, h = c["x"], c["z"], c["height"]
        lake = next(f for f in self.D["terrain"]["features"] if f["kind"] == "lake")
        dx, dz = lake["x"] - px, lake["z"] - pz
        n = math.hypot(dx, dz)
        ux, uz = dx / n, dz / n
        jyaw = yaw_of(ux, uz)
        jx, jz = px + ux * 13, pz + uz * 13
        self.add("wk_jetty", jx, jz, y=0.0, yaw=jyaw, role="feature", description="The Mirrormist jetty", radius=0.0)
        for t in range(0, 33, 4):
            self.hash.add(jx + ux * t, jz + uz * t, 2.2, "jetty")
        bx, bz = jx + ux * 22 - uz * 3.2, jz + uz * 22 + ux * 3.2
        self.add("wk_rowboat", bx, bz, y=0.0, yaw=jyaw + 8, role="feature", description="A moored rowboat", radius=2.0)
        for k in range(4):
            a = math.radians(jyaw + 45 + k * 90)
            x, z = px + 9 * math.sin(a), pz + 9 * math.cos(a)
            self.add("wk_lantern_stone", x, z, y=h, role="lantern", action="create light color=ffb060 brightness=0.6 radius=12", radius=0.6)
        for s in (-1, 1):
            x, z = px + ux * 4 - uz * 5 * s, pz + uz * 4 + ux * 5 * s
            self.add("wk_bench", x, z, y=h, yaw=jyaw, role="furniture", radius=1.0)
        isl = self.meta["island"]
        self.add("wk_lantern_stone", isl["x"], isl["z"], role="lantern", description="The lantern isle",
                 action="create light color=ffb060 brightness=0.7 radius=14", radius=1.0)
        placed, tries = 0, 0
        while placed < 30 and tries < 4000:
            tries += 1
            a = self.rng.uniform(0, math.tau)
            r = self.rng.uniform(18, 96)
            x, z = lake["x"] + r * math.sin(a), lake["z"] + r * math.cos(a)
            if self.ground(x, z) > -0.9 or math.hypot(x - isl["x"], z - isl["z"]) < 14:
                continue
            if not self.hash.clear(x, z, 3.0, reach=10):
                continue
            action = "create light color=ffb060 brightness=0.5 radius=10" if placed % 3 == 0 else ""
            self.add("wk_lantern_float", x, z, y=0.0, yaw=self.rng.uniform(0, 360), role="lantern", action=action, radius=1.5,
                     description="A floating lantern")
            placed += 1
        reeds, tries = 0, 0
        while reeds < 80 and tries < 20000:
            tries += 1
            a = self.rng.uniform(0, math.tau)
            r = self.rng.uniform(100, 150)
            x, z = lake["x"] + r * math.sin(a), lake["z"] + r * math.cos(a)
            g = self.ground(x, z)
            if not (-0.35 < g < 0.45) or self.in_clearing(x, z, 2.0) or self.path_distance(x, z) < 3.0 or not self.hash.clear(x, z, 0.6, reach=8):
                continue
            self.add("wk_reeds", x, z, y=g - 0.02, yaw=self.rng.uniform(0, 360), role="understory", radius=0.5)
            reeds += 1

    def beacon(self):
        c = self.clearings["beacon"]
        cx, cz, h = c["x"], c["z"], c["height"]
        self.add("wk_beacon", cx, cz, y=h, role="landmark", description="The Beacon of Wickwood",
                 action="create light color=ff9a3c brightness=1 radius=40 fx=flicker", radius=6.5)
        self.add("wk_spire", cx + 11, cz + 11, y=h, yaw=225, role="landmark", description="The watch spire",
                 action="create light color=ffb65c brightness=0.8 radius=20", radius=1.2)
        for k in range(4):
            a = math.radians(45 + k * 90)
            x, z = cx + 12 * math.sin(a), cz + 12 * math.cos(a)
            if math.hypot(x - (cx + 11), z - (cz + 11)) < 3:
                continue
            self.add("wk_lantern_stone", x, z, y=h, role="lantern", action="create light color=ffb060 brightness=0.6 radius=12", radius=0.6)
        for k in range(3):
            a = math.radians(160 + k * 70)
            x, z = cx + 18 * math.sin(a), cz + 18 * math.cos(a)
            self.add("wk_bench", x, z, y=h, yaw=yaw_of(x - cx, z - cz) + 180, role="furniture", radius=1.0)
        for k in range(7):
            a = self.rng.uniform(0, math.tau)
            r = self.rng.uniform(24, 36)
            x, z = cx + r * math.sin(a), cz + r * math.cos(a)
            if self.path_distance(x, z) > 4 and self.hash.clear(x, z, 2.0, reach=10):
                self.add("wk_boulder", x, z, yaw=self.rng.uniform(0, 360), role="scenery", sink=0.4, radius=1.8)

    def oak(self):
        c = self.clearings["oak"]
        cx, cz, h = c["x"], c["z"], c["height"]
        rt = next(r for r in self.meta["routes"] if r["id"] == "rootwalk")
        ex, ez = rt["stations"][-1][0], rt["stations"][-1][1]
        self.add("wk_giant_oak", cx, cz, y=h, yaw=yaw_of(ex - cx, ez - cz), role="landmark", description="The Hollow Oak",
                 action="create light color=ffb65c brightness=0.8 radius=12", radius=7.0)
        for k in range(18):
            a = math.radians(k * 20 + 5)
            r = 15.5 + (k % 3) * 0.8
            x, z = cx + r * math.sin(a), cz + r * math.cos(a)
            model = "wk_glowcap_amber" if k % 2 else "wk_glowcap_teal"
            self.add(model, x, z, y=h, yaw=k * 47, role="fungus", radius=0.6)
        for k in range(8):
            a = math.radians(k * 45)
            x, z = cx + 26 * math.sin(a), cz + 26 * math.cos(a)
            if self.path_distance(x, z) < 2.5 or not self.hash.clear(x, z, 0.7, reach=8):
                continue
            self.add("wk_lantern_post", x, z, y=h, yaw=yaw_of(cx - x, cz - z), role="lantern", action=LIGHT if k % 2 == 0 else "", radius=0.7)
        for k in range(4):
            a = math.radians(k * 90 + 30)
            x, z = cx + 21 * math.sin(a), cz + 21 * math.cos(a)
            if self.path_distance(x, z) > 2.5:
                self.add("wk_bench", x, z, y=h, yaw=yaw_of(cx - x, cz - z), role="furniture", radius=1.0)

    def circle(self):
        c = self.clearings["circle"]
        cx, cz, h = c["x"], c["z"], c["height"]
        for k in range(9):
            theta = k * 40 + 10
            a = math.radians(theta)
            x, z = cx + 12 * math.sin(a), cz + 12 * math.cos(a)
            self.add(("wk_stone_a", "wk_stone_b", "wk_stone_c")[k % 3], x, z, y=h, yaw=theta + 180, role="landmark", description="A standing stone", radius=1.0)
            x2, z2 = cx + 10.2 * math.sin(a), cz + 10.2 * math.cos(a)
            self.add("wk_candles", x2, z2, y=h, yaw=theta, role="lantern", action="create light color=ffc070 brightness=0.5 radius=8" if k % 3 == 0 else "", radius=0.4)
        self.add("wk_altar", cx, cz, y=h, role="landmark", description="The candlestone altar", action="create light color=ffc070 brightness=0.7 radius=14", radius=1.6)

    def dell(self):
        c = self.clearings["dell"]
        cx, cz, h = c["x"], c["z"], c["height"]
        lit = 0
        for k in range(8):
            a = math.radians(k * 45 + 10)
            r = 22 + (k % 2) * 16
            x, z = cx + r * math.sin(a), cz + r * math.cos(a)
            if self.path_distance(x, z) < 4 or not self.hash.clear(x, z, 2.2, reach=10):
                continue
            action = "create light color=4de8c4 brightness=0.5 radius=16" if lit < 6 else ""
            lit += 1
            self.add("wk_glowcap_giant", x, z, yaw=k * 33, role="fungus", action=action, radius=2.0, description="A giant glowcap")
        for model, count, radius in (("wk_glowmoss", 45, 1.8), ("wk_glowcap_teal", 30, 0.8), ("wk_glowcap_violet", 22, 0.8), ("wk_glowcap_amber", 14, 0.8),
                                     ("wk_fireflies_a", 12, 2.5), ("wk_fireflies_b", 8, 3.5), ("wk_log_a", 5, 2.4), ("wk_fern_b", 30, 1.0), ("wk_stump", 4, 0.6)):
            placed, tries = 0, 0
            while placed < count and tries < count * 60:
                tries += 1
                a = self.rng.uniform(0, math.tau)
                r = math.sqrt(self.rng.uniform(0, 1)) * (c["radius"] + 18)
                x, z = cx + r * math.sin(a), cz + r * math.cos(a)
                if self.path_distance(x, z) < 2.8 or not self.hash.clear(x, z, radius, reach=10):
                    continue
                ground_kind = "glow" if model == "wk_glowmoss" else "fungus"
                self.add(model, x, z, yaw=self.rng.uniform(0, 360), role=ground_kind, radius=radius * (0.4 if model == "wk_glowmoss" else 1.0), sink=0.0)
                placed += 1

    # ------------------------------------------------------------------ forest and understory
    def forest(self, seed):
        T = self.D["terrain"]
        noise_rng = np.random.default_rng([seed, 99])
        coarse = noise_rng.random((40, 40))
        def patch(x, z):
            gx, gz = (x + 2000) / 100.0, (z + 2000) / 100.0
            i, j = int(gx) % 39, int(gz) % 39
            fx, fz = gx - int(gx), gz - int(gz)
            return (coarse[j, i] * (1 - fx) * (1 - fz) + coarse[j, i + 1] * fx * (1 - fz) + coarse[j + 1, i] * (1 - fx) * fz + coarse[j + 1, i + 1] * fx * fz)
        spacing = {"wk_pine_s": 5.5, "wk_pine_m": 6.5, "wk_pine_l": 7.5, "wk_spruce_m": 5.5, "wk_spruce_l": 6.5, "wk_oak_s": 8, "wk_oak_m": 10,
                   "wk_oak_l": 12, "wk_birch_s": 5, "wk_birch_m": 5.5, "wk_birch_l": 6, "wk_snag_a": 6, "wk_snag_b": 6}
        centres = {pid: (p["x"], p["z"]) for pid, p in self.places.items()}
        def weights(x, z, h):
            w = {k: 1.0 for k in spacing}
            conifer = 0.5 + 0.5 * math.tanh((h - 34) / 18) * 0.8 + (patch(x, z) - 0.5) * 0.9
            conifer = min(1.0, max(0.0, conifer))
            for k in ("wk_pine_s", "wk_pine_m", "wk_pine_l", "wk_spruce_m", "wk_spruce_l"):
                w[k] *= 0.3 + 1.7 * conifer
            for k in ("wk_oak_s", "wk_oak_m", "wk_oak_l", "wk_birch_s", "wk_birch_m", "wk_birch_l"):
                w[k] *= 0.3 + 1.7 * (1 - conifer)
            def near(pid, r):
                px, pz = centres[pid]
                return max(0.0, 1 - math.hypot(x - px, z - pz) / r)
            for k in ("wk_birch_s", "wk_birch_m", "wk_birch_l"):
                w[k] *= 0.45 * (1 + 3 * near("tarn", 330))
            for k in ("wk_pine_s", "wk_pine_m", "wk_pine_l", "wk_spruce_m", "wk_spruce_l"):
                w[k] *= 1.25
            for k in ("wk_oak_s", "wk_oak_m", "wk_oak_l"):
                w[k] *= 1 + 2.5 * near("oak", 320) + 1.5 * near("hollow", 260) + 1.5 * near("glade", 200)
            for k in ("wk_spruce_m", "wk_spruce_l", "wk_snag_a", "wk_snag_b"):
                w[k] *= 1 + 3 * near("dell", 340)
            for k in ("wk_snag_a", "wk_snag_b"):
                w[k] *= 0.25
            return w
        lake = next(f for f in T["features"] if f["kind"] == "lake")
        rng = random.Random(f"{seed}:forest")
        step = 3.2
        count = 0
        for gz in np.arange(-1180, 1280, step):
            for gx in np.arange(-1480, 1480, step):
                x = gx + rng.uniform(-1.4, 1.4)
                z = gz + rng.uniform(-1.4, 1.4)
                i = int(round((x - self._fx[0]) / self._res))
                j = int(round((z - self._fz[0]) / self._res))
                if not (0 <= j < self._field.shape[0] and 0 <= i < self._field.shape[1]):
                    continue
                d = self._field[j, i]
                if d > 360:
                    continue
                if d < 12:
                    d = self.path_distance(x, z)
                    if d < 5.2:
                        continue
                band = 1.0 if d < 60 else (1.7 if d < 160 else 2.8)
                keep = 1.0 if d < 60 else (0.5 if d < 160 else 0.2)
                if rng.random() > keep * 0.29:
                    continue
                if self.in_clearing(x, z, margin=4.0) or math.hypot(x - lake["x"], z - lake["z"]) < lake["radius"] + 16:
                    continue
                h = self.ground(x, z)
                if h < 1.2:
                    continue
                gradient = abs(float(self.S.height(x + 2, z)) - float(self.S.height(x - 2, z))) + abs(float(self.S.height(x, z + 2)) - float(self.S.height(x, z - 2)))
                if gradient > 3.2:
                    continue
                w = weights(x, z, h)
                total = sum(w.values())
                pick = rng.uniform(0, total)
                for model, wt in w.items():
                    pick -= wt
                    if pick <= 0:
                        break
                sp = spacing[model] * band
                if not self.hash.clear(x, z, 0.6, reach=max(sp, 8) + 2, spacing=sp):
                    continue
                self.add(model, x, z, yaw=rng.uniform(0, 360), role="tree", sink=0.05, radius=0.6, kind="tree")
                count += 1
        return count

    def understory(self, seed):
        rng = random.Random(f"{seed}:understory")
        choices = [("wk_fern_a", 28), ("wk_fern_b", 18), ("wk_shrub_a", 12), ("wk_shrub_b", 7), ("wk_rock_a", 8), ("wk_rock_b", 5),
                   ("wk_log_a", 4), ("wk_log_b", 2), ("wk_stump", 4), ("wk_toadstools", 6), ("wk_boulder", 1.2)]
        radius = {"wk_fern_a": 0.9, "wk_fern_b": 1.3, "wk_shrub_a": 1.0, "wk_shrub_b": 1.4, "wk_rock_a": 0.6, "wk_rock_b": 1.0, "wk_log_a": 2.3,
                  "wk_log_b": 3.3, "wk_stump": 0.5, "wk_toadstools": 0.4, "wk_boulder": 1.8, "wk_glowcap_teal": 0.6, "wk_glowcap_violet": 0.6,
                  "wk_glowcap_amber": 0.6, "wk_fireflies_a": 2.5, "wk_glowmoss": 1.0}
        total = sum(w for _, w in choices)
        count = 0
        for rt in self.meta["routes"]:
            glow = rt["id"] in ("mosswalk", "firefly-run")
            st = rt["stations"]
            for k in range(len(st) - 1):
                (ax, az, _), (bx, bz, _) = st[k], st[k + 1]
                L = math.hypot(bx - ax, bz - az) or 1.0
                nx, nz = -(bz - az) / L, (bx - ax) / L
                for _ in range(3):
                    t = rng.random()
                    side = rng.choice((-1, 1))
                    off = rng.uniform(3.0, 13.0)
                    x, z = ax + (bx - ax) * t + nx * off * side, az + (bz - az) * t + nz * off * side
                    if glow and rng.random() < 0.22:
                        model = rng.choice(["wk_glowcap_teal", "wk_glowcap_violet", "wk_glowmoss", "wk_fireflies_a"])
                    else:
                        pick = rng.uniform(0, total)
                        for model, wt in choices:
                            pick -= wt
                            if pick <= 0:
                                break
                        if rng.random() < 0.03:
                            model = "wk_glowcap_amber"
                    rr = radius[model]
                    if self.in_clearing(x, z, margin=-3.0) or self.path_distance(x, z) < 2.4 + rr * 0.6 or self.ground(x, z) < 0.6:
                        continue
                    if not self.hash.clear(x, z, rr, reach=10):
                        continue
                    sink = {"wk_boulder": 0.5, "wk_rock_b": 0.2, "wk_rock_a": 0.1}.get(model, 0.02)
                    self.add(model, x, z, yaw=rng.uniform(0, 360), role="understory", sink=sink, radius=rr)
                    count += 1
        return count

    # ------------------------------------------------------------------ build
    def build(self, seed):
        stats = {}
        stats["trailSegments"] = self.trails()
        self.trailheads()
        self.glade()
        self.hollow()
        self.tarn()
        self.beacon()
        self.oak()
        self.circle()
        self.dell()
        # destinations (arrival points) then signs, which need every destination
        self.set_destination("glade", 0.0, -30.0, self.clearings["glade"]["height"], 0.0)
        for pid, rid in (("hollow", "hollow-lane"), ("oak", "rootwalk"), ("circle", "candle-road"), ("dell", "firefly-run"), ("beacon", "beacon-climb")):
            self.set_destination(pid, *self.arrival_on_route(rid, pid))
        tc = self.clearings["tarn"]
        lake = next(f for f in self.D["terrain"]["features"] if f["kind"] == "lake")
        ux, uz = lake["x"] - tc["x"], lake["z"] - tc["z"]
        n = math.hypot(ux, uz)
        self.set_destination("tarn", tc["x"] - ux / n * 8, tc["z"] - uz / n * 8, tc["height"], yaw_of(ux, uz))
        self.glade_signs()
        for pid, targets in (("hollow", ["glade", "circle", "dell"]), ("tarn", ["glade", "circle"]), ("beacon", ["glade", "tarn"]),
                             ("oak", ["glade", "dell"]), ("circle", ["glade", "hollow", "tarn"]), ("dell", ["glade", "oak", "hollow"])):
            d = self.destinations[pid]
            c = self.clearings[pid]
            fx, fz = c["x"] - d["x"], c["z"] - d["z"]
            n = math.hypot(fx, fz) or 1.0
            anchor = (d["x"] + fx / n * 7.0, d["z"] + fz / n * 7.0)
            self.signs(pid, targets, anchor, yaw_of(-fx, -fz))
        stats["pathLanterns"] = self.trail_lanterns()
        stats["trees"] = self.forest(seed)
        stats["understory"] = self.understory(seed)
        return stats
