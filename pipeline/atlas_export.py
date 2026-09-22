"""Export compact browser-atlas data for Wickwood from a frozen release.

Models are re-read from the published ZIP archives with the strict RWX reader, so the atlas renders exactly the
geometry clients download. Output (all little-endian):
  index.json        world settings, models, materials, destinations, routes, specials, region grid
  models.bin        per-part Float32 positions (m), Float32 UVs, Float32 prelight RGB (glow parts only)
  terrain-h.i16     512x512 Int16 heights in cm, Z-major (row = cell z from -256), X fastest
  terrain-t.u8      512x512 UInt8 terrain texture ids, same layout
  objects.bin       per object: UInt16 model index, UInt16 region, Float32 x, y, z (m), Float32 yaw (rad)
"""
from __future__ import annotations

import argparse, hashlib, json, math, pathlib, re, struct, sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rwx_reader import read_rwx  # noqa: E402

SIZE, MIN_CELL, REGION = 512, -256, 640.0
LIGHT = re.compile(r"create\s+light\b([^;]*)", re.I)
SIGN = re.compile(r'create\s+sign\s+"([^"]*)"([^;]*)', re.I)
TELEPORT = re.compile(r"activate\s+teleport\s+([^;]+)", re.I)


def kv(text):
    return {k.lower(): v for k, v in re.findall(r"(\w+)=([^\s;]+)", text)}


def teleport_to_xyz(t):
    m = re.match(r"\s*([\d.]+)([NS])\s+([\d.]+)([WE])(?:\s+([-\d.]+)a)?(?:\s+([-\d.]+))?", t, re.I)
    if not m:
        return None
    z = float(m.group(1)) * 10 * (1 if m.group(2).upper() == "N" else -1)
    x = float(m.group(3)) * 10 * (1 if m.group(4).upper() == "W" else -1)
    y = float(m.group(5) or 0) * 10
    yaw = float(m.group(6) or 0)
    return {"x": x, "y": y, "z": z, "yaw": yaw}


def region_of(x, z):
    i = min(7, max(0, int((x + 2560) // REGION)))
    j = min(7, max(0, int((z + 2560) // REGION)))
    return j * 8 + i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=pathlib.Path)
    ap.add_argument("out", type=pathlib.Path)
    a = ap.parse_args()
    rel, out = a.release, a.out
    if not (rel / "FROZEN.json").exists():
        raise SystemExit("atlas export reads frozen releases only")
    out.mkdir(parents=True, exist_ok=True)
    scene = json.loads((rel / "scene.json").read_text(encoding="utf-8"))["objects"]
    attrs = json.loads((rel / "world-attributes.json").read_text(encoding="utf-8"))
    design = json.loads(pathlib.Path(rel.parent.parent, "DESIGN.json").read_text(encoding="utf-8"))
    names = sorted({o["model"][:-4] if o["model"].lower().endswith(".rwx") else o["model"] for o in scene})
    index_of = {n: i for i, n in enumerate(names)}

    blob = bytearray()
    materials, mat_key = [], {}
    models = []
    for n in names:
        m = read_rwx(rel / "models" / f"{n}.zip", n)
        parts = []
        for p in m["parts"]:
            mat = m["materials"][p["material"]]
            glow = mat["surface"][0] == 0 and mat["surface"][1] == 0 and "prelight" in p
            sign = p["tag"] == 100
            key = (tuple(round(c, 4) for c in mat["color"]), mat["texture"], round(mat["opacity"], 3), glow, sign)
            if key not in mat_key:
                mat_key[key] = len(materials)
                materials.append({"color": [round(c, 4) for c in mat["color"]], "texture": mat["texture"], "opacity": round(mat["opacity"], 3),
                                  "glow": glow, "sign": sign, "surface": [round(s, 3) for s in mat["surface"]]})
            tri = [i for t in p["triangles"] for i in t]
            pos = np.array([p["vertices"][i] for i in tri], dtype="<f4").reshape(-1)
            uv = np.array([p["uv"][i] if p["uv"][i] is not None else (0.0, 0.0) for i in tri], dtype="<f4").reshape(-1)
            entry = {"material": mat_key[key], "count": len(tri), "pos": len(blob)}
            blob += pos.tobytes()
            entry["uv"] = len(blob)
            blob += uv.tobytes()
            if glow:
                pl = np.array([p["prelight"][i] or (1.0, 1.0, 1.0) for i in tri], dtype="<f4").reshape(-1)
                entry["prelight"] = len(blob)
                blob += pl.tobytes()
            if sign:
                entry["signBounds"] = p["bounds"]
            parts.append(entry)
        glow_vs = [v for p in m["parts"] if "prelight" in p and m["materials"][p["material"]]["surface"][:2] == [0, 0] for v in p["vertices"]]
        glow_center = [round(sum(v[i] for v in glow_vs) / len(glow_vs), 3) for i in range(3)] if glow_vs else None
        models.append({"name": n, "triangles": m["triangles"], "bounds": m["bounds"], "parts": parts, "glowCenter": glow_center,
                       "zipSha256": m["source"]["sha256"]})
    (out / "models.bin").write_bytes(bytes(blob))

    heights = np.fromfile(rel / "terrain" / "terrain-heights-cm.i32le", dtype="<i4")
    textures = np.fromfile(rel / "terrain" / "terrain-textures.u16le", dtype="<u2")
    if heights.min() < -32768 or heights.max() > 32767 or textures.max() > 255:
        raise SystemExit("terrain does not fit the atlas encodings")
    heights.astype("<i2").tofile(out / "terrain-h.i16")
    textures.astype("u1").tofile(out / "terrain-t.u8")

    objects = bytearray()
    specials = []
    counts = [0] * 64
    for k, o in enumerate(scene):
        n = o["model"][:-4] if o["model"].lower().endswith(".rwx") else o["model"]
        x, y, z = o["x"] / 100, o["y"] / 100, o["z"] / 100
        r = region_of(x, z)
        counts[r] += 1
        objects += struct.pack("<HHffff", index_of[n], r, x, y, z, math.radians(o["yaw"] / 10))
        act = o["action"] or ""
        sp = {}
        if (m := LIGHT.search(act)):
            q = kv(m.group(1))
            sp["light"] = {"color": q.get("color", "ffffff"), "brightness": float(q.get("brightness", 1)), "radius": float(q.get("radius", 10)),
                           "flicker": q.get("fx", "").lower() == "flicker"}
        if (m := SIGN.search(act)):
            q = kv(m.group(2))
            sp["sign"] = {"text": m.group(1), "color": q.get("color", "ffffff"), "bcolor": q.get("bcolor", "0000c0")}
        if (m := TELEPORT.search(act)):
            sp["teleport"] = {"text": m.group(1).strip(), **(teleport_to_xyz(m.group(1)) or {})}
        if sp:
            sp.update({"object": k, "number": o["number"], "model": index_of[n], "x": x, "y": y, "z": z, "yaw": math.radians(o["yaw"] / 10)})
            specials.append(sp)
    (out / "objects.bin").write_bytes(bytes(objects))

    index = {
        "world": "Wickwood", "size": SIZE, "minCell": MIN_CELL, "spacing": 10, "regionSize": REGION, "regionCounts": counts,
        "objects": len(scene), "objectRecordBytes": 20,
        "attributes": {k: attrs[k] for k in attrs if k.startswith(("Fog", "Sky", "Ambient", "Light", "Water", "Minimum", "MaxLight", "EntryPoint", "Title", "Welcome"))},
        "terrainTextures": design["terrain"].get("textures"),
        "terrainColors": {t["name"]: t["meanRGB"] for t in json.loads((rel / "textures" / "texture-provenance.json").read_text(encoding="utf-8"))["textures"]
                          if t["name"].startswith("terrain")},
        "materials": materials, "models": models,
        "destinations": json.loads((rel / "destinations.json").read_text(encoding="utf-8")),
        "routes": [{k: r[k] for k in ("id", "name", "from", "to", "length", "maxGradePercent", "points")} for r in json.loads((rel / "routes.json").read_text(encoding="utf-8"))],
        "specials": specials,
        "source": {"release": rel.name, "sceneSha256": hashlib.sha256((rel / "scene.json").read_bytes()).hexdigest()},
        "approximation": "Browser approximation rendered with Three.js from the published RWX models and the frozen terrain. It is not the native Active Worlds renderer.",
    }
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in out.iterdir() if p.is_file()}
    print(json.dumps({"models": len(models), "materials": len(materials), "specials": len(specials), "bytes": sizes}, indent=1))


if __name__ == "__main__":
    main()
