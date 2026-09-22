"""Build Wickwood from DESIGN.json into generated/ and validate it.

    python pipeline/build.py            # generate + validate into generated/
    python pipeline/build.py --freeze   # additionally copy generated/ to releases/<stamp>/ with a manifest

The build never contacts a server. A frozen release directory is never overwritten.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
import math
import re
import shutil
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.dont_write_bytecode = True

import wk_layout  # noqa: E402
import wk_models  # noqa: E402
import wk_qa  # noqa: E402
import wk_terrain  # noqa: E402
import wk_textures  # noqa: E402
from wk_mesh import MATERIALS, write_zip  # noqa: E402

BUILD_TIME = datetime.datetime(2026, 9, 22, 3, 0, 0, tzinfo=datetime.timezone.utc)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_reader():
    spec = importlib.util.spec_from_file_location("wk_rwx_reader", ROOT / "pipeline" / "rwx_reader.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def world_attributes(design, destinations):
    A = design["atmosphere"]
    fog, sky = A["fog"], A["sky"]
    rgb = lambda prefix, c: {f"{prefix}Red": str(c[0]), f"{prefix}Green": str(c[1]), f"{prefix}Blue": str(c[2])}
    attrs = {
        "Title": "Wickwood - a fog-bound forest of lanterns",
        "Keywords": "Wickwood forest fog lanterns trails hamlet tarn beacon oak candles glowmoss",
        "WelcomeMessage": "Welcome to Wickwood. Follow the lantern trails into the fog, or use the wayboard signs to travel. Flight is enabled.",
        "EntryPoint": destinations["glade"]["teleport"],
        "ObjectPath": design["objectPath"],
        "ObjectRefresh": "30",
        "EnterRight": "",  # private while importing; opened to "*" after readback passes
        "SpeakRight": "*",
        "AllowTouristBuild": "N",
        "AllowFlying": "Y", "AllowTeleport": "Y", "AllowPassthru": "Y", "AllowAvatarCollision": "Y",
        "Gravity": "1", "Friction": "1",
        "EnableTerrain": "Y", "TerrainOffset": "0", "Ground": "", "RepeatingGround": "N",
        "FogEnable": "Y" if fog["enabled"] else "N", "FogTinted": "Y" if fog["tinted"] else "N",
        "FogMinimum": str(fog["minimum"]), "FogMaximum": str(fog["maximum"]), "MinimumVisibility": str(A["minimumVisibility"]),
        "MaxLightRadius": str(A["maxLightRadius"]),
        "LightX": str(A["moonlight"]["direction"][0]), "LightY": str(A["moonlight"]["direction"][1]), "LightZ": str(A["moonlight"]["direction"][2]),
        "WaterEnabled": "Y", "WaterLevel": str(design["terrain"]["waterLevelMetres"]), "WaterUnderTerrain": "Y", "WaterTexture": "wk_water",
        "WaterOpacity": "210", "WaterVisibility": "12", "WaterSpeed": "0.2", "WaterWaveMove": "0.03", "WaterSurfaceMove": "0.02",
        "CellLimit": "11000",  # classic "ultra" code; the server stores its 32768-byte tier for classic clients
    }
    attrs.update(rgb("Fog", fog["color"]))
    for side in ("Top", "North", "East", "South", "West", "Bottom"):
        colour = sky["top"] if side == "Top" else (sky["bottom"] if side == "Bottom" else sky["horizon"])
        attrs.update(rgb(f"Sky{side}", colour))
    attrs.update(rgb("AmbientLight", A["ambient"]))
    attrs.update(rgb("Light", A["moonlight"]["color"]))
    attrs.update(rgb("Water", (18, 40, 44)))
    return attrs


def native_records(placements):
    """Integer native fields: centimetres, tenths of degrees, model names with .rwx."""
    records = []
    stamp = int(BUILD_TIME.timestamp())
    for number, p in enumerate(placements, 1):
        records.append({
            "number": number, "owner": 1, "timestamp": stamp, "type": 0,
            "x": int(round(p["x"] * 100)), "y": int(round(p["y"] * 100)), "z": int(round(p["z"] * 100)),
            "yaw": int(round(p["yaw"] * 10)) % 3600, "tilt": 0, "roll": 0,
            "model": p["model"] + ".rwx", "description": p["description"], "action": p["action"],
            "role": p["role"],
        })
    return records


def write_propdb(path, records):
    """Portable property SQLite (the aw_prop2sqlite/Lumenwood source layout) for offline inspection."""
    path = Path(path)
    if path.exists():
        path.unlink()
    schema = """CREATE TABLE cell_data (rowid INTEGER PRIMARY KEY, number INTEGER, citizen INTEGER, timestamp TEXT,
        z INTEGER, x INTEGER, y INTEGER, yaw INTEGER, tilt INTEGER, roll INTEGER, type INTEGER, model TEXT, description TEXT,
        action TEXT, data BLOB,
        cell_z INTEGER GENERATED ALWAYS AS (CASE WHEN z < 0 THEN ((z - 999) / 1000) ELSE z / 1000 END) STORED,
        cell_x INTEGER GENERATED ALWAYS AS (CASE WHEN x < 0 THEN ((x - 999) / 1000) ELSE x / 1000 END) STORED);
        CREATE INDEX cell_idx ON cell_data(cell_z, cell_x);"""
    with sqlite3.connect(path) as db:
        db.executescript(schema)
        stamp = BUILD_TIME.strftime("%Y-%m-%d %H:%M:%S")
        db.executemany("INSERT INTO cell_data(number, citizen, timestamp, x, y, z, yaw, tilt, roll, type, model, description, action, data) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)",
                       [(r["number"], r["owner"], stamp, r["x"], r["y"], r["z"], r["yaw"], r["tilt"], r["roll"], r["type"], r["model"], r["description"], r["action"]) for r in records])
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        return db.execute("SELECT count(*), count(DISTINCT cell_x || ',' || cell_z) FROM cell_data").fetchone()


def cell_of(cm):
    return math.floor(cm / 1000)


def validate(design, meta, sampler, layout, records, model_meta, reader_results):
    problems, notes = [], []
    R = design["range"]
    # bounds and cell budgets
    cells = collections.defaultdict(int)
    counts = collections.defaultdict(int)
    for r in records:
        cx, cz = cell_of(r["x"]), cell_of(r["z"])
        if abs(cx) > R or abs(cz) > R:
            problems.append(f"object {r['number']} outside P{R}: cell {cx},{cz}")
        size = 32 + len(r["model"]) + len(r["description"]) + len(r["action"])
        cells[(cx, cz)] += size
        counts[(cx, cz)] += 1
    worst = max(cells.values())
    if worst > 32768:
        problems.append(f"cell byte budget exceeded: {worst}")
    over_normal = sum(1 for v in cells.values() if v > 2048)
    # model and texture references
    used = collections.Counter(p["model"] for p in layout.placements)
    for name in used:
        if name not in model_meta:
            problems.append(f"placed model missing: {name}")
    for name, rr in reader_results.items():
        if rr.get("warnings"):
            problems.append(f"{name}: reader warnings {rr['warnings'][:3]}")
    # trails: ribbon surface against the terrain
    deviations = []
    for rt in meta["routes"]:
        for p in rt["planes"]:
            (ax, az), (bx, bz) = p["a"], p["b"]
            L = p["length"]
            dx, dz = (bx - ax) / L, (bz - az) / L
            for t in (0.0, L / 2, L):
                for side in (-1.5, 0.0, 1.5):
                    x, z = ax + dx * t - dz * side, az + dz * t + dx * side
                    top = p["h"] + p["grade"] * t + 0.07
                    for c in sampler.candidates(x, z):
                        deviations.append(top - float(c))
    buried = sum(1 for d in deviations if d < 0)
    floating = max(deviations)
    # trees against trails, lanterns and buildings
    trees = [p for p in layout.placements if p["role"] == "tree"]
    near_trail = [p for p in trees if layout.path_distance(p["x"], p["z"]) < 5.0]
    if near_trail:
        problems.append(f"{len(near_trail)} trees within 5 m of a trail centreline")
    lanterns = [p for p in layout.placements if p["model"] == "wk_lantern_post"]
    for p in lanterns:
        if layout.path_distance(p["x"], p["z"]) < 2.2:
            problems.append(f"lantern post on a trail at {p['x']:.1f},{p['z']:.1f}")
    # building pads must be planar (both terrain diagonals agree) under every footprint
    for p in layout.placements:
        if p["role"] != "building":
            continue
        b = model_meta[p["model"]]["bounds"]
        spread = 0.0
        for fx in (b["min"][0], 0.0, b["max"][0]):
            for fz in (b["min"][2], 0.0, b["max"][2]):
                c, s = math.cos(math.radians(p["yaw"])), math.sin(math.radians(p["yaw"]))
                x, z = p["x"] + fx * c + fz * s, p["z"] - fx * s + fz * c
                spread = max(spread, float(max(sampler.candidates(x, z))) - float(min(sampler.candidates(x, z))), abs(float(sampler.height(x, z)) - p["y"]))
        if spread > 0.03:
            problems.append(f"{p['model']} at {p['x']:.0f},{p['z']:.0f}: pad not flat ({spread:.3f} m)")
    # signs and destinations
    tele = re.compile(r"activate teleport (\d+\.\d\d)([NS]) (\d+\.\d\d)([WE]) (-?\d+\.\d\d)a (\d+)$")
    for p in layout.placements:
        if p["role"] == "sign" and "teleport" in p["action"]:
            m = tele.search(p["action"])
            if not m:
                problems.append(f"bad teleport action: {p['action']}")
    for d in layout.destinations.values():
        g = float(sampler.ground(d["x"], d["z"]))
        if d["y"] < g - 0.05:
            problems.append(f"destination {d['id']} below ground")
        if not layout.hash.clear(d["x"], d["z"], 0.6, reach=8):
            problems.append(f"destination {d['id']} is blocked by an object")
    numbers = [r["number"] for r in records]
    if len(set(numbers)) != len(numbers):
        problems.append("duplicate object numbers")
    notes.append(f"{over_normal} cells exceed the 2048-byte normal tier (the world uses the 32768-byte tier)")
    for name, info in model_meta.items():
        for c in info.get("floatingParts", []):
            problems.append(f"{name}: {'/'.join(c['materials'])} part floats {c['clearance']:.2f} m clear of the rest of the model")
    notes.append(f"floating-part check: parts more than {wk_qa.MIN_CLEARANCE:.2f} m clear fail the build; "
                 f"exempt by design {sorted(wk_qa.INTENTIONAL)}, accepted as they are {sorted(wk_qa.ACCEPTED)}")
    return {
        "problems": problems, "notes": notes,
        "objects": len(records), "cells": len(cells), "maxCellBytes": worst, "maxObjectsPerCell": max(counts.values()),
        "trailSurface": {"samples": len(deviations), "belowTerrainSamples": buried, "maxRibbonTopAboveTerrain": round(floating, 3),
                         "maxRibbonTopBelowTerrain": round(min(deviations), 3)},
        "treesWithin5mOfTrail": len(near_trail),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    design = json.loads((ROOT / "DESIGN.json").read_text(encoding="utf-8"))
    seed = design["seed"]
    out = ROOT / "generated"
    if out.exists():
        shutil.rmtree(out)
    (out / "models").mkdir(parents=True)
    (out / "avatars").mkdir()

    textures = wk_textures.build(out / "textures", seed)
    meta, _ = wk_terrain.build(design, out / "terrain")
    meta, sampler = wk_terrain.load(out / "terrain")  # use the quantized centimetre grid from here on

    layout = wk_layout.Layout(design, meta, sampler, seed)
    stats = layout.build(seed)
    catalogue = wk_models.catalogue(seed)
    used = collections.Counter(p["model"] for p in layout.placements)
    reader = load_reader()
    model_meta, reader_results = {}, {}
    for name in sorted(used):
        mesh = catalogue[name]
        info = mesh.export(out / "models")
        vols = mesh.closed_volumes()
        info["closedParts"] = len(vols)
        info["inwardClosedParts"] = sum(1 for _, v in vols if v <= 0)
        info["placements"] = used[name]
        parsed = reader.read_rwx(out / "models" / f"{name}.rwx", name=name)
        reader_results[name] = {"warnings": parsed.get("warnings", []), "triangles": sum(len(p["triangles"]) for p in parsed["parts"])}
        info["readerTriangles"] = reader_results[name]["triangles"]
        info["zipSha256"] = sha(out / "models" / f"{name}.zip")
        floating = wk_qa.scan({name: mesh}).get(name)
        if floating:
            info["floatingParts"] = floating
        model_meta[name] = info
    texture_names = {t["name"] for t in textures}
    missing_textures = sorted({t for m in model_meta.values() for t in m["textures"]} - texture_names)

    # avatars: static geometry, version 3 catalogue, ASCII with CRLF line endings
    records_av = []
    for index, (geo, (title, mesh)) in enumerate(wk_models.avatars().items()):
        info = mesh.export(out / "avatars", crlf=True)
        b = info["bounds"]
        records_av.append({"index": index, "name": title, "geometry": f"{geo}.rwx", "bounds": b, "triangles": info["triangles"],
                           "textures": info["textures"], "static": True})
    cat = ["version 3", ""]
    for r in records_av:
        cat += ["avatar", f" name={r['name']}", f" geometry={r['geometry']}", " beginimp", " endimp", " beginexp", " endexp", "endavatar", ""]
    data = ("\r\n".join(cat) + "\r\n").encode("ascii")
    (out / "avatars" / "avatars.dat").write_bytes(data)
    write_zip(out / "avatars" / "avatars.zip", "avatars.dat", data)

    records = native_records(layout.placements)
    (out / "scene.json").write_text(json.dumps({"world": design["world"], "objects": records}, separators=(",", ":")) + "\n", encoding="utf-8")
    counts = write_propdb(out / "properties.sqlite", records)
    attrs = world_attributes(design, layout.destinations)
    (out / "world-attributes.json").write_text(json.dumps(attrs, indent=1) + "\n", encoding="utf-8")
    (out / "destinations.json").write_text(json.dumps(list(layout.destinations.values()), indent=1) + "\n", encoding="utf-8")
    routes = [{"id": r["id"], "name": r["name"], "from": r["from"], "to": r["to"], "length": r["length"], "maxGradePercent": r["maxGradePercent"],
               "points": r["stations"]} for r in meta["routes"]]
    (out / "routes.json").write_text(json.dumps(routes, indent=1) + "\n", encoding="utf-8")
    (out / "models" / "manifest.json").write_text(json.dumps(model_meta, indent=1) + "\n", encoding="utf-8")

    report = validate(design, meta, sampler, layout, records, model_meta, reader_results)
    if missing_textures:
        report["problems"].append(f"textures referenced but not generated: {missing_textures}")
    for name, info in model_meta.items():
        if info["inwardClosedParts"]:
            report["problems"].append(f"{name}: {info['inwardClosedParts']} inward-facing closed parts")
    for r in records_av:
        h = r["bounds"]["max"][1] - r["bounds"]["min"][1]
        if not (1.6 <= h <= 2.2) or abs(r["bounds"]["min"][1]) > 0.001:
            report["problems"].append(f"avatar {r['name']} height/base outside the tested envelope")
    lights = sum(1 for r in records if "create light" in r["action"])
    signs = sum(1 for r in records if "create sign" in r["action"])
    summary = {
        "world": design["world"], "seed": seed, "range": design["range"], "built": datetime.datetime.now().astimezone().isoformat(),
        "objects": len(records), "cellsWithObjects": counts[1], "byRole": dict(collections.Counter(r["role"] for r in records)),
        "layoutStats": stats, "uniqueModels": len(model_meta), "modelTriangles": sum(m["triangles"] for m in model_meta.values()),
        "placedTriangles": sum(m["triangles"] * m["placements"] for m in model_meta.values()),
        "lights": lights, "signs": signs, "textures": len(textures), "avatars": len(records_av),
        "downloadBytes": sum(m["zipBytes"] for m in model_meta.values()) + sum(t["bytes"] for t in textures),
        "terrain": {"min": meta["heights"]["minMetres"], "max": meta["heights"]["maxMetres"], "heightsSha256": meta["heights"]["sha256"],
                    "texturesSha256": meta["textures"]["sha256"]},
        "validation": report, "seconds": round(time.time() - t0, 1),
    }
    files = {str(p.relative_to(out)).replace("\\", "/"): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    summary["files"] = len(files)
    (out / "BUILD.json").write_text(json.dumps({**summary, "sha256": files}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "validation"}, indent=1))
    print(json.dumps(report, indent=1))
    if report["problems"]:
        print("VALIDATION FAILED", file=sys.stderr)
        sys.exit(2)
    if args.freeze:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = ROOT / "releases" / stamp
        if dest.exists():
            sys.exit(f"release {dest} already exists; refusing to overwrite")
        shutil.copytree(out, dest)
        (dest / "FROZEN.json").write_text(json.dumps({"frozen": stamp, "source": "generated/", "files": files}, indent=1) + "\n", encoding="utf-8")
        print(f"frozen release: {dest}")


if __name__ == "__main__":
    main()
