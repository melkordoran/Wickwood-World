"""Fetch every Wickwood client asset over the world's object path URL and prove it is the frozen release file.

Checks per model: HTTP 200, not an HTML page, SHA-256 equal to the frozen release, ZIP CRC valid, exactly one
member named <model>.rwx, strict RWX parse of the downloaded bytes. Textures: JPEG decode at the expected size.
Avatars: catalogue, archive and every referenced geometry. Negative checks: private files and listings are refused.
"""
from __future__ import annotations

import argparse, hashlib, io, json, pathlib, re, sys, tempfile, urllib.error, urllib.request, zipfile

from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rwx_reader import read_rwx  # noqa: E402


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "WickwoodAssetCheck/1"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=pathlib.Path)
    ap.add_argument("base")
    ap.add_argument("--report", type=pathlib.Path, required=True)
    a = ap.parse_args()
    base = a.base.rstrip("/") + "/"
    frozen = json.loads((a.release / "FROZEN.json").read_text(encoding="utf-8"))["files"]
    scene = json.loads((a.release / "scene.json").read_text(encoding="utf-8"))["objects"]
    models = sorted({o["model"][:-4] for o in scene})
    results, failures = [], []

    def check(kind, rel, ok, detail):
        results.append({"kind": kind, "path": rel, "pass": ok, **detail})
        if not ok:
            failures.append(f"{kind} {rel}: {detail}")

    textures = set()
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="wk-http-"))
    for n in models:
        for ext in (".zip", ".rwx"):
            rel = f"models/{n}{ext}"
            status, ctype, body = fetch(base + rel)
            sha = hashlib.sha256(body).hexdigest()
            d = {"status": status, "contentType": ctype, "bytes": len(body), "sha256Match": sha == frozen.get(rel)}
            ok = status == 200 and "html" not in ctype.lower() and not body.lstrip()[:15].lower().startswith(b"<!doctype") and d["sha256Match"]
            if ok and ext == ".zip":
                (tmp / f"{n}.zip").write_bytes(body)
                with zipfile.ZipFile(io.BytesIO(body)) as z:
                    d["members"] = z.namelist()
                    d["crcValid"] = z.testzip() is None
                ok = d["crcValid"] and d["members"] == [f"{n}.rwx"]
                try:
                    m = read_rwx(tmp / f"{n}.zip", n)
                    d["strictParse"] = True
                    d["triangles"] = m["triangles"]
                    textures.update(t["texture"] for t in m["materials"].values() if t.get("texture"))
                except Exception as e:  # noqa: BLE001
                    d["strictParse"] = False
                    d["error"] = str(e)
                    ok = False
            check("model", rel, ok, d)
    textures.update({f"terrain{i}" for i in range(5)})
    textures.add(json.loads((a.release / "world-attributes.json").read_text(encoding="utf-8"))["WaterTexture"])
    for t in sorted(textures):
        rel = f"textures/{t}.jpg"
        status, ctype, body = fetch(base + rel)
        d = {"status": status, "contentType": ctype, "bytes": len(body), "sha256Match": hashlib.sha256(body).hexdigest() == frozen.get(rel)}
        ok = status == 200 and ctype.lower().startswith("image/jpeg") and d["sha256Match"]
        if ok:
            im = Image.open(io.BytesIO(body))
            im.load()
            d.update({"format": im.format, "size": list(im.size), "mode": im.mode})
            ok = im.format == "JPEG" and im.size == (256, 256)
        check("texture", rel, ok, d)
    status, ctype, body = fetch(base + "avatars/avatars.dat")
    refs = re.findall(r"^\s*geometry\s*=\s*(\S+)", body.decode("ascii", "replace"), re.M)
    check("avatar", "avatars/avatars.dat", status == 200 and hashlib.sha256(body).hexdigest() == frozen["avatars/avatars.dat"] and body.startswith(b"version 3\r\n"),
          {"status": status, "geometry": refs})
    status, ctype, body = fetch(base + "avatars/avatars.zip")
    with zipfile.ZipFile(io.BytesIO(body)) as z:
        check("avatar", "avatars/avatars.zip", status == 200 and z.testzip() is None and z.read("avatars.dat") == (a.release / "avatars/avatars.dat").read_bytes(),
              {"status": status, "members": z.namelist()})
    for g in refs:
        stem = g[:-4] if g.lower().endswith(".rwx") else g
        for rel in (f"avatars/{stem}.zip", f"avatars/{stem}.rwx"):
            status, ctype, body = fetch(base + rel)
            ok = status == 200 and hashlib.sha256(body).hexdigest() == frozen.get(rel)
            if ok and rel.endswith(".zip"):
                (tmp / f"{stem}.zip").write_bytes(body)
                try:
                    read_rwx(tmp / f"{stem}.zip", stem)
                except Exception as e:  # noqa: BLE001
                    ok = False
            check("avatar", rel, ok, {"status": status})
    # Private material must not be reachable through the object path.
    for rel in ["scene.json", "properties.sqlite", "world-attributes.json", "BUILD.json", "FROZEN.json", "terrain/terrain-heights-cm.i32le",
                "textures/texture-provenance.json", "models/manifest.json", "../../Caddyfile", "%2e%2e/%2e%2e/Caddyfile", "..%2f..%2fCaddyfile",
                "%2e%2e/%2e%2e/universe/secrets/db-password.hex", ".git/config", "models/", ""]:
        status, ctype, body = fetch(base + rel)
        listing = b"<a href" in body.lower() and status == 200
        check("private", rel or "(root)", status in (403, 404) or (status == 200 and not listing and len(body) == 0), {"status": status, "bytes": len(body)})
    summary = {
        "base": base, "release": a.release.name, "models": len(models), "requests": len(results),
        "passed": sum(r["pass"] for r in results), "failed": len(failures), "failures": failures[:20],
        "textures": sorted(textures),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps({"summary": summary, "results": results}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
