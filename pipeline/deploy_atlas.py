"""Publish the Wickwood browser atlas (page, vendored Three.js and exported data) under the object path.

Only the page files, the vendored library with its licence, and the exported atlas data are copied.
The atlas is a derived preview, so republishing replaces it; a manifest of hashes is written privately.
"""
import hashlib, json, pathlib, shutil, sys

root = pathlib.Path(__file__).resolve().parent.parent
src_page = root / "preview"
src_data = root / "build" / "atlas" / "data"
target = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
files = {f"{n}": src_page / n for n in ("index.html", "app.js", "style.css")}
files.update({f"vendor/{n}": src_page / "vendor" / n for n in ("three.module.js", "three.core.js", "OrbitControls.js", "LICENSE")})
files.update({f"data/{p.name}": p for p in sorted(src_data.iterdir()) if p.is_file()})
expected = {"data/index.json", "data/models.bin", "data/objects.bin", "data/terrain-h.i16", "data/terrain-t.u8"}
missing = expected - set(files)
if missing:
    raise SystemExit(f"atlas data missing: {sorted(missing)}")
if target.exists():
    stray = [str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() and str(p.relative_to(target)).replace("\\", "/") not in files]
    if stray:
        raise SystemExit(f"target holds unexpected files: {stray[:10]}")
out = {}
for rel, src in files.items():
    dst = target / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    h = hashlib.sha256(dst.read_bytes()).hexdigest()
    if h != hashlib.sha256(src.read_bytes()).hexdigest():
        raise SystemExit(f"copy verification failed: {rel}")
    out[rel] = h
manifest.parent.mkdir(parents=True, exist_ok=True)
manifest.write_text(json.dumps({"target": str(target), "files": out}, indent=1), encoding="utf-8")
print(f"published {len(out)} atlas files to {target}")
