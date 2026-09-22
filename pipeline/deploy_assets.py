"""Copy Wickwood's client assets from a frozen release into the object path.

Only deliberate client files are published: models (.zip plus the plain .rwx fallback), textures (.jpg) and
avatars (avatars.dat, avatars.zip, avatar models). Scene data, databases, terrain buffers, build metadata and
provenance files stay private. Existing files are never overwritten with different content, except with
--update-from <previous release>: then a published file is replaced only when it still matches that previous
release exactly (anything else is drift and is refused), and each replacement is swapped in atomically.
"""
from __future__ import annotations

import argparse, hashlib, json, os, pathlib, shutil, sys

ALLOW = {
    "models": (".zip", ".rwx"),
    "textures": (".jpg",),
    "avatars": (".dat", ".zip", ".rwx"),
}


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=pathlib.Path)
    ap.add_argument("target", type=pathlib.Path)
    ap.add_argument("--manifest", type=pathlib.Path, required=True)
    ap.add_argument("--update-from", type=pathlib.Path, help="the frozen release currently published at the target")
    a = ap.parse_args()
    if not (a.release / "FROZEN.json").exists():
        raise SystemExit("refusing to deploy from a folder that is not a frozen release")
    frozen = json.loads((a.release / "FROZEN.json").read_text(encoding="utf-8"))["files"]
    previous = json.loads((a.update_from / "FROZEN.json").read_text(encoding="utf-8"))["files"] if a.update_from else {}
    copied, same, conflicts, replaced = [], [], [], []
    for folder, exts in ALLOW.items():
        for src in sorted((a.release / folder).iterdir()):
            rel = f"{folder}/{src.name}"
            if src.suffix.lower() not in exts or src.name.startswith("."):
                continue
            if sha(src) != frozen[rel]:
                raise SystemExit(f"release file changed since freezing: {rel}")
            dst = a.target / folder / src.name
            if dst.exists():
                current = sha(dst)
                if current == frozen[rel]:
                    same.append(rel)
                elif previous.get(rel) == current:
                    tmp = dst.with_name(f".{dst.name}.updating")
                    shutil.copyfile(src, tmp)
                    if sha(tmp) != frozen[rel]:
                        tmp.unlink()
                        raise SystemExit(f"copy verification failed: {rel}")
                    os.replace(tmp, dst)
                    replaced.append(rel)
                else:
                    conflicts.append(rel)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            if sha(dst) != frozen[rel]:
                raise SystemExit(f"copy verification failed: {rel}")
            copied.append(rel)
    if conflicts:
        raise SystemExit(f"target already holds different content for: {conflicts[:10]}")
    published = sorted(str(p.relative_to(a.target)).replace("\\", "/") for folder in ALLOW for p in (a.target / folder).rglob("*") if p.is_file())
    stray = [p for p in published if p not in frozen]
    if stray:
        raise SystemExit(f"target folder holds files that are not release assets: {stray[:10]}")
    report = {"release": str(a.release), "target": str(a.target), "copied": len(copied), "alreadyPresent": len(same),
              "updateFrom": str(a.update_from) if a.update_from else None, "replaced": replaced,
              "published": {p: frozen[p] for p in published}}
    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    a.manifest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "published"} | {"files": len(published)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
