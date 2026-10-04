"""Download the raw datasets listed in configs/sources.yaml into data/raw/<dest>/.

    uv run python -m src.data.download                       # everything
    uv run python -m src.data.download --group core comparison
    uv run python -m src.data.download --only naukri_2025 --force
    uv run python -m src.data.download --list

This runs once, by hand. After it finishes, track each folder with `dvc add`
(see README, Step 2). DVC and the DagsHub remote then hold the bytes, so teammates
run `dvc pull` and never need this script.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.utils import CONFIGS, RAW, get_logger, load_yaml

log = get_logger("download")


def fetch(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "career-guidance-mlops/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 20)


def extract(zip_path: Path, files: dict[str, str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for target, member in files.items():
            if member not in names:
                raise FileNotFoundError(f"{member!r} not in {zip_path.name}. Contents: {sorted(names)[:20]}")
            with zf.open(member) as src, open(out_dir / target, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1 << 20)


def main() -> None:
    sources = load_yaml(CONFIGS / "sources.yaml")["sources"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=sorted(sources), help="download just these sources")
    ap.add_argument("--group", nargs="+", choices=sorted({s["group"] for s in sources.values()}))
    ap.add_argument("--force", action="store_true", help="re-download even if the files exist")
    ap.add_argument("--list", action="store_true", help="show the sources and exit")
    args = ap.parse_args()

    if args.list:
        for name, s in sources.items():
            print(f"{name:24s} {s['group']:11s} -> data/raw/{s['dest']}/  [{s['licence']}]  {s['page']}")
        return

    chosen = {
        n: s
        for n, s in sources.items()
        if (not args.only or n in args.only) and (not args.group or s["group"] in args.group)
    }
    for name, src in chosen.items():
        out_dir = RAW / src["dest"]
        shown = out_dir.relative_to(RAW.parents[1])
        if not args.force and all((out_dir / f).exists() for f in src["files"]):
            log.info("%s: already in %s, skipping (use --force to refresh)", name, shown)
            continue
        log.info("%s: downloading %s", name, src["url"])
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "download.zip"
            fetch(src["url"], zip_path)
            extract(zip_path, src["files"], out_dir)
        for f in src["files"]:
            log.info("  saved %s/%s (%.1f MB)", shown, f, (out_dir / f).stat().st_size / 1e6)


if __name__ == "__main__":
    main()
