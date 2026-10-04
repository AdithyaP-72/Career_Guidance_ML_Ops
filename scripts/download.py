"""Download the raw datasets from Kaggle's public API (no Kaggle login needed). Works on Windows, macOS and Linux.

  uv run python scripts/download.py            # everything the pipeline needs
  uv run python scripts/download.py --only naukri_2025

Only needed to REBUILD the model. The app itself runs from the committed models/champion/ bundle.
"""
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.kaggle.com/api/v1/datasets/download/"

# name -> (kaggle dataset, optional list of members to extract / rename)
DATASETS = {
    "naukri_2025": ("shivamshrivastava21/indian-job-market-dataset-2025-2026", None),
    "naukri_2019": ("promptcloud/jobs-on-naukricom", "naukri_2019.csv"),
    "naukri_2022": ("kuchhbhi/latest-30k-jobs-data", None),
    "naukri_2017": ("PromptCloudHQ/jobs-on-naukricom", None),
    "internshala": ("jayaantanaath/internship-opportunities-in-india-2025", None),
    "nptel": ("lakshyyaaaa/nptel-swayam-course-catalog", None),
    "coursera": ("longnguyen3774/coursera-courses-metadata-for-analytics-2025", None),
    "plfs": ("pradnyakalvikatte/plfs-india-2017-18-to-2023-24", ["perv1_2023_24.csv"]),
}


def fetch(name: str) -> None:
    slug, extra = DATASETS[name]
    dest = ROOT / "data" / "raw" / name
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / "_download.zip"
    print(f"[{name}] downloading {slug} ...", flush=True)
    urllib.request.urlretrieve(API + slug, zpath)
    with zipfile.ZipFile(zpath) as z:
        if isinstance(extra, list):
            for m in extra:
                z.extract(m, dest)
        else:
            z.extractall(dest)
    zpath.unlink()
    if name == "naukri_2019":  # the zip nests the csv in a long path
        csv = next(dest.rglob("*.csv"))
        shutil.move(str(csv), str(dest / "naukri_2019.csv"))
        for d in [p for p in dest.iterdir() if p.is_dir()]:
            shutil.rmtree(d)
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(DATASETS))
    a = ap.parse_args()
    for n in (a.only or list(DATASETS)):
        try:
            fetch(n)
        except Exception as e:  # noqa: BLE001
            print(f"[{n}] FAILED: {e}\n  Download it manually from https://www.kaggle.com/datasets/{DATASETS[n][0]} "
                  f"and unzip into data/raw/{n}/", file=sys.stderr)
    print("\nNote: Naukri 2025 ships as indian-job-market-dataset-2025.xlsx; ensure it sits in data/raw/naukri_2025/.")
