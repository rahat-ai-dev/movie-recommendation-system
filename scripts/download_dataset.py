"""
Downloads and extracts the ml-32m MovieLens dataset from the official
GroupLens source into data/raw/ml-32m/.

Run:  python scripts/download_dataset.py

If you already have ml-32m.zip locally (e.g. downloaded manually or
uploaded to a Colab session), pass its path instead of re-downloading:

    python scripts/download_dataset.py --zip /path/to/ml-32m.zip

Safe to re-run: skips the download/extract if data/raw/ml-32m/ratings.csv
already exists (pass --force to redo it anyway).
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=str, default=None,
                         help="Path to an already-downloaded ml-32m.zip (skips the network download).")
    parser.add_argument("--force", action="store_true", help="Re-download/extract even if data is already present.")
    args = parser.parse_args()

    target_dir = config.RAW_DATASET_DIR  # data/raw/ml-32m/
    ratings_path = target_dir / "ratings.csv"

    if ratings_path.exists() and not args.force:
        print(f"{ratings_path} already exists -- nothing to do (use --force to redo).")
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if args.zip:
        zip_path = Path(args.zip)
        if not zip_path.exists():
            raise SystemExit(f"--zip path not found: {zip_path}")
    else:
        import urllib.request
        zip_path = target_dir.parent / "ml-32m.zip"
        print(f"Downloading {config.ML32M_DOWNLOAD_URL} -> {zip_path} (about 900MB extracted, "
              f"~240MB compressed) ...")
        urllib.request.urlretrieve(config.ML32M_DOWNLOAD_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(target_dir.parent)  # archive already contains an "ml-32m/" top folder

    if not ratings_path.exists():
        raise SystemExit(f"Extraction finished but {ratings_path} still missing -- check the archive layout.")

    if not args.zip:
        zip_path.unlink(missing_ok=True)  # only clean up what we downloaded ourselves

    print(f"Done. Dataset ready at {target_dir}/")


if __name__ == "__main__":
    main()
