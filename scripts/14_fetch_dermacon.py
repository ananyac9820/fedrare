#!/usr/bin/env python3
"""
Download DermaCon-IN (Indian dermatology, NeurIPS 2025) from Harvard Dataverse.

DermaCon-IN is 5,450 clinical photographs from 3,002 patients, collected across three
tertiary-care hospitals in North Karnataka, annotated by board-certified dermatologists:
8 main classes, 19 sub-classes, 245 disease labels, plus Fitzpatrick and Monk skin tone.
Licence CC BY-NC-SA 4.0 - non-commercial, share-alike, attribution required.

    Madarkar et al., "DermaCon-IN: A Multi-concept Annotated Dermatological Image Dataset
    of Indian Skin Disorders for Clinical AI Research", NeurIPS 2025 Datasets & Benchmarks.
    doi:10.7910/DVN/W7OUZM

One thing to know before building on it: the released metadata has no hospital or site
column, so which of the three hospitals an image came from is not recoverable. Any centre
split on this dataset is therefore constructed by us, and has to be reported that way.

    python scripts/14_fetch_dermacon.py              # metadata + both image archives (~3.6 GB)
    python scripts/14_fetch_dermacon.py --metadata   # just the tables (~4 MB)

Writes to data/dermacon_in/ (gitignored). Skips files already downloaded at the right size.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "dermacon_in"
DOI = "doi:10.7910/DVN/W7OUZM"
API = "https://dataverse.harvard.edu/api"
UA = {"User-Agent": "fedrare-research/1.0 (https://github.com/ananyac9820/fedrare)"}

# The trained checkpoints in the same deposit (~350 MB each) are not needed here.
METADATA = ["Skin_Metadata.tab", "train_split.tab", "test_split.tab",
            "Metadata_schema.md", "README.md"]
IMAGES = ["DATASET_0.zip", "DATASET_1.zip"]


def dataset_files() -> list[dict]:
    req = urllib.request.Request(f"{API}/datasets/:persistentId/?persistentId={DOI}",
                                 headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        payload = json.loads(r.read())
    if payload.get("status") != "OK":
        raise SystemExit(f"Dataverse said: {payload.get('message')}")
    return payload["data"]["latestVersion"]["files"]


def fetch(file_entry: dict, out_dir: Path) -> Path:
    data_file = file_entry["dataFile"]
    name, file_id = data_file["filename"], data_file["id"]
    size = int(data_file.get("filesize", 0))
    path = out_dir / name

    if path.exists() and (size == 0 or abs(path.stat().st_size - size) < size * 0.02):
        print(f"  have {name} ({path.stat().st_size / 1e6:.1f} MB)")
        return path

    url = f"{API}/access/datafile/{file_id}"
    print(f"  fetching {name} ({size / 1e6:.1f} MB)...", flush=True)
    req = urllib.request.Request(url, headers=UA)
    done = 0
    with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
            done += len(chunk)
            if size and done % (200 << 20) < (1 << 20):
                print(f"    {done / 1e6:>7.0f} / {size / 1e6:.0f} MB", flush=True)
    print(f"    wrote {path.stat().st_size / 1e6:.1f} MB")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download DermaCon-IN.")
    parser.add_argument("--metadata", action="store_true", help="skip the image archives")
    parser.add_argument("--no-unzip", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    wanted = METADATA + ([] if args.metadata else IMAGES)
    files = {f["dataFile"]["filename"]: f for f in dataset_files()}
    missing = [w for w in wanted if w not in files]
    if missing:
        print(f"  not in the deposit (skipping): {missing}", file=sys.stderr)

    print(f"DermaCon-IN -> {OUT.relative_to(ROOT)}")
    got = [fetch(files[w], OUT) for w in wanted if w in files]

    if not args.no_unzip:
        for path in got:
            if path.suffix == ".zip":
                target = OUT / path.stem
                if target.exists():
                    print(f"  already unpacked {path.name}")
                    continue
                print(f"  unpacking {path.name}...", flush=True)
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(target)
                n = sum(1 for _ in target.rglob("*") if _.is_file())
                print(f"    {n:,} files in {target.name}/")

    print("\nCite: Madarkar et al., DermaCon-IN, NeurIPS 2025 Datasets & Benchmarks "
          "(doi:10.7910/DVN/W7OUZM). CC BY-NC-SA 4.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
