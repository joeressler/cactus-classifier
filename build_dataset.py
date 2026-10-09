#!/usr/bin/env python3
"""
Build a Trichocereus vs. lookalike image dataset from iNaturalist.

Pulls research-grade observation photos per taxon, downloads them at a
higher resolution, then splits them into train/val/test (70/20/10).

Usage:
    python build_dataset.py                       # defaults: 250 imgs/class, ./dataset
    python build_dataset.py --per-class 100 --size medium --out data
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://api.inaturalist.org/v1/observations"
USER_AGENT = "cactus-classifier-dataset-builder/1.0 (contact: joe.a.ressler@gmail.com)"

# Class name -> iNaturalist taxon ID.
# The IDs were checked against /v1/taxa. Several IDs in the original spec pointed to
# unrelated plants, such as Lapageria and Agave, so they were replaced with the
# current accepted taxa. taxon_id queries also include descendant taxa.
CLASSES: dict[str, int] = {
    "trichocereus_pachanoi": 829212,             # Trichocereus macrogonus var. pachanoi
    "trichocereus_peruvianus": 870382,           # Trichocereus macrogonus subsp. peruvianus
    "trichocereus_bridgesii": 870017,            # Trichocereus bridgesii (syn. Echinopsis lageniformis)
    "miss_cereus_repandus": 320319,              # Cereus repandus
    "miss_myrtillocactus_geometrizans": 274227,  # Myrtillocactus geometrizans
    "miss_pilosocereus_pachycladus": 442108,     # Pilosocereus pachycladus
}

SPLITS = {"train": 0.7, "val": 0.2, "test": 0.1}
PER_PAGE = 100            # API max for v1/observations
API_DELAY = 1.0           # iNat asks for ~1 request/sec on the API
DOWNLOAD_DELAY = 0.2      # pause between image downloads
SIZE_RE = re.compile(r"/(square|thumb|small|medium|large|original)\.(jpe?g|png|gif)", re.I)

log = logging.getLogger("build_dataset")


def make_session() -> requests.Session:
    """Requests session with a UA header and automatic retry/backoff on flaky responses."""
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    retry = Retry(
        total=5,
        backoff_factor=1.5,                       # 0, 1.5, 3, 6, 12s...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def upscale_url(url: str, size: str) -> str:
    """Swap the size segment in an iNat photo URL (square/small/...) for the requested size."""
    return SIZE_RE.sub(lambda m: f"/{size}.{m.group(2)}", url, count=1)


def fetch_photo_records(
    session: requests.Session,
    taxon_id: int,
    limit: int,
    size: str,
    photos_per_obs: int,
    licensed_only: bool,
) -> list[dict]:
    """Page through research-grade observations for a taxon and collect up to `limit` photos."""
    records: list[dict] = []
    seen_photo_ids: set[int] = set()
    id_below: int | None = None  # id-based cursor; avoids the API's deep-pagination cap

    page = 1
    while len(records) < limit:
        params = {
            "taxon_id": taxon_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": PER_PAGE,
            "order_by": "id",
            "order": "desc",
            "page": 1 if id_below else page,
        }
        if id_below:
            params["id_below"] = id_below
        if licensed_only:
            params["photo_license"] = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"

        try:
            resp = session.get(API_URL, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
        except (requests.RequestException, ValueError) as exc:
            log.error("API request failed for taxon %s (page %s): %s", taxon_id, page, exc)
            break

        results = payload.get("results", [])
        if not results:
            break  # no more observations

        for obs in results:
            taken = 0
            for photo in obs.get("photos", []):
                if len(records) >= limit or taken >= photos_per_obs:
                    break
                pid, url = photo.get("id"), photo.get("url")
                if not pid or not url or pid in seen_photo_ids:
                    continue
                seen_photo_ids.add(pid)
                taken += 1
                records.append({
                    "observation_id": obs["id"],
                    "photo_id": pid,
                    "url": upscale_url(url, size),
                    "license": photo.get("license_code") or "all-rights-reserved",
                    "attribution": photo.get("attribution", ""),
                    "observed_on": obs.get("observed_on") or "",
                    "taxon_name": (obs.get("taxon") or {}).get("name", ""),
                })
            if len(records) >= limit:
                break

        log.info("  taxon %s: %d/%d photos collected", taxon_id, len(records), limit)
        id_below = results[-1]["id"]
        page += 1
        if len(results) < PER_PAGE:
            break  # last page
        time.sleep(API_DELAY)

    return records


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    """Download one image to `dest`. Skips files that already exist, which makes reruns resumable."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with session.get(url, timeout=30, stream=True) as resp:
            resp.raise_for_status()
            if not resp.headers.get("Content-Type", "").startswith("image/"):
                log.warning("Not an image (%s): %s", resp.headers.get("Content-Type"), url)
                return False
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
        tmp.replace(dest)
        return True
    except (requests.RequestException, OSError) as exc:
        log.warning("Download failed %s: %s", url, exc)
        tmp.unlink(missing_ok=True)
        return False


def split_by_observation(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """
    70/20/10 split grouped by observation ID, so photos of the same plant never
    land in both train and test (this avoids data leakage).
    """
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["observation_id"]].append(r)
    obs_ids = sorted(groups)
    random.Random(seed).shuffle(obs_ids)

    n = len(obs_ids)
    n_train = round(n * SPLITS["train"])
    n_val = round(n * SPLITS["val"])
    buckets = {
        "train": obs_ids[:n_train],
        "val": obs_ids[n_train:n_train + n_val],
        "test": obs_ids[n_train + n_val:],
    }
    return {split: [r for oid in ids for r in groups[oid]] for split, ids in buckets.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="output root (default: ./dataset)")
    parser.add_argument("--per-class", type=int, default=250, help="max images per class (default: 250)")
    parser.add_argument("--size", choices=("medium", "large", "original"), default="large",
                        help="photo size to download (default: large, ~1024px)")
    parser.add_argument("--photos-per-obs", type=int, default=1,
                        help="max photos kept per observation; 1 = most diverse (default: 1)")
    parser.add_argument("--licensed-only", action="store_true",
                        help="only keep Creative Commons licensed photos")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the split")
    parser.add_argument("--clean", action="store_true", help="delete the output folder before starting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out: Path = args.out
    raw_dir = out / "_raw"  # staging area; images are moved into splits afterwards
    if args.clean and out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        for cls in CLASSES:
            (out / split / cls).mkdir(parents=True, exist_ok=True)

    session = make_session()
    manifest_rows: list[dict] = []
    summary: dict[str, dict[str, int]] = {}

    for cls, taxon_id in CLASSES.items():
        log.info("== %s (taxon %s)", cls, taxon_id)
        records = fetch_photo_records(session, taxon_id, args.per_class, args.size,
                                      args.photos_per_obs, args.licensed_only)
        if len(records) < args.per_class:
            log.warning("  only %d photos available for %s (wanted %d)", len(records), cls, args.per_class)

        # 1) Download into the staging folder.
        cls_raw = raw_dir / cls
        cls_raw.mkdir(parents=True, exist_ok=True)
        downloaded: list[dict] = []
        for i, rec in enumerate(records, 1):
            ext = Path(rec["url"].split("?")[0]).suffix.lower() or ".jpg"
            rec["filename"] = f"{rec['observation_id']}_{rec['photo_id']}{ext}"
            if download_image(session, rec["url"], cls_raw / rec["filename"]):
                downloaded.append(rec)
            if i % 25 == 0:
                log.info("  downloaded %d/%d", i, len(records))
            time.sleep(DOWNLOAD_DELAY)

        # 2) Split and move into train/val/test.
        splits = split_by_observation(downloaded, args.seed)
        summary[cls] = {}
        for split, recs in splits.items():
            for rec in recs:
                src = cls_raw / rec["filename"]
                dst = out / split / cls / rec["filename"]
                if src.exists():
                    shutil.move(str(src), str(dst))
                manifest_rows.append({"split": split, "class": cls, "taxon_id": taxon_id, **rec})
            summary[cls][split] = len(recs)
        log.info("  split: %s", summary[cls])

    shutil.rmtree(raw_dir, ignore_errors=True)

    # Manifest with source URLs + license/attribution for every image (needed for reuse/credits).
    manifest_path = out / "manifest.csv"
    if manifest_rows:
        with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print("\nDataset summary")
    print(f"{'class':38} {'train':>6} {'val':>6} {'test':>6} {'total':>6}")
    for cls, counts in summary.items():
        total = sum(counts.values())
        print(f"{cls:38} {counts.get('train', 0):>6} {counts.get('val', 0):>6} {counts.get('test', 0):>6} {total:>6}")
    print(f"\nImages in {out.resolve()}; manifest at {manifest_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
