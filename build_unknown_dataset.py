#!/usr/bin/env python3
"""Add a diverse, licensed iNaturalist ``unknown`` class to an existing dataset.

Samples a seeded subset of recent research-grade observations from each selected
taxon, keeps one photo per observation, and splits each taxon separately so every
split contains a mix of cacti, succulents, and other plants.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
import sys
import time
from pathlib import Path

from PIL import Image

from build_dataset import (
    DOWNLOAD_DELAY,
    SPLITS,
    download_image,
    fetch_photo_records,
    make_session,
    split_by_observation,
)

# Each selected taxon is distinct from the six known classes. The mix includes
# other cacti/succulents and unrelated garden/wild plants to broaden the negatives.
UNKNOWN_TAXA: dict[str, int] = {
    "opuntia_ficus_indica": 64119,
    "mammillaria_elongata": 273594,
    "schlumbergera_truncata": 51223,
    "agave_americana": 64103,
    "aloe_vera": 126882,
    "echeveria_elegans": 206749,
    "crassula_ovata": 76465,
    "haworthiopsis_attenuata": 577654,
    "taraxacum_officinale": 47602,
    "helianthus_annuus": 57983,
    "rosa_canina": 55884,
    "quercus_robur": 1520309,
}

log = logging.getLogger("build_unknown_dataset")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("dataset"), help="existing dataset root")
    parser.add_argument("--per-taxon", type=int, default=20,
                        help="photos sampled per unknown taxon (default: 20)")
    parser.add_argument("--candidate-multiplier", type=int, default=6,
                        help="candidate observations sampled per taxon before random selection")
    parser.add_argument("--size", choices=("medium", "large", "original"), default="large")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replace", action="store_true",
                        help="replace only existing unknown folders and unknown manifest rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    out: Path = args.out
    if not all((out / split).is_dir() for split in SPLITS):
        raise SystemExit(f"Expected existing train/val/test folders under {out}")
    unknown_dirs = [out / split / "unknown" for split in SPLITS]
    if any(path.exists() and any(path.iterdir()) for path in unknown_dirs) and not args.replace:
        raise SystemExit("An unknown class already exists; pass --replace to replace only that class")

    session = make_session()
    rng = random.Random(args.seed)
    raw_root = out / "_unknown_raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    raw_root.mkdir(parents=True)
    manifest_rows: list[dict] = []
    staged_files: list[tuple[Path, Path]] = []
    summary: dict[str, dict[str, int]] = {}
    seen_photo_ids: set[int] = set()

    try:
        for taxon_index, (taxon_name, taxon_id) in enumerate(UNKNOWN_TAXA.items()):
            candidate_limit = args.per_taxon * args.candidate_multiplier
            log.info("== %s (taxon %s), sampling %d from up to %d candidates",
                     taxon_name, taxon_id, args.per_taxon, candidate_limit)
            candidates = fetch_photo_records(
                session, taxon_id, candidate_limit, args.size, photos_per_obs=1,
                licensed_only=True,
            )
            # The API candidate pool is sorted by observation ID. Seeded shuffling
            # avoids always taking only its newest items while keeping reruns reproducible.
            random.Random(args.seed + taxon_index).shuffle(candidates)
            selected = [record for record in candidates if record["photo_id"] not in seen_photo_ids]
            selected = selected[:args.per_taxon]
            if len(selected) < args.per_taxon:
                log.warning("Only %d licensed candidates available for %s (wanted %d)",
                            len(selected), taxon_name, args.per_taxon)

            cls_raw = raw_root / taxon_name
            cls_raw.mkdir(parents=True)
            downloaded: list[dict] = []
            for index, record in enumerate(selected, 1):
                seen_photo_ids.add(record["photo_id"])
                extension = Path(record["url"].split("?")[0]).suffix.lower() or ".jpg"
                record["filename"] = f"{record['observation_id']}_{record['photo_id']}{extension}"
                path = cls_raw / record["filename"]
                if download_image(session, record["url"], path):
                    try:
                        with Image.open(path) as image:
                            image.verify()
                        record["taxon_id"] = taxon_id
                        downloaded.append(record)
                    except (OSError, SyntaxError):
                        log.warning("Invalid image, skipping %s", path.name)
                        path.unlink(missing_ok=True)
                if index % 10 == 0:
                    log.info("  downloaded %d/%d", index, len(selected))
                time.sleep(DOWNLOAD_DELAY)

            splits = split_by_observation(downloaded, args.seed + taxon_index)
            summary[taxon_name] = {}
            for split, records in splits.items():
                for record in records:
                    src = cls_raw / record["filename"]
                    dst_dir = out / split / "unknown"
                    staged_files.append((src, dst_dir / record["filename"]))
                    manifest_rows.append({"split": split, "class": "unknown", **record})
                summary[taxon_name][split] = len(records)
            log.info("  split: %s", summary[taxon_name])

        # Commit the new class only after all downloads have completed.
        if args.replace:
            for path in unknown_dirs:
                if path.exists():
                    shutil.rmtree(path)
        for src, dst in staged_files:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

        # Ensure the ImageFolder class exists in all splits, including a sparse one.
        for path in unknown_dirs:
            path.mkdir(parents=True, exist_ok=True)

        manifest_path = out / "manifest.csv"
        existing_rows: list[dict] = []
        fields = ["split", "class", "taxon_id", "observation_id", "photo_id", "url",
                  "license", "attribution", "observed_on", "taxon_name", "filename"]
        if manifest_path.exists():
            with manifest_path.open(newline="", encoding="utf-8") as source:
                reader = csv.DictReader(source)
                fields = reader.fieldnames or fields
                existing_rows = [row for row in reader if row.get("class") != "unknown"]
        with manifest_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerows(manifest_rows)

        print("\nUnknown-class dataset summary (licensed iNaturalist images)")
        print(f"{'source taxon':32} {'train':>6} {'val':>6} {'test':>6} {'total':>6}")
        totals = {split: sum(counts[split] for counts in summary.values()) for split in SPLITS}
        for taxon_name, counts in summary.items():
            print(f"{taxon_name:32} {counts['train']:>6} {counts['val']:>6} "
                  f"{counts['test']:>6} {sum(counts.values()):>6}")
        print(f"{'unknown total':32} {totals['train']:>6} {totals['val']:>6} "
              f"{totals['test']:>6} {sum(totals.values()):>6}")
        print(f"\nImages saved under {out.resolve()}; attribution in {manifest_path.resolve()}")
    finally:
        shutil.rmtree(raw_root, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
