"""
Create the Phase 2 Track 2 observational-corpus registry.

This script is intentionally lightweight: it records the new real-data targets
that expand the MIRAGE validation corpus beyond WASP-39b. Actual archive
downloads remain manual/MAST driven so raw data is not committed to the repo.

Run from the repository root:
  python scripts/register_phase2_track2_corpus.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_MANIFEST = Path("configs/observational_corpus_phase2_track2.csv")
DEFAULT_OUTPUT = Path("data/observational_corpus/phase2_track2_manifest.csv")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "instrument",
        "mode",
        "archive",
        "program_hint",
        "product_level",
        "status",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    write_manifest(rows, args.output)

    print(f"Registered {len(rows)} Phase 2 Track 2 corpus targets:")
    for row in rows:
        print(f"  - {row['target']}: {row['instrument']} {row['mode']} ({row['archive']})")
    print(f"Wrote local registry to {args.output}")


if __name__ == "__main__":
    main()
