#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

from scvi.experimental.velovi_improvements.datasets import (
    VELOVI_DATASETS,
    ensure_dataset_file,
)


def download_dataset(name: str, output_dir: Path, force: bool = False) -> None:
    config = VELOVI_DATASETS[name]
    dest_path = ensure_dataset_file(output_dir, config, force=force)
    if dest_path.exists():
        print(f"[ready] {dest_path}")
    else:
        print(f"[warn] Missing file for {name}: expected {dest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VELOVI benchmark datasets via scvelo.")
    default_cache = Path(
        os.environ.get("SCVELO_CACHE_DIR", Path.home() / ".cache/scvelo")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_cache,
        help="Directory where datasets will be stored (default: SCVELO cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of datasets even if files already exist.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in VELOVI_DATASETS:
        download_dataset(dataset_name, output_dir, force=args.force)

    print("[done] Dataset download complete.")


if __name__ == "__main__":
    main()
