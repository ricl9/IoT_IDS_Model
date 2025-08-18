#!/usr/bin/env python3
"""
CLI to prepare cleaned Parquet datasets from CSVs.
Usage:
  python -m src.cli_prepare --level packet --out data/ --seed 42
  python -m src.cli_prepare --level flow   --out data/ --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .data_preparation import PrepConfig, prepare_level_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["packet", "flow"], required=True)
    ap.add_argument("--base", type=Path, default=Path("datasets"))
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-scale", action="store_true", help="Disable numeric scaling")
    ap.add_argument("--iqr", type=float, default=3.0, help="IQR clip factor")
    ap.add_argument("--sample-frac", type=float, default=None, help="Debug: process a random fraction of rows from each CSV")
    args = ap.parse_args()

    cfg = PrepConfig(
        base_dir=args.base.resolve(),
        level=args.level,
        out_dir=args.out.resolve(),
        seed=args.seed,
        scale_numeric=not args.no_scale,
        iqr_clip_factor=args.iqr,
        sample_frac=args.sample_frac,
    )
    _, meta = prepare_level_dataset(cfg)
    out_file = cfg.out_dir / (cfg.parquet_filename or f"{cfg.level}_clean.parquet")
    print(f"Wrote {meta['rows']} rows to {out_file}")


if __name__ == "__main__":
    main()
