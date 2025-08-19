#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .phase2_anomaly import Phase2Config, run_phase2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/packet_clean.parquet"))
    ap.add_argument("--meta", type=Path, default=Path("data/packet_clean.parquet.meta.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase2"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pca", type=int, default=20)
    ap.add_argument("--kmeans-k", type=int, nargs="*", default=[5, 10, 15])
    ap.add_argument("--methods", type=str, nargs="*", default=["kmeans", "iforest", "lof", "autoencoder"],
                    help='Subset of methods to run: any of {"kmeans","iforest","lof","autoencoder"}')
    ap.add_argument("--target-fpr", type=float, default=0.02, help="Target FPR used to set anomaly threshold from benign train scores")
    args = ap.parse_args()

    cfg = Phase2Config(
        data_parquet=args.data.resolve(),
        meta_json=args.meta.resolve(),
        out_dir=args.out.resolve(),
        seed=args.seed,
        pca_components=args.pca,
        kmeans_k_list=tuple(args.kmeans_k),
    methods=tuple(args.methods),
    target_fpr=args.target_fpr,
    )
    res = run_phase2(cfg)
    print("Best method:", res["best"]["name"]) 
    print("Validation F1:", res["best"]["val"]["f1"]) 
    print("Test F1:", res["best"]["test"]["f1"]) 


if __name__ == "__main__":
    main()
