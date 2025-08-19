#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .phase3_supervised import Phase3Config, run_phase3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", type=Path, default=Path("data/packet_clean.parquet"))
    ap.add_argument("--packet-meta", type=Path, default=Path("data/packet_clean.parquet.meta.json"))
    ap.add_argument("--flow", type=Path, default=Path("data/flow_clean.parquet"))
    ap.add_argument("--flow-meta", type=Path, default=Path("data/flow_clean.parquet.meta.json"))
    ap.add_argument("--out", type=Path, default=Path("reports/phase3"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--no-ae", action="store_true", help="Disable autoencoder and use IsolationForest for stage-1")
    ap.add_argument("--classifiers", type=str, nargs="*", default=["rf", "gb", "logreg"], help="Subset of classifiers to try (rf, gb, hgb, logreg)")
    # Sampling knobs
    ap.add_argument("--benign-target", type=int, default=200_000)
    ap.add_argument("--attack-min", type=int, default=4_000)
    ap.add_argument("--attack-max", type=int, default=6_200)
    ap.add_argument("--per-class-min", type=int, default=300)
    # Model hyperparams
    ap.add_argument("--rf-n-estimators", type=int, default=400)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=1)
    ap.add_argument("--hgb-max-depth", type=int, default=None)
    ap.add_argument("--hgb-learning-rate", type=float, default=0.1)
    ap.add_argument("--hgb-max-iter", type=int, default=200)
    ap.add_argument("--logreg-max-iter", type=int, default=500)
    ap.add_argument("--logreg-solver", type=str, default="lbfgs")
    args = ap.parse_args()

    cfg = Phase3Config(
        packet_parquet=args.packet.resolve(),
        packet_meta_json=args.packet_meta.resolve(),
        flow_parquet=args.flow.resolve(),
        flow_meta_json=args.flow_meta.resolve(),
        out_dir=args.out.resolve(),
        seed=args.seed,
        target_fpr=args.target_fpr,
        use_autoencoder=not args.no_ae,
        classifiers=tuple(args.classifiers),
    benign_target=args.benign_target,
    attack_min=args.attack_min,
    attack_max=args.attack_max,
    per_class_min=args.per_class_min,
    rf_n_estimators=args.rf_n_estimators,
    rf_min_samples_leaf=args.rf_min_samples_leaf,
    hgb_max_depth=args.hgb_max_depth,
    hgb_learning_rate=args.hgb_learning_rate,
    hgb_max_iter=args.hgb_max_iter,
    logreg_max_iter=args.logreg_max_iter,
    logreg_solver=args.logreg_solver,
    )

    res = run_phase3(cfg)
    print("Classifier:", res["classifier"])
    print("Test macro F1:", res["test_macro_f1"])
    print("Stage-1 flagged packets:", res["stage1"]["flagged_packets"]) 
    print("Flagged flows:", res["stage2"]["flagged_flows_total"]) 
    print("FP reduction (Stage2 vs Stage1):", res["stage2"]["false_positive_reduction"]) 


if __name__ == "__main__":
    main()
