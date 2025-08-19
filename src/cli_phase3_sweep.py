#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
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
    # Use smaller benign target for faster sweeps by default
    ap.add_argument("--benign-target", type=int, default=60_000)
    ap.add_argument("--attack-min", type=int, default=4_000)
    ap.add_argument("--attack-max", type=int, default=6_200)
    args = ap.parse_args()

    base_cfg = Phase3Config(
        packet_parquet=args.packet.resolve(),
        packet_meta_json=args.packet_meta.resolve(),
        flow_parquet=args.flow.resolve(),
        flow_meta_json=args.flow_meta.resolve(),
        out_dir=args.out.resolve(),
        seed=args.seed,
        target_fpr=0.02,
        use_autoencoder=True,
        classifiers=("rf", "hgb"),
        benign_target=args.benign_target,
        attack_min=args.attack_min,
        attack_max=args.attack_max,
        per_class_min=600,
        rf_n_estimators=600,
        rf_min_samples_leaf=2,
        hgb_max_depth=8,
        hgb_learning_rate=0.05,
        hgb_max_iter=350,
        logreg_max_iter=500,
        logreg_solver="lbfgs",
    )

    runs = [
        {"id": "A", "target_fpr": 0.02,  "per_class_min": 600, "classifiers": ("rf", "hgb"), "rf_n": 600, "leaf": 2,  "hgb_depth": 8,  "hgb_lr": 0.05, "hgb_iter": 350},
        {"id": "B", "target_fpr": 0.025, "per_class_min": 800, "classifiers": ("rf",),        "rf_n": 800, "leaf": 3,  "hgb_depth": 8,  "hgb_lr": 0.05, "hgb_iter": 350},
        {"id": "C", "target_fpr": 0.025, "per_class_min": 800, "classifiers": ("hgb",),      "rf_n": 600, "leaf": 2,  "hgb_depth": 10, "hgb_lr": 0.05, "hgb_iter": 450},
        {"id": "D", "target_fpr": 0.03,  "per_class_min": 800, "classifiers": ("rf", "hgb"), "rf_n": 800, "leaf": 3,  "hgb_depth": 8,  "hgb_lr": 0.05, "hgb_iter": 450},
    ]

    results = []
    for r in runs:
        cfg = Phase3Config(
            packet_parquet=base_cfg.packet_parquet,
            packet_meta_json=base_cfg.packet_meta_json,
            flow_parquet=base_cfg.flow_parquet,
            flow_meta_json=base_cfg.flow_meta_json,
            out_dir=base_cfg.out_dir,
            seed=base_cfg.seed,
            target_fpr=r["target_fpr"],
            use_autoencoder=True,
            classifiers=tuple(r["classifiers"]),
            benign_target=base_cfg.benign_target,
            attack_min=base_cfg.attack_min,
            attack_max=base_cfg.attack_max,
            per_class_min=r["per_class_min"],
            rf_n_estimators=r["rf_n"],
            rf_min_samples_leaf=r["leaf"],
            hgb_max_depth=r["hgb_depth"],
            hgb_learning_rate=r["hgb_lr"],
            hgb_max_iter=r["hgb_iter"],
            logreg_max_iter=base_cfg.logreg_max_iter,
            logreg_solver=base_cfg.logreg_solver,
        )
        t0 = time.time()
        out = run_phase3(cfg)
        dt = time.time() - t0
        results.append({
            "id": r["id"],
            "target_fpr": r["target_fpr"],
            "per_class_min": r["per_class_min"],
            "classifiers": list(r["classifiers"]),
            "rf_n": r["rf_n"],
            "leaf": r["leaf"],
            "hgb_depth": r["hgb_depth"],
            "hgb_lr": r["hgb_lr"],
            "hgb_iter": r["hgb_iter"],
            "chosen": out.get("classifier"),
            "macro_f1": out.get("test_macro_f1"),
            "stage2_fp_reduction": out.get("stage2", {}).get("false_positive_reduction"),
            "time_sec": round(dt, 2),
        })

    # Write summary
    out_dir = base_cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "sweep_results.json", "w", encoding="utf-8") as f:
        json.dump({"runs": results}, f, indent=2)

    # Print compact table
    print("ID\tmacroF1\tchosen\ttargetFPR\tperClassMin\tRFn\tHGBd/LR/Iter\tsec")
    for r in results:
        print(
            f"{r['id']}\t{r['macro_f1']:.4f}\t{r['chosen']}\t{r['target_fpr']}\t{r['per_class_min']}\t{r['rf_n']}\t{r['hgb_depth']}/{r['hgb_lr']}/{r['hgb_iter']}\t{r['time_sec']}"
        )


if __name__ == "__main__":
    main()
