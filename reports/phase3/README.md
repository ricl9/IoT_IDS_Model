# Phase 3: Flow-level Supervised Refinement

This stage refines Stage-1 packet flags using a supervised classifier on aggregated flows. Packets are mapped to flows via `[src_ip]-[dst_ip]-[src_port]-[dst_port]` using unscaled ports. Flows split into 2-minute segments are aggregated by key (mean of numeric features; majority `attack_type`).

Pipeline:
- Stage 1: Packet anomaly scoring (Autoencoder by default; falls back to IsolationForest if torch is unavailable). Threshold calibrated on benign to target FPR (default 2%).
- Stage 2: Aggregate flow features and train a multiclass classifier (Random Forest, Logistic Regression). Best model chosen by validation macro-F1.

How to run
```bash
python -m src.cli_phase3 \
	--packet data/packet_clean.parquet --packet-meta data/packet_clean.parquet.meta.json \
	--flow data/flow_clean.parquet   --flow-meta   data/flow_clean.parquet.meta.json \
	--out reports/phase3 --classifiers rf logreg --target-fpr 0.02
```

Flags:
- `--no-ae`: use IsolationForest for Stage 1
- `--target-fpr`: adjust Stage-1 sensitivity (0.01–0.05 reasonable)
- `--classifiers`: subset to try, e.g. `rf gb logreg`

Current results (snapshot)
- Best classifier: RandomForest
- Test macro F1: ~0.790
- Stage-1 flagged packets: 233,681  | Unique flagged flow keys: 219,459
- Benign false positives among flagged flows: Stage-1 = 343 → Stage-2 = 0 (100% reduction in this run)

Artifacts
- `phase3_metrics.json` with:
	- `classifier_val_results` (model selection)
	- `test_report` (per-class precision/recall/F1)
	- `test_confusion_matrix`
	- `stage1` and `stage2` summaries

Notes
- If some classes are absent in a split, metrics still render via explicit labels. Consider k-fold CV for more stable estimates.
- Tune `--target-fpr` to trade detection vs false alarms feeding Stage-2.
- Try `gb` (Gradient Boosting) for an additional baseline.
