# IoT IDS Model

Two-stage IoT Intrusion Detection on CIC IoT-DIAD 2024:
- Phase 1: Data preparation and Parquet consolidation
- Phase 2: Packet-level anomaly detection (KMeans+PCA, IsolationForest, LOF, Autoencoder)
- Phase 3: Flow-level supervised refinement (Random Forest, Logistic Regression)

## Setup

Prereqs: Python 3.10+ and pip. Optional: PyTorch for the autoencoder.

Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data prep (Phase 1)

Generate cleaned Parquet from the provided CSVs:

```bash
python -m src.cli_prepare --level packet --out data
python -m src.cli_prepare --level flow --out data
```

Tips:
- Use `--sample-frac 0.2` during development to speed up.

## Phase 2: Packet anomaly detection

Run all methods and select the best by validation F1 at the target FPR:

```bash
python -m src.cli_phase2 --data data/packet_clean.parquet --out reports/phase2 --methods kmeans iforest lof autoencoder
```

Common flags:
- `--methods autoencoder` to run AE only (requires torch)
- `--target-fpr 0.02` to change alert thresholding

Outputs (reports/phase2):
- `phase2_metrics.json`, `roc_val.png`, `roc_test.png`, `pca_kmeans_scatter.png`, `ae_loss.png`

## Phase 3: Flow-level supervised refinement

Use Stage-1 scores to flag packets, map to flows, then train flow classifier:

```bash
python -m src.cli_phase3 \
	--packet data/packet_clean.parquet --packet-meta data/packet_clean.parquet.meta.json \
	--flow data/flow_clean.parquet   --flow-meta   data/flow_clean.parquet.meta.json \
	--out reports/phase3 --classifiers rf logreg --target-fpr 0.02
```

Common flags:
- `--no-ae` to force IsolationForest at Stage-1
- `--classifiers rf gb logreg` to try a subset

Outputs (reports/phase3):
- `phase3_metrics.json` with classifier scores, confusion matrix, Stage-1 vs Stage-2 FP reduction

## Repo structure
- `datasets/` raw CSVs (packet and flow)
- `data/` cleaned Parquet outputs
- `src/` code (prep + phases 2 & 3 CLIs)
- `reports/` results and documentation

## Current results (snapshot)
- Phase 2 (Autoencoder best): Test F1 ≈ 0.847 at ~2% FPR (see `reports/phase2/phase2_metrics.json`)
- Phase 3 (RandomForest): Test macro F1 ≈ 0.790; Stage-2 reduced flagged benign flows to zero (see `reports/phase3/phase3_metrics.json`)
