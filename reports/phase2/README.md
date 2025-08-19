# Phase 2: Packet-level Anomaly Detection

This stage evaluates multiple unsupervised methods on packet features and selects the best model by validation F1 at a benign-calibrated threshold.

Methods tried:
- KMeans + PCA (distance to nearest centroid as anomaly score)
- Isolation Forest (negative decision_function as anomaly score)
- Local Outlier Factor, novelty mode (negative decision_function)
- Autoencoder (reconstruction MSE; optional if PyTorch is installed)

Evaluation:
- Threshold chosen from benign training scores to target ~2% FPR.
- Metrics on validation and test: precision, recall, F1, FPR, FNR, AUC.
- Per-attack recall reported when `attack_type` is available.

Artifacts written here:
- phase2_metrics.json: sorted results (best first) + feature list
- roc_val.png, roc_test.png: ROC curves for the best method
- pca_kmeans_scatter.png: 2D PCA scatter colored by ground-truth labels

Run:
```bash
python -m src.cli_phase2 --data data/packet_clean.parquet --out reports/phase2
```

Notes:
- Plots and metrics depend on data availability and class balance.
- Autoencoder training is capped for speed and uses early stopping on benign-only validation.
