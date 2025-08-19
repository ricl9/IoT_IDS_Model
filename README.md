# IoT IDS Model

Multi-stage IoT Intrusion Detection on CIC IoT-DIAD 2024 using a two-phase pipeline:
- Phase 1: Data preparation and Parquet consolidation
- Phase 2: Packet-level anomaly detection (KMeans+PCA, IsolationForest, LOF, Autoencoder)
- Phase 3: Flow-level supervised refinement (planned)

## Quick start

Prereqs: Python 3.10+ and pip.

1) Prepare Parquet datasets from CSVs

```bash
python -m src.cli_prepare --level packet --out data
python -m src.cli_prepare --level flow --out data
```

2) Run Phase 2 anomaly detection on packet-level data

```bash
python -m src.cli_phase2 --data data/packet_clean.parquet --out reports/phase2
```

Artifacts will appear under `reports/phase2/` (metrics JSON, ROC plots, PCA scatter).

Notes:
- Autoencoder is optional and requires PyTorch; if not installed, it is skipped gracefully.
- Use `--sample-frac` in `cli_prepare` to speed up development runs.

## Repo structure
- `datasets/` raw CSVs (packet and flow)
- `data/` cleaned Parquet outputs
- `src/` code (prep + phase 2)
- `reports/` results and documentation

## Next steps
- Implement Phase 3 flow-level supervised classifier and end-to-end evaluation.
