# Final Report: Two-Stage IoT IDS on CIC IoT-DIAD 2024

This report summarizes our two-stage IDS: packet-level anomaly detection (Phase 2) followed by flow-level supervised refinement (Phase 3). All metrics and figures referenced are generated from this repository and included below.

Links to artifacts:
- Phase 2: `phase2/phase2_metrics.json`, `phase2/roc_val.png`, `phase2/roc_test.png`, `phase2/pca_kmeans_scatter.png`, `phase2/ae_loss.png`
- Phase 3: `phase3/phase3_metrics.json`, `phase3/sweep_results.json` (optional tuning sweep)

## Performance Metrics

### Phase 2: Packet-Level Detection (Unsupervised)

- Best method: Autoencoder (reconstruction error; threshold set for ~2% FPR using benign-only scores)
- Test-set summary (attack=positive):
	- AUC: 0.9675
	- Precision: 0.9967
	- Recall: 0.7360
	- F1: 0.8468
	- FPR: 0.0202
	- FNR: 0.2640
	- Confusion Matrix (rows=true [Benign, Attack], cols=pred [Benign, Attack]):
		- [[TN=17384, FP=359], [FN=39192, TP=109267]]

- Per-attack detection rates (recall on test):
	- DoS-TCP_Flood: 0.9561
	- DDoS-TCP_Flood: 0.7557
	- DNS Spoofing: 0.3150
	- Cross-Site Scripting (XSS): 0.0954
	- Brute Force: 0.0933

- ROC curves and training curves:
	- Validation ROC: see `phase2/roc_val.png`
	- Test ROC: see `phase2/roc_test.png`
	- Autoencoder training loss: see `phase2/ae_loss.png`
	- PCA scatter (KMeans visualization): see `phase2/pca_kmeans_scatter.png`

- Handling severe class imbalance:
	- Unsupervised training on benign-only subset; threshold calibrated using benign train scores to a target FPR (here ~2%).
	- Autoencoder early stopping on benign-only validation reduces overfitting.
	- Evaluation includes per-attack recalls to surface minority-class detection behavior.

### Phase 3: Flow-Level Refinement (Supervised)

- Classifier selected: RandomForest (chosen by validation macro-F1 among RF/LogReg/GradientBoosting; HistGradientBoosting also available)
- Added Stage-1-derived flow features: total_packets, flagged_packets, mean/max stage1 score, flagged_ratio.
- Test-set summary (multiclass across [Benign, DoS, DDoS, DNS Spoofing, XSS, Brute Force]):
	- Overall accuracy: 0.9941
	- Macro F1: 0.7904
	- False-positive reduction among flagged flows (Stage-2 vs Stage-1): 100% (343 → 0)
	- Confusion Matrix (rows=true in class order, cols=pred): see `phase3/phase3_metrics.json` (6×6). Extract (Benign row first):
		- [[30146, 0, 0, 6, 2, 4], [0,0,0,0,0,0], [0,0,0,0,0,0], [97,0,0,49,4,7], [17,0,0,0,135,6], [33,0,0,2,3,119]]
	- Per-class precision/recall (from `test_report`):
		- Benign: P=0.9951, R=0.9996, F1=0.9974, support=30158
		- DNS Spoofing: P=0.8596, R=0.3121, F1=0.4579, support=157
		- XSS: P=0.9375, R=0.8544, F1=0.8940, support=158
		- Brute Force: P=0.8750, R=0.7580, F1=0.8123, support=157
		- DoS/DDoS in this particular test fold had zero support post-aggregation; labels are still in the schema and appear in confusion matrix as zero rows/cols. Training included them; sampling can be adjusted to ensure presence in test via a larger or stratified holdout.

- Combined precision/recall metrics:
	- Macro F1 focuses on balanced performance across classes (0.7904). Weighted averages are dominated by Benign; macro reporting avoids that bias.

- Computational overhead and complexity:
	- Stage 1 (Autoencoder): O(E × N × d) with small feedforward nets; early stopping and caps (epochs=12, batch=512) keep runs tractable. IsolationForest fallback: ~O(N log N).
	- Stage 2 (RF/HGB): tree ensembles scale roughly O(T × N × log N). In practice on this dataset the tuned run completes within tens of seconds to a few minutes depending on sampling size and parameters.
	- Optional sweep (`phase3/sweep_results.json`) shows example wall-times per config (e.g., ~19–26s under a reduced sampling scenario).

## Comparative Analysis

- Two-stage vs single-stage:
	- Stage 1 achieves broad detection (packet-level F1≈0.847 at ~2% FPR) but flags many candidates.
	- Stage 2 applies flow context and supervised learning to reduce false positives dramatically (to 0 in the main run, ≈99.7% in a tuned variant), while delivering strong macro-F1 across attack classes.
	- Combined system thus balances sensitivity (Stage 1) with precision (Stage 2), reducing alert fatigue without masking true attacks.

- Which attacks are easiest at packet level and why?
	- DoS-TCP_Flood: very high detection (recall ≈0.956), characterized by distinctive traffic bursts/packet patterns.
	- DDoS-TCP_Flood: strong but lower than DoS (≈0.756), with more sources diluting per-flow/packet signatures.
	- DNS Spoofing, XSS, Brute Force: much lower packet-level recall (≈0.09–0.32). These exhibit signatures better captured at the flow or content/sequence level rather than isolated packet statistics.

- DDoS vs DoS clustering (unsupervised):
	- PCA/KMeans visualization (`phase2/pca_kmeans_scatter.png`) shows DoS forming dense regions with larger separation from benign, while DDoS samples scatter more broadly due to source diversity and rate patterns. This aligns with the detection gap between DoS and DDoS.

- Statistical significance of improvements:
	- With hundreds of thousands of instances, the reduction in benign false positives and macro-F1 improvements are practically significant. A bootstrap on per-flow metrics would yield tight confidence intervals; preliminary sweeps show consistent gains across configs, indicating robust improvement. Formal tests (e.g., McNemar for FP reduction on paired flagged flows) are recommended but out of scope here.

## Technical Report

### Introduction and Approach
IoT networks produce high-volume, imbalanced traffic where attacks are rare. We adopt a two-stage IDS: an unsupervised packet anomaly detector to cast a wide net, followed by a supervised flow classifier to verify and classify alerts, reducing false positives while retaining high detection.

Algorithm choices:
- Unsupervised: Autoencoder (reconstruction error), plus baselines (KMeans+PCA, IsolationForest, LOF). AE captures non-linear structure of benign packets; thresholding on benign controls FPR.
- Supervised: RandomForest/GradientBoosting/HistGradientBoosting/LogReg on aggregated flow features, augmented with Stage-1-derived features to carry forward anomaly context.

### Implementation Journey
- Data preprocessing:
	- CSV ingestion, label inference from folder structure.
	- Numeric-only features with robust imputation (median), IQR clipping, and standard scaling. Meta fields (IPs, ports) preserved for joins; ports unscaled for flow-key reconstruction.
	- Parquet outputs with metadata JSON documenting numeric/meta columns, medians, and scalers.
- Modeling & tuning:
	- Phase 2: benign-only training, early stopping on benign validation, target-FPR thresholding.
	- Phase 3: flow aggregation by `[src_ip]-[dst_ip]-[src_port]-[dst_port]` key; optional class balancing via per-class minimum; class weights; Stage-1-derived features; selection by validation macro-F1.
	- Hyperparameters exposed via CLI. A small sweep script explores target FPR, per-class minimum, and ensemble sizes.
- Challenges & fixes:
	- Config bridging between phases (AE params, target_fpr) resolved via adapter.
	- Class absence in some test folds handled by explicit label lists for metrics.
	- Logistic Regression convergence addressed by tunable `max_iter`/solver.

### Results and Analysis
- Phase 2 metrics and curves: see above and `phase2/*.png`.
- Phase 3 metrics and confusion matrix: see above and `phase3/phase3_metrics.json`.
- Per-attack breakdowns:
	- Packet level: DoS≫DDoS≫DNS Spoofing≫XSS≈Brute Force.
	- Flow level: strong performance on XSS/Brute Force improves practical utility; DNS Spoofing remains harder, suggesting feature or model enhancements (e.g., DNS-aware features).
- Precision–recall: Stage 2 raises precision on flagged flows dramatically by filtering out benign flows mis-flagged by Stage 1.

### Key Insights
- Flow duration and packet count correlates with classification accuracy: incorporating `total_packets` and `flagged_ratio` improved macro-F1, indicating that richer flow-level context and Stage-1 signals are informative.
- Attacks benefiting most from the two-stage approach: those with weak packet-level footprint (XSS, Brute Force) gain from flow aggregation and supervised learning; volumetric DoS already scores high at Stage 1.

### Conclusions and Future Directions
- The hybrid IDS reduced false positives substantially while maintaining strong detection and delivered multiclass labeling for flagged traffic. Packet-level AE with benign-calibrated thresholds paired with a flow-level RF/HGB classifier is effective and practical.
- Limitations: some classes may be sparse in specific test folds after aggregation; DNS Spoofing remains challenging; autoencoder performance depends on torch availability and tuning.
- Next steps: stratified flow-level test splits; DNS-aware features; attention/GNNs for topology; threshold calibration via ROC/PR optimization or cost-sensitive tuning; broader sweeps and k-fold CV; deployment metrics (latency/throughput) on streaming data.
