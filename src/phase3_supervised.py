from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from .data_preparation import generate_sampled_dataset
from .phase2_anomaly import method_autoencoder, method_isolation_forest


@dataclass
class Phase3Config:
    packet_parquet: Path = Path("data/packet_clean.parquet")
    packet_meta_json: Path = Path("data/packet_clean.parquet.meta.json")
    flow_parquet: Path = Path("data/flow_clean.parquet")
    flow_meta_json: Path = Path("data/flow_clean.parquet.meta.json")
    out_dir: Path = Path("reports/phase3")
    seed: int = 42
    # Stage-1 threshold target FPR
    target_fpr: float = 0.02
    # AE fallback to IsolationForest when torch is missing
    use_autoencoder: bool = True
    # Flow sampling targets (approximate, matched to project brief)
    benign_target: int = 200_000
    attack_min: int = 4_000
    attack_max: int = 6_200
    # Classifier choices to try
    classifiers: Tuple[str, ...] = ("rf", "gb", "logreg")
    # Oversampling floor per class after sampling (to stabilize macro-F1)
    per_class_min: int = 300
    # Tunables for models
    rf_n_estimators: int = 400
    rf_min_samples_leaf: int = 1
    hgb_max_depth: int | None = None
    hgb_learning_rate: float = 0.1
    hgb_max_iter: int = 200
    logreg_max_iter: int = 500
    logreg_solver: str = "lbfgs"


ATTACK_CLASS_ORDER = [
    "Benign",
    "DoS-TCP_Flood",
    "DDoS-TCP_Flood",
    "DNS Spoofing",
    "Cross-Site Scripting (XSS)",
    "Brute Force",
]
CLASS_TO_ID = {c: i for i, c in enumerate(ATTACK_CLASS_ORDER)}


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _read_meta(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _unscale(df: pd.DataFrame, scale_params: Dict, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns and c in scale_params:
            mean = scale_params[c].get("mean", 0.0)
            std = scale_params[c].get("std", 1.0) or 1.0
            out[c + "_orig"] = out[c] * std + mean
    return out


def _build_flow_key(df: pd.DataFrame, src_col: str, dst_col: str, sport_col: str, dport_col: str) -> pd.Series:
    # Round ports after unscaling and coerce to int strings
    sport = pd.to_numeric(df[sport_col], errors="coerce").round().astype("Int64").astype(str)
    dport = pd.to_numeric(df[dport_col], errors="coerce").round().astype("Int64").astype(str)
    src = df[src_col].astype(str)
    dst = df[dst_col].astype(str)
    return src + "-" + dst + "-" + sport + "-" + dport


def _flag_packets_with_stage1(packet_df: pd.DataFrame, packet_meta: Dict, cfg: Phase3Config) -> pd.DataFrame:
    # Use Autoencoder if available else IsolationForest
    num_cols = [c for c in packet_meta.get("numeric_cols", []) if c != "label"]
    X = packet_df[num_cols].astype(np.float32).to_numpy()
    y = packet_df["label"].to_numpy().astype(int)

    # Train scores on benign only
    X_train = X[y == 0]
    scores_train = None
    scores_all = None
    used = "iforest"

    if cfg.use_autoencoder:
        try:
            scores = method_autoencoder(X_train, X, X, y, _ae_cfg_like_phase2(cfg))
            scores_train = scores["scores_train"]
            scores_all = scores["scores_val"]  # we passed X as val
            used = "autoencoder"
        except Exception:
            scores_if = method_isolation_forest(X_train, X, X, _ae_cfg_like_phase2(cfg))
            scores_train = scores_if["scores_train"]
            scores_all = scores_if["scores_val"]
            used = "iforest"
    else:
        scores_if = method_isolation_forest(X_train, X, X, _ae_cfg_like_phase2(cfg))
        scores_train = scores_if["scores_train"]
        scores_all = scores_if["scores_val"]
        used = "iforest"

    # Threshold by target FPR on benign train scores
    thr = _threshold_from_benign_like(scores_train, target_fpr=cfg.target_fpr)
    flagged = (scores_all >= thr).astype(int)

    out = packet_df.copy()
    out["stage1_score"] = scores_all.astype(np.float32)
    out["stage1_pred"] = flagged
    out["stage1_method"] = used
    out["stage1_threshold"] = thr
    return out


def _ae_cfg_like_phase2(cfg: Phase3Config):
    # Minimal adapter object providing attributes used by method_autoencoder/method_isolation_forest
    class C:
        ae_max_train = 150_000
        ae_max_val_benign = 30_000
        ae_hidden = (256, 128)
        ae_latent = 32
        ae_batch = 512
        ae_epochs = 12
        ae_lr = 1e-3
        ae_weight_decay = 1e-5
        ae_patience = 3
        # Needed by IsolationForest and thresholding
        target_fpr = float(cfg.target_fpr)
        seed = cfg.seed
        # For AE loss plotting
        out_dir = cfg.out_dir
    return C()


def _threshold_from_benign_like(scores_benign: np.ndarray, target_fpr: float) -> float:
    q = 1.0 - float(np.clip(target_fpr, 0.0, 0.5))
    q = np.clip(q, 0.5, 0.999)
    return float(np.quantile(scores_benign, q))


def _aggregate_flows(flow_df: pd.DataFrame, flow_meta: Dict) -> pd.DataFrame:
    # Unscale ports to reconstruct original integers for keying
    flow_df = _unscale(flow_df, flow_meta.get("scale_params", {}), ["src_port", "dst_port"])
    # Build key from unscaled ports
    key = _build_flow_key(flow_df, "src_ip", "dst_ip", "src_port_orig", "dst_port_orig")
    flow_df = flow_df.assign(flow_key=key)
    # Group by key; mean aggregate numeric features, and propagate label/attack_type via mode
    num_cols = [c for c in flow_meta.get("numeric_cols", []) if c not in ("label",)]
    agg = flow_df.groupby("flow_key").agg({**{c: "mean" for c in num_cols},
                                            **{"label": "max"}})
    # For attack_type, take most frequent within key
    atk = flow_df.groupby("flow_key")["attack_type"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else (s.dropna().iat[0] if len(s.dropna()) else "Benign"))
    agg = agg.join(atk, how="left")
    agg = agg.reset_index()
    return agg


def _stage1_flow_features(pkt_scored: pd.DataFrame) -> pd.DataFrame:
    # Derive flow-level features from Stage-1 packet scores
    grp = pkt_scored.groupby("flow_key")
    df = grp.agg(
        total_packets=("stage1_score", "size"),
        flagged_packets=("stage1_pred", "sum"),
        mean_stage1_score=("stage1_score", "mean"),
        max_stage1_score=("stage1_score", "max"),
    ).reset_index()
    df["flagged_ratio"] = (df["flagged_packets"].astype(float) / df["total_packets"].clip(lower=1)).astype(float)
    return df


def _prepare_flow_samples(flow_df: pd.DataFrame, cfg: Phase3Config) -> pd.DataFrame:
    # Use existing sampling utility to create imbalanced dataset for training
    df = generate_sampled_dataset(
        flow_df,
        benign_target=cfg.benign_target,
        attack_min=cfg.attack_min,
        attack_max=cfg.attack_max,
        seed=cfg.seed,
    )
    # Optional: ensure a minimum number per class by upsampling
    if cfg.per_class_min and cfg.per_class_min > 0 and "attack_type" in df.columns:
        parts: List[pd.DataFrame] = []
        for k, g in df.groupby("attack_type"):
            if len(g) >= cfg.per_class_min:
                parts.append(g)
            else:
                # upsample with replacement to reach the floor
                if len(g) > 0:
                    reps = cfg.per_class_min
                    idx = np.random.default_rng(cfg.seed).choice(g.index.to_numpy(), size=reps, replace=True)
                    parts.append(g.loc[idx])
                else:
                    # if class missing entirely, skip; it won't appear in this sample
                    pass
        if parts:
            df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)
    return df


def _train_classifiers(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, methods: Tuple[str, ...], seed: int, cfg: Phase3Config):
    results = []
    models = {}
    # Class weights for imbalance
    classes = np.unique(y_tr)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
    # Sample weights vector for learners without class_weight param
    sw_tr = np.array([class_weight[int(c)] for c in y_tr], dtype=np.float32)
    for m in methods:
        if m == "rf":
            clf = RandomForestClassifier(
                n_estimators=cfg.rf_n_estimators,
                max_depth=None,
                min_samples_leaf=cfg.rf_min_samples_leaf,
                n_jobs=-1,
                random_state=seed,
                class_weight=class_weight,
            )
        elif m == "gb":
            clf = GradientBoostingClassifier(random_state=seed)
        elif m == "hgb":
            clf = HistGradientBoostingClassifier(
                random_state=seed,
                max_depth=cfg.hgb_max_depth,
                learning_rate=cfg.hgb_learning_rate,
                max_iter=cfg.hgb_max_iter,
            )
        elif m == "logreg":
            clf = LogisticRegression(
                max_iter=cfg.logreg_max_iter,
                n_jobs=-1,
                random_state=seed,
                class_weight=class_weight,
                solver=cfg.logreg_solver,
                multi_class="auto",
            )
        else:
            continue
        if m == "hgb":
            clf.fit(X_tr, y_tr, sample_weight=sw_tr)
        else:
            clf.fit(X_tr, y_tr)
        yv = clf.predict(X_val)
        f1 = f1_score(y_val, yv, average="macro")
        results.append({"name": m, "val_macro_f1": float(f1)})
        models[m] = clf
    results = sorted(results, key=lambda r: r["val_macro_f1"], reverse=True)
    best_name = results[0]["name"]
    return best_name, models[best_name], results


def run_phase3(cfg: Phase3Config) -> Dict:
    packet_df = _read_parquet(cfg.packet_parquet)
    packet_meta = _read_meta(cfg.packet_meta_json)
    flow_df_raw = _read_parquet(cfg.flow_parquet)
    flow_meta = _read_meta(cfg.flow_meta_json)

    # 1) Stage-1: flag packets
    pkt_scored = _flag_packets_with_stage1(packet_df, packet_meta, cfg)
    pkt_scored = _unscale(pkt_scored, packet_meta.get("scale_params", {}), ["src_port", "dst_port"])  # reconstruct orig ports
    pkt_scored = pkt_scored.assign(
        flow_key=_build_flow_key(pkt_scored, "src_ip", "dst_ip", "src_port_orig", "dst_port_orig")
    )
    flagged_pkt = pkt_scored[pkt_scored["stage1_pred"] == 1].copy()

    # 2) Aggregate flows (segment consolidation)
    flows_agg = _aggregate_flows(flow_df_raw, flow_meta)
    # 2b) Derive Stage-1 flow features and join
    s1f = _stage1_flow_features(pkt_scored)
    flows_agg = flows_agg.merge(s1f, on="flow_key", how="left")
    # Fill missing Stage-1 features with zeros for flows not seen at packet level
    for c in ["total_packets", "flagged_packets", "mean_stage1_score", "max_stage1_score", "flagged_ratio"]:
        if c not in flows_agg.columns:
            flows_agg[c] = 0.0
        flows_agg[c] = flows_agg[c].fillna(0.0)

    # 3) Prepare flow training data via sampling
    flows_sampled = _prepare_flow_samples(flows_agg, cfg)

    # Encode multiclass target
    flows_sampled["class_id"] = flows_sampled["attack_type"].map(CLASS_TO_ID).fillna(0).astype(int)

    # Feature matrix
    feat_cols = [c for c in flow_meta.get("numeric_cols", []) if c not in ("label",)]
    # Add Stage-1 derived features
    feat_cols += ["total_packets", "flagged_packets", "mean_stage1_score", "max_stage1_score", "flagged_ratio"]
    X = flows_sampled[feat_cols].astype(np.float32).to_numpy()
    y = flows_sampled["class_id"].to_numpy().astype(int)

    # Split train/val/test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=cfg.seed, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=cfg.seed, stratify=y_tmp)

    # 4) Train classifiers and select best
    best_name, best_model, cls_results = _train_classifiers(X_tr, y_tr, X_val, y_val, cfg.classifiers, cfg.seed, cfg)

    # 5) Evaluate on test (overall)
    y_hat = best_model.predict(X_te)
    labels_full = list(range(len(ATTACK_CLASS_ORDER)))
    report = classification_report(y_te, y_hat, labels=labels_full, output_dict=True, zero_division=0, target_names=ATTACK_CLASS_ORDER)
    cm = confusion_matrix(y_te, y_hat, labels=labels_full).tolist()
    macro_f1 = f1_score(y_te, y_hat, average="macro")

    # 6) Evaluate only on flows referenced by flagged packets (false-positive reduction)
    flagged_keys = pd.Series(flagged_pkt["flow_key"].unique())
    flows_flagged = flows_agg[flows_agg["flow_key"].isin(flagged_keys)].copy()
    flows_flagged["class_id"] = flows_flagged["attack_type"].map(CLASS_TO_ID).fillna(0).astype(int)
    Xf = flows_flagged[feat_cols].astype(np.float32).to_numpy()
    yf = flows_flagged["class_id"].to_numpy().astype(int)
    yfh = best_model.predict(Xf) if len(Xf) else np.array([], dtype=int)

    # Stage-1 false positives (benign flows among flagged)
    stage1_fp = int((flows_flagged["class_id"] == 0).sum())
    # Stage-2 false positives (benign predicted as attack among flagged)
    stage2_fp = int(((flows_flagged["class_id"] == 0) & (pd.Series(yfh, index=flows_flagged.index) != 0)).sum()) if len(yfh) else 0
    fp_reduction = float(1.0 - (stage2_fp / stage1_fp)) if stage1_fp > 0 else 1.0

    out_dir = cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "classifier": best_name,
        "classifier_val_results": cls_results,
        "test_macro_f1": float(macro_f1),
        "test_report": report,
        "test_confusion_matrix": cm,
        "stage1": {
            "method": str(pkt_scored.get("stage1_method").iloc[0]) if len(pkt_scored) else "",
            "threshold": float(pkt_scored.get("stage1_threshold").iloc[0]) if len(pkt_scored) else 0.0,
            "flagged_packets": int(len(flagged_pkt)),
            "unique_flagged_flow_keys": int(len(flagged_keys)),
        },
        "stage2": {
            "flagged_flows_total": int(len(flows_flagged)),
            "stage1_false_positives": stage1_fp,
            "stage2_false_positives": stage2_fp,
            "false_positive_reduction": fp_reduction,
        },
        "features_used": feat_cols,
    }
    with open(out_dir / "phase3_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    return out


__all__ = [
    "Phase3Config",
    "run_phase3",
]
