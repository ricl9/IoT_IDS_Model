from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import math

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = None


RANDOM_SEED = 42


@dataclass
class Phase2Config:
    data_parquet: Path = Path("data/packet_clean.parquet")
    meta_json: Path = Path("data/packet_clean.parquet.meta.json")
    out_dir: Path = Path("reports/phase2")
    seed: int = RANDOM_SEED
    # Visualization sampling to keep plots snappy
    viz_sample: int = 50000
    # Which methods to run: any of {"kmeans","iforest","lof","autoencoder"}
    methods: Tuple[str, ...] = ("kmeans", "iforest", "lof", "autoencoder")
    # PCA dimensionality for clustering
    pca_components: int = 20
    # Candidate K for KMeans
    kmeans_k_list: Tuple[int, ...] = (5, 10, 15)
    # Target FPR when calibrating threshold on benign train scores
    target_fpr: float = 0.02
    # Fraction of benign for train/val/test
    train_frac: float = 0.7
    val_frac: float = 0.15  # remainder goes to test
    # Autoencoder settings
    ae_latent: int = 32
    ae_hidden: Tuple[int, ...] = (256, 128)
    ae_epochs: int = 12
    ae_batch: int = 512
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 3
    ae_max_train: int = 150_000  # cap for speed
    ae_max_val_benign: int = 30_000


def _load_packet_data(cfg: Phase2Config) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_parquet(cfg.data_parquet)
    # Numeric feature columns
    feat_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # label is numeric; keep it but not as feature
    feat_cols = [c for c in feat_cols if c != "label"]
    return df, feat_cols


def _split_benign_attack(df: pd.DataFrame, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    benign = df[df["label"] == 0].copy()
    attack = df[df["label"] == 1].copy()
    benign = benign.sample(frac=1.0, random_state=seed)
    attack = attack.sample(frac=1.0, random_state=seed)
    return benign, attack


def _make_splits(cfg: Phase2Config, benign: pd.DataFrame, attack: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    n_b = len(benign)
    n_train = int(n_b * cfg.train_frac)
    n_val = int(n_b * cfg.val_frac)
    train_b = benign.iloc[:n_train]
    val_b = benign.iloc[n_train : n_train + n_val]
    test_b = benign.iloc[n_train + n_val :]

    # For evaluation, combine benign with all attacks split in half for val/test
    n_a = len(attack)
    a_mid = n_a // 2
    val_a = attack.iloc[:a_mid]
    test_a = attack.iloc[a_mid:]

    val = pd.concat([val_b, val_a], ignore_index=True).sample(frac=1.0, random_state=cfg.seed)
    test = pd.concat([test_b, test_a], ignore_index=True).sample(frac=1.0, random_state=cfg.seed)

    return {
        "train": train_b,
        "val": val,
        "test": test,
    }


def _compute_metrics(y_true: np.ndarray, scores: np.ndarray, y_pred: np.ndarray, attack_labels: List[str]) -> Dict:
    # Positive class is attack (1). Higher score should indicate more anomalous.
    try:
        auc = roc_auc_score(y_true, scores)
    except Exception:
        auc = float("nan")
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report = classification_report(y_true, y_pred, labels=[0, 1], target_names=["Benign", "Attack"], zero_division=0, output_dict=True)

    # Per-attack recall (detection rate)
    per_attack = {}
    if attack_labels:
        df_eval = pd.DataFrame({"y": y_true, "pred": y_pred, "attack_type": attack_labels})
        mask_attack = df_eval["y"] == 1
        if mask_attack.any():
            grouped = df_eval[mask_attack].groupby("attack_type")
            for k, g in grouped:
                tp = ((g["y"] == 1) & (g["pred"] == 1)).sum()
                fn = ((g["y"] == 1) & (g["pred"] == 0)).sum()
                per_attack[k] = {
                    "n": int(len(g)),
                    "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                }

    # FPR/FNR
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = float(fp / (fp + tn)) if (fp + tn) else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) else 0.0

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": fpr,
        "fnr": fnr,
        "cm": cm.tolist(),
        "report": report,
        "per_attack_recall": per_attack,
    }


def _threshold_from_benign(scores_benign: np.ndarray, target_fpr: float = 0.02) -> float:
    # choose threshold so that at most ~target_fpr of benign would be flagged as anomalies
    # scores are anomaly scores: higher means more anomalous
    q = 1.0 - target_fpr
    q = np.clip(q, 0.5, 0.999)
    return float(np.quantile(scores_benign, q))


def method_kmeans_pca(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, cfg: Phase2Config) -> Tuple[Dict, Dict[str, np.ndarray]]:
    # PCA reduce for stability and speed
    pca = PCA(n_components=min(cfg.pca_components, X_train.shape[1]))
    Z_train = pca.fit_transform(X_train)
    Z_val = pca.transform(X_val)
    Z_test = pca.transform(X_test)

    best = None
    best_scores = {}
    for k in cfg.kmeans_k_list:
        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed)
        km.fit(Z_train)

        # anomaly score: distance to nearest centroid
        def scores(Z):
            dists = np.min(((Z[:, None, :] - km.cluster_centers_) ** 2).sum(axis=2), axis=1) ** 0.5
            return dists

        s_train = scores(Z_train)
        s_val = scores(Z_val)
        s_test = scores(Z_test)
        thr = _threshold_from_benign(s_train, target_fpr=cfg.target_fpr)

        res = {
            "k": k,
            "thr": thr,
            "pca_var_ratio": pca.explained_variance_ratio_.sum().item() if hasattr(pca, "explained_variance_ratio_") else None,
            "scores_train": s_train,
            "scores_val": s_val,
            "scores_test": s_test,
        }

        # Keep the last result; selection is done by validation F1 later.
        best = res

    return best, {"Z_train": Z_train, "Z_val": Z_val, "Z_test": Z_test, "pca": pca}


def method_isolation_forest(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, cfg: Phase2Config) -> Dict:
    iforest = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=cfg.target_fpr,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    iforest.fit(X_train)
    # decision_function: positive for inliers; more negative=more anomalous
    def anomaly_score(X):
        return -iforest.decision_function(X)

    return {
        "scores_train": anomaly_score(X_train),
        "scores_val": anomaly_score(X_val),
        "scores_test": anomaly_score(X_test),
        "model": iforest,
    }


def method_lof(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, cfg: Phase2Config) -> Dict:
    # Use novelty=True to allow predict on unseen data
    lof = LocalOutlierFactor(n_neighbors=35, novelty=True)
    lof.fit(X_train)
    def anomaly_score(X):
        # Higher negative_outlier_factor_ means more normal; decision_function larger => inlier
        return -lof.decision_function(X)

    return {
        "scores_train": anomaly_score(X_train),
        "scores_val": anomaly_score(X_val),
        "scores_test": anomaly_score(X_test),
        "model": lof,
    }


def _evaluate_method(name: str, scores: Dict, y_val: np.ndarray, y_test: np.ndarray, attack_type_val: List[str], attack_type_test: List[str], target_fpr: float) -> Dict:
    thr = _threshold_from_benign(scores["scores_train"], target_fpr=target_fpr)
    yhat_val = (scores["scores_val"] >= thr).astype(int)
    yhat_test = (scores["scores_test"] >= thr).astype(int)

    metrics_val = _compute_metrics(y_val, scores["scores_val"], yhat_val, attack_type_val)
    metrics_test = _compute_metrics(y_test, scores["scores_test"], yhat_test, attack_type_test)
    return {
        "name": name,
        "threshold": float(thr),
        "val": metrics_val,
        "test": metrics_test,
    }


def _plot_kmeans_pca(cfg: Phase2Config, Z: np.ndarray, labels: np.ndarray, km: KMeans, out_path: Path, sample: int = 30000):
    # 2D visualization with PCA already applied: if Z has >2 dims, reduce to 2 via PCA
    if Z.shape[1] > 2:
        pca2 = PCA(n_components=2, random_state=cfg.seed)
        Z2 = pca2.fit_transform(Z)
    else:
        Z2 = Z[:, :2]

    n = len(Z2)
    if n > sample:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(n, size=sample, replace=False)
        Z2 = Z2[idx]
        labels = labels[idx]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=Z2[:, 0], y=Z2[:, 1], hue=labels, palette={0: "#2b8a3e", 1: "#e03131"}, s=5, alpha=0.5, legend=False)
    plt.title("KMeans clusters in PCA space (colored by ground truth)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def run_phase2(cfg: Phase2Config) -> Dict:
    df, feat_cols = _load_packet_data(cfg)
    benign, attack = _split_benign_attack(df, cfg.seed)
    splits = _make_splits(cfg, benign, attack)

    # Build matrices
    def XY(frame: pd.DataFrame):
        X = frame[feat_cols].astype(np.float32).to_numpy()
        y = frame["label"].to_numpy().astype(int)
        attack_type = frame.get("attack_type", pd.Series(["" for _ in range(len(frame))])).tolist()
        return X, y, attack_type

    X_tr, y_tr, _ = XY(splits["train"])  # y_tr should be all zeros
    X_val, y_val, attack_val = XY(splits["val"]) 
    X_te, y_te, attack_te = XY(splits["test"]) 

    results: List[Dict] = []

    # Method 1: KMeans + PCA
    km_best, km_artifacts = method_kmeans_pca(X_tr, X_val, X_te, cfg)
    # Evaluate with last fit's scores (already computed)
    if "kmeans" in cfg.methods:
        res_km = _evaluate_method("KMeans+PCA(k={})".format(km_best["k"]), km_best, y_val, y_te, attack_val, attack_te, cfg.target_fpr)
        results.append(res_km)

    # Method 2: Isolation Forest
    res_if_scores = method_isolation_forest(X_tr, X_val, X_te, cfg)
    if "iforest" in cfg.methods:
        res_if = _evaluate_method("IsolationForest", res_if_scores, y_val, y_te, attack_val, attack_te, cfg.target_fpr)
        results.append(res_if)

    # Method 3: Local Outlier Factor (novelty)
    res_lof_scores = method_lof(X_tr, X_val, X_te, cfg)
    if "lof" in cfg.methods:
        res_lof = _evaluate_method("LOF(novelty)", res_lof_scores, y_val, y_te, attack_val, attack_te, cfg.target_fpr)
        results.append(res_lof)

    # Method 4: Autoencoder (if torch available)
    res_ae_scores = None
    if "autoencoder" in cfg.methods and torch is not None:
        try:
            res_ae_scores = method_autoencoder(X_tr, X_val, X_te, y_val, cfg)
            res_ae = _evaluate_method("Autoencoder", res_ae_scores, y_val, y_te, attack_val, attack_te, cfg.target_fpr)
            results.append(res_ae)
        except Exception:
            pass

    # Select best by validation F1
    results_sorted = sorted(results, key=lambda r: r["val"]["f1"], reverse=True)
    best = results_sorted[0]

    # Save metrics JSON (best-effort; tolerate low-disk)
    out_dir = cfg.out_dir
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "phase2_metrics.json", "w", encoding="utf-8") as f:
            json.dump({"results": results_sorted, "features": feat_cols}, f, indent=2)
    except OSError:
        pass

    # Plots: ROC curve for best method
    # Recompute scores for ROC using best method mapping
    name = best["name"]
    if name.startswith("KMeans"):
        s_val = km_best["scores_val"]; s_te = km_best["scores_test"]
    elif name.startswith("IsolationForest"):
        s_val = res_if_scores["scores_val"]; s_te = res_if_scores["scores_test"]
    elif name.startswith("Autoencoder") and res_ae_scores is not None:
        s_val = res_ae_scores["scores_val"]; s_te = res_ae_scores["scores_test"]
    else:
        s_val = res_lof_scores["scores_val"]; s_te = res_lof_scores["scores_test"]

    for split_name, y_split, s_split in [("val", y_val, s_val), ("test", y_te, s_te)]:
        try:
            fpr, tpr, _ = roc_curve(y_split, s_split)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, label=f"ROC ({name})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"ROC Curve ({split_name})")
            plt.legend()
            plt.tight_layout()
            try:
                plt.savefig(out_dir / f"roc_{split_name}.png", dpi=200)
            except OSError:
                pass
            plt.close()
        except Exception:
            pass

    # PCA scatter (KMeans)
    try:
        Z_all = np.vstack([km_artifacts["Z_val"], km_artifacts["Z_test"]])
        labels_all = np.concatenate([y_val, y_te])
        try:
            _plot_kmeans_pca(cfg, Z_all, labels_all, km=None, out_path=out_dir / "pca_kmeans_scatter.png")
        except OSError:
            pass
    except Exception:
        pass

    return {"best": best, "all": results_sorted}


def method_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_val: np.ndarray,
    cfg: Phase2Config,
) -> Dict:
    if torch is None:
        raise RuntimeError("PyTorch not installed")

    # Reproducibility
    try:
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cap sizes for speed
    n_train = min(len(X_train), cfg.ae_max_train)
    Xtr = X_train[:n_train]
    # For early stopping, validate on benign subset of val
    mask_val_b = (y_val == 0)
    Xvb = X_val[mask_val_b]
    if len(Xvb) > cfg.ae_max_val_benign:
        Xvb = Xvb[: cfg.ae_max_val_benign]

    in_dim = Xtr.shape[1]

    class AE(nn.Module):
        def __init__(self, d: int, hidden: Tuple[int, ...], latent: int):
            super().__init__()
            enc_layers: List[nn.Module] = []
            last = d
            for h in hidden:
                enc_layers += [nn.Linear(last, h), nn.ReLU()]
                last = h
            self.encoder = nn.Sequential(*enc_layers, nn.Linear(last, latent))
            dec_layers: List[nn.Module] = []
            last = latent
            for h in reversed(hidden):
                dec_layers += [nn.Linear(last, h), nn.ReLU()]
                last = h
            self.decoder = nn.Sequential(*dec_layers, nn.Linear(last, d))
        def forward(self, x):
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return x_hat, z

    model = AE(in_dim, cfg.ae_hidden, cfg.ae_latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.ae_lr, weight_decay=cfg.ae_weight_decay)
    crit = nn.MSELoss()

    ds_tr = TensorDataset(torch.from_numpy(Xtr))
    dl_tr = DataLoader(ds_tr, batch_size=cfg.ae_batch, shuffle=True)
    # Val benign loader
    ds_vb = TensorDataset(torch.from_numpy(Xvb)) if len(Xvb) > 0 else None
    dl_vb = DataLoader(ds_vb, batch_size=cfg.ae_batch) if ds_vb is not None else None

    best_state = None
    best_loss = math.inf
    patience = cfg.ae_patience
    epochs_no_improve = 0
    train_curve: List[float] = []
    val_curve: List[float] = []

    for epoch in range(cfg.ae_epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in dl_tr:
            xb = xb.to(device)
            opt.zero_grad()
            xh, _ = model(xb)
            loss = crit(xh, xb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= max(1, len(ds_tr))
        train_curve.append(float(epoch_loss))

        # Validation on benign-only portion
        val_loss = epoch_loss
        if dl_vb is not None:
            model.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for (xb,) in dl_vb:
                    xb = xb.to(device)
                    xh, _ = model(xb)
                    l = crit(xh, xb).item()
                    tot += l * len(xb)
                    n += len(xb)
            val_loss = tot / max(1, n)
        val_curve.append(float(val_loss))

        # Early stopping
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Plot training curve if possible
    try:
        plt.figure(figsize=(6,4))
        plt.plot(train_curve, label="train")
        if val_curve:
            plt.plot(val_curve, label="val")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Autoencoder training curve")
        plt.legend()
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(cfg.out_dir / "ae_loss.png", dpi=160)
        plt.close()
    except Exception:
        pass

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute anomaly scores as per-sample MSE
    def recon_err(X: np.ndarray) -> np.ndarray:
        model.eval()
        errs = np.zeros(len(X), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), cfg.ae_batch):
                xb = torch.from_numpy(X[i:i+cfg.ae_batch]).to(device)
                xh, _ = model(xb)
                se = (xh - xb).pow(2).mean(dim=1).detach().cpu().numpy()
                errs[i:i+len(se)] = se
        return errs

    s_train = recon_err(Xtr)
    s_val = recon_err(X_val)
    s_test = recon_err(X_test)

    return {
        "scores_train": s_train,
        "scores_val": s_val,
        "scores_test": s_test,
    "model": "autoencoder",
    "train_curve": train_curve,
    "val_curve": val_curve,
    }


__all__ = [
    "Phase2Config",
    "run_phase2",
]
