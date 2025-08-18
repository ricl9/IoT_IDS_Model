"""
Data preparation utilities for CIC IoT-DIAD 2024 packet- and flow-level CSVs.

Features:
- Discover and load CSVs under datasets/packet and datasets/flow.
- Harmonize columns, infer labels from folder names, and map attack types.
- Clean: drop constant/duplicate cols, handle missing values, clip outliers (IQR),
  and optionally scale numeric features.
- Write cleaned datasets to Parquet with a compact schema and metadata JSON.
- Provide sampling utilities to create imbalanced train/val/test splits with
  target benign/attack proportions.

Assumptions:
- Directory layout follows the provided repo structure.
- Label can be inferred from path (Benign vs Attacks/<Type>). If label columns exist
  in CSV, they are ignored in favor of folder-derived labels to ensure consistency.

"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ATTACK_FOLDERS = {
    "DoS-TCP_Flood": "DoS-TCP_Flood",
    "DDoS-TCP_Flood": "DDoS-TCP_Flood",
    "DNS_Spoofing": "DNS Spoofing",
    "DNS Spoofing": "DNS Spoofing",
    "XSS": "Cross-Site Scripting (XSS)",
    "Brute_Force": "Brute Force",
    "Brute-Force": "Brute Force",
}


DEFAULT_META_KEEP = {
    # Common network key fields if present; kept unscaled for joins/aggregation
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "sport",
    "dport",
    "source_port",
    "destination_port",
    "protocol",
    "flow_id",
    "flow_key",
}


@dataclass
class PrepConfig:
    base_dir: Path
    level: str  # "packet" or "flow"
    out_dir: Path
    seed: int = 42
    scale_numeric: bool = True
    iqr_clip_factor: float = 3.0
    write_parquet: bool = True
    parquet_filename: Optional[str] = None
    sample_frac: Optional[float] = None  # for debugging; randomly downsample input before processing
    keep_meta_cols: Iterable[str] = tuple(DEFAULT_META_KEEP)


def _list_csvs(base_dir: Path, level: str) -> List[Path]:
    level_dir = base_dir / level
    if not level_dir.exists():
        raise FileNotFoundError(f"Level directory not found: {level_dir}")
    # All CSVs under Benign and Attacks subfolders
    csvs = list(level_dir.rglob("*.csv"))
    return csvs


def _infer_label_from_path(p: Path) -> Tuple[int, str]:
    # Determine benign/attack and type from path parts
    parts = [s for s in p.parts]
    if "Benign" in parts:
        return 0, "Benign"
    if "Attacks" in parts:
        # attack folder name is the part directly under Attacks
        try:
            idx = parts.index("Attacks")
            folder = parts[idx + 1]
        except Exception:
            folder = "Unknown"
        attack_type = ATTACK_FOLDERS.get(folder, folder)
        return 1, attack_type
    # Fallback: treat as benign but mark unknown
    return 0, "Unknown"


def _safe_read_csv(path: Path, sample_frac: Optional[float] = None, seed: int = 42) -> pd.DataFrame:
    # Read CSV with robust settings; optionally sample a fraction for speed
    df = pd.read_csv(path, low_memory=False)
    if sample_frac is not None and 0 < sample_frac < 1.0 and len(df) > 0:
        df = df.sample(frac=sample_frac, random_state=seed)
    return df


def _select_columns(df: pd.DataFrame, keep_meta_cols: Iterable[str]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    # Keep numeric features + selected meta columns; drop purely constant and duplicate columns
    df = df.copy()

    # Normalize column names to lower snake for matching, but preserve originals for output
    original_cols = list(df.columns)
    norm_map = {c: c.strip().lower().replace(" ", "_") for c in original_cols}
    df.rename(columns=norm_map, inplace=True)

    # Identify meta cols present
    keep_meta = [c for c in keep_meta_cols if c in df.columns]

    # Identify numeric columns (exclude bools treated as numeric here)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop columns with single unique value
    nunq = df.nunique(dropna=False)
    constant_cols = nunq[nunq <= 1].index.tolist()

    # Drop obvious non-useful textual columns (e.g., timestamps as strings); keep meta
    drop_candidates = [
        c
        for c in df.columns
        if c not in num_cols and c not in keep_meta
    ]

    df.drop(columns=[c for c in constant_cols if c in df.columns], inplace=True, errors="ignore")

    # Recompute numeric cols after drops
    num_cols = [c for c in num_cols if c in df.columns]

    # Retain only num + meta
    cols = sorted(set(num_cols).union(set(keep_meta)))
    df = df[cols]

    return df, num_cols, keep_meta


def _clip_iqr(df: pd.DataFrame, numeric_cols: List[str], factor: float) -> pd.DataFrame:
    if not numeric_cols:
        return df
    df = df.copy()
    for c in numeric_cols:
        q1 = df[c].quantile(0.25)
        q3 = df[c].quantile(0.75)
        iqr = q3 - q1
        if not np.isfinite(iqr) or iqr == 0:
            continue
        lo = q1 - factor * iqr
        hi = q3 + factor * iqr
        df[c] = df[c].clip(lower=lo, upper=hi)
    return df


def _impute_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    df = df.copy()
    medians: Dict[str, float] = {}
    for c in numeric_cols:
        med = df[c].median()
        if not np.isfinite(med):
            med = 0.0
        medians[c] = float(med)
        df[c] = df[c].fillna(med)
        # Replace inf with median too
        df[c] = df[c].replace([np.inf, -np.inf], med)
    return df, medians


def _scale_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    df = df.copy()
    scaler = StandardScaler(with_mean=True, with_std=True)
    if numeric_cols:
        arr = scaler.fit_transform(df[numeric_cols].astype(float))
        df[numeric_cols] = arr
    params = {c: (float(m), float(s) if float(s) != 0 else 1.0) for c, m, s in zip(numeric_cols, scaler.mean_ if numeric_cols else [], scaler.scale_ if numeric_cols else [])}
    return df, params


def prepare_level_dataset(cfg: PrepConfig) -> Tuple[pd.DataFrame, Dict]:
    np.random.seed(cfg.seed)
    csvs = _list_csvs(cfg.base_dir, cfg.level)
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {cfg.base_dir}/{cfg.level}")

    frames: List[pd.DataFrame] = []
    meta_records: List[Dict] = []
    for path in csvs:
        label, attack_type = _infer_label_from_path(path)
        df = _safe_read_csv(path, sample_frac=cfg.sample_frac, seed=cfg.seed)
        if df.empty:
            continue
        df, num_cols, meta_cols = _select_columns(df, cfg.keep_meta_cols)
        df["label"] = label
        df["attack_type"] = attack_type
        frames.append(df)
        meta_records.append({
            "file": str(path),
            "rows": int(len(df)),
            "label": label,
            "attack_type": attack_type,
        })

    if not frames:
        raise RuntimeError("All CSVs were empty after selection; nothing to prepare.")

    data = pd.concat(frames, axis=0, ignore_index=True)

    # Collect numeric columns after concat
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    # Remove the synthetic label from numeric processing
    numeric_cols = [c for c in numeric_cols if c != "label"]

    # Clean: impute, clip outliers, scale
    data, medians = _impute_numeric(data, numeric_cols)
    data = _clip_iqr(data, numeric_cols, cfg.iqr_clip_factor)
    scale_params: Dict[str, Tuple[float, float]] = {}
    if cfg.scale_numeric:
        data, scale_params = _scale_numeric(data, numeric_cols)

    # Persist parquet + metadata
    meta = {
        "level": cfg.level,
        "seed": cfg.seed,
        "rows": int(len(data)),
        "numeric_cols": numeric_cols,
        "meta_cols": [c for c in data.columns if c not in numeric_cols + ["label", "attack_type"] and not np.issubdtype(data[c].dtype, np.number)],
        "medians": medians,
        "scale_params": {k: {"mean": m, "std": s} for k, (m, s) in scale_params.items()},
        "source_files": meta_records,
    }

    if cfg.write_parquet:
        out_dir = cfg.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = cfg.parquet_filename or f"{cfg.level}_clean.parquet"
        out_parquet = out_dir / fname
        # Use pyarrow if available for best perf
        try:
            data.to_parquet(out_parquet, engine="pyarrow", index=False)
        except Exception:
            data.to_parquet(out_parquet, index=False)
        with open(out_dir / f"{fname}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    return data, meta


def _balanced_attack_sampling(df_attack: pd.DataFrame, target_total: int, seed: int = 42) -> pd.DataFrame:
    # Distribute samples approximately evenly across attack types present
    rng = np.random.default_rng(seed)
    grouped = df_attack.groupby("attack_type")
    types = list(grouped.groups.keys())
    if not types:
        return df_attack.sample(n=min(len(df_attack), target_total), random_state=seed)
    per_type = max(1, target_total // len(types))
    samples: List[pd.DataFrame] = []
    for t, g in grouped:
        n = min(len(g), per_type)
        if n < per_type and len(g) > 0:
            # sample with replacement to reach per_type
            idx = rng.choice(g.index.to_numpy(), size=per_type, replace=True)
            samples.append(g.loc[idx])
        else:
            samples.append(g.sample(n=per_type, random_state=seed))
    result = pd.concat(samples, ignore_index=False)
    # If we are short/excess due to rounding, adjust
    if len(result) < target_total and len(df_attack) > 0:
        extra = target_total - len(result)
        result = pd.concat([
            result,
            df_attack.sample(n=min(extra, len(df_attack)), random_state=seed)
        ], ignore_index=False)
    elif len(result) > target_total:
        result = result.sample(n=target_total, random_state=seed)
    return result.sample(frac=1.0, random_state=seed)  # shuffle


def generate_sampled_dataset(
    df: pd.DataFrame,
    benign_target: int = 200_000,
    attack_min: int = 4_000,
    attack_max: int = 6_200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create an imbalanced dataset reflecting real-world conditions.
    - ~97-98% benign, 2-3% attack, with attacks balanced across types.
    If available rows are insufficient, samples with replacement.
    """
    rng = np.random.default_rng(seed)
    benign = df[df["label"] == 0]
    attack = df[df["label"] == 1]

    # Choose attack total in [min, max]
    attack_total = int(rng.integers(attack_min, attack_max + 1))

    # Sample benign
    if len(benign) >= benign_target:
        benign_s = benign.sample(n=benign_target, random_state=seed)
    elif len(benign) > 0:
        idx = rng.choice(benign.index.to_numpy(), size=benign_target, replace=True)
        benign_s = benign.loc[idx]
    else:
        benign_s = benign.copy()

    # Sample attacks balanced across types
    if attack_total > 0:
        attack_s = _balanced_attack_sampling(attack, attack_total, seed=seed)
    else:
        attack_s = attack.head(0)

    combined = pd.concat([benign_s, attack_s], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return combined


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out_path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(out_path, index=False)


__all__ = [
    "PrepConfig",
    "prepare_level_dataset",
    "generate_sampled_dataset",
    "write_parquet",
]
