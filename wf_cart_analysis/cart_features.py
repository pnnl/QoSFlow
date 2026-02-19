# cart_features.py
from __future__ import annotations
from typing import List, Tuple
import re
import numpy as np
import pandas as pd

def detect_store_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c.endswith("_stor") or c.endswith("_store")]
    if not cols:
        raise RuntimeError("No *_stor or *_store columns found in input.")
    return cols

def stage_name_from_store_col(c: str) -> str:
    return c[:-5] if c.endswith("_stor") else (c[:-6] if c.endswith("_store") else c)

def parse_critical_tokens(s: str) -> List[str]:
    s = s or ""
    return [t.strip() for t in s.split("->") if t.strip()]

def _as_float(x):
    try:
        if isinstance(x, str) and x.strip() == "-":
            return 0.0
        return float(x)
    except Exception:
        return 0.0

def compute_rowwise_sensitivities(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build lam_* components and share columns per row using 'critical_path' tokens.
    Robust to tokens like 'read_stage(3)' by stripping parenthetical suffixes.
    """
    store_cols = detect_store_cols(df)
    stage_names = [stage_name_from_store_col(c) for c in store_cols]
    stores = sorted({v for c in store_cols for v in df[c].dropna().unique()})
    local_stores = [s for s in stores if s != "beegfs"]

    # numeric coercion for potential timing columns
    for s in stage_names:
        for base in ("read_", "write_", "in_", "out_"):
            col = base + s
            if col in df.columns:
                df[col] = df[col].apply(_as_float)

    # initialize lambdas
    for s in stores:
        df[f"lam_read_{s}"]  = 0.0
        df[f"lam_write_{s}"] = 0.0
    df["lam_in_total"]  = 0.0
    df["lam_out_total"] = 0.0

    # attribute along the critical path
    for idx, row in df.iterrows():
        tokens = parse_critical_tokens(row.get("critical_path", ""))
        store_of = {stage_name_from_store_col(c): row[c] for c in store_cols}
        for tok in tokens:
            m = re.match(r"^(read|write|stage_in|stage_out)_(.+)$", tok)
            if not m:
                continue
            kind, stg_raw = m.group(1), m.group(2).strip()
            stg = re.sub(r"\(.*\)$", "", stg_raw)  # strip "(...)"
            if kind == "read":
                col = f"read_{stg}"; val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
                    df.at[idx, f"lam_read_{store}"] += val
            elif kind == "write":
                col = f"write_{stg}"; val = row[col] if col in df.columns else 0.0
                store = store_of.get(stg, None)
                if store is not None:
                    df.at[idx, f"lam_write_{store}"] += val
            elif kind == "stage_in":
                col = f"in_{stg}"; val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_in_total"] += val
            elif kind == "stage_out":
                col = f"out_{stg}"; val = row[col] if col in df.columns else 0.0
                df.at[idx, "lam_out_total"] += val

    eps = 1e-12
    if "beegfs" in stores:
        df["exec_beegfs_share"] = (df["lam_read_beegfs"] + df["lam_write_beegfs"]) / (df["total"] + eps)
    else:
        df["exec_beegfs_share"] = 0.0
    if local_stores:
        df["exec_local_share"] = sum(df[f"lam_read_{s}"] + df[f"lam_write_{s}"] for s in local_stores) / (df["total"] + eps)
    else:
        df["exec_local_share"] = 0.0
    df["movement_share"] = (df["lam_in_total"] + df["lam_out_total"]) / (df["total"] + eps)

    for c in ["exec_beegfs_share", "exec_local_share", "movement_share"]:
        df[c] = df[c].fillna(0.0).clip(lower=0.0, upper=1.0)

    return df, store_cols
