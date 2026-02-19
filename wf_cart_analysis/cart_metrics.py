# cart_metrics.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
import math
import numpy as np
import pandas as pd

def qhat(x: np.ndarray, q: float, robust: bool = False) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0:
        return float("nan")
    if not robust:
        return float(np.quantile(x, q))
    try:
        return float(np.quantile(x, q, method="hazen"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="linear"))

def _pair_weight(n_i: int, n_j: int, mode: str = "min") -> float:
    if mode == "hmean":
        if n_i <= 0 or n_j <= 0:
            return 0.0
        return 2.0 / (1.0 / n_i + 1.0 / n_j)
    return float(min(n_i, n_j))

def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Unbiased standardized mean difference (Hedges’ g)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return np.nan
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    denom = (nx + ny - 2)
    if denom <= 0:
        return np.nan
    s_p = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / denom) if denom > 0 else np.nan
    if not np.isfinite(s_p) or s_p == 0:
        return np.nan
    d = (my - mx) / s_p
    J = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0)) if (nx + ny) > 2 else 1.0
    return float(J * d)

def _cv_value(arr: np.ndarray, estimator: str = "std") -> float:
    """CV = sd/mean or (1.4826*MAD)/median with guards."""
    x = np.asarray(arr, dtype=float)
    if x.size == 0:
        return np.inf
    if estimator == "mad":
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        sd = 1.4826 * mad
        mu = med
    else:
        sd = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
        mu = float(np.mean(x))
    if not np.isfinite(mu) or mu <= 0:
        return np.inf
    return sd / mu

def _pooled_cv(cv_i: float, cv_j: float) -> float:
    if not np.isfinite(cv_i) or not np.isfinite(cv_j):
        return np.inf
    return math.sqrt(0.5 * (cv_i * cv_i + cv_j * cv_j))

def _pair_g_threshold(delta: Optional[float],
                      cv_pooled: float,
                      g_min_fixed: float,
                      use_auto: bool,
                      g_floor: float,
                      g_cap: float) -> float:
    """Adaptive per-pair threshold for |g| based on δ and pooled CV."""
    if use_auto and (delta is not None) and np.isfinite(cv_pooled) and cv_pooled > 0:
        g_auto = delta / cv_pooled
        g_thr = max(g_floor, min(g_cap, g_auto))
        return float(g_thr)
    return float(g_min_fixed)

def _leaf_stats_with_rows(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool):
    df = pd.DataFrame({"leaf": leaf_ids, "y": y, "_row": np.arange(len(y))})
    rows = []
    for leaf, grp in df.groupby("leaf"):
        arr = grp["y"].to_numpy(dtype=float)
        rows.append({
            "leaf": int(leaf),
            "size": int(arr.size),
            "median": qhat(arr, 0.5, robust=robust_q),
            "rows": tuple(grp["_row"].to_numpy())
        })
    g = pd.DataFrame(rows)
    return g.sort_values("median").reset_index(drop=True)

def adjacent_effect_metric_all(
    y: np.ndarray,
    leaf_ids: np.ndarray,
    g_min: float = 0.30,
    n_min_pair: int = 5,
    pair_weight: str = "hmean",
    robust_q: bool = False,
    score_mode: str = "product",
    # adaptive-g knobs:
    g_min_auto: bool = False,
    delta: Optional[float] = None,
    cv_estimator: str = "std",
    g_floor: float = 0.0,
    g_cap: float = 1.0,
    # >>> NEW: debug plumbing <<<
    debug_rows: Optional[list] = None,
    debug_ctx: Optional[dict] = None,
) -> dict:
    """
    Adjacent-pair statistical separation (ALL adjacencies) using Hedges’ g.
    If debug_rows is provided, appends a row per adjacent pair with:
      leaf ids, sizes, medians, CVs, g, g_thr, weight, significant flag, and skip reasons.
    """
    stats = _leaf_stats_with_rows(y, leaf_ids, robust_q=robust_q)
    L = len(stats)
    if L <= 1:
        return {"skip": True, "n_leaves": L, "pairs_eval": 0, "pairs_sig": 0,
                "avg_effect": 0.0, "frac_sig": 0.0, "score": float("nan")}

    sum_g_sig, W, Ws = 0.0, 0.0, 0.0
    pairs_eval, pairs_sig = 0, 0

    for i in range(L - 1):
        leaf_i = int(stats.loc[i, "leaf"])
        leaf_j = int(stats.loc[i+1, "leaf"])
        rows_i = stats.loc[i, "rows"]; rows_j = stats.loc[i+1, "rows"]
        n_i, n_j = len(rows_i), len(rows_j)
        med_i = float(stats.loc[i, "median"]); med_j = float(stats.loc[i+1, "median"])

        # Pre-populate debug row
        dbg = {
            "pair_idx": i,
            "leaf_i": leaf_i, "leaf_j": leaf_j,
            "n_i": n_i, "n_j": n_j,
            "median_i": med_i, "median_j": med_j,
            "cv_i": np.nan, "cv_j": np.nan, "cv_pooled": np.nan,
            "g": np.nan, "g_thr": np.nan, "weight": np.nan,
            "significant": False,
            "skipped_nmin": False, "skipped_nan_g": False,
            "used_g_min_auto": bool(g_min_auto),
            "g_min_fixed": float(g_min),
            "delta": float(delta) if delta is not None else np.nan,
            "cv_estimator": cv_estimator,
            "g_floor": float(g_floor), "g_cap": float(g_cap),
        }
        if debug_ctx:
            dbg.update(debug_ctx)

        # Skip small pairs
        if min(n_i, n_j) < n_min_pair:
            dbg["skipped_nmin"] = True
            if debug_rows is not None:
                debug_rows.append(dbg)
            continue

        yi = y[list(rows_i)]; yj = y[list(rows_j)]
        g = abs(hedges_g(yi, yj))
        if not np.isfinite(g):
            dbg["skipped_nan_g"] = True
            if debug_rows is not None:
                debug_rows.append(dbg)
            continue

        cv_i = _cv_value(yi, estimator=cv_estimator)
        cv_j = _cv_value(yj, estimator=cv_estimator)
        cv_p = _pooled_cv(cv_i, cv_j)
        g_thr = _pair_g_threshold(delta=delta, cv_pooled=cv_p,
                                  g_min_fixed=g_min, use_auto=g_min_auto,
                                  g_floor=g_floor, g_cap=g_cap)
        w = _pair_weight(n_i, n_j, mode=pair_weight)

        # Update debug row
        dbg.update({"cv_i": cv_i, "cv_j": cv_j, "cv_pooled": cv_p,
                    "g": g, "g_thr": g_thr, "weight": w})

        W += w; pairs_eval += 1
        if g >= g_thr:
            Ws += w; pairs_sig += 1; sum_g_sig += g * w
            dbg["significant"] = True

        if debug_rows is not None:
            debug_rows.append(dbg)

    avg_effect = (sum_g_sig / Ws) if Ws > 0 else 0.0
    frac_sig   = (Ws / W) if W > 0 else 0.0
    if score_mode == "avg_effect":
        score = avg_effect
    elif score_mode == "frac_sig":
        score = frac_sig
    else:
        score = avg_effect * frac_sig

    return {"skip": False, "n_leaves": L, "pairs_eval": pairs_eval, "pairs_sig": pairs_sig,
            "avg_effect": float(avg_effect), "frac_sig": float(frac_sig), "score": float(score)}
