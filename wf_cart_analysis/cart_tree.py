# cart_tree.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from cart_metrics import qhat

# Add just below the existing imports in cart_tree.py
EPS_ALPHA = 1e-12

def _sanitize_alpha(a: float) -> float:
    """
    Ensure ccp_alpha is valid for sklearn:
      - non-finite -> 0.0
      - tiny negative in (-EPS_ALPHA, 0) -> 0.0
      - otherwise clamp to >= 0
    """
    import numpy as np
    if a is None or not np.isfinite(a):
        return 0.0
    if a < 0.0 and a > -EPS_ALPHA:
        return 0.0
    return float(max(0.0, a))

def _ensure_dense(X):
    try:
        return X.toarray()
    except Exception:
        return np.asarray(X)

def build_cart_design(df: pd.DataFrame, cat_cols: List[str], num_cols: List[str]):
    all_cols = cat_cols + num_cols
    try:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
            remainder="passthrough", sparse_threshold=0.0
        )
    except TypeError:
        ct = ColumnTransformer(
            [("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)],
            remainder="passthrough", sparse_threshold=0.0
        )
    X = _ensure_dense(ct.fit_transform(df[all_cols]))
    return X, ct, all_cols

def cart_alpha_grid(X: np.ndarray, y: np.ndarray,
                    min_leaf: int, max_depth: Optional[int],
                    criterion: str, random_state: int = 42) -> np.ndarray:
    try:
        base = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state
        )
        base.fit(X, y)
    except ValueError:
        base = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state
        )
        base.fit(X, y)
    path = base.cost_complexity_pruning_path(X, y)
    alphas = np.unique(path.ccp_alphas)
    # sanitize any tiny negative/invalid values
    alphas = np.array([_sanitize_alpha(x) for x in alphas], dtype=float)
    alphas = np.unique(alphas)
    if alphas.size == 0:
        return np.array([0.0], dtype=float)
    return alphas

def inner_gate_alpha(
    X_tr: np.ndarray, y_tr: np.ndarray,
    max_depth: Optional[int], min_leaf: int, criterion: str,
    sep_kind: str, sep_kwargs: dict,
    gate_seed: int, val_frac: float = 0.2,
    tol_mae: float = 0.0, tol_sep: float = 0.0
) -> Tuple[float, pd.DataFrame]:
    """
    Split-acceptance gate on inner 80/20 of the training fold.
    Traverse α from simple→complex; accept a step only if
    MAE_val does not increase (<= tol_mae) AND separation does not decrease (>= -tol_sep).
    """
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=val_frac, random_state=gate_seed, shuffle=True
    )

    try:
        base = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=gate_seed
        )
        base.fit(X_fit, y_fit)
    except ValueError:
        base = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=gate_seed
        )
        base.fit(X_fit, y_fit)

    path = base.cost_complexity_pruning_path(X_fit, y_fit)
    alphas = np.unique(path.ccp_alphas)
    # sanitize & order simple→complex (descending α)
    alphas = np.array([_sanitize_alpha(x) for x in alphas], dtype=float)
    alphas = np.unique(alphas)
    alphas = np.sort(alphas)[::-1]

    audit = []
    prev_mae = float("inf")
    prev_sep = -float("inf")
    accepted_alpha = alphas[0]

    # lazy import to avoid cycle
    from cart_metrics import adjacent_effect_metric_all
    from cart_metrics import qhat as _qhat  # not used here; keeps symmetry

    def compute_sep(tree, Xv, yv) -> float:
        leaves_v = tree.apply(Xv)
        if sep_kind == "stat_g_all":
            m = adjacent_effect_metric_all(y=yv, leaf_ids=leaves_v, **sep_kwargs)
            return float(m["score"])
        else:
            # legacy: pass_wfrac path was removed here for simplicity
            return float("nan")

    for a in alphas:
        try:
            a_use = _sanitize_alpha(float(a))
            tree = DecisionTreeRegressor(
                criterion=criterion, min_samples_leaf=min_leaf,
                max_depth=max_depth, random_state=gate_seed, ccp_alpha=a_use
            )
            tree.fit(X_fit, y_fit)
        except ValueError:
            a_use = _sanitize_alpha(float(a))
            tree = DecisionTreeRegressor(
                criterion="squared_error", min_samples_leaf=min_leaf,
                max_depth=max_depth, random_state=gate_seed, ccp_alpha=a_use
            )
            tree.fit(X_fit, y_fit)

        pred = tree.predict(X_val)
        mae = float(np.mean(np.abs(pred - y_val)))
        sep = compute_sep(tree, X_val, y_val)

        accept = (mae <= prev_mae + tol_mae) and (sep >= prev_sep - tol_sep)
        audit.append({"alpha": float(a), "mae_val": mae, "sep_val": sep, "accept": bool(accept)})
        if accept:
            accepted_alpha = float(a)
            prev_mae = mae
            prev_sep = sep
        else:
            break

    return accepted_alpha, pd.DataFrame(audit)

def _leaf_stats_by_median_simple(y: np.ndarray, leaf_ids: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"leaf": leaf_ids, "y": y})
    g = df.groupby("leaf")["y"].agg(size="size", mean="mean", median="median").reset_index()
    return g.sort_values("median").reset_index(drop=True)

def fit_cart_and_label(
    df: pd.DataFrame, store_cols: List[str],
    ccp_alpha: float, min_leaf: int, max_depth: Optional[int],
    criterion: str = "absolute_error", random_state: int = 42
) -> Tuple[pd.DataFrame, Dict[str, int], DecisionTreeRegressor]:
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)
    try:
        tree = DecisionTreeRegressor(
            criterion=criterion, min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        ); tree.fit(X, y)
    except ValueError:
        tree = DecisionTreeRegressor(
            criterion="squared_error", min_samples_leaf=min_leaf,
            max_depth=max_depth, random_state=random_state, ccp_alpha=float(ccp_alpha)
        ); tree.fit(X, y)

    leaves = tree.apply(X)
    stats = _leaf_stats_by_median_simple(y, leaves)  # order by median
    ordered = stats["leaf"].tolist()
    leaf_to_region = {int(ordered[i]): i for i in range(len(ordered))}
    regions = np.array([leaf_to_region[int(l)] for l in leaves], dtype=int)

    out = df.copy()
    out["region_provisional"] = pd.Series(leaves, index=df.index).astype(str)
    out["region"] = regions
    prov_to_final = {str(k): int(v) for k, v in leaf_to_region.items()}
    return out, prov_to_final, tree

def _leaf_order_stats(y: np.ndarray, leaf_ids: np.ndarray, robust_q: bool) -> pd.DataFrame:
    df = pd.DataFrame({"leaf": leaf_ids, "y": y, "_row": np.arange(len(y))})
    rows = []
    for leaf, grp in df.groupby("leaf"):
        arr = grp["y"].to_numpy(dtype=float)
        rows.append({
            "leaf": int(leaf),
            "size": int(arr.size),
            "median": qhat(arr, 0.5, robust=robust_q),
            "q25": qhat(arr, 0.25, robust=robust_q),
            "q75": qhat(arr, 0.75, robust=robust_q),
            "rows": tuple(grp["_row"].to_numpy())
        })
    g = pd.DataFrame(rows)
    return g.sort_values("median").reset_index(drop=True)

def _pool_stats(y_pool: np.ndarray, robust_q: bool):
    return (qhat(y_pool, 0.5, robust=robust_q),
            qhat(y_pool, 0.25, robust=robust_q),
            qhat(y_pool, 0.75, robust=robust_q))

def cart_leaf_postmerge_union(
    y: np.ndarray,
    leaf_ids: np.ndarray,
    y_sep: float = 0.10,
    abs_tau: float = 0.0,
    min_region_size: int = 3,
    use_overlap_guard: bool = True,
    min_guard_n: int = 10,
    robust_q: bool = False,
):
    stats = _leaf_order_stats(y, leaf_ids, robust_q=robust_q)
    if len(stats) == 0:
        return {}, pd.DataFrame()
    leaf_rows = {int(r.leaf): list(r.rows) for r in stats.itertuples(index=False)}

    blocks = []
    cur_leaves = [int(stats.loc[0, "leaf"])]
    cur_idxs = leaf_rows[cur_leaves[0]].copy()
    cur_n = len(cur_idxs)
    cur_med = float(stats.loc[0, "median"])
    cur_q25 = float(stats.loc[0, "q25"])
    cur_q75 = float(stats.loc[0, "q75"])

    audit_rows = []
    for k in range(1, len(stats)):
        leaf_k = int(stats.loc[k, "leaf"]); rows_k = leaf_rows[leaf_k]
        med_k = float(stats.loc[k, "median"]); q25_k = float(stats.loc[k, "q25"]); q75_k = float(stats.loc[k, "q75"])
        n_k = len(rows_k)

        must_merge = (cur_n < min_region_size)
        rel_ok = (med_k >= (1.0 + y_sep) * cur_med)
        abs_ok = ((med_k - cur_med) >= abs_tau)
        gap_ok = (rel_ok or abs_ok)

        ov_ok = True
        if use_overlap_guard and (cur_n >= min_guard_n) and (n_k >= min_guard_n):
            ov_ok = (cur_q75 * (1.0 + 0.5*y_sep) <= q25_k)

        decision = "start_new" if (not must_merge and gap_ok and ov_ok) else "merge_into_prev"
        audit_rows.append({
            "prev_block_size": cur_n, "prev_block_median": cur_med, "prev_block_q75": cur_q75,
            "leaf_k": leaf_k, "leaf_n": n_k, "leaf_median": med_k, "leaf_q25": q25_k,
            "rel_gap_ok": bool(rel_ok), "abs_gap_ok": bool(abs_ok), "gap_ok": bool(gap_ok),
            "overlap_ok": bool(ov_ok), "must_merge": bool(must_merge), "decision": decision
        })

        if decision == "start_new":
            blocks.append({"leaves": cur_leaves.copy(), "rows": cur_idxs.copy(),
                           "n": cur_n, "median": cur_med, "q25": cur_q25, "q75": cur_q75})
            cur_leaves = [leaf_k]
            cur_idxs = rows_k.copy()
            cur_n = len(cur_idxs)
            cur_med, cur_q25, cur_q75 = med_k, q25_k, q75_k
        else:
            cur_leaves.append(leaf_k)
            cur_idxs.extend(rows_k)
            cur_n = len(cur_idxs)
            med_new, q25_new, q75_new = _pool_stats(np.asarray(y)[cur_idxs], robust_q=robust_q)
            cur_med, cur_q25, cur_q75 = med_new, q25_new, q75_new

    blocks.append({"leaves": cur_leaves, "rows": cur_idxs,
                   "n": cur_n, "median": cur_med, "q25": cur_q25, "q75": cur_q75})

    leaf_to_region = {}
    for rid, b in enumerate(blocks):
        for lf in b["leaves"]:
            leaf_to_region[int(lf)] = rid

    audit = pd.DataFrame(audit_rows)
    return leaf_to_region, audit
