# cart_cv.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

from cart_tree import build_cart_design, cart_alpha_grid, inner_gate_alpha
from cart_metrics import adjacent_effect_metric_all

EPS_ALPHA = 1e-12
def _sanitize_alpha(a: float) -> float:
    import numpy as np
    if a is None or not np.isfinite(a):
        return 0.0
    if a < 0.0 and a > -EPS_ALPHA:
        return 0.0
    return float(max(0.0, a))

def cv_select_alpha_cart_repeated(
    df: pd.DataFrame,
    store_cols: List[str],
    # separation metric config
    separation_metric: str,
    g_min: float,
    n_min_pair: int,
    stat_score: str,
    pair_weight: str,
    robust_q: bool,
    # adaptive g controls
    g_min_auto: bool = False,
    delta: Optional[float] = None,
    cv_estimator: str = "std",
    g_floor: float = 0.0,
    g_cap: float = 1.0,
    # CV controls
    kfolds: int = 5,
    repeats: int = 5,
    min_leaf: int = 5,
    max_depth: Optional[int] = None,
    criterion: str = "absolute_error",
    # gating
    gate_splits: bool = False,
    gate_val_frac: float = 0.2,
    gate_mae_tol: float = 0.0,
    gate_sep_tol: float = 0.0,
    seed: int = 42,
    # selection blend
    select_mode: str = "constrained",
    mae_budget: float = 0.02,
    mae_weight: float = 0.30,
    # >>> NEW: debug toggle <<<
    debug: bool = False,
) -> Tuple[float, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Repeated-K-fold alpha selection; if debug=True, returns a third DataFrame with per-pair diagnostics.
    """
    num_cols = ["nodes"] if "nodes" in df.columns else []
    X, _, _ = build_cart_design(df, store_cols, num_cols)
    y = df["total"].to_numpy(dtype=float)

    alphas = cart_alpha_grid(X, y, min_leaf, max_depth, criterion, random_state=seed)
    if alphas.size == 0:
        alphas = np.array([0.0], dtype=float)
    # extra safety: sanitize grid here too
    alphas = np.array([_sanitize_alpha(x) for x in alphas], dtype=float)
    alphas = np.unique(alphas)

    sep_kwargs = dict(
        g_min=g_min, n_min_pair=n_min_pair, pair_weight=pair_weight, robust_q=robust_q, score_mode=stat_score,
        g_min_auto=g_min_auto, delta=delta, cv_estimator=cv_estimator, g_floor=g_floor, g_cap=g_cap,
    )

    rows = []
    dbg_rows_all = [] if debug else None

    for r in range(repeats):
        rep_seed = seed + 137 * r
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=rep_seed)
        gated_alpha_per_fold: Dict[int, float] = {}

        for ai, a in enumerate(alphas):
            leaves_sum = 0.0
            pairs_sum  = 0
            skipped_folds = 0
            sep_vals = []
            mae_num = 0.0; mae_den = 0

            for fidx, (tr, va) in enumerate(kf.split(X)):
                Xt, Xv = X[tr], X[va]
                yt, yv = y[tr], y[va]

                if gate_splits and fidx not in gated_alpha_per_fold:
                    a_gate, _ = inner_gate_alpha(
                        Xt, yt, max_depth=max_depth, min_leaf=min_leaf, criterion=criterion,
                        sep_kind=separation_metric, sep_kwargs=sep_kwargs,
                        gate_seed=rep_seed + fidx,
                        val_frac=gate_val_frac, tol_mae=gate_mae_tol, tol_sep=gate_sep_tol
                    )
                    gated_alpha_per_fold[fidx] = float(a_gate)
                a_eff_raw = float(max(a, gated_alpha_per_fold.get(fidx, a)))
                a_eff = _sanitize_alpha(a_eff_raw)

                try:
                    tree = DecisionTreeRegressor(
                        criterion=criterion, min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    ); tree.fit(Xt, yt)
                except ValueError:
                    tree = DecisionTreeRegressor(
                        criterion="squared_error", min_samples_leaf=min_leaf,
                        max_depth=max_depth, random_state=rep_seed + fidx, ccp_alpha=a_eff
                    ); tree.fit(Xt, yt)

                leaves_v = tree.apply(Xv)

                # Build a per-call debug sink and context
                dbg_rows = [] if debug else None
                dbg_ctx  = {
                    "repeat": r, "fold": fidx, "alpha_idx": ai,
                    "ccp_alpha": float(a), "a_eff_raw": float(a_eff_raw), "a_eff": float(a_eff)
                }

                m = adjacent_effect_metric_all(
                    y=yv, leaf_ids=leaves_v, **sep_kwargs,
                    debug_rows=dbg_rows, debug_ctx=dbg_ctx
                )
                sep_val = float(m["score"])
                leaves_here = int(m["n_leaves"])
                pairs_here  = int(m["pairs_eval"])
                skipped = bool(m["skip"])

                # Accumulate fold-level metrics
                pred = tree.predict(Xv)
                err = np.abs(pred - yv)
                mae_fold = float(np.mean(err))
                mae_num += float(err.sum()); mae_den += int(len(yv))

                if skipped:
                    skipped_folds += 1
                else:
                    leaves_sum += leaves_here
                    pairs_sum  += pairs_here
                    sep_vals.append(sep_val)

                rows.append({
                    "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                    "fold": fidx, "n_val": int(len(yv)),
                    "n_leaves_val": int(leaves_here),
                    "pairs_val": int(pairs_here),
                    "sep_val": float(sep_val),
                    "skipped": bool(skipped),
                    "a_eff": float(a_eff),
                    "mae_val": float(mae_fold),
                })

                # Flush per-pair debug rows into global sink (optional cap later)
                if debug and dbg_rows:
                    dbg_rows_all.extend(dbg_rows)

            cv_sep = float(np.nanmedian(sep_vals)) if len(sep_vals) else float("nan")
            cv_mae = float(mae_num / mae_den) if mae_den > 0 else float("nan")

            rows.append({
                "repeat": r, "alpha_idx": ai, "ccp_alpha": float(a),
                "fold": "CV-agg", "n_val": int(len(y)),
                "n_leaves_val": float(leaves_sum / max(1, (kfolds - skipped_folds))) if (kfolds - skipped_folds) > 0 else float("nan"),
                "pairs_val": int(pairs_sum),
                "sep_val": float(cv_sep),
                "skipped": (skipped_folds > 0),
                "a_eff": np.nan,
                "mae_val": float(cv_mae),
            })

    df_diag = pd.DataFrame(rows)

    # ---- Final α selection (unchanged, prefers simpler on ties) ----
    by_alpha = (
        df_diag[df_diag["fold"].eq("CV-agg")]
        .groupby("ccp_alpha", as_index=False)
        .agg(
            sep_med=("sep_val", "median"),
            mae_med=("mae_val", "median"),
            n_leaves_mean=("n_leaves_val", "mean"),
        )
        .sort_values("ccp_alpha")
    )

    if select_mode == "pass_only":
        by_alpha = by_alpha.sort_values(["sep_med", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, True, False]).reset_index(drop=True)
        chosen_alpha = float(by_alpha.iloc[0]["ccp_alpha"])

    elif select_mode == "constrained":
        best_mae_med = by_alpha["mae_med"].min()
        eligible = by_alpha[by_alpha["mae_med"] <= (1.0 + mae_budget) * best_mae_med].copy()
        if eligible.empty:
            eligible = by_alpha.nsmallest(1, "mae_med").copy()
        eligible = eligible.sort_values(["sep_med", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, False, True]).reset_index(drop=True)
        chosen_alpha = float(eligible.iloc[0]["ccp_alpha"])

    else:  # 'weighted'
        a = by_alpha["sep_med"].to_numpy()
        b = by_alpha["mae_med"].to_numpy()
        a_min, a_max = np.nanmin(a), np.nanmax(a)
        b_min, b_max = np.nanmin(b), np.nanmax(b)
        a_norm = np.zeros_like(a) if not np.isfinite(a_min + a_max) or (a_max - a_min) == 0 else (a - a_min) / (a_max - a_min)
        b_norm = np.zeros_like(b) if not np.isfinite(b_min + b_max) or (b_max - b_min) == 0 else (b - b_min) / (b_max - b_min)
        J = (1.0 - mae_weight) * a_norm + mae_weight * (1.0 - b_norm)
        by_alpha = by_alpha.assign(J=J)
        by_alpha = by_alpha.sort_values(["J", "n_leaves_mean", "ccp_alpha"],
                                        ascending=[False, False, True]).reset_index(drop=True)
        chosen_alpha = float(by_alpha.iloc[0]["ccp_alpha"])

    # Build a single debug DF if needed
    df_pairs_dbg = None
    if debug and dbg_rows_all:
        df_pairs_dbg = pd.DataFrame(dbg_rows_all)

    return chosen_alpha, df_diag, df_pairs_dbg
