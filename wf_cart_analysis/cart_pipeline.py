# cart_pipeline.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

from cart_features import compute_rowwise_sensitivities, detect_store_cols
from cart_cv import cv_select_alpha_cart_repeated
from cart_tree import fit_cart_and_label, cart_leaf_postmerge_union
from cart_summaries import strict_wildcard_summary_joint_unique, compute_regret, critical_path_prevalence

def run_pipeline(
    input_csv: str,
    outdir: str,
    # tree & CV
    min_leaf: int,
    max_depth: Optional[int],
    criterion: str,
    kfolds: int,
    repeats: int,
    seed: int,
    # separation
    separation_metric: str,
    g_min: float,
    n_min_pair: int,
    stat_score: str,
    pair_weight: str,
    robust_quantiles: bool,
    # adaptive g
    g_min_auto: bool,
    delta: Optional[float],
    cv_estimator: str,
    g_floor: float,
    g_cap: float,
    # legacy (pass_wfrac) knobs (ignored here but kept for compatibility)
    top_frac: float,
    max_topN: int,
    y_sep: float,
    use_overlap_guard: bool,
    # gating
    gate_splits: bool,
    gate_val_frac: float,
    gate_mae_tol: float,
    gate_sep_tol: float,
    # selection blend
    select_mode: str,
    mae_budget: float,
    mae_weight: float,
    # post-merge
    postmerge: bool,
    min_region_size: int,
    abs_gap_tau: float,
    min_guard_n: int,
    # >>> DEBUG control (new) <<<
    debug: bool = False,
    debug_max_rows: Optional[int] = None,
):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)

    df0 = pd.read_csv(input_csv)
    df, store_cols = compute_rowwise_sensitivities(df0.copy())

    chosen_alpha, cv_diag, df_pairs_dbg = cv_select_alpha_cart_repeated(
        df=df, store_cols=store_cols,
        separation_metric=separation_metric,
        g_min=g_min, n_min_pair=n_min_pair, stat_score=stat_score,
        pair_weight=pair_weight, robust_q=robust_quantiles,
        # adaptive g
        g_min_auto=g_min_auto, delta=delta, cv_estimator=cv_estimator, g_floor=g_floor, g_cap=g_cap,
        # CV + tree
        kfolds=kfolds, repeats=repeats, min_leaf=min_leaf, max_depth=max_depth, criterion=criterion,
        # gating
        gate_splits=gate_splits, gate_val_frac=gate_val_frac, gate_mae_tol=gate_mae_tol, gate_sep_tol=gate_sep_tol,
        seed=seed,
        # selection
        select_mode=select_mode, mae_budget=mae_budget, mae_weight=mae_weight,
        # debug
        debug=debug,
    )
    cv_diag.to_csv(out / "cart_cv_candidates.csv", index=False)

    # >>> write pair-level debug CSV if requested
    if debug and df_pairs_dbg is not None:
        if debug_max_rows is not None:
            df_pairs_dbg = df_pairs_dbg.head(int(debug_max_rows))
        df_pairs_dbg.to_csv(out / "cart_debug_pairs.csv", index=False)


    df_cf, prov_to_final, tree = fit_cart_and_label(
        df=df, store_cols=store_cols, ccp_alpha=chosen_alpha,
        min_leaf=min_leaf, max_depth=max_depth, criterion=criterion, random_state=seed
    )

    if postmerge:
        leaf_key = df_cf["region"].to_numpy(dtype=int)
        y = df_cf["total"].to_numpy(dtype=float)
        leaf_to_region, pm_audit = cart_leaf_postmerge_union(
            y=y, leaf_ids=leaf_key, y_sep=y_sep, abs_tau=abs_gap_tau,
            min_region_size=min_region_size, use_overlap_guard=use_overlap_guard,
            min_guard_n=min_guard_n, robust_q=robust_quantiles
        )
        df_cf["leaf_region"] = df_cf["region"].astype(int)
        df_cf["region"] = df_cf["leaf_region"].map(leaf_to_region).astype(int)
        pm_audit.to_csv(out / "cart_postmerge_adjacent_audit.csv", index=False)
        pd.DataFrame([{"leaf_region": k, "merged_region": v} for k, v in leaf_to_region.items()]) \
            .to_csv(out / "cart_postmerge_map.csv", index=False)

    df_cf.to_csv(out / "workflow_rowwise_sensitivities.csv", index=False)
    pd.DataFrame([{"region_provisional": k, "region": v} for k, v in prov_to_final.items()]) \
        .to_csv(out / "region_merge_map.csv", index=False)

    stat_cols_map = {
        "mean_total":   ("total", "mean"),
        "median_total": ("total", "median"),
        "std_total":    ("total", "std"),
        "cv_total":     ("total", "cv"),
        "q25_total":    ("total", "q25"),
        "q75_total":    ("total", "q75"),
        "min_total":    ("total", "min"),
        "max_total":    ("total", "max"),
    }
    regions_df = strict_wildcard_summary_joint_unique(
        df_all=df, df_labeled=df_cf, store_cols=store_cols,
        stat_cols_map=stat_cols_map,
        additional_cols=["exec_beegfs_share", "exec_local_share", "movement_share"],
        include_nodes=True
    )
    regions_df.to_csv(out / "regions_by_total.csv", index=False)

    regret_region, regret_config = compute_regret(df_cf)
    regret_region.to_csv(out / "regret_per_region.csv", index=False)
    regret_config.to_csv(out / "regret_per_config.csv", index=False)

    cp_df = critical_path_prevalence(df_cf)
    cp_df.to_csv(out / "critical_path_prevalence_by_region.csv", index=False)

    print("Chosen α (ccp_alpha):", chosen_alpha)
    print("Wrote:")
    for p in [
        out / "workflow_rowwise_sensitivities.csv",
        out / "regions_by_total.csv",
        out / "regret_per_region.csv",
        out / "regret_per_config.csv",
        out / "critical_path_prevalence_by_region.csv",
        out / "region_merge_map.csv",
        out / "cart_cv_candidates.csv",
    ]:
        print(" -", p)
    if postmerge:
        print(" -", out / "cart_postmerge_adjacent_audit.csv")
        print(" -", out / "cart_postmerge_map.csv")
