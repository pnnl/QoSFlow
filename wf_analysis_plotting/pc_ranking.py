#!/usr/bin/env python3
"""
Policy comparison (2x2) + Pairwise Concordance (PC) report.

Policies shown:
  - Fastest-Storage First (FSF): sort by (#tmpfs desc, #ssd desc), tie-break by total asc.
  - Low-Transition Layout (LTL): transition_score = count of nonzero in_* + out_*; sort asc.
  - Hybrid: score = FSF_norm + (1 - LTL_norm); sort desc; tie-break by total asc.
  - QoSFlow: order regions by region median makespan asc; list configs in that region order.

Input:
  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Output:
  Figure:  ../<wf_name>/sens_out/<wf>_policy6_subplots_scale<scale>.{pdf|png}
  PC CSV:  ../<wf_name>/sens_out/<wf>_policy_pc_scale<scale>.csv
  PC text: printed to stdout

Usage:
  python make_policy_six_subplots_from_sens.py --wf 1kgenome --scale 10 --out-format pdf
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Global font & style (≥18pt everywhere) ----
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# ---------- helpers ----------

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _stor_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.endswith("_stor")])

def _in_out_cols(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    in_cols  = sorted([c for c in df.columns if c.startswith("in_")])
    out_cols = sorted([c for c in df.columns if c.startswith("out_")])
    return in_cols, out_cols

def _pairwise_concordance(order_idx: np.ndarray, y: np.ndarray) -> float:
    """
    PC = fraction of pairs (i<j in the *policy order*) with y_i <= y_j.
    Assumes lower y (makespan) is better. Ties in y count as correct.
    """
    n = len(order_idx)
    if n < 2:
        return np.nan
    total_pairs = n * (n - 1) // 2
    conc = 0
    for a in range(n - 1):
        ia = order_idx[a]
        ya = y[ia]
        for b in range(a + 1, n):
            ib = order_idx[b]
            if ya <= y[ib]:
                conc += 1
    return conc / total_pairs

def _rank_minmax(x: np.ndarray) -> np.ndarray:
    """Min–max normalize to [0,1]; constant arrays map to 0."""
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)

# ---------- policy orderings ----------

def order_fsf(df: pd.DataFrame, stor_cols: list[str]) -> np.ndarray:
    tmpfs_cnt = (df[stor_cols] == "tmpfs").sum(axis=1)
    ssd_cnt   = (df[stor_cols] == "ssd").sum(axis=1)
    # tie-break by total asc for a stable "best-first" feel
    key = pd.DataFrame({"neg_tmpfs": -tmpfs_cnt, "neg_ssd": -ssd_cnt, "total": df["total"].values})
    order = np.lexsort((key["total"].to_numpy(),
                        key["neg_ssd"].to_numpy(),
                        key["neg_tmpfs"].to_numpy()))
    return order

def order_ltl(df: pd.DataFrame, in_cols: list[str], out_cols: list[str]) -> np.ndarray:
    # Transition score: count stage-boundary actions (only in_ / out_)
    in_present  = (df[in_cols].fillna(0) > 0).sum(axis=1) if in_cols else 0
    out_present = (df[out_cols].fillna(0) > 0).sum(axis=1) if out_cols else 0
    trans = in_present + out_present
    key = pd.DataFrame({"trans": trans, "total": df["total"].values})
    order = np.lexsort((key["total"].to_numpy(),
                        key["trans"].to_numpy()))  # ascending
    return order

def order_hybrid(df: pd.DataFrame, stor_cols: list[str], in_cols: list[str], out_cols: list[str]) -> np.ndarray:
    # FSF component
    tmpfs_cnt = (df[stor_cols] == "tmpfs").sum(axis=1).astype(float)
    ssd_cnt   = (df[stor_cols] == "ssd").sum(axis=1).astype(float)
    fsf_score = 2.0 * tmpfs_cnt + 1.0 * ssd_cnt

    # LTL component (only boundary in/out)
    in_present  = (df[in_cols].fillna(0) > 0).sum(axis=1).astype(float) if in_cols else 0.0
    out_present = (df[out_cols].fillna(0) > 0).sum(axis=1).astype(float) if out_cols else 0.0
    ltl_score = in_present + out_present

    fsf_norm = _rank_minmax(fsf_score)
    ltl_norm = _rank_minmax(ltl_score)

    hybrid = fsf_norm + (1.0 - ltl_norm)   # higher is better
    key = pd.DataFrame({"neg_h": -hybrid, "total": df["total"].values})
    order = np.lexsort((key["total"].to_numpy(),
                        key["neg_h"].to_numpy()))
    return order

def order_qosflow(df: pd.DataFrame) -> np.ndarray:
    """
    Regions already exist (CART-derived). Order regions by median total asc,
    list configs by that region order; within a region keep measured best-first.
    """
    if "region" not in df.columns:
        raise ValueError("QoSFlow requires a 'region' column in the CSV.")
    reg_stats = (df.groupby("region")["total"].median()
                 .sort_values(ascending=True))
    # build full index order
    order_idx = []
    for r in reg_stats.index:
        sub = df[df["region"] == r].copy()
        sub = sub.sort_values("total", ascending=True)  # keep good first inside region
        order_idx.extend(sub.index.tolist())
    # convert to 0..n-1 positions for the filtered df
    # (the caller passes df that is a filtered view; convert to positional array)
    pos_map = {idx: i for i, idx in enumerate(df.index)}
    return np.array([pos_map[i] for i in order_idx], dtype=int)

# ---------- plotting ----------

def _plot_policy(ax, order_idx: np.ndarray, y: np.ndarray, title: str):
    x = np.arange(1, len(order_idx) + 1)
    yy = y[order_idx]
    ax.scatter(x, yy, s=40)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_title(title)

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Policy comparisons with Pairwise Concordance (PC) report.")
    ap.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    ap.add_argument("--scale", type=int, required=True, help="Node scale to visualize")
    ap.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                    help="Output format for the figure (default: pdf)")
    args = ap.parse_args()

    # Paths
    sens_csv = os.path.join("..", args.wf, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", args.wf, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(args.wf))
    fig_base = os.path.join(out_dir, f"{prefix}_policy6_subplots_scale{args.scale}")
    pc_csv   = os.path.join(out_dir, f"{prefix}_policy_pc_scale{args.scale}.csv")

    # Load & filter
    df0 = pd.read_csv(sens_csv)
    if "nodes" not in df0.columns or "total" not in df0.columns:
        raise ValueError("CSV must contain 'nodes' and 'total' columns.")
    df0["nodes"] = pd.to_numeric(df0["nodes"], errors="coerce").astype("Int64")
    df0["total"] = pd.to_numeric(df0["total"], errors="coerce")
    df = df0[df0["nodes"] == args.scale].copy()
    if df.empty:
        raise ValueError(f"No rows for nodes == {args.scale}")

    stor_cols = _stor_cols(df)
    for c in stor_cols:
        df[c] = _norm_str(df[c])
    in_cols, out_cols = _in_out_cols(df)

    # Compute policy orders
    y = df["total"].to_numpy()
    fsf_ord   = order_fsf(df, stor_cols)
    ltl_ord   = order_ltl(df, in_cols, out_cols)
    hyb_ord   = order_hybrid(df, stor_cols, in_cols, out_cols)
    qos_ord   = order_qosflow(df)

    # ---- Pairwise Concordance (PC) ----
    pc_fsf = _pairwise_concordance(fsf_ord, y)
    pc_ltl = _pairwise_concordance(ltl_ord, y)
    pc_hyb = _pairwise_concordance(hyb_ord, y)
    pc_qos = _pairwise_concordance(qos_ord, y)

    # Relative % better of QoSFlow vs each baseline
    def pct_better(pc_base):
        return np.nan if (pc_base is None or not np.isfinite(pc_base) or pc_base == 0) \
                      else 100.0 * (pc_qos - pc_base) / pc_base

    better_fsf = pct_better(pc_fsf)
    better_ltl = pct_better(pc_ltl)
    better_hyb = pct_better(pc_hyb)

    # Lift over best heuristic
    best_heur_pc = np.nanmax([pc_fsf, pc_ltl, pc_hyb])
    lift_vs_best = pct_better(best_heur_pc)

    # Print the report
    print("\n=== Pairwise Concordance (PC) — scale =", args.scale, "===")
    print(f"FSF      PC: {pc_fsf:0.4f}")
    print(f"LTL      PC: {pc_ltl:0.4f}")
    print(f"Hybrid   PC: {pc_hyb:0.4f}")
    print(f"QoSFlow  PC: {pc_qos:0.4f}")
    print("\nQoSFlow relative improvement:")
    print(f"  vs FSF   : {better_fsf:0.2f}%")
    print(f"  vs LTL   : {better_ltl:0.2f}%")
    print(f"  vs Hybrid: {better_hyb:0.2f}%")
    print(f"\nLift over best heuristic: {lift_vs_best:0.2f}%\n")

    # Save PCs as CSV
    pc_rows = [
        {"policy": "FSF",     "PC": pc_fsf, "qosflow_vs_policy_%": better_fsf},
        {"policy": "LTL",     "PC": pc_ltl, "qosflow_vs_policy_%": better_ltl},
        {"policy": "Hybrid",  "PC": pc_hyb, "qosflow_vs_policy_%": better_hyb},
        {"policy": "QoSFlow", "PC": pc_qos, "qosflow_vs_policy_%": np.nan},
        {"policy": "QoSFlow_lift_over_best_heuristic(%)", "PC": np.nan, "qosflow_vs_policy_%": lift_vs_best},
    ]
    pd.DataFrame(pc_rows).to_csv(pc_csv, index=False)
    print("Saved PC CSV:", pc_csv)

    # ---- Plot (2x2) ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    axes = axes.ravel().tolist()

    _plot_policy(axes[0], fsf_ord, y, "Fastest-Storage First (FSF)")
    _plot_policy(axes[1], ltl_ord, y, "Low-Transition Layout (LTL)")
    _plot_policy(axes[2], hyb_ord, y, "Hybrid Heuristic")
    _plot_policy(axes[3], qos_ord, y, "QoSFlow")

    for ax in axes[:2]:
        ax.set_xlabel("")  # top row
    for ax in axes[2:]:
        ax.set_xlabel("Configuration #")
    axes[0].set_ylabel("Makespan")
    axes[2].set_ylabel("Makespan")

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.95, wspace=0.20, hspace=0.22)

    # Save figure
    fmt = args.out_format.lower()
    if fmt in ("png", "both"):
        p = fig_base + ".png"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved figure:", p)
    if fmt in ("pdf", "both"):
        p = fig_base + ".pdf"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("Saved figure:", p)

if __name__ == "__main__":
    main()
