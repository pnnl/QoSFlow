#!/usr/bin/env python3
"""
Policy comparisons (2x2 subplots) at a single node scale, using:

  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Included policies:
  • Fastest-Storage First (FSF)
  • Low-Transition Layout (LTL)
  • Hybrid Heuristic = FSF_norm + (1 - LTL_norm)
  • QoSFlow (was “CART Region Order”, orders by region ascending)

Notes
  - Random Shuffle and Bottleneck-First are intentionally disabled.
  - LTL counts ONLY boundary actions (in_<stage>, out_<stage> != 0).
  - Y label: “Makespan”; bottom row X label: “Configuration #”.
  - Output format selectable via --out-format {png,pdf}.
  - Tight layout via constrained_layout (no big gutters).

"""

import argparse
import os
import math
from typing import List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---- Global font & style (≥18pt everywhere) ----
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _stage_cols_in_order(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.endswith("_stor")]

def _in_col(stage: str) -> str:  return f"in_{stage}"
def _out_col(stage: str) -> str: return f"out_{stage}"

def _transition_score(row: pd.Series, stages: List[str]) -> int:
    """LTL: ONLY boundary I/O presence (+1 for each nonzero in_<st> and out_<st>)."""
    score = 0
    for st in stages:
        ci, co = _in_col(st), _out_col(st)
        if ci in row.index:
            try:
                if float(row[ci]) != 0: score += 1
            except Exception:
                pass
        if co in row.index:
            try:
                if float(row[co]) != 0: score += 1
            except Exception:
                pass
    return score

def generate_policy_six_subplots_from_sens(
    wf_name: str,
    scale: int,
    major_every: int = 10,
    seed: int = 42,           # kept for CLI compatibility; Random is disabled
    out_format: str = "pdf",  # "png" or "pdf"
):
    # ---- Paths ----
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))
    out_path = os.path.join(out_dir, f"{prefix}_policy6_subplots.{out_format.lower()}")

    # ---- Load ----
    df = pd.read_csv(sens_csv)

    stor_cols = _stage_cols_in_order(df)
    if not stor_cols:
        raise ValueError("No '*_stor' columns in workflow_rowwise_sensitivities.csv")
    for c in stor_cols:
        df[c] = _norm_str(df[c])

    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df.get("total", np.nan), errors="coerce")

    df = df[df["nodes"] == scale].copy()
    if df.empty:
        raise ValueError(f"No rows for nodes == {scale}")

    stages = [c[:-5] for c in stor_cols]  # keep file order

    # FSF counts
    df["tmpfs_count"] = sum((df[f"{st}_stor"] == "tmpfs").astype(int) for st in stages)
    df["ssd_count"]   = sum((df[f"{st}_stor"] == "ssd").astype(int)   for st in stages)

    # LTL
    df["transition_score"] = [_transition_score(row, stages) for _, row in df.iterrows()]

    # Hybrid components
    df["FSF_score"] = 2.0 * df["tmpfs_count"] + 1.0 * df["ssd_count"]
    df["LTL_score"] = df["transition_score"].astype(float)

    def _minmax(col: pd.Series) -> pd.Series:
        vmin, vmax = float(col.min()), float(col.max())
        if vmax == vmin:
            return pd.Series(np.zeros(len(col)), index=col.index)
        return (col - vmin) / (vmax - vmin)

    df["FSF_norm"] = _minmax(df["FSF_score"])   # higher is better
    df["LTL_norm"] = _minmax(df["LTL_score"])   # higher = worse
    df["Hybrid"]   = df["FSF_norm"] + (1.0 - df["LTL_norm"])

    # ---- Orders to plot (4) ----
    orders = {}
    orders["Fastest-Storage First (FSF)"] = (
        df.sort_values(["tmpfs_count", "ssd_count"], ascending=[False, False]).index.tolist()
    )
    orders["Low-Transition Layout (LTL)"] = (
        df.sort_values(["transition_score"], ascending=[True]).index.tolist()
    )
    orders["Hybrid Heuristic"] = df.sort_values(["Hybrid"], ascending=[False]).index.tolist()

    if "region" not in df.columns:
        raise ValueError("Column 'region' not in sensitivities file.")
    def _region_key(val):
        try:    return (0, float(val))
        except: return (1, str(val))
    orders["QoSFlow"] = (
        df.sort_values(["region", "total"], key=lambda s: s.map(_region_key), ascending=[True, True]).index.tolist()
    )

    # ---- Shared Y range ----
    ymin, ymax = float(df["total"].min()), float(df["total"].max())
    pad = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # ---- Plot (tight layout) ----
    # Use constrained_layout to trim gutters nicely, no manual tight_layout needed.
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9), sharey=True, constrained_layout=True
    )
    axes = axes.ravel()
    names = list(orders.keys())

    for ax, name in zip(axes, names):
        idx = orders[name]
        x = np.arange(1, len(idx) + 1)
        y = df.loc[idx, "total"].values
        ax.scatter(x, y, s=36)
        ax.xaxis.set_major_locator(MultipleLocator(major_every))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_title(name)

    # Labels
    axes[0].set_ylabel("Makespan (seconds)")
    axes[2].set_ylabel("Makespan (seconds)")
    axes[2].set_xlabel("Configuration #")
    axes[3].set_xlabel("Configuration #")

    # Save
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved:", out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True, help="Workflow name, e.g. 1kgenome")
    ap.add_argument("--scale", type=int, required=True, help="Single node scale (e.g., 10)")
    ap.add_argument("--xtick-major", type=int, default=10, help="Major tick spacing for X axis")
    ap.add_argument("--seed", type=int, default=42, help="Kept for compatibility; Random is disabled")
    ap.add_argument("--out-format", choices=["png", "pdf"], default="pdf",
                    help="Output format for the figure (default: pdf)")
    args = ap.parse_args()

    generate_policy_six_subplots_from_sens(
        wf_name=args.wf,
        scale=args.scale,
        major_every=args.xtick_major,
        seed=args.seed,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    main()
