#!/usr/bin/env python3
"""
Total-cost (Makespan) comparison subplots across node scales.

Reads:
  ../<wf_name>/workflow_makespan_stageorder.csv

Behavior:
  - Build configuration labels from *_stor columns (deterministic order).
  - Filter to desired node scales.
  - For each scale, sort configurations by makespan ascending (per subplot).
  - Plot one subplot per scale (shared Y), X as configuration index (#).
  - Adjustable X tick density via --xtick-major.

Outputs (controlled by --out-format {png,pdf,both}):
  ../<wf_name>/sens_out/<wf_name>_total_cost_subplots.{png|pdf}
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import Patch

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

def _stor_cols(df: pd.DataFrame) -> list[str]:
    return sorted([c for c in df.columns if c.endswith("_stor")])

def _make_config_label(row: pd.Series, scols: list[str]) -> str:
    parts = []
    for c in scols:
        parts.append(f"{c[:-5]}:{row[c]}")
    return " | ".join(parts)

def generate_total_cost_subplots(
    wf_name: str,
    desired_scales: list[int],
    major_every: int = 10,
    out_format: str = "pdf",  # "png", "pdf", or "both"
):
    # ---- Paths ----
    csv_path = os.path.join("..", wf_name, "workflow_makespan_stageorder.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))
    out_base = os.path.join(out_dir, f"{prefix}_total_cost_subplots")

    # ---- Load ----
    df = pd.read_csv(csv_path)
    if "nodes" not in df.columns or "total" not in df.columns:
        raise ValueError("CSV must include 'nodes' and 'total' columns.")

    scols = _stor_cols(df)
    if not scols:
        raise ValueError("No '*_stor' columns found to define configurations.")

    # Normalize storage values for stable labels
    for c in scols:
        df[c] = _norm_str(df[c])

    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    # Configuration label
    df = df.copy()
    df["configuration"] = df.apply(lambda r: _make_config_label(r, scols), axis=1)

    # Keep only requested scales that exist
    desired_scales = [int(s) for s in desired_scales]
    present = [s for s in desired_scales if s in set(df["nodes"].dropna().unique().tolist())]
    if not present:
        raise ValueError("None of the desired scales are present in the CSV.")

    # Aggregate (mean) in case of duplicates
    agg = (df[df["nodes"].isin(present)]
           .groupby(["configuration", "nodes"], as_index=False)["total"].mean())

    # Prepare plot data per scale (sorted ascending by makespan per subplot)
    scales_present = [s for s in present if s in agg["nodes"].unique()]
    per_scale = {}
    all_vals = []
    for s in scales_present:
        s_df = agg[agg["nodes"] == s].sort_values("total", ascending=True).reset_index(drop=True)
        s_df["index_in_subplot"] = np.arange(1, len(s_df) + 1)
        per_scale[s] = s_df
        all_vals.extend(s_df["total"].tolist())

    if not all_vals:
        raise ValueError("No totals available to plot.")

    # Shared Y range
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    pad  = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # ---- Plot ----
    n = len(scales_present)
    fig, axes = plt.subplots(1, n, figsize=(5.6*n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]

    legend_proxy = [Patch(label="Makespan")]

    for ax, s in zip(axes, scales_present):
        s_df = per_scale[s]
        x = s_df["index_in_subplot"].to_numpy()
        y = s_df["total"].to_numpy()

        ax.bar(x, y)
        # X ticks: every Nth
        step = max(1, int(major_every))
        tick_positions = np.arange(1, len(x) + 1, step=step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions], rotation=0)

        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(handles=legend_proxy, loc="upper left", frameon=False)
        ax.set_title(f"Nodes = {s}")

    axes[0].set_ylabel("Makespan")
    fig.supxlabel("Configuration (#)")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.92, wspace=0.18)

    # ---- Save ----
    fmt = out_format.lower()
    if fmt in ("png", "both"):
        png_path = out_base + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print("Saved:", png_path)
    if fmt in ("pdf", "both"):
        pdf_path = out_base + ".pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print("Saved:", pdf_path)

def main():
    p = argparse.ArgumentParser(description="Compare makespan across node scales (subplots).")
    p.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    p.add_argument("--scales", required=True, help="Comma-separated node scales, e.g., 2,5,10")
    p.add_argument("--xtick-major", type=int, default=10, help="Show every Nth configuration index on X (default: 10)")
    p.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                   help="Output format for the figure (default: pdf)")
    args = p.parse_args()

    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    generate_total_cost_subplots(
        wf_name=args.wf,
        desired_scales=scales,
        major_every=args.xtick_major,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    main()
