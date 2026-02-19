#!/usr/bin/env python3
"""
Region clusters at a SINGLE node scale with median line.
X = region (renumbered per scale), Y = makespan.

Reads:
  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Behavior:
  - Filter to a SINGLE node scale (--scale).
  - Renumber regions present at that scale to 0..R-1 (ascending original IDs;
    numeric-first, then lexicographic).
  - Scatter all configs (slight X jitter for visibility).
  - Overlay a dashed line connecting per-region medians.
  - Output format controlled by --out-format {png,pdf,both}.

Outputs:
  ../<wf_name>/sens_out/<wf_name>_region_clusters_scale<scale>.{png|pdf}
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

# ---- Global font & style (≥18pt everywhere) ----
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

def _region_sort_key(x):
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))

def generate_region_cluster_plot(
    wf_name: str,
    scale: int,
    jitter: float = 0.18,
    seed: int = 42,
    out_format: str = "pdf",  # "png", "pdf", or "both"
):
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))

    df = pd.read_csv(sens_csv)
    if "region" not in df.columns:
        raise ValueError("Column 'region' not found in sensitivities CSV.")

    df["nodes"] = pd.to_numeric(df.get("nodes", np.nan), errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df.get("total", np.nan), errors="coerce")

    # Filter to single scale
    df = df[df["nodes"] == scale].copy()
    if df.empty:
        raise ValueError(f"No rows for nodes == {scale}")

    # Renumber regions present at this scale to 0..R-1 by ascending original IDs
    unique_regions_sorted = sorted(df["region"].unique().tolist(), key=_region_sort_key)
    rmap = {orig: i for i, orig in enumerate(unique_regions_sorted)}
    df["region_mapped"] = df["region"].map(rmap).astype(int)

    # Build jittered X
    rng = np.random.default_rng(seed)
    x_base = df["region_mapped"].to_numpy(dtype=float)
    x = x_base + rng.uniform(-jitter, jitter, size=len(df))

    # Y padding
    ymin = float(df["total"].min())
    ymax = float(df["total"].max())
    pad  = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # Color by region (discrete)
    nreg = int(df["region_mapped"].max()) + 1
    cmap = plt.get_cmap("tab20") if nreg <= 20 else plt.get_cmap("viridis")
    norm = colors.BoundaryNorm(range(-1, nreg+1), cmap.N)

    # Plot (tight spacing)
    fig, ax = plt.subplots(figsize=(12, 6.5), constrained_layout=True)
    sc = ax.scatter(
        x, df["total"].to_numpy(),
        c=df["region_mapped"], cmap=cmap, norm=norm,
        s=90, alpha=0.9, linewidths=0.5, edgecolors="black"   # bigger dots
    )

    # Per-region median line
    med = df.groupby("region_mapped")["total"].median().sort_index()
    line_handle = ax.plot(
        med.index.values, med.values, linestyle="--", linewidth=2.5,
        color="tab:blue", marker="o", markersize=6, label="Region median", zorder=3
    )[0]

    # Ticks at integer region ids
    ax.set_xlim(-0.6, nreg - 1 + 0.6)
    ax.set_xticks(range(nreg))
    ax.set_xticklabels([str(i) for i in range(nreg)], rotation=0)
    ax.set_xlabel("Region #")
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Makespan (seconds)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Colorbar for region IDs
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_ticks(range(nreg))
    cbar.set_ticklabels([str(i) for i in range(nreg)])
    cbar.set_label("Region")

    # Legend: what a dot means + median line
    dot_handle = Line2D([], [], marker='o', linestyle='None', markersize=9,
                        markerfacecolor='gray', markeredgecolor='black', markeredgewidth=0.6,
                        label='= Configuration')
    ax.legend(handles=[dot_handle, line_handle], loc="upper left", frameon=False)

    # Save (png/pdf/both)
    out_base = os.path.join(out_dir, f"{prefix}_region_clusters_scale{scale}")
    fmt = out_format.lower()
    if fmt in ("png", "both"):
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        print("Saved:", out_base + ".png")
    if fmt in ("pdf", "both"):
        fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
        print("Saved:", out_base + ".pdf")

def main():
    p = argparse.ArgumentParser(description="Plot makespan clusters per region (single scale) with median line.")
    p.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    p.add_argument("--scale", type=int, required=True, help="Single node scale to visualize, e.g., 10")
    p.add_argument("--jitter", type=float, default=0.18, help="Horizontal jitter (default: 0.18)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for jitter (default: 42)")
    p.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                   help="Output format for the figure (default: pdf)")
    args = p.parse_args()

    generate_region_cluster_plot(
        wf_name=args.wf,
        scale=args.scale,
        jitter=args.jitter,
        seed=args.seed,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    main()
