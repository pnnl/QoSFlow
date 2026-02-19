#!/usr/bin/env python3
"""
Region clusters across MULTIPLE node scales.
X = region (original labels), Y = makespan (total), color = node scale.

Reads:
  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Behavior:
  - Filter to a SET of node scales (--scales).
  - DO NOT renumber regions; keep original region labels on the X axis.
    (We place categories at x=0..R-1 internally but keep tick labels as originals.)
  - Color points by 'nodes' (scale). No median line.
  - Slight X jitter so per-region clusters are visible.

Output:
  ../<wf_name>/sens_out/<wf_name>_region_clusters_multiscale_scales<joined>.png
  (joined = scales joined with '-'; e.g., scales2-5-10)
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

def _region_sort_key(x):
    # numeric-first sort; fallback to string
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))

def _pick_scale_colors(scales):
    """Return {scale: color} with distinct colors for however many scales are requested."""
    scales_sorted = sorted(scales)
    n = len(scales_sorted)
    # Prefer tab10/tab20; if more than 20 scales, sample a continuous map
    if n <= 10:
        cmap = plt.get_cmap("tab10")
        return {s: cmap(i) for i, s in enumerate(scales_sorted)}
    elif n <= 20:
        cmap = plt.get_cmap("tab20")
        return {s: cmap(i) for i, s in enumerate(scales_sorted)}
    else:
        cmap = plt.get_cmap("nipy_spectral")
        return {s: cmap(i / max(1, n - 1)) for i, s in enumerate(scales_sorted)}

def generate_region_cluster_multiscale_plot(
    wf_name: str,
    scales: list[int],
    jitter: float = 0.18,
    seed: int = 42,
):
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))
    scales_tag = "-".join(str(s) for s in sorted(scales))
    out_png  = os.path.join(out_dir, f"{prefix}_region_clusters_multiscale_scales{scales_tag}.png")

    df = pd.read_csv(sens_csv)
    if "region" not in df.columns:
        raise ValueError("Column 'region' not found in sensitivities CSV.")

    # Basic coercions
    df["nodes"] = pd.to_numeric(df.get("nodes", np.nan), errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df.get("total", np.nan), errors="coerce")

    # Filter to requested scales that actually exist
    present_scales = sorted(x for x in set(scales) if x in set(df["nodes"].dropna().unique().tolist()))
    if not present_scales:
        raise ValueError(f"No rows found for requested scales: {scales}")
    d = df[df["nodes"].isin(present_scales)].copy()

    # Build category positions for original region labels (NOT renumbered for labeling)
    unique_regions = sorted(d["region"].dropna().unique().tolist(), key=_region_sort_key)
    region_to_x = {r: i for i, r in enumerate(unique_regions)}
    x_base = d["region"].map(region_to_x).to_numpy(dtype=float)

    # Jitter for visibility
    rng = np.random.default_rng(seed)
    x = x_base + rng.uniform(-jitter, jitter, size=len(d))

    # Colors by scale
    scale_to_color = _pick_scale_colors(present_scales)
    point_colors = d["nodes"].map(scale_to_color).to_list()

    # Y range with padding
    ymin = float(d["total"].min())
    ymax = float(d["total"].max())
    pad  = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x, d["total"].to_numpy(), c=point_colors, s=24, alpha=0.9, linewidths=0.2, edgecolors="black")

    # X ticks show original region labels
    ax.set_xlim(-0.6, len(unique_regions) - 1 + 0.6)
    ax.set_xticks(range(len(unique_regions)))
    ax.set_xticklabels([str(r) for r in unique_regions], rotation=0)
    ax.set_xlabel("Region")
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Makespan (total)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Legend by scale
    handles = []
    labels  = []
    for s in present_scales:
        dot = plt.Line2D([], [], marker='o', linestyle='None', markersize=7,
                         markerfacecolor=scale_to_color[s], markeredgecolor='black', markeredgewidth=0.3)
        handles.append(dot)
        labels.append(f"nodes = {s}")
    ax.legend(handles, labels, title="Scale", loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.14, top=0.93)
    fig.suptitle(f"Region clusters across scales: {', '.join(map(str, present_scales))}", y=0.98, fontsize=12)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Saved:", out_png)

def main():
    p = argparse.ArgumentParser(description="Region cluster plot across multiple scales (color by scale).")
    p.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    p.add_argument("--scales", required=True, help="Comma-separated node scales, e.g., 2,5,10")
    p.add_argument("--jitter", type=float, default=0.18, help="Horizontal jitter (default: 0.18)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for jitter (default: 42)")
    args = p.parse_args()

    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    generate_region_cluster_multiscale_plot(
        wf_name=args.wf,
        scales=scales,
        jitter=args.jitter,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()
