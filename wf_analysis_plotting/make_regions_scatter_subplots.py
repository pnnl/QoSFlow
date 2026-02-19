#!/usr/bin/env python3
"""
Regions scatter subplots by node scale (colored by REGION with smooth gradient).

Reads:
  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Behavior:
  - For each requested node scale, filter rows.
  - Within each scale's subplot, sort rows by makespan (total) ascending and plot:
      X = rank index (1..N), Y = total
      Color = region (continuous gradient via viridis; numeric-first, then lexicographic mapping)
  - Show every Nth x tick via --xtick-major (default: 10).
  - Y-axis label: "Makespan"
  - X-axis label: "Configuration (#)"
  - Output format controlled by --out-format {png, pdf, both}.

Outputs:
  ../<wf_name>/sens_out/<wf_name>_regions_scatter_subplots.{png|pdf}
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter

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
    # Deterministic order for reproducible labels if needed
    return sorted([c for c in df.columns if c.endswith("_stor")])

def _region_sort_key(x):
    """Sort regions numeric-first then lexicographic to avoid 1,10,11,... issues."""
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))

def generate_regions_scatter_subplots(
    wf_name: str,
    desired_scales: list[int],
    major_every: int = 10,
    out_format: str = "pdf",  # "png", "pdf", or "both"
):
    # ---- Paths ----
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))
    out_base = os.path.join(out_dir, f"{prefix}_regions_scatter_subplots")

    # ---- Load ----
    df = pd.read_csv(sens_csv)
    if "nodes" not in df.columns or "total" not in df.columns or "region" not in df.columns:
        raise ValueError("CSV must include 'nodes', 'total', and 'region' columns.")

    # Normalize types
    scols = _stor_cols(df)  # not used for plotting, but keep normalization consistent
    for c in scols:
        df[c] = _norm_str(df[c])
    df["nodes"]  = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    df["total"]  = pd.to_numeric(df["total"], errors="coerce")
    df["region"] = df["region"].astype(str)

    # Keep only requested scales that exist
    desired_scales = [int(s) for s in desired_scales]
    present = [s for s in desired_scales if s in set(df["nodes"].dropna().unique().tolist())]
    if not present:
        raise ValueError("None of the desired scales are present in the CSV.")

    # Build region→numeric mapping once (numeric-first, then lexicographic)
    df_plot = df[df["nodes"].isin(present)].copy()
    all_regions_used = sorted(df_plot["region"].unique().tolist(), key=_region_sort_key)
    region_to_code = {r: i for i, r in enumerate(all_regions_used)}
    code_to_region = {i: r for r, i in region_to_code.items()}
    nreg = len(all_regions_used)
    vmin, vmax = 0, max(0, nreg - 1)

    # Continuous colormap & normalization (smooth transition)
    cmap = plt.get_cmap("viridis")
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Build per-scale sorted frames
    per_scale = {}
    all_vals = []
    for s in present:
        sdf = df_plot[df_plot["nodes"] == s].copy()
        if sdf.empty:
            continue
        # Sort by total ascending within the subplot
        sdf = sdf.sort_values("total", ascending=True).reset_index(drop=True)
        sdf["x_idx"] = np.arange(1, len(sdf) + 1)
        sdf["region_code"] = sdf["region"].map(region_to_code).astype(int)
        per_scale[s] = sdf
        all_vals.extend(sdf["total"].tolist())

    if not all_vals:
        raise ValueError("No data to plot after filtering.")

    # Shared Y range
    ymin = float(np.nanmin(all_vals))
    ymax = float(np.nanmax(all_vals))
    pad  = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # ---- Plot ----
    n = len(per_scale)
    fig, axes = plt.subplots(1, n, figsize=(5.8*n, 5.2), sharey=True)
    axes = np.atleast_1d(axes).ravel().tolist()

    last_sc = None
    for ax, s in zip(axes, present):
        sdf = per_scale.get(s, None)
        if sdf is None or sdf.empty:
            ax.text(0.5, 0.5, f"No data for nodes={s}", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        x = sdf["x_idx"].to_numpy()
        y = sdf["total"].to_numpy()
        c = sdf["region_code"].to_numpy()

        # Color by REGION code with smooth colormap
        last_sc = ax.scatter(
            x, y, c=c, cmap=cmap, norm=norm,
            s=50, edgecolors="black", linewidths=0.5
        )

        # X ticks: every Nth index
        step = max(1, int(major_every))
        tick_positions = np.arange(1, len(x) + 1, step=step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([str(int(t)) for t in tick_positions], rotation=0)

        ax.set_ylim(ylo, yhi)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if wf_name == "ddmd":
            ax.set_title(f"GPUs = {s*6}")
        else:
            ax.set_title(f"Nodes = {s}")

    # Labels
    axes[0].set_ylabel("Makespan (seconds)")
    fig.supxlabel("Configuration (#)")

    # Continuous colorbar with integer region ticks
    if last_sc is not None:
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(mappable, ax=axes, orientation="vertical", fraction=0.035, pad=0.02)
        cbar.ax.yaxis.set_major_locator(MultipleLocator(1))
        cbar.ax.yaxis.set_major_formatter(
            FuncFormatter(lambda val, pos: str(code_to_region.get(int(round(val)), "")))
        )
        cbar.set_label("Region")

    # Layout
    fig.subplots_adjust(wspace=0.05, left=0.07, right=0.86, bottom=0.20, top=0.92)

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
    p = argparse.ArgumentParser(description="Scatter subplots of regions per node scale (Makespan vs configuration index).")
    p.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    p.add_argument("--scales", required=True, help="Comma-separated node scales, e.g., 2,5,10")
    p.add_argument("--xtick-major", type=int, default=10, help="Show every Nth configuration index on X (default: 10)")
    p.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                   help="Output format for the figure (default: pdf)")
    args = p.parse_args()

    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    generate_regions_scatter_subplots(
        wf_name=args.wf,
        desired_scales=scales,
        major_every=args.xtick_major,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    main()
