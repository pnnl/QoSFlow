#!/usr/bin/env python3
"""
Total cost overlay bar chart driven by sensitivities, ordered by a target scale.

Reads:
  ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv

Behavior:
  - Build configuration labels from the *_stor columns (deterministic order).
  - Filter to desired scales.
  - Determine configuration ordering from the TARGET scale by ascending total.
  - Plot a SINGLE axes:
      * First, draw all NON-target scales as red bars (one pass per scale).
      * Then overlay the TARGET scale bars on top with transparency so the
        red bars remain visible beneath.
  - Horizontal grid, shared Y (across all scales by construction), numeric
    configuration indices on X with adjustable major tick step.

Outputs (controlled by --out-format {png,pdf,both}):
  ../<wf_name>/sens_out/<wf_name>_sens_totals_target<target_scale>.{png|pdf}
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

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _stage_stor_cols(df: pd.DataFrame) -> list[str]:
    # Deterministic order: alphabetical over *_stor columns
    cols = [c for c in df.columns if c.endswith("_stor")]
    return sorted(cols)

def _make_config_label(row: pd.Series, stor_cols: list[str]) -> str:
    # stage label uses column name minus '_stor'
    parts = []
    for c in stor_cols:
        parts.append(f"{c[:-5]}:{row[c]}")
    return " | ".join(parts)

def generate_total_cost_subplots_from_sens(
    wf_name: str,
    desired_scales: list[int],
    target_scale: int,
    major_every: int = 10,
    out_format: str = "pdf",  # "png", "pdf", or "both"
):
    # --- Paths ---
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix   = os.path.basename(os.path.normpath(wf_name))
    out_base = os.path.join(out_dir, f"{prefix}_sens_totals_target{target_scale}")

    # --- Load & normalize ---
    df = pd.read_csv(sens_csv)
    if "nodes" not in df.columns or "total" not in df.columns:
        raise ValueError("CSV must include 'nodes' and 'total' columns.")

    stor_cols = _stage_stor_cols(df)
    if not stor_cols:
        raise ValueError("No '*_stor' columns found to define configurations.")

    for c in stor_cols:
        df[c] = _norm_str(df[c])

    df["nodes"] = pd.to_numeric(df["nodes"], errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df["total"], errors="coerce")

    # Build configuration label
    df = df.copy()
    df["configuration"] = df.apply(lambda r: _make_config_label(r, stor_cols), axis=1)

    # Filter to desired scales that actually exist
    desired_scales = list(dict.fromkeys(int(s) for s in desired_scales))
    present = [s for s in desired_scales if s in set(df["nodes"].dropna().unique().tolist())]
    if not present:
        raise ValueError("None of the desired scales are present in the CSV.")
    if target_scale not in present:
        raise ValueError(f"target_scale={target_scale} not found among present scales: {present}")

    # Aggregate (mean) per configuration & scale (in case of duplicates)
    agg = (df[df["nodes"].isin(present)]
           .groupby(["configuration", "nodes"], as_index=False)["total"].mean())

    # Determine order from target scale (ascending total)
    tgt = agg[agg["nodes"] == target_scale].copy()
    if tgt.empty:
        raise ValueError(f"No rows for target_scale={target_scale}.")
    tgt = tgt.sort_values("total", ascending=True)
    config_order = tgt["configuration"].tolist()

    # Align each scale to the target config order
    scales_present = [s for s in present if s in agg["nodes"].unique()]
    plot_data = {}
    global_vals = []
    for s in scales_present:
        s_vals = (agg[agg["nodes"] == s]
                  .set_index("configuration")["total"]
                  .reindex(config_order))  # keep target order
        plot_data[s] = s_vals
        global_vals.extend(s_vals.dropna().tolist())

    if not global_vals:
        raise ValueError("No totals available to plot after reindexing.")

    # Shared Y with padding
    ymin = float(np.nanmin(global_vals))
    ymax = float(np.nanmax(global_vals))
    pad  = 0.08 * max(1.0, ymax - ymin)
    ylo, yhi = ymin - pad, ymax + pad

    # --- Plot: Overlay bars ---
    fig_width = max(8.0, 0.12 * len(config_order) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    Ncfg = len(config_order)
    x_all = np.arange(1, Ncfg + 1, dtype=float)
    bar_width = 0.85

    # Draw NON-target scales first (red), so target overlays them
    other_scales = [s for s in scales_present if s != target_scale]
    for s in other_scales:
        vals = plot_data[s].values
        ax.bar(
            x_all, vals,
            width=bar_width,
            color="red",
            alpha=0.85,
            edgecolor="none",
            label=f"Nodes = {s}",
            zorder=1,
        )

    # Draw TARGET scale on top with transparency
    vals_tgt = plot_data[target_scale].values
    ax.bar(
        x_all, vals_tgt,
        width=bar_width,
        color="blue",
        alpha=0.9,          # transparent enough to see red underneath
        edgecolor="black",
        linewidth=0.4,
        label=f"Nodes = {target_scale}",
        zorder=2,
    )

    # X ticks: show every Nth configuration index
    step = max(1, int(major_every))
    tick_positions = np.arange(1, Ncfg + 1, step=step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(int(t)) for t in tick_positions], rotation=0)

    ax.set_ylim(ylo, yhi)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylabel("Makespan (seconds)")
    ax.set_xlabel("Configuration (#)")

    # Deduplicate legend labels while preserving order
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            uniq_handles.append(h)
            uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, loc="upper left", frameon=False, ncol=1)

    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.14, top=0.94)

    # Save according to out_format
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
    p = argparse.ArgumentParser(description="Overlay total makespan bars from sensitivities, ordered by a target scale.")
    p.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    p.add_argument("--scales", required=True, help="Comma-separated node scales to plot, e.g., 2,5,10")
    p.add_argument("--target-scale", type=int, required=True, help="Scale whose ordering to apply to all bars")
    p.add_argument("--xtick-major", type=int, default=10, help="Show every Nth configuration index on X (default: 10)")
    p.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                   help="Output format for the figure (default: pdf)")
    args = p.parse_args()

    scales = [int(x.strip()) for x in args.scales.split(",") if x.strip()]
    generate_total_cost_subplots_from_sens(
        wf_name=args.wf,
        desired_scales=scales,
        target_scale=args.target_scale,
        major_every=args.xtick_major,
        out_format=args.out_format,
    )

if __name__ == "__main__":
    main()
