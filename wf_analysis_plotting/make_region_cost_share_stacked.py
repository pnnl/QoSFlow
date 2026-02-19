#!/usr/bin/env python3
import argparse
import os
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

COLOR_SHARED = "#1f77b4"  # blue
COLOR_LOCAL  = "#ff7f0e"  # orange
COLOR_MOVE   = "#2ca02c"  # green
COLOR_LINE   = "#d62728"  # red dashed line

import matplotlib as mpl  # add if not already imported

# ---- Global font & style (≥18pt everywhere) ----
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def _stor_class(val: str) -> str:
    s = str(val).strip().lower()
    return "shared" if s == "beegfs" else "local"

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _build_cfg(df: pd.DataFrame, cols: List[str]) -> pd.Series:
    parts=[]
    for c in sorted(cols):
        parts.append(_norm_str(df[c]).map(lambda v: f"{c[:-5]}:{v}"))
    return pd.Series([" | ".join(v) for v in zip(*parts)], index=df.index)

def generate_region_cost_share_stacked(wf_name: str, desired_scales: List[int], major_every: int = 10, out_format: str = "pdf"):
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    cfg_csv  = os.path.join("..", wf_name, "workflow_makespan_stageorder.csv")
    out_dir  = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.basename(os.path.normpath(wf_name))
    out_png = os.path.join(out_dir, f"{prefix}_region_shares_stacked")

    sdf = pd.read_csv(sens_csv)
    cdf = pd.read_csv(cfg_csv)

    stor_cols_s = sorted([c for c in sdf.columns if c.endswith("_stor")])
    stor_cols_c = sorted([c for c in cdf.columns if c.endswith("_stor")])
    stor_cols   = [c for c in stor_cols_s if c in stor_cols_c]
    if not stor_cols:
        raise ValueError("No overlapping '*_stor' columns between sensitivities and configs.")

    for c in stor_cols:
        sdf[c] = _norm_str(sdf[c])
        cdf[c] = _norm_str(cdf[c])

    sdf["configuration"] = _build_cfg(sdf, stor_cols)
    cdf["configuration"] = _build_cfg(cdf, stor_cols)

    sdf["nodes"] = pd.to_numeric(sdf["nodes"], errors="coerce").astype("Int64")
    cdf["nodes"] = pd.to_numeric(cdf["nodes"], errors="coerce").astype("Int64")
    cdf["total"] = _to_num(cdf.get("total", 0.0))
    cdf["critical_path"] = _to_num(cdf.get("critical_path", 0.0))

    stages = [c[:-5] for c in stor_cols_c]
    read  = {st: _to_num(cdf.get(f"read_{st}", 0.0))  for st in stages}
    write = {st: _to_num(cdf.get(f"write_{st}", 0.0)) for st in stages}
    sin   = {st: _to_num(cdf.get(f"in_{st}", 0.0))    for st in stages}
    sout  = {st: _to_num(cdf.get(f"out_{st}", 0.0))   for st in stages}

    shared_exec = pd.Series(0.0, index=cdf.index)
    local_exec  = pd.Series(0.0, index=cdf.index)
    movement    = pd.Series(0.0, index=cdf.index)

    for st, stor_col in zip(stages, stor_cols_c):
        kind = cdf[stor_col].map(_stor_class)
        exec_cost = read[st] + write[st]
        move_cost = sin[st] + sout[st]
        movement += move_cost
        shared_exec += exec_cost.where(kind.eq("shared"), 0.0)
        local_exec  += exec_cost.where(kind.eq("local"),  0.0)

    cdf["shared_exec"] = shared_exec
    cdf["local_exec"]  = local_exec
    cdf["movement"]    = movement

    joined = pd.merge(
        sdf[["region","nodes","configuration"]],
        cdf[["nodes","configuration","total","critical_path","shared_exec","local_exec","movement"]],
        on=["nodes","configuration"], how="inner"
    )

    denom = joined["critical_path"].copy()
    fallback = denom <= 0
    denom.loc[fallback] = (joined.loc[fallback, "shared_exec"]
                           + joined.loc[fallback, "local_exec"]
                           + joined.loc[fallback, "movement"])
    denom = denom.replace(0, np.nan)
    joined["s_shared"] = (joined["shared_exec"]/denom).fillna(0.0)
    joined["s_local"]  = (joined["local_exec"] /denom).fillna(0.0)
    joined["s_move"]   = (joined["movement"]   /denom).fillna(0.0)

    agg = joined.groupby(["nodes","region"], as_index=False).agg(
        mean_shared=("s_shared","mean"),
        mean_local =("s_local","mean"),
        mean_move  =("s_move","mean"),
        avg_total  =("total","mean"),
    )
    ssum = (agg["mean_shared"]+agg["mean_local"]+agg["mean_move"]).replace(0,np.nan)
    agg["mean_shared"] = (agg["mean_shared"]/ssum).fillna(0.0)
    agg["mean_local"]  = (agg["mean_local"] /ssum).fillna(0.0)
    agg["mean_move"]   = (agg["mean_move"]  /ssum).fillna(0.0)

    present = [s for s in desired_scales if s in agg["nodes"].dropna().unique().tolist()]
    scales_present = present if present else sorted(agg["nodes"].dropna().unique().tolist())
    per_scale: Dict[int, pd.DataFrame] = {}
    for s in scales_present:
        d = agg[agg["nodes"]==s].copy()
        # sort by avg_total asc
        try:
            d["_rnum"] = pd.to_numeric(d["region"], errors="coerce")
            d = d.sort_values(["avg_total","_rnum"]).drop(columns=["_rnum"])
        except Exception:
            d = d.sort_values(["avg_total","region"])
        per_scale[s] = d.reset_index(drop=True)

    # global y2
    all_tot = pd.concat([per_scale[s]["avg_total"] for s in scales_present], ignore_index=True)
    y2_min, y2_max = 0.0, 1.05*float(all_tot.max()) if not all_tot.empty else (0.0, 1.0)

    fig, axes = plt.subplots(1, len(scales_present), figsize=(6.6*len(scales_present), 5.2), sharey=True)
    if len(scales_present)==1:
        axes=[axes]

    handles=None; labels=None
    end_ax = axes[-1]
    for ax, s in zip(axes, scales_present):
        d = per_scale[s]
        x = np.arange(len(d))
        b1 = ax.bar(x, d["mean_shared"].values, width=0.8, color=COLOR_SHARED, label="Shared exec")
        b2 = ax.bar(x, d["mean_local"].values, width=0.8,
                    bottom=d["mean_shared"].values, color=COLOR_LOCAL, label="Local exec")
        bottom2 = d["mean_shared"].values + d["mean_local"].values
        b3 = ax.bar(x, d["mean_move"].values, width=0.8,
                    bottom=bottom2, color=COLOR_MOVE, label="Data movement")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Region #")
        ax.set_xticks(x)
        ax.set_xticklabels(d["region"].astype(str).tolist(), rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if wf_name == "ddmd":
            ax.set_title(f"GPUs = {s*6}")
        else:
            ax.set_title(f"Nodes = {s}")

        ax2 = ax.twinx()
        (line,) = ax2.plot(x, d["avg_total"].values, linestyle="--", color=COLOR_LINE,
                           marker="o", linewidth=2, markersize=4, label="Avg total makespan")
        ax2.set_ylim(y2_min, y2_max)
        if end_ax == ax:
            ax2.set_ylabel("Average Makespan")

        if handles is None:
            handles = [b1,b2,b3,line]
            labels  = ["Shared storage I/O","Local storage I/O","Data movement","Avg Makespan"]

    axes[0].set_ylabel("Makespan Share")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.05), borderaxespad=0.0, columnspacing=1.8, handlelength=3.0)
    fig.subplots_adjust(wspace=0.20, left=0.07, right=0.93, bottom=0.18, top=0.88)

    # ---- Save ----
    fmt = out_format.lower()
    if fmt in ("png", "both"):
        png_path = out_png + ".png"
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print("Saved:", png_path)
    if fmt in ("pdf", "both"):
        pdf_path = out_png + ".pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print("Saved:", pdf_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wf", required=True)
    ap.add_argument("--scales", required=True, help="e.g. 2,5,10")
    ap.add_argument("--xtick-major", type=int, default=10)  # not used here, kept for API symmetry
    ap.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                   help="Output format for the figure (default: pdf)")
    a = ap.parse_args()
    scales = [int(x.strip()) for x in a.scales.split(",") if x.strip()]
    generate_region_cost_share_stacked(wf_name=a.wf, desired_scales=scales, major_every=a.xtick_major, out_format=a.out_format)

if __name__ == "__main__":
    main()
