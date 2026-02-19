#!/usr/bin/env python3
"""
DAG-of-Glyphs: Region Signature Panels (single row)

Reads:
  - ../<wf_name>/sens_out/workflow_rowwise_sensitivities.csv
  - ../<wf_name>/*_script_order.json

JSON support:
  1) List-of-lists format: {"levels": [["s1"], ["s2","s3"], ...], "edges": [["s1","s2"], ...]}  (edges optional)
  2) Stage-map format (e.g., 1kg_script_order.json):
         {
           "stageA": {"stage_order": 1, "predecessors": {...}},
           "stageB": {"stage_order": 2, "predecessors": {"stageA": {...}}},
           ...
         }
     Levels are built by grouping by stage_order; edges from predecessors
     whose names match known stages (ignores "initial_data", etc.).

Behavior:
  - Filter to --scale.
  - Renumber regions present at that scale to 0..R-1 (ascending original IDs; numeric-first, then lexicographic).
  - Compute per-renumbered-region median makespan ("total").
  - Select regions to show:
      * default: lowest, 25th, 50th, 75th, highest by median (if fewer than 5, show all);
      * with --top5: five best (lowest) medians.
  - Single row of panels: first = Stage DAG (labeled), then the selected region DAGs.
  - Region panels show ONLY the tri-dot glyph at each node (no big node circle):
        left dot = tmpfs, middle = ssd, right = beegfs
        a dot is filled if that medium occurs in that region for the stage;
        if all three occur, all dots filled (legend label: "don't care").
  - Directed edges; dashed grey border; larger nodes/glyphs.
  - Legend at top-center in ONE LINE with a bounding box (18pt font).
  - Output format via --out-format {png, pdf, both}.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

# ---- Global font & style (≥18pt everywhere) ----
mpl.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

# Colors for storages
COLOR_TMPFS  = "tab:blue"
COLOR_SSD    = "tab:orange"
COLOR_BEEGFS = "tab:green"
EDGE_COLOR   = (0, 0, 0, 0.25)   # subdued edges
BORDER_COLOR = (0.6, 0.6, 0.6, 0.85)

# ----------------- Helpers -----------------

def _norm_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()

def _region_sort_key(x):
    """numeric-first, then lexicographic."""
    try:
        return (0, float(x))
    except Exception:
        return (1, str(x))

def _find_script_order_json(wf_name: str) -> str:
    root = os.path.join("..", wf_name)
    matches = sorted(glob.glob(os.path.join(root, "*_script_order.json")))
    if not matches:
        raise FileNotFoundError(f"No *_script_order.json found under {root}")
    return matches[0]

def _load_stage_levels_edges(json_path: str) -> Tuple[List[List[str]], List[Tuple[str, str]]]:
    """
    Supports:
      - {"levels": [[...],[...],...], "edges":[["a","b"],...]}   (edges optional)
      - stage-map with "stage_order" and optional "predecessors" per stage
        (edges inferred from predecessors).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Case 1: list-of-lists style
    for key in ("levels", "stage_levels", "orders", "order"):
        if isinstance(data, dict) and key in data and isinstance(data[key], list) and all(isinstance(x, list) for x in data[key]):
            levels = data[key]
            edges = []
            if "edges" in data and isinstance(data["edges"], list):
                for e in data["edges"]:
                    if isinstance(e, (list, tuple)) and len(e) == 2:
                        edges.append((e[0], e[1]))
            if not edges:
                # connect adjacent levels fully
                for i in range(len(levels) - 1):
                    for u in levels[i]:
                        for v in levels[i + 1]:
                            edges.append((u, v))
            return levels, edges

    # Case 2: stage-map style
    if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
        stage_orders: Dict[str, int] = {}
        for st, meta in data.items():
            if isinstance(meta, dict) and "stage_order" in meta:
                try:
                    stage_orders[st] = int(meta["stage_order"])
                except Exception:
                    pass
        if not stage_orders:
            raise ValueError("Could not find 'stage_order' entries in stage-map JSON.")

        # levels by increasing stage_order (stable by name)
        order_vals = sorted(set(stage_orders.values()))
        levels: List[List[str]] = []
        for o in order_vals:
            group = [st for st, so in stage_orders.items() if so == o]
            group.sort()
            levels.append(group)

        # edges from predecessors that are known stages
        edges: List[Tuple[str, str]] = []
        stage_set = set(stage_orders.keys())
        for st, meta in data.items():
            preds = meta.get("predecessors", {}) if isinstance(meta, dict) else {}
            if isinstance(preds, dict):
                for pred_name in preds.keys():
                    if pred_name in stage_set:
                        edges.append((pred_name, st))
        if not edges:
            for i in range(len(levels) - 1):
                for u in levels[i]:
                    for v in levels[i + 1]:
                        edges.append((u, v))
        return levels, edges

    raise ValueError(
        "JSON must be either list-of-lists (levels) or a stage map with 'stage_order' per stage."
    )

def _get_stage_storage_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.endswith("_stor")])

def _panel_border(ax):
    rect = Rectangle((0,0), 1, 1, transform=ax.transAxes, fill=False,
                     linestyle=(0,(4,4)), linewidth=1.8, edgecolor=BORDER_COLOR)
    ax.add_patch(rect)

def _node_positions_top_to_bottom(levels: List[List[str]]) -> Dict[str, Tuple[float, float]]:
    """Place levels from top (y=0.85) to bottom (y=0.15). Spread nodes horizontally within each level."""
    nlev = len(levels)
    y_top, y_bot = 0.85, 0.15
    ys = np.linspace(y_top, y_bot, nlev)
    pos = {}
    for li, stage_list in enumerate(levels):
        n = len(stage_list)
        xs = [0.5] if n == 1 else np.linspace(0.2, 0.8, n)
        for x, st in zip(xs, stage_list):
            pos[st] = (x, ys[li])
    return pos

def _draw_directed_edges(ax, edges, pos):
    for u, v in edges:
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                arrowstyle='-|>', mutation_scale=20,
                                lw=1.6, color=EDGE_COLOR, zorder=0,
                                shrinkA=8, shrinkB=10)
        ax.add_patch(arrow)

def _draw_stage_dag(ax, levels, edges, pos, node_radius=0.058):
    ax.set_axis_off()
    _panel_border(ax)
    _draw_directed_edges(ax, edges, pos)
    # nodes (hollow) with labels
    for st, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), radius=node_radius, facecolor='white',
                            edgecolor='black', linewidth=1.4, zorder=2))
        ax.text(x, y + node_radius + 0.04, st, ha="center", va="bottom", fontsize=18)

def _draw_tri_glyph(ax, xy, size=0.085, filled=(True, True, True)):
    """
    Draw 3 adjacent circles centered around xy (no big node circle in region panels).
    filled: tuple for (tmpfs, ssd, beegfs)
    """
    x, y = xy
    r = size / 3.0  # radius for each dot
    dx = r * 2.2

    # tmpfs (left)
    ax.add_patch(Circle((x - dx, y), radius=r,
                        facecolor=(COLOR_TMPFS if filled[0] else 'white'),
                        edgecolor='black', linewidth=1.0, zorder=3))
    # ssd (middle)
    ax.add_patch(Circle((x, y), radius=r,
                        facecolor=(COLOR_SSD if filled[1] else 'white'),
                        edgecolor='black', linewidth=1.0, zorder=3))
    # beegfs (right)
    ax.add_patch(Circle((x + dx, y), radius=r,
                        facecolor=(COLOR_BEEGFS if filled[2] else 'white'),
                        edgecolor='black', linewidth=1.0, zorder=3))

def _draw_region_panel(ax, levels, edges, pos,
                       allowed: Dict[str, List[str]],
                       glyph_size=0.090,
                       title_line: str = ""):
    ax.set_axis_off()
    _panel_border(ax)
    _draw_directed_edges(ax, edges, pos)

    # ONLY glyphs (no big node circle)
    for st, (x, y) in pos.items():
        allow = allowed.get(st, [])
        filled = ("tmpfs" in allow, "ssd" in allow, "beegfs" in allow)
        _draw_tri_glyph(ax, (x, y), size=glyph_size, filled=filled)

    if title_line:
        ax.text(0.5, 0.96, title_line, transform=ax.transAxes,
                ha="center", va="top", fontsize=24)

def _region_allowed_sets(df_region: pd.DataFrame, stor_cols: List[str]) -> Dict[str, List[str]]:
    """For each stage, which storage labels appear in this (renumbered) region (⊆ {tmpfs, ssd, beegfs})."""
    allowed = {}
    for c in stor_cols:
        stage = c[:-5]
        vals = set(_norm_str(df_region[c]).unique())
        allowed[stage] = sorted([v for v in ["tmpfs", "ssd", "beegfs"] if v in vals])
    return allowed

def _pick_representative_regions_by_median(reg_medians: pd.Series, k: int = 5) -> List[int]:
    """Pick lowest, 25th, 50th, 75th, highest by index on the *sorted* medians."""
    reg_sorted = reg_medians.sort_values(ascending=True)
    regs = reg_sorted.index.tolist()
    if len(regs) <= k:
        return regs
    idxs = [
        0,
        int(round(0.25 * (len(regs) - 1))),
        int(round(0.50 * (len(regs) - 1))),
        int(round(0.75 * (len(regs) - 1))),
        len(regs) - 1,
    ]
    keep, seen = [], set()
    for i in idxs:
        if i not in seen:
            seen.add(i)
            keep.append(regs[i])
    return keep

# ----------------- Main generator -----------------

def generate_dag_region_signature_panels(
    wf_name: str,
    scale: int,
    out_format: str = "pdf",  # "png" | "pdf" | "both"
    top5: bool = False,       # if True, show five best (lowest median)
):
    # ---- paths ----
    sens_csv = os.path.join("..", wf_name, "sens_out", "workflow_rowwise_sensitivities.csv")
    script_json = _find_script_order_json(wf_name)
    out_dir = os.path.join("..", wf_name, "sens_out")
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.basename(os.path.normpath(wf_name))
    out_base = os.path.join(out_dir, f"{prefix}_dag_region_signatures_scale{scale}")

    # ---- load ----
    levels, edges = _load_stage_levels_edges(script_json)
    df = pd.read_csv(sens_csv)
    stor_cols = _get_stage_storage_columns(df)
    if not stor_cols:
        raise ValueError("No '*_stor' stage storage columns found in sensitivities CSV.")

    # normalize storage strings
    for c in stor_cols:
        df[c] = _norm_str(df[c])

    df["nodes"] = pd.to_numeric(df.get("nodes", np.nan), errors="coerce").astype("Int64")
    df["total"] = pd.to_numeric(df.get("total", np.nan), errors="coerce")

    df = df[df["nodes"] == scale].copy()
    if df.empty:
        raise ValueError(f"No rows for nodes == {scale}")

    if "region" not in df.columns:
        raise ValueError("Column 'region' not found in sensitivities CSV.")

    # ---- Renumber regions present at this scale to 0..R-1 ----
    unique_regions_sorted = sorted(df["region"].unique().tolist(), key=_region_sort_key)
    rmap = {orig: i for i, orig in enumerate(unique_regions_sorted)}
    df["region_mapped"] = df["region"].map(rmap).astype(int)

    # per-renumbered-region median
    reg_median = df.groupby("region_mapped")["total"].median()

    # ---- Choose regions to show ----
    if top5:
        # five best regions (lowest medians)
        chosen_regions = reg_median.nsmallest(min(5, len(reg_median))).index.tolist()
    else:
        # representative set by percentiles
        chosen_regions = _pick_representative_regions_by_median(reg_median, k=5)

    # order panels left→right by median, then id
    chosen_regions = sorted(chosen_regions, key=lambda r: (reg_median.loc[r], r))

    # positions for DAG nodes
    pos = _node_positions_top_to_bottom(levels)

    # figure: single row -> 1 (legend) + len(chosen_regions)
    n_panels = 1 + len(chosen_regions)
    fig_w = max(12.0, 4.5 * n_panels)   # wide enough to avoid overlaps
    fig_h = 5.6
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h))  # no constrained_layout
    if n_panels == 1:
        axes = [axes]

    # ---- panel 0: Stage DAG legend ----
    _draw_stage_dag(axes[0], levels, edges, pos, node_radius=0.058)
    axes[0].set_title("Stage DAG")

    # ---- region panels ----
    for i, reg_id in enumerate(chosen_regions, start=1):
        df_r = df[df["region_mapped"] == reg_id]
        allowed = _region_allowed_sets(df_r, stor_cols)
        #title = f"Region {reg_id}, median: {int(round(reg_median.loc[reg_id]))}"
        title = f"Region {reg_id}"
        _draw_region_panel(
            axes[i], levels, edges, pos,
            allowed=allowed,
            glyph_size=0.092,
            title_line=title
        )

    # ---- top-center legend (one line) with bounding box ----
    h_tmp = Line2D([], [], marker='o', linestyle='None', markersize=9,
                   color='black', markerfacecolor=COLOR_TMPFS, label='tmpFS')
    h_ssd = Line2D([], [], marker='o', linestyle='None', markersize=9,
                   color='black', markerfacecolor=COLOR_SSD, label='SSD')
    h_bfs = Line2D([], [], marker='o', linestyle='None', markersize=9,
                   color='black', markerfacecolor=COLOR_BEEGFS, label='BeeGFS')
    h_dc = (
        Line2D([], [], marker='o', linestyle='None', markersize=9, color='black', markerfacecolor=COLOR_TMPFS),
        Line2D([], [], marker='o', linestyle='None', markersize=9, color='black', markerfacecolor=COLOR_SSD),
        Line2D([], [], marker='o', linestyle='None', markersize=9, color='black', markerfacecolor=COLOR_BEEGFS),
    )

    leg = fig.legend(
        [h_tmp, h_ssd, h_bfs, h_dc],
        ["tmpFS", "SSD", "BeeGFS", "Don't Care"],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper center", ncol=4, frameon=True, fancybox=True,
        edgecolor=(0.4, 0.4, 0.4, 1.0), facecolor="white",
        bbox_to_anchor=(0.5, 0.98)
    )
    for txt in leg.get_texts():
        txt.set_fontsize(mpl.rcParams["legend.fontsize"])

    # generous spacing (no tight layout) + moderate panel spacing
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.06, top=0.90, wspace=0.18)

    # ---- save ----
    fmt = out_format.lower()
    if fmt in ("png", "both"):
        fig.savefig(out_base + ".png", dpi=300, bbox_inches="tight")
        print("Saved:", out_base + ".png")
    if fmt in ("pdf", "both"):
        fig.savefig(out_base + ".pdf", dpi=300, bbox_inches="tight")
        print("Saved:", out_base + ".pdf")

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="DAG-of-Glyphs: region signatures (single row).")
    ap.add_argument("--wf", required=True, help="Workflow name/folder (e.g., 1kgenome)")
    ap.add_argument("--scale", type=int, required=True, help="Node scale to visualize")
    ap.add_argument("--out-format", choices=["png", "pdf", "both"], default="pdf",
                    help="Output format for the figure (default: pdf)")
    ap.add_argument("--top5", action="store_true",
                    help="Show five best regions (lowest median) instead of percentile representatives.")
    args = ap.parse_args()

    generate_dag_region_signature_panels(
        wf_name=args.wf,
        scale=args.scale,
        out_format=args.out_format,
        top5=args.top5,
    )

if __name__ == "__main__":
    main()
