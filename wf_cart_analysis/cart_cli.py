# cart_cli.py
from __future__ import annotations
import argparse
from typing import Optional

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="CART regions with repeated-CV statistical separation and MAE-aware alpha selection.")
    ap.add_argument("--input", required=True, help="Input CSV (e.g., workflow_makespan_stageorder.csv)")
    ap.add_argument("--outdir", required=True, help="Directory to write outputs")

    # Tree & CV
    ap.add_argument("--min-leaf", type=int, default=8, help="Minimum samples per CART leaf.")
    ap.add_argument("--max-depth", type=int, default=12, help="Maximum tree depth.")
    ap.add_argument("--criterion", choices=["squared_error","absolute_error"], default="absolute_error", help="Tree impurity.")
    ap.add_argument("--kfolds", type=int, default=5, help="Outer CV folds.")
    ap.add_argument("--repeats", type=int, default=5, help="Repeated CV runs.")
    ap.add_argument("--seed", type=int, default=42)

    # Separation metric (stat_g_all)
    ap.add_argument("--separation-metric", choices=["stat_g_all"], default="stat_g_all")
    ap.add_argument("--g-min", type=float, default=0.30, help="Fixed |g| threshold when not using adaptive-g.")
    # Redundancy removed: we default these to None and fill from --min-leaf later.
    ap.add_argument("--n-min-pair", type=int, default=None, help="Min rows per side to evaluate a pair (defaults to --min-leaf).")
    ap.add_argument("--stat-score", choices=["avg_effect","frac_sig","product"], default="product")
    ap.add_argument("--pair-weight", choices=["min","hmean"], default="hmean")
    ap.add_argument("--robust-quantiles", action="store_true", help="Use Hazen smoothing for medians/quartiles.")
    # Adaptive g
    ap.add_argument("--g-min-auto", action="store_true", help="Use per-pair g_thresh = max(g_floor, min(g_cap, delta / CV_pooled)).")
    ap.add_argument("--delta", type=float, default=0.10, help="Target relative gap δ (e.g., 0.10).")
    ap.add_argument("--cv-estimator", choices=["std","mad"], default="mad")
    ap.add_argument("--g-floor", type=float, default=0.0)
    ap.add_argument("--g-cap", type=float, default=1.0)

    # (Legacy placeholders kept to avoid breaking flags; not used in stat_g_all)
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--max-topN", type=int, default=12)
    ap.add_argument("--y-sep", type=float, default=0.10)
    ap.add_argument("--no-overlap-guard", action="store_true")

    # Gating
    ap.add_argument("--gate-splits", action="store_true", help="Enable inner 80/20 split-acceptance gate.")
    ap.add_argument("--gate-val-frac", type=float, default=0.2)
    ap.add_argument("--gate-mae-tol", type=float, default=0.0)
    ap.add_argument("--gate-sep-tol", type=float, default=0.0)

    # Selection blend
    ap.add_argument("--select-mode", choices=["pass_only","constrained","weighted"], default="constrained")
    ap.add_argument("--mae-budget", type=float, default=0.02)
    ap.add_argument("--mae-weight", type=float, default=0.30)

    # Post-merge (union of leaves)
    ap.add_argument("--postmerge", action="store_true")
    ap.add_argument("--min-region-size", type=int, default=None, help="Defaults to --min-leaf if not set.")
    ap.add_argument("--abs-gap-tau", type=float, default=0.0)
    ap.add_argument("--min-guard-n", type=int, default=None, help="Defaults to --min-leaf if not set.")

    # >>> NEW: Debug switch (pair-level CSV)
    ap.add_argument("--debug", action="store_true", help="If set, writes cart_debug_pairs.csv with per-pair metrics.")
    ap.add_argument("--debug-max-rows", type=int, default=None, help="Optional cap on debug rows written.")

    return ap

def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    # Tie redundant knobs to --min-leaf if not explicitly set
    if args.n_min_pair is None:
        args.n_min_pair = args.min_leaf
    if args.min_region_size is None:
        args.min_region_size = args.min_leaf
    if args.min_guard_n is None:
        args.min_guard_n = args.min_leaf
    return args
