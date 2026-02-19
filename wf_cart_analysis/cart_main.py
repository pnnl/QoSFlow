# cart_main.py  (replace the run_pipeline(...) call)
from __future__ import annotations
from cart_cli import build_parser, normalize_args
from cart_pipeline import run_pipeline

def main():
    ap = build_parser()
    args = ap.parse_args()
    args = normalize_args(args)

    run_pipeline(
        input_csv=args.input,
        outdir=args.outdir,
        # tree & CV
        min_leaf=args.min_leaf, max_depth=args.max_depth, criterion=args.criterion,
        kfolds=args.kfolds, repeats=args.repeats, seed=args.seed,
        # separation
        separation_metric=args.separation_metric,
        g_min=args.g_min, n_min_pair=args.n_min_pair, stat_score=args.stat_score,
        pair_weight=args.pair_weight, robust_quantiles=args.robust_quantiles,
        # adaptive g
        g_min_auto=args.g_min_auto, delta=args.delta, cv_estimator=args.cv_estimator,
        g_floor=args.g_floor, g_cap=args.g_cap,
        # legacy placeholders (unused for stat_g_all)
        top_frac=args.top_frac, max_topN=args.max_topN, y_sep=args.y_sep,
        use_overlap_guard=not args.no_overlap_guard,
        # gating
        gate_splits=args.gate_splits, gate_val_frac=args.gate_val_frac,
        gate_mae_tol=args.gate_mae_tol, gate_sep_tol=args.gate_sep_tol,
        # selection
        select_mode=args.select_mode, mae_budget=args.mae_budget, mae_weight=args.mae_weight,
        # post-merge
        postmerge=args.postmerge, min_region_size=args.min_region_size,
        abs_gap_tau=args.abs_gap_tau, min_guard_n=args.min_guard_n,
        # >>> DEBUG flags (new) <<<
        debug=args.debug, debug_max_rows=args.debug_max_rows,
    )

if __name__ == "__main__":
    main()
