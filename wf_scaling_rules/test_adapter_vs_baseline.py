# test_adapter_vs_baseline.py  (top-level; imports assume pwd = workflow_analysis)

import argparse, sys, pandas as pd
from adapter_flowforecaster import (
    build_adapter_input_from_csv, apply_flowforecaster_scales, FFRules, run_spm_with_ff
)
from workflow_analyzer import analyze_workflow_from_csv
from modules.workflow_config import STORAGE_LIST, TEST_CONFIGS

import math

def _as_scalar(v):
    # SPM/rank dicts store values as [scalar]; normalize to a float
    if isinstance(v, list):
        return float(v[0]) if v else float("nan")
    try:
        return float(v)
    except Exception:
        return float("nan")

def compare_spm(a: dict, b: dict, atol: float = 1e-9, rtol: float = 1e-6):
    diffs = []
    pairs = set(a.keys()) | set(b.keys())
    for pair in sorted(pairs):
        if pair not in a:
            diffs.append(f"[missing-in-A] {pair}")
            continue
        if pair not in b:
            diffs.append(f"[missing-in-B] {pair}")
            continue

        # compare both rank and SPM maps (values are lists)
        for metric in ("rank", "SPM"):
            da = a[pair].get(metric, {})
            db = b[pair].get(metric, {})
            keys = set(da.keys()) | set(db.keys())
            for e in sorted(keys):
                if e not in da:
                    diffs.append(f"[missing-{metric}-in-A] {pair}:{e}")
                    continue
                if e not in db:
                    diffs.append(f"[missing-{metric}-in-B] {pair}:{e}")
                    continue
                va = _as_scalar(da[e])
                vb = _as_scalar(db[e])
                if not math.isclose(va, vb, rel_tol=rtol, abs_tol=atol):
                    diffs.append(f"[{metric}-mismatch] {pair}:{e} → {va} vs {vb}")

    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--workflow-name", default=None)
    ap.add_argument("--ior", default="../perf_profiles/updated_master_ior_df.csv")
    ap.add_argument("--mode", choices=["identity","copy_processed"], default="identity")
    args = ap.parse_args()

    # Baseline analyzer (full pipeline)
    results = analyze_workflow_from_csv(args.csv, workflow_name=args.workflow_name, ior_data_path=args.ior, save_results=True)
    if not isinstance(results, dict) or "spm_results" not in results or "workflow_df" not in results:
        print("Analyzer must return {'spm_results','workflow_df'}"); sys.exit(2)
    baseline_spm = results["spm_results"]
    processed_df = results["workflow_df"]

    # Adapter path: same pre-steps, optional FF scales, then SPM
    adapter_df = build_adapter_input_from_csv(args.csv, workflow_name=args.workflow_name, debug=False)
    allowed_parallelism = None
    if args.workflow_name in TEST_CONFIGS:
        cp_scp = set(adapter_df.loc[adapter_df['operation'].isin(['cp','scp','none']),
                                    'parallelism'].unique())
        cfg = set(TEST_CONFIGS[args.workflow_name]["ALLOWED_PARALLELISM"])
        allowed_parallelism = sorted(cfg.union(cp_scp))

    if args.mode == "copy_processed":
        # Align and copy target inputs from analyzer’s processed DF
        join_cols = ["taskName","taskPID","fileName","operation","stageOrder","numNodes","parallelism"]
        for c in join_cols:
            if c not in adapter_df.columns or c not in processed_df.columns:
                print("Missing join column for copy_processed:", c); sys.exit(3)
        cols_to_copy = ["aggregateFilesizeMB","transferSize"]
        copy_src = processed_df[join_cols + cols_to_copy].copy()
        adapter_df = adapter_df.merge(copy_src, on=join_cols, how="left", suffixes=("","_proc"))
        for c in cols_to_copy:
            if f"{c}_proc" in adapter_df.columns:
                adapter_df[c] = adapter_df[f"{c}_proc"].combine_first(adapter_df[c])
                adapter_df.drop(columns=[f"{c}_proc"], inplace=True)

    # FF identity scales (no change)
    rules = FFRules(global_factors=None, overrides=None)
    adapter_df = apply_flowforecaster_scales(adapter_df, rules, debug=False)

    ior_df = pd.read_csv(args.ior)
    adapter_spm = run_spm_with_ff(adapter_df, ior_df, storage_list=STORAGE_LIST,
                                  allowed_parallelism=allowed_parallelism, multi_nodes=True,
                                  debug=False, workflow_name=args.workflow_name)

    diffs = compare_spm(baseline_spm, adapter_spm)
    if diffs:
        print("❌ MISMATCHES:"); [print("  ", d) for d in diffs[:200]]
        print(f"... total mismatches: {len(diffs)}"); sys.exit(1)
    print("✅ Adapter reproduces baseline SPM."); sys.exit(0)

if __name__ == "__main__":
    main()
