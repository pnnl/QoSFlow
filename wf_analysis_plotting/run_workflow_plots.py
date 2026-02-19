#!/usr/bin/env python3
import argparse
import os
from typing import List

# Existing generators
from make_total_cost_plot import generate_total_cost_subplots
from make_regions_scatter_subplots import generate_regions_scatter_subplots
from make_region_cost_share_stacked import generate_region_cost_share_stacked
from make_policy_six_subplots_from_sens import generate_policy_six_subplots_from_sens
from make_region_cluster_plot import generate_region_cluster_plot
from make_region_cluster_multiscale_plot import generate_region_cluster_multiscale_plot
from make_total_cost_plot_from_sens import generate_total_cost_subplots_from_sens

# NEW: DAG-of-Glyphs generator
from make_dag_region_signature_panels import generate_dag_region_signature_panels
import matplotlib as mpl

# ---- Global font & style (≥24pt everywhere) ----
mpl.rcParams.update({
    "font.size": 24,
    "axes.titlesize": 24,
    "axes.labelsize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
})

def parse_scales(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def main():
    p = argparse.ArgumentParser(
        description="Run workflow plots: totals, regions scatter, region shares, 6-policy (sens), and DAG region signatures."
    )
    p.add_argument("--wf", required=True, help="Workflow folder name, e.g. 1kgenome")
    p.add_argument("--scales", required=True,
                   help="Comma-separated node scales for standard plots, e.g. 2,5,10")
    p.add_argument("--xtick-major", type=int, default=10,
                   help="Major tick spacing for config-index X axes (default: 10)")
    p.add_argument("--which",
               choices=["all","totals","regions_scatter","region_shares","policies6","dag_signatures","region_clusters","region_clusters_multi","totals_from_sens"],
               default="all",
               help="Which plot(s) to run (default: all)")


    # Policy-6 controls
    p.add_argument("--policy-scale", type=int, default=None,
                   help="Single node scale for the 6-policy analysis; if omitted and --scales has one value, it's used")
    p.add_argument("--policy-seed", type=int, default=42,
                   help="Random seed for Random Shuffle baseline (default: 42)")

    # DAG-of-Glyphs controls
    p.add_argument("--dag-scale", type=int, default=None,
                   help="Single node scale for the DAG region signature panels; if omitted and --scales has one value, it's used")

    p.add_argument("--cluster-scale", type=int, default=None,
                help="Single node scale for region cluster plot; if omitted and --scales has one value, it's used")
    p.add_argument("--target-scale", type=int, default=None,
               help="Target node scale whose config ordering will be applied to all subplots for --which totals_from_sens")
    
    p.add_argument(
        "--out-format",
        choices=["png", "pdf"],
        default="pdf",
        help="Output format (default: pdf)"
    )
    
    p.add_argument(
        "--top5",
        action="store_true",
        help="For dag_signatures: show five best regions (lowest medians)."
    )


    args = p.parse_args()

    wf_name = args.wf
    scales  = parse_scales(args.scales)
    xtick   = args.xtick_major

    # Ensure local <wf>/sens_out exists for the standard plots
    os.makedirs(os.path.join("..", wf_name, "sens_out"), exist_ok=True)

    if args.which in ("all", "totals"):
        generate_total_cost_subplots(wf_name=wf_name, desired_scales=scales, major_every=xtick, out_format=args.out_format)

    if args.which in ("all", "regions_scatter"):
        generate_regions_scatter_subplots(wf_name=wf_name, desired_scales=scales, major_every=xtick, out_format=args.out_format)

    if args.which in ("all", "region_shares"):
        generate_region_cost_share_stacked(wf_name=wf_name, desired_scales=scales, out_format=args.out_format)

    if args.which in ("all", "policies6"):
        if args.policy_scale is not None:
            pscale = args.policy_scale
        else:
            if len(scales) == 1:
                pscale = scales[0]
            else:
                raise SystemExit("For 6-policy plot, provide --policy-scale or pass a single value to --scales.")
        generate_policy_six_subplots_from_sens(
            wf_name=wf_name,
            scale=pscale,
            major_every=xtick,
            seed=args.policy_seed,
            out_format=args.out_format,
        )

    if args.which in ("all", "dag_signatures"):
        if args.dag_scale is not None:
            dscale = args.dag_scale
        else:
            if len(scales) == 1:
                dscale = scales[0]
            else:
                raise SystemExit("For DAG signatures, provide --dag-scale or pass a single value to --scales.")
        generate_dag_region_signature_panels(
            wf_name=wf_name,
            scale=dscale,
            out_format=args.out_format,
            top5=args.top5,
        )

    if args.which in ("all", "region_clusters"):
        if args.cluster_scale is not None:
            cscale = args.cluster_scale
        else:
            if len(scales) == 1:
                cscale = scales[0]
            else:
                raise SystemExit("For region cluster plot, provide --cluster-scale or pass a single value to --scales.")
        generate_region_cluster_plot(
            wf_name=wf_name,
            scale=cscale,
            out_format=args.out_format,
        )

    if args.which in ("all", "region_clusters_multi"):
        # uses the global --scales list directly
        generate_region_cluster_multiscale_plot(
            wf_name=wf_name,
            scales=scales,
        )

    if args.which in ("all", "totals_from_sens"):
        if args.target_scale is None:
            raise SystemExit("Please provide --target-scale for totals_from_sens.")
        generate_total_cost_subplots_from_sens(
            wf_name=wf_name,
            desired_scales=scales,       # uses global --scales list
            target_scale=args.target_scale,
            major_every=xtick,
            out_format=args.out_format,
        )

    print("Done.")

if __name__ == "__main__":
    main()
