# QoSFlow

QoSFlow is a framework for **QoS-aware configuration search** on scientific workflows. It combines (i) *workflow scaling rules* with (ii) an SPM-driven makespan table and (iii) *region identification* via decision trees to produce **interpretable regions** of configurations and fast QoS-oriented recommendations.

> The associated research manuscript is under double‑blind review. This repository is organized to reproduce the core pipeline while keeping manuscript identity anonymous.

---

## Repository Layout (top-level)

```
QoSFlow/
├── wf_scaling_rules/        # Human-authored stage-level scaling rules
├── wf_raw_data_csvs/        # Raw per-stage measurements/primitives (observed at baseline scales)
├── wf_spm_result_csvs/      # Per-configuration makespan tables produced by SPM (derived from scaled raw data)
├── wf_cost_modeling/        # Cost formulation and decomposition built *from SPM results*
├── wf_cart_analysis/        # CART-based region identification (cross-fitting, pruning, exports)
├── wf_analysis_results/     # Aggregated analysis artifacts (labeled configs, per-region stats, QoS tables)
└── wf_analysis_plotting/    # Plotting utilities (region scatters, stacked costs, sensitivity panels)
```

> Folder names above reflect the canonical layout; workflow-specific subfolders (e.g., 1kgenome, pyflextrkr, ddmd) appear under the results directories when you run the pipeline at different scales.

---

## End-to-End Pipeline (Execution Flow)

QoSFlow follows a **three-phase** flow. The corrections below make explicit how **SPM** is used and where the cost formulation occurs.

### Phase 1 — Inputs and Scale Projection
1. **Author scaling rules** → place rule files for each workflow stage in `wf_scaling_rules/`. Rules describe how data volumes/accesses/concurrency **change with scale** (task/data fan-in/out, replication, etc.).  
2. **Provide raw observations** → place the baseline, *observed* per‑stage primitives in `wf_raw_data_csvs/`.  
3. **Apply scaling rules to raw data** → the rules are applied to the files in `wf_raw_data_csvs/` to **project** per‑stage I/O and concurrency to the **target scale**. The result is a *scaled* dataset that becomes the input to SPM.

> ✅ **Clarification**: *Scaling rules act on raw_data_csvs to synthesize target‑scale per‑stage stats. These scaled stats are then consumed by SPM.*

### Phase 2 — SPM Results (External to this work)
4. **Run SPM** → feed the *scaled* per‑stage data into the **SPM framework** (external component; **not the scope of this repository**) to compute per‑configuration makespans.  
   - Output: `wf_spm_result_csvs/<workflow>/<scale>_filtered_spm_results.csv` (naming may vary).  
   - These CSVs are the **SPM results derived by processing raw_data_csvs through scaling**.

> ✅ **Clarification**: *SPM results are collected from the SPM framework. They are **derived** by first processing `wf_raw_data_csvs/` with scaling rules and then evaluating with SPM. This repository treats SPM as a producer of makespan tables and does not re‑implement SPM.*

### Phase 3 — Cost, Regions, and QoS
5. **Cost formulation & decomposition (from SPM results)** → the code in `wf_cost_modeling/` **consumes SPM CSVs** from `wf_spm_result_csvs/` to compute cost breakdowns (e.g., shared vs. local I/O vs. movement) and any auxiliary summaries needed downstream.  
   - **Important**: cost formulation happens **after** SPM and is **based on SPM outputs**, not raw data directly.
6. **CART-based region identification** → run `wf_cart_analysis/` to:
   - encode configurations,
   - train CART with **cost–complexity pruning**,
   - use **repeated K‑fold cross‑fitting** (no train/test leakage),
   - select pruning via a **joint objective** balancing *region separability* (effect sizes, variance-aware thresholds) and *prediction error* (MAE),
   - export region labels & summaries to `wf_analysis_results/`.
7. **Plot & inspect** → use `wf_analysis_plotting/` to produce region scatters, *stacked cost* plots (built from step 5), sensitivity panels, and QoS tables/reports.

---

## What Goes Where (I/O Contracts)

- **Inputs**
  - `wf_scaling_rules/` → scaling rules (CSV/YAML/JSON; see module docs).
  - `wf_raw_data_csvs/` → baseline per‑stage observed primitives.

- **Intermediate via SPM (external)**
  - `wf_spm_result_csvs/` → **SPM‑produced** per‑configuration makespans: *scaled raw data → SPM → CSVs*.  
    (*SPM is an external framework; this repo only consumes its outputs.*)

- **Downstream (this repo)**
  - `wf_cost_modeling/` → **cost formulation from SPM results** + decomposition exports.
  - `wf_cart_analysis/` → regions (labels, stats, ordered lists).
  - `wf_analysis_results/` → final tables for QoS querying and reporting.
  - `wf_analysis_plotting/` → figures created from the above.

---

## Minimal Repro Checklist

1. Place scaling rules in `wf_scaling_rules/` and raw observations in `wf_raw_data_csvs/`.
2. Apply the rules to project to the target scale (scripted/automated as per your workflow).
3. **Run SPM** on the *scaled* data and write CSVs to `wf_spm_result_csvs/`. *(External step; not implemented here.)*
4. Run cost formulation in `wf_cost_modeling/` **on the SPM CSVs**.
5. Train regions with `wf_cart_analysis/` and generate plots with `wf_analysis_plotting/`.

---

## Environment

- Python 3.9+
- NumPy, Pandas, scikit‑learn, Matplotlib
- (Optional) Stats helpers for effect sizes / ANOVA-style checks

Create/activate a virtual environment and install module‑level requirements as needed.

---

## Reproducibility Notes

- SPM is treated as a **black box producer** of makespan tables. This repo documents **how SPM outputs are used**, not how SPM itself is implemented.
- Keep workflow‑specific artifacts grouped under the results directories to avoid cluttering the repo root.
- For double‑blind review, avoid manuscript-identifying strings in filenames/figures.

---

## License

MIT License (see `LICENSE`).
