<!-- -*-Mode: markdown;-*- -->
<!-- $Id: 4098d4ffce45696ec3497ad9e08e712906c9d8fe $ -->

QoSFlow
=============================================================================

**Home**:
  - [QoSFlow](https://github.com/pnnl/QoSFlow), part of [DataFlowDrs](https://github.com/pnnl/DataFlowDrs)
  
  - [Performance Lab for EXtreme Computing and daTa](https://github.com/PerfLab-EXaCT)


**About**: 

🆕 To enable Quality of Service scheduling constraints (e.g., minimize time, limit execution to resource subsets) for scientific workflows, QoSFlow uses rapid reasoning over the large configuration space that is driven by predictive models rather than costly executions. QoSFlow partitions a workflow's execution configuration space into regions with similar behavior. Each region groups configurations with comparable execution times according to a given statistical sensitivity, enabling efficient QoS-driven scheduling through analytical reasoning rather than exhaustive testing. The analytical reasoning is enabled with interpretable models that highlight the key workflow paths that determine performance; distinguish which configuration parameters are critical vs. flexible; and in turn explain the critical path's performance using analytical dataflow expressions.

QoSFlow is a framework for **QoS-aware configuration search** on scientific workflows. It combines (i) *workflow scaling rules* with (ii) an DPM-driven makespan table and (iii) *region identification* via decision trees to produce **interpretable regions** of configurations and fast QoS-oriented recommendations.


------------------------------------------------------------------------------

## Contacts

**Contacts**: (_firstname_._lastname_@pnnl.gov)
  - Nathan R. Tallent ([www](https://nathantallent.github.io))
  - Md Hasanur Rashid ([www](https://www.linkedin.com/in/hasanurrashid95/))


**Contributors**:
  - Md Hasanur Rashid ([www](https://www.linkedin.com/in/hasanurrashid95/))


References
-----------------------------------------------------------------------------
- **Overview**: Nathan R. Tallent, Meng Tang, Zhen Peng, Jesun Firoz, Luanzheng Guo, Anthony Kougkas, and Xian-He Sun. "DataFlowDrs: Automating Performance Optimization of Data Flow Within HPC Workflows" IEEE Transactions on Parallel and Distributed Systems, pp. 1-18, September 2026 ([doi: 10.1109/TPDS.2026.3722592](https://doi.org/10.1109/IPDPS65963.2026.00112))

* **Specific**: M. H. Rashid, J. Firoz, N. R. Tallent, L. Guo, M. Tang, and D. Dai, “QoSFlow: Ensuring Service Quality of Distributed Workflows Using Interpretable Sensitivity Models,” in Proc. of the 40th IEEE Intl. Parallel and Distributed Processing Symp., IEEE Computer Society, May 2026.

- All related: [DataFlowDrs](https://github.com/pnnl/DataFlowDrs)


## License

BSD 2-clause license: [README-License.txt](/README-License.txt)


Acknowledgements
-----------------------------------------------------------------------------
This work was supported by the U.S. Department of Energy's Office of
Advanced Scientific Computing Research:

- Orchestration for Distributed & Data-Intensive Scientific Exploration


---

# Getting Started

## Repository Layout (top-level)

```
QoSFlow/
├── wf_scaling_rules/        # Human-authored stage-level scaling rules
├── wf_raw_data_csvs/        # Raw per-stage measurements/primitives (observed at baseline scales)
├── wf_spm_result_csvs/      # Per-configuration makespan tables produced by DPM (derived from scaled raw data)
├── wf_cost_modeling/        # Cost formulation and decomposition built *from DPM results*
├── wf_cart_analysis/        # CART-based region identification (cross-fitting, pruning, exports)
├── wf_analysis_results/     # Aggregated analysis artifacts (labeled configs, per-region stats, QoS tables)
└── wf_analysis_plotting/    # Plotting utilities (region scatters, stacked costs, sensitivity panels)
```

> Folder names above reflect the canonical layout; workflow-specific subfolders (e.g., 1kgenome, pyflextrkr, ddmd) appear under the results directories when you run the pipeline at different scales.

---

## End-to-End Pipeline (Execution Flow)

QoSFlow follows a **three-phase** flow. The corrections below make explicit how **DPM** is used and where the cost formulation occurs.

### Phase 1 — Inputs and Scale Projection
1. **Author scaling rules** → place rule files for each workflow stage in `wf_scaling_rules/`. Rules describe how data volumes/accesses/concurrency **change with scale** (task/data fan-in/out, replication, etc.).  
2. **Provide raw observations** → place the baseline, *observed* per‑stage primitives in `wf_raw_data_csvs/`.  
3. **Apply scaling rules to raw data** → the rules are applied to the files in `wf_raw_data_csvs/` to **project** per‑stage I/O and concurrency to the **target scale**. The result is a *scaled* dataset that becomes the input to DPM.

> ✅ **Clarification**: *Scaling rules act on raw_data_csvs to synthesize target‑scale per‑stage stats. These scaled stats are then consumed by DPM.*

### Phase 2 — DPM Results (External to this work)
4. **Run DPM** → feed the *scaled* per‑stage data into the **DPM framework** (external component; **not the scope of this repository**) to compute per‑configuration makespans.  
   - Output: `wf_spm_result_csvs/<workflow>/<scale>_filtered_spm_results.csv` (naming may vary).  
   - These CSVs are the **DPM results derived by processing raw_data_csvs through scaling**.

> ✅ **Clarification**: *DPM results are collected from the DPM framework. They are **derived** by first processing `wf_raw_data_csvs/` with scaling rules and then evaluating with DPM. This repository treats DPM as a producer of makespan tables and does not re‑implement DPM.*

### Phase 3 — Cost, Regions, and QoS
5. **Cost formulation & decomposition (from DPM results)** → the code in `wf_cost_modeling/` **consumes DPM CSVs** from `wf_spm_result_csvs/` to compute cost breakdowns (e.g., shared vs. local I/O vs. movement) and any auxiliary summaries needed downstream.  
   - **Important**: cost formulation happens **after** DPM and is **based on DPM outputs**, not raw data directly.
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

- **Intermediate via DPM (external)**
  - `wf_spm_result_csvs/` → **DPM‑produced** per‑configuration makespans: *scaled raw data → DPM → CSVs*.  
    (*DPM is an external framework; this repo only consumes its outputs.*)

- **Downstream (this repo)**
  - `wf_cost_modeling/` → **cost formulation from DPM results** + decomposition exports.
  - `wf_cart_analysis/` → regions (labels, stats, ordered lists).
  - `wf_analysis_results/` → final tables for QoS querying and reporting.
  - `wf_analysis_plotting/` → figures created from the above.

---

## Minimal Repro Checklist

1. Place scaling rules in `wf_scaling_rules/` and raw observations in `wf_raw_data_csvs/`.
2. Apply the rules to project to the target scale (scripted/automated as per your workflow).
3. **Run DPM** on the *scaled* data and write CSVs to `wf_spm_result_csvs/`. *(External step; not implemented here.)*
4. Run cost formulation in `wf_cost_modeling/` **on the DPM CSVs**.
5. Train regions with `wf_cart_analysis/` and generate plots with `wf_analysis_plotting/`.

---

## Environment

- Python 3.9+
- NumPy, Pandas, scikit‑learn, Matplotlib
- (Optional) Stats helpers for effect sizes / ANOVA-style checks

Create/activate a virtual environment and install module‑level requirements as needed.

---

## Reproducibility Notes

- DPM is treated as a **black box producer** of makespan tables. This repo documents **how DPM outputs are used**, not how DPM itself is implemented.
- Keep workflow‑specific artifacts grouped under the results directories to avoid cluttering the repo root.
- For double‑blind review, avoid manuscript-identifying strings in filenames/figures.

