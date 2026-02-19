# model.py
# =============================================================================
# WORKFLOW MAKESPAN — RULES & LIMITATIONS (Concise)
# -----------------------------------------------------------------------------
# Barriered timeline per stage_order k:
#   in_k   = max over all stage_in actions needed BEFORE running stages at order k
#   exec_k = max over execution time of stages at order k
#            (for each stage C: READ on σ(C) + WRITE on σ(C))
#   out_k  = max over all stage_out actions AFTER finishing stages at order k
#   TOTAL  = Σ_k (in_k + exec_k + out_k)
#
# Associations of costs:
#   • stage_in/out: ONLY data movement (cp/scp).
#   • execution:    ONLY read/write of stages.
#
# Execution (per stage C)
#   • READ(C): the stage must read ALL of its inputs.
#       Time = SUM over predecessors P of consumer-side read on σ(C).
#       Use SPM rows with (producer=P, consumer=C, storage_dst=σ(C));
#       prefer cons_tpn == tpn(C) (we do not assert any equality with write TPN).
#     For “initial-only” C (only predecessor is initial_data):
#       - if σ(C) is local, use local-read from stage_in row;
#       - otherwise use shared-read from stage_in row.
#
#   • WRITE(C):
#       - If C has successors D with ORDER[D] > ORDER[C]:
#           take the MAX producer-side write on σ(C) using direct execution rows:
#             (producer = C, consumer = D, storage_src = σ(C))   # storage_dst unconstrained
#           Prefer prod_tpn == tpn(C); do not assert equality with read TPN.
#       - If C is FINAL (no successors):
#           • If σ(C) is LOCAL (e.g., ssd/tmpfs):
#               use the producer-side WRITE from the final stage_out row:
#                 (producer = C, consumer = stage_out-C,
#                  storage_src = σ(C), storage_dst = σ(C)-beegfs)
#               Prefer prod_tpn == tpn(C), fallback to cons_tpn==tpn(C), else any.
#           • If σ(C) is SHARED (beegfs):
#               use the producer-side WRITE from stage_out rows with beegfs as source:
#                 (producer = C, consumer = stage_out-C,
#                  storage_src = beegfs, storage_dst starts with "beegfs-")
#               Prefer prod_tpn == tpn(C), fallback to cons_tpn==tpn(C), else any.
#
#   • NOTE: σ(P) and σ(C) may differ; any cross-storage movement required because σ(P) ≠ σ(C)
#     is charged as cp/scp in the stage_out phases (not in execution).
#
# Stage_in (in_k)
#   • ANY order where a stage at that order needs data from shared:
#       beegfs→local cp via stage_in rows: charge PRODUCER-side cp only (prod_spm).
#       We select the SPM row with the LOWEST cost across ALL available TPNs and record
#       the chosen prod_tpn. Execution READ then uses local-read for those stages.
#
# Stage_out (out_k)
#   • For any order:
#       - beegfs→local cp (from producer’s stage_out rows): CONSUMER-side cp only (cons_spm).
#       - local→beegfs cp (from producer’s stage_out rows): CONSUMER-side cp only (cons_spm).
#       - local→local scp fan-out: CONSUMER-side scp only (cons_spm).
#       For each movement, select the SPM row with the LOWEST cost across ALL available TPNs
#       and record the chosen cons_tpn. De-duplicate by (producer, target storage) and take the
#       per-order max to form out_k.
#
# TPN handling
#   • Execution (READ/WRITE) uses the stage’s TPN: reads prefer cons_tpn == tpn(C), writes prefer prod_tpn == tpn(C).
#     We annotate critical path tokens as read_C(Y) and write_C(Y) where Y = tpn(C) used by execution.
#   • Movement (stage_in / stage_out) selects the LOWEST SPM across ALL TPNs. We record the TPN from
#     the chosen row and annotate critical path tokens as stage_in_C(X) / stage_out_C(Z).
#   • We do NOT assume or assert that stage_in and stage_out use the same TPN for a stage.
#
# Validity (reflects CSV availability)
#   • Cross-local transitions between different local types (e.g., ssd↔tmpfs) are not modeled (unmeasured).
#     Configurations that require these movements are rejected.
#
# CSV Reporting (this file)
#   • For each stage S (ordered by stage_order, then alphabetically), we report four columns:
#       in_S, read_S, write_S, out_S
#     Meaning of each column and when it is populated:
#       - in_S    : stage-in (cp) cost CONSIDERED for S at its order (beegfs→local cp, producer-side; min SPM across all TPNs).
#       - read_S  : execution READ cost CONSIDERED for S (SUM over feasible predecessors P on σ(S), or initial-read).
#       - write_S : execution WRITE cost CONSIDERED for S (MAX over feasible successors D on σ(S); if S is final on local,
#                   use the producer-side WRITE from the final stage_out row; if final on beegfs, use beegfs-source stage_out row).
#       - out_S   : stage-out (cp/scp) cost CONSIDERED for S at its order (beegfs→local cp, local→beegfs cp, and/or local→local fan-out;
#                   min SPM across all TPNs; consumer-side).
#
#   • IMPORTANT: These per-stage fields show the value that was actually CONSIDERED in the per-order comparison,
#     even if they did NOT win the barrier and are NOT on the critical path. A hyphen “-” is used ONLY when the component
#     was NOT APPLICABLE / NOT EVALUATED under the current storage assignment (e.g., no stage_in needed, no stage_out path, etc.).
#
#   • Per-order aggregates still appear:
#       in_oK   : max of all in_S considered at order K
#       exec_oK : max of per-stage (read_S + write_S) at order K
#       out_oK  : max of all out_S considered at order K
#     TOTAL = Σ_K (in_oK + exec_oK + out_oK).
#
#   • The critical_path field records which stage/phase achieved each order’s barrier max and
#     includes TPNs:
#       stage_in_S(X) -> read_S(Y) -> write_S(Y) -> ... -> stage_out_T(Z)
#
#   • The CSV also includes nodes, storage choices (σ(S) per stage), and is row-sorted by total (then nodes).
# =============================================================================

import os
import itertools
import math
import numpy as np
import pandas as pd

from constants import STORAGES, GLOBAL_NODES, OUT_DIR, FORBIDDEN_CROSS
from parser import load_rank, load_meta

DF = load_rank()
TASKS, ORDER, PREDS, CONSUMERS = load_meta()
ORDERS = sorted(set(ORDER.values()))
os.makedirs(OUT_DIR, exist_ok=True)

LOCAL_STORES = tuple(s for s in STORAGES if s != "beegfs")
O_MIN = min(ORDERS)
O_MAX = max(ORDERS)

# -----------------------------
# Helpers & lookups
# -----------------------------
def tpn_map(N: int):
    return {s: max(1, math.ceil(TASKS[s] / N)) for s in TASKS}

def _nz(x) -> float:
    try:
        xv = float(x)
        return 0.0 if np.isnan(xv) else xv
    except Exception:
        return 0.0

def _min_col(d: pd.DataFrame, col: str) -> float:
    if d is None or d.empty or col not in d.columns:
        return np.nan
    vals = d[col]
    return float(np.nanmin(vals.values)) if len(vals) else np.nan

def _stage_sort_key(name: str):
    return (ORDER[name], name.lower())

def only_initial(c: str) -> bool:
    preds = [p for p in PREDS[c] if p is not None]
    return len(preds) == 1 and preds[0] == "initial_data"

def _order_groups():
    grp = {}
    for s, o in ORDER.items():
        grp.setdefault(o, []).append(s)
    for o in grp:
        grp[o].sort(key=str.lower)
    return dict(sorted(grp.items()))

# -----------------------------
# TPN-aware filters for EXECUTION
# -----------------------------
def _tpn_filter_exec_read(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    if base is None or base.empty or ("cons_tpn" not in base.columns):
        return base
    sel = base[base["cons_tpn"] == c_tpn_val]
    return sel if not sel.empty else base

def _tpn_filter_exec_write(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    if base is None or base.empty or ("prod_tpn" not in base.columns):
        return base
    sel = base[base["prod_tpn"] == c_tpn_val]
    return sel if not sel.empty else base

def _tpn_filter_final_write_stageout(base: pd.DataFrame, c_tpn_val: int) -> pd.DataFrame:
    if base is None or base.empty:
        return base
    have_p = "prod_tpn" in base.columns
    have_c = "cons_tpn" in base.columns
    if have_p:
        psel = base[base["prod_tpn"] == c_tpn_val]
        if not psel.empty:
            return psel
    if have_c:
        csel = base[base["cons_tpn"] == c_tpn_val]
        if not csel.empty:
            return csel
    return base

# -----------------------------
# Cost accessors
# -----------------------------
# movement (stage_out cp/scp) — choose MIN SPM across ALL TPNs; return (cost, chosen_cons_tpn)
def stage_out_consumer_cost(stage: str, src: str, combined_dst: str):
    d = DF[
        (DF.producer == stage)
        & (DF.consumer == f"stage_out-{stage}")
        & (DF.storage_src == src)
        & (DF.storage_dst == combined_dst)
    ]
    if d is None or d.empty or "cons_spm" not in d.columns:
        return (float("nan"), None)
    idx = d["cons_spm"].astype(float).idxmin()
    row = d.loc[idx]
    chosen_cost = float(row["cons_spm"])
    chosen_tpn = int(row["cons_tpn"]) if "cons_tpn" in d.columns and not pd.isna(row["cons_tpn"]) else None
    return (chosen_cost, chosen_tpn)

# stage_in cp — choose MIN SPM across ALL TPNs; return (cost, chosen_prod_tpn)
def stage_in_cp_producer_cost(stage: str, dst_local: str):
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == f"beegfs-{dst_local}")
        & (DF.storage_dst == dst_local)
    ]
    if d is None or d.empty or "prod_spm" not in d.columns:
        return (float("nan"), None)
    idx = d["prod_spm"].astype(float).idxmin()
    row = d.loc[idx]
    chosen_cost = float(row["prod_spm"])
    chosen_tpn = int(row["prod_tpn"]) if "prod_tpn" in d.columns and not pd.isna(row["prod_tpn"]) else None
    return (chosen_cost, chosen_tpn)

# shared read for initial-only (consumer side)
def stage_in_shared_read_runtime(stage: str, cons_tpn_val: int) -> float:
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == "beegfs")
        & (DF.storage_dst == "beegfs")
    ]
    d = _tpn_filter_exec_read(d, cons_tpn_val)
    return _min_col(d, "cons_spm")

# local read for initial-only after cp (consumer side)
def stage_in_initial_local_read_runtime(stage: str, dst_local: str, cons_tpn_val: int) -> float:
    d = DF[
        (DF.producer == f"stage_in-{stage}")
        & (DF.consumer == stage)
        & (DF.storage_src == f"beegfs-{dst_local}")
        & (DF.storage_dst == dst_local)
    ]
    d = _tpn_filter_exec_read(d, cons_tpn_val)
    return _min_col(d, "cons_spm")

# execution READ for (P->C): measured on σ(C); storage_dst fixed, storage_src unconstrained
def exec_read_cost(p: str, c: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == p)
        & (DF.consumer == c)
        & (DF.storage_dst == store_y)  # src unconstrained; movement handled by stage_in/out
    ]
    d = _tpn_filter_exec_read(d, c_tpn_val)
    return _min_col(d, "cons_spm")

# execution WRITE for (C->D): measured on σ(C); storage_src fixed, storage_dst unconstrained
def exec_write_cost(c: str, d: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == d)
        & (DF.storage_src == store_y)  # dst unconstrained; movement handled by stage_out
    ]
    d = _tpn_filter_exec_write(d, c_tpn_val)
    return _min_col(d, "prod_spm")

# final WRITE for stage C via stage_out row (local->beegfs): producer side only
def final_exec_write_from_stageout(c: str, store_y: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == f"stage_out-{c}")
        & (DF.storage_src == store_y)
        & (DF.storage_dst == f"{store_y}-beegfs")
    ]
    d = _tpn_filter_final_write_stageout(d, c_tpn_val)
    return _min_col(d, "prod_spm")

# final WRITE for stage C when it runs on beegfs (producer-side via stage_out beegfs->*)
def final_exec_write_on_beegfs_from_stageout(c: str, c_tpn_val: int) -> float:
    d = DF[
        (DF.producer == c)
        & (DF.consumer == f"stage_out-{c}")
        & (DF.storage_src == "beegfs")
    ]
    if not d.empty and "storage_dst" in d.columns:
        d = d[d["storage_dst"].astype(str).str.startswith("beegfs-")]
    d = _tpn_filter_final_write_stageout(d, c_tpn_val)
    return _min_col(d, "prod_spm")

# -----------------------------
# Validity filter
# -----------------------------
def config_is_valid(cfg: dict) -> bool:
    # forbid cross-local transitions (e.g., ssd<->tmpfs) which are unmeasured
    for c, preds in PREDS.items():
        if not preds:
            continue
        for p in preds:
            if p == "initial_data":
                continue
            X, Y = cfg[p], cfg[c]
            if X != Y and (X, Y) in FORBIDDEN_CROSS:
                return False
    return True

# -----------------------------
# Stage_in actions (in_k) for order k
# -----------------------------
def _stage_in_actions_for_order(order_k: int, cfg: dict):
    """
    ANY order:
      - stage_in cp (beegfs->local) for stages on local that need shared data
      - cost: PRODUCER-only (prod_spm), picked as MIN across all TPNs; record chosen prod_tpn.
    Returns list of actions: {stage, cost, tpn}
    """
    actions = []
    for c in [s for s in TASKS if ORDER[s] == order_k]:
        Y = cfg[c]
        if Y in LOCAL_STORES:
            need_cp = any((p == "initial_data") or (cfg.get(p) == "beegfs") for p in PREDS[c])
            if need_cp:
                cost, chosen_tpn = stage_in_cp_producer_cost(c, Y)
                cost = _nz(cost)
                if cost > 0:
                    actions.append({"stage": c, "cost": cost, "tpn": chosen_tpn})
    return actions

# -----------------------------
# Stage_out actions (out_k) for order k
# -----------------------------
def _stage_out_actions_for_order(order_k: int, cfg: dict):
    """
    For any order:
      - For each producer p in this order, and for each downstream consumer storage Y:
          * if cfg[p] == 'beegfs' and Y in local: beegfs->Y cp (CONSUMER-only, min SPM across TPNs)
          * if cfg[p] in local and Y == 'beegfs': X->beegfs cp (CONSUMER-only, min SPM across TPNs)
          * if cfg[p] in local and Y == cfg[p]:  X->X fan-out scp (CONSUMER-only, min SPM across TPNs)
      - De-dup by (p, Y); attach action to stage p with chosen cons_tpn from selected row.
      - Additionally, at FINAL order, if p is local, also allow X->beegfs cp even without successors (final flush).
    Returns actions: list of {stage, cost, tpn}
    """
    actions = []
    stages_k = [s for s in TASKS if ORDER[s] == order_k]

    so_targets = {}   # map (p, Y) -> (cost, tpn)
    fan_targets = {}  # map (p, Y) -> (cost, tpn)

    for p in stages_k:
        succs = [c for c in TASKS if ORDER[c] > order_k and p in PREDS[c]]
        targets = sorted({cfg[c] for c in succs}) if succs else []
        Xp = cfg[p]

        # consider successors' storages
        for Y in targets:
            if Xp == "beegfs" and Y in LOCAL_STORES:
                val, tpn = stage_out_consumer_cost(p, "beegfs", f"beegfs-{Y}")
                if (p, Y) not in so_targets or _nz(val) > _nz(so_targets[(p, Y)][0]):
                    so_targets[(p, Y)] = (val, tpn)
            if Xp in LOCAL_STORES and Y == "beegfs":
                val2, tpn2 = stage_out_consumer_cost(p, Xp, f"{Xp}-beegfs")
                if (p, Y) not in so_targets or _nz(val2) > _nz(so_targets[(p, Y)][0]):
                    so_targets[(p, Y)] = (val2, tpn2)
            if Xp in LOCAL_STORES and Y == Xp:
                valf, tpnf = stage_out_consumer_cost(p, Xp, f"{Xp}-{Xp}")
                if (p, Y) not in fan_targets or _nz(valf) > _nz(fan_targets[(p, Y)][0]):
                    fan_targets[(p, Y)] = (valf, tpnf)

        # final flush even without successors
        if order_k == O_MAX and Xp in LOCAL_STORES:
            val3, tpn3 = stage_out_consumer_cost(p, Xp, f"{Xp}-beegfs")
            if (p, "beegfs") not in so_targets or _nz(val3) > _nz(so_targets[(p, "beegfs")][0]):
                so_targets[(p, "beegfs")] = (val3, tpn3)

    # combine per (p,Y): prefer whichever movement has larger cost (cp vs fan-out)
    combined = {}
    for (p, Y), (v, tpnv) in so_targets.items():
        if (p, Y) not in combined or _nz(v) > _nz(combined[(p, Y)][0]):
            combined[(p, Y)] = (v, tpnv)
    for (p, Y), (v, tpnv) in fan_targets.items():
        if (p, Y) not in combined or _nz(v) > _nz(combined[(p, Y)][0]):
            combined[(p, Y)] = (v, tpnv)

    for (p, Y), (v, tpnv) in combined.items():
        if _nz(v) > 0:
            actions.append({"stage": p, "cost": _nz(v), "tpn": tpnv})

    return actions

# -----------------------------
# Execution per stage (READ + WRITE)
# -----------------------------
def _exec_time_for_stage(c: str, cfg: dict, N: int):
    """
    Compute execution for stage c:
      read_sum  = SUM over predecessors (consumer-side read on σ(c)) or initial-read path
      write_max = MAX over successors (producer-side write on σ(c)) or final-write via stage_out
      exec_time = read_sum + write_max

    Returns:
      exec_time, read_sum, write_max, read_considered, write_considered
    """
    Y = cfg[c]
    tpn = tpn_map(N)
    ctpn = tpn.get(c, 1)

    # READ side: sum over predecessors
    read_components = []
    for p in PREDS[c]:
        if p == "initial_data":
            continue
        val = _nz(exec_read_cost(p, c, Y, ctpn))
        read_components.append(val)

    if only_initial(c):
        if Y in LOCAL_STORES:
            init_val = _nz(stage_in_initial_local_read_runtime(c, Y, ctpn))
        else:
            init_val = _nz(stage_in_shared_read_runtime(c, ctpn))
        read_components.append(init_val)

    read_considered = len(read_components) > 0
    read_sum = sum(read_components) if read_components else 0.0

    # WRITE side: max over successors (or final write)
    write_candidates = []
    succs = CONSUMERS.get(c, [])
    real_succs = [d for d in succs if ORDER[d] > ORDER[c]]
    if real_succs:
        for d in real_succs:
            val = _nz(exec_write_cost(c, d, Y, ctpn))
            write_candidates.append(val)
    else:
        # final stage write via stage_out
        if Y in LOCAL_STORES:
            val = _nz(final_exec_write_from_stageout(c, Y, ctpn))
            write_candidates.append(val)
        elif Y == "beegfs":
            val = _nz(final_exec_write_on_beegfs_from_stageout(c, ctpn))
            write_candidates.append(val)

    write_considered = len(write_candidates) > 0
    write_max = max(write_candidates) if write_candidates else 0.0

    return read_sum + write_max, read_sum, write_max, read_considered, write_considered

# -----------------------------
# One configuration evaluation
# -----------------------------
def one_config(N: int, cfg: dict):
    order_map = _order_groups()

    # per-stage output cells
    stage_cells = {s: {"in": "-", "read": "-", "write": "-", "out": "-"} for s in TASKS}

    in_times = {}
    exec_times = {}
    out_times = {}
    critical_tokens = []

    # ---- per-order barriers ----
    for ok, stages_k in order_map.items():
        # IN_k
        in_actions = _stage_in_actions_for_order(ok, cfg)
        in_k = max([a["cost"] for a in in_actions], default=0.0)
        in_times[ok] = in_k

        # stamp all considered in_S
        per_stage_in = {}
        for a in in_actions:
            per_stage_in[a["stage"]] = max(per_stage_in.get(a["stage"], 0.0), a["cost"])
        for s, v in per_stage_in.items():
            stage_cells[s]["in"] = f"{v:.6f}"

        if in_k > 0:
            winners = [a for a in in_actions if abs(a["cost"] - in_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x["stage"]))
            win = winners[0]
            tpn_tag = win.get("tpn", "NA")
            critical_tokens.append(f"stage_in_{win['stage']}({tpn_tag})")

        # EXEC_k
        per_stage_exec = []
        for c in stages_k:
            et, rsum, wmax, r_cons, w_cons = _exec_time_for_stage(c, cfg, N)
            per_stage_exec.append((c, et, rsum, wmax, r_cons, w_cons))

        # stamp all considered read/write
        for (c, et, rsum, wmax, r_cons, w_cons) in per_stage_exec:
            if r_cons:
                stage_cells[c]["read"] = f"{rsum:.6f}"
            if w_cons:
                stage_cells[c]["write"] = f"{wmax:.6f}"

        exec_k = max([et for (_, et, _, _, _, _) in per_stage_exec], default=0.0)
        exec_times[ok] = exec_k
        if exec_k > 0:
            winners = [t for t in per_stage_exec if abs(t[1] - exec_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x[0]))
            c, _, rsum, wmax, r_cons, w_cons = winners[0]
            stage_tpn = tpn_map(N).get(c, "NA")
            if r_cons and rsum > 0:
                critical_tokens.append(f"read_{c}({stage_tpn})")
            if w_cons and wmax > 0:
                critical_tokens.append(f"write_{c}({stage_tpn})")

        # OUT_k
        out_actions = _stage_out_actions_for_order(ok, cfg)
        out_k = max([a["cost"] for a in out_actions], default=0.0)
        out_times[ok] = out_k

        # stamp all considered out_S
        per_stage_out = {}
        for a in out_actions:
            per_stage_out[a["stage"]] = max(per_stage_out.get(a["stage"], 0.0), a["cost"])
        for s, v in per_stage_out.items():
            stage_cells[s]["out"] = f"{v:.6f}"

        if out_k > 0:
            winners = [a for a in out_actions if abs(a["cost"] - out_k) < 1e-12]
            winners.sort(key=lambda x: _stage_sort_key(x["stage"]))
            win = winners[0]
            tpn_tag = win.get("tpn", "NA")
            critical_tokens.append(f"stage_out_{win['stage']}({tpn_tag})")

    total = float(sum(in_times.values()) + sum(exec_times.values()) + sum(out_times.values()))
    critical_path = "->".join(critical_tokens) if critical_tokens else ""
    return total, in_times, exec_times, out_times, stage_cells, critical_path

# -----------------------------
# Sweep and CSV output
# -----------------------------
def sweep(thresh=None) -> pd.DataFrame:
    rows = []
    ordered_stages = sorted(TASKS.keys(), key=_stage_sort_key)

    for N in GLOBAL_NODES:
        for vals in itertools.product(STORAGES, repeat=len(TASKS)):
            cfg = dict(zip(TASKS, vals))
            if not config_is_valid(cfg):
                continue

            total, in_t, exe_t, out_t, stage_cells, crit = one_config(N, cfg)
            if thresh and total >= thresh:
                continue

            row = {f"{k}_stor": v for k, v in cfg.items()}
            row.update(nodes=N, total=total)
            for o in ORDERS:
                row[f"in_o{o}"] = in_t.get(o, 0.0)
                row[f"exec_o{o}"] = exe_t.get(o, 0.0)
                row[f"out_o{o}"] = out_t.get(o, 0.0)
            row["critical_path"] = crit

            # per-stage reporting: in_S, read_S, write_S, out_S
            for s in ordered_stages:
                row[f"in_{s}"]   = stage_cells[s]["in"]
                row[f"read_{s}"] = stage_cells[s]["read"]
                row[f"write_{s}"]= stage_cells[s]["write"]
                row[f"out_{s}"]  = stage_cells[s]["out"]

            rows.append(row)

    df = pd.DataFrame(rows)

    base_cols = ["nodes", "total"] \
                + [f"in_o{o}" for o in ORDERS] \
                + [f"exec_o{o}" for o in ORDERS] \
                + [f"out_o{o}" for o in ORDERS] \
                + ["critical_path"]

    stor_cols = [f"{s}_stor" for s in ordered_stages]

    per_stage_cols = []
    for s in ordered_stages:
        per_stage_cols += [f"in_{s}", f"read_{s}", f"write_{s}", f"out_{s}"]

    ordered_all = [c for c in base_cols + stor_cols + per_stage_cols if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_all]
    df = df[ordered_all + remaining]

    # sort rows by total then nodes
    return df.sort_values(["total", "nodes"], ascending=[True, True], kind="mergesort").reset_index(drop=True)

def write_csvs() -> pd.DataFrame:
    full = sweep()
    out_path = f"{OUT_DIR}/workflow_makespan_stageorder.csv"
    full.to_csv(out_path, index=False)
    return full
