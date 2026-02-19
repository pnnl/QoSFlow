# parser.py ----------------------------------------------------
import json
import pandas as pd
from constants import SPM_CSV_FILE, META_FILE

# ---------------------------------------------------------------------------
# New CSV schema (robustly parsed):
#   Required id columns:
#     producer, consumer
#     producerStorageType, consumerStorageType
#     producerTasksPerNode, consumerTasksPerNode
#
#   New timing columns: separate producer/consumer SPM for each entry.
#   Current canonical fields: "estT_prod", "estT_cons" (case-insensitive).
#   For backward compatibility, we also accept: producer_spm / consumer_spm variations.
#
# We normalize to the columns the model expects:
#   producer, consumer, storage_src, storage_dst, prod_tpn, cons_tpn,
#   prod_spm, cons_spm
# ---------------------------------------------------------------------------

def _normalize_spm_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Lower-case and strip for robust matching
    lc = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lc)

    # Canonical mapping for id fields
    rename_map = {}
    for src, dst in [
        ("producerstoragetype", "storage_src"),
        ("consumerstoragetype", "storage_dst"),
        ("producertaskspernode", "prod_tpn"),
        ("consumertaskspernode", "cons_tpn"),
    ]:
        if src in df.columns:
            rename_map[src] = dst
    if "producer" not in df.columns or "consumer" not in df.columns:
        raise ValueError("CSV must include 'producer' and 'consumer' columns")

    # Find producer/consumer SPM columns (robust)
    def _find_col(candidates):
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    # Prefer new names, fall back to legacy variations
    prod_spm_col = _find_col(["estt_prod", "producer_spm", "producerspm", "producer_spm_value"])
    cons_spm_col = _find_col(["estt_cons", "consumer_spm", "consumerspm", "consumer_spm_value"])

    if prod_spm_col is None or cons_spm_col is None:
        raise ValueError(
            "CSV must include producer/consumer SPM columns (e.g., 'estT_prod' and 'estT_cons')"
        )

    rename_map[prod_spm_col] = "prod_spm"
    rename_map[cons_spm_col] = "cons_spm"

    df = df.rename(columns=rename_map)

    # Keep only normalized columns; tolerate missing TPNs by filling with NA
    keep_cols = [
        "producer", "consumer",
        "storage_src", "storage_dst",
        "prod_tpn", "cons_tpn",
        "prod_spm", "cons_spm",
    ]
    for k in keep_cols:
        if k not in df.columns:
            # optional TPNs may be missing in some files
            if k in ("prod_tpn", "cons_tpn"):
                df[k] = pd.NA
            else:
                raise ValueError(f"Missing required column: {k}")

    # normalize dtypes
    for k in ("prod_spm", "cons_spm"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    for k in ("prod_tpn", "cons_tpn"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors="coerce")

    return df[keep_cols].copy()


def load_rank() -> pd.DataFrame:
    raw = pd.read_csv(SPM_CSV_FILE)
    return _normalize_spm_columns(raw)


def load_meta():
    meta = json.load(open(META_FILE))
    tasks = {k: int(v["num_tasks"]) for k, v in meta.items()}
    order = {k: int(v["stage_order"]) for k, v in meta.items()}
    preds = {k: list(v.get("predecessors", {})) for k, v in meta.items()}
    # build reverse edges (who consumes a stage)
    consumers = {k: [] for k in meta}
    for c, ps in preds.items():
        for p in ps:
            if p != "initial_data":
                consumers[p].append(c)
    return tasks, order, preds, consumers
