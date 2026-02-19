# cart_summaries.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from cart_features import detect_store_cols

def strict_wildcard_summary_joint_unique(
    df_all: pd.DataFrame,
    df_labeled: pd.DataFrame,
    store_cols: List[str],
    stat_cols_map: Dict[str, Tuple[str, str]],
    additional_cols: List[str] | None = None,
    include_nodes: bool = True,
) -> pd.DataFrame:
    additional_cols = additional_cols or []
    wc_cols = list(store_cols)
    if include_nodes and ("nodes" in df_all.columns):
        wc_cols.append("nodes")

    def _vals_set(series_like) -> tuple:
        vals = pd.Series(series_like).dropna().astype(str).unique().tolist()
        return tuple(sorted(vals))

    reg_order = (
        df_labeled.groupby("region")["total"]
        .agg(median_total="median").reset_index()
        .sort_values("median_total", ascending=True)
    )
    order = reg_order["region"].tolist()
    pos = {r: i for i, r in enumerate(order)}

    reg_present: Dict[int, Dict[str, tuple]] = {}
    for reg, grp in df_labeled.groupby("region"):
        reg_present[reg] = {c: _vals_set(grp[c]) for c in wc_cols}

    rows = []
    for reg in order:
        grp = df_labeled[df_labeled["region"] == reg]

        fixed_cols = [c for c in wc_cols if grp[c].nunique(dropna=True) == 1]
        cond = pd.Series(True, index=df_all.index)
        for c in fixed_cols:
            cond &= (df_all[c] == grp[c].iloc[0])
        df_cond = df_all[cond]

        present_sets = {c: set(map(str, grp[c].dropna().unique())) for c in wc_cols}
        domain_sets  = {c: set(map(str, df_cond[c].dropna().unique())) for c in wc_cols}

        eligible = {c: (len(present_sets[c]) > 1 and len(domain_sets[c]) > 1 and present_sets[c] == domain_sets[c]) for c in wc_cols}
        S = {c for c, ok in eligible.items() if ok}

        later = order[pos[reg] + 1:]

        def sig_exc(region_id: int, exclude: set[str]) -> tuple:
            parts = []
            for k in wc_cols:
                if k in exclude:
                    continue
                parts.append((k, reg_present[region_id][k]))
            return tuple(sorted(parts, key=lambda x: x[0]))

        def clashes(Sset: set[str]) -> int:
            my = sig_exc(reg, Sset)
            return sum(sig_exc(r2, Sset) == my for r2 in later)

        ccount = clashes(S)
        while ccount > 0 and len(S) > 0:
            drop = None; best = ccount
            for c in list(S):
                S_try = S - {c}
                cc = clashes(S_try)
                if cc < best or (cc == best and len(present_sets[c]) < len(present_sets.get(drop, []))):
                    best = cc; drop = c
            if drop is None:
                break
            S.remove(drop); ccount = best

        wc_out = {}
        for c in wc_cols:
            if len(present_sets[c]) == 1:
                wc_out[c] = next(iter(present_sets[c]))
            elif c in S:
                wc_out[c] = "*"
            else:
                wc_out[c] = ",".join(sorted(present_sets[c]))

        def stat(series: pd.Series, how: str):
            x = series.to_numpy(dtype=float)
            if how == "n": return int(x.size)
            if how == "mean": return float(np.mean(x))
            if how == "std": return float(np.std(x, ddof=1) if x.size > 1 else 0.0)
            if how == "cv":
                m = float(np.mean(x)); s = float(np.std(x, ddof=1) if x.size > 1 else 0.0)
                return (s / m) if m else np.nan
            if how == "q25": return float(np.quantile(x, 0.25))
            if how == "median": return float(np.quantile(x, 0.5))
            if how == "q75": return float(np.quantile(x, 0.75))
            if how == "min": return float(np.min(x)) if x.size else np.nan
            if how == "max": return float(np.max(x)) if x.size else np.nan
            return np.nan

        stats_out = {name: stat(grp[src], how) for name, (src, how) in stat_cols_map.items()}
        for col in additional_cols:
            stats_out[f"mean_{col}"] = stat(grp[col], "mean")
            stats_out[f"cv_{col}"]   = stat(grp[col], "cv")

        rows.append({"region": int(reg), "size": int(len(grp)), **stats_out, **wc_out})

    out = pd.DataFrame(rows).sort_values("median_total", ascending=True).reset_index(drop=True)
    return out

def compute_regret(df: pd.DataFrame):
    global_best = df["total"].min()
    cfg = df[["region", "nodes", "total"] + detect_store_cols(df)].copy()
    cfg["regret_vs_global_best"] = cfg["total"] - global_best
    reg = df.groupby("region")["total"].mean().reset_index().rename(columns={"total": "mean_total"})
    reg["regret_vs_global_best"] = reg["mean_total"] - global_best
    return reg, cfg

def critical_path_prevalence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stores = detect_store_cols(df)
    for reg, grp in df.groupby("region"):
        size = len(grp)
        counts = grp["critical_path"].value_counts(dropna=False)
        prevalent = counts.index[0] if not counts.empty else ""
        for path, cnt in counts.items():
            rep = grp[grp["critical_path"] == path].iloc[0] if path in grp["critical_path"].values else grp.iloc[0]
            base = {
                "region": reg,
                "region_size": size,
                "prevalent_critical_path": prevalent,
                "critical_path": path if pd.notna(path) else "",
                "count": int(cnt),
                "fraction": float(cnt / size) if size else 0.0,
                "config_nodes": int(rep["nodes"]) if "nodes" in rep else None,
            }
            for c in stores:
                base[c] = rep[c]
            rows.append(base)
    out = pd.DataFrame(rows).sort_values(["region", "count"], ascending=[True, False])
    return out
