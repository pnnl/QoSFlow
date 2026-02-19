# adapter_flowforecaster.py  (top-level; imports assume pwd = workflow_analysis)

from __future__ import annotations
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd

from modules.workflow_data_staging import insert_data_staging_rows
from modules.workflow_interpolation import (
    calculate_aggregate_filesize_per_node,
    estimate_transfer_rates_for_workflow,
)
from modules.workflow_spm_calculator import (
    calculate_spm_from_wfg,
)
from modules.workflow_config import STORAGE_LIST

def build_adapter_input_from_csv(csv_file_path: str,
                                 workflow_name: Optional[str] = None,
                                 debug: bool = False) -> pd.DataFrame:
    """Analyzer’s pre-steps: load → insert staging → compute aggregateFilesizeMB per node."""
    df = pd.read_csv(csv_file_path)
    if debug:
        print(f"[adapter] Loaded {len(df)} rows")
    df = insert_data_staging_rows(df, debug=debug)
    df = calculate_aggregate_filesize_per_node(df, debug=debug)
    if 'transferSize' not in df.columns:
        df['transferSize'] = 4096  # conservative fallback; staging normally sets this
    return df

@dataclass
class FFRules:
    """Minimal rules to override ONLY aggregateFilesizeMB / transferSize."""
    # e.g. {'aggregateFilesizeMB': 2.0, 'transferSize': 1.5}
    global_factors: Optional[Dict[str, float]] = None
    # list of rules: {'where': {'taskName':'foo','operation':'read'},
    #                 'set': {'aggregateFilesizeMB':1234}, 'scale': {'transferSize':2.0}}
    overrides: Optional[List[Dict[str, Any]]] = None

def apply_flowforecaster_scales(df: pd.DataFrame,
                                rules: Optional[FFRules] = None,
                                debug: bool = False) -> pd.DataFrame:
    """Mutate ONLY the two target inputs used by the 4D interpolation."""
    out = df.copy()
    if rules and rules.global_factors:
        if 'aggregateFilesizeMB' in rules.global_factors:
            out['aggregateFilesizeMB'] = out['aggregateFilesizeMB'].astype(float) * float(rules.global_factors['aggregateFilesizeMB'])
        if 'transferSize' in rules.global_factors:
            out['transferSize'] = out['transferSize'].astype(float) * float(rules.global_factors['transferSize'])
    if rules and rules.overrides:
        for i, rule in enumerate(rules.overrides, start=1):
            where = rule.get('where', {})
            set_vals = rule.get('set', {})
            scale_vals = rule.get('scale', {})
            mask = pd.Series([True]*len(out))
            for col, val in where.items():
                mask &= (out[col] == val)
            if set_vals:
                if 'aggregateFilesizeMB' in set_vals:
                    out.loc[mask, 'aggregateFilesizeMB'] = float(set_vals['aggregateFilesizeMB'])
                if 'transferSize' in set_vals:
                    out.loc[mask, 'transferSize'] = float(set_vals['transferSize'])
            if scale_vals:
                if 'aggregateFilesizeMB' in scale_vals:
                    out.loc[mask, 'aggregateFilesizeMB'] = out.loc[mask, 'aggregateFilesizeMB'].astype(float) * float(scale_vals['aggregateFilesizeMB'])
                if 'transferSize' in scale_vals:
                    out.loc[mask, 'transferSize'] = out.loc[mask, 'transferSize'].astype(float) * float(scale_vals['transferSize'])
    return out

def run_spm_with_ff(df_prepared: pd.DataFrame,
                    ior_df: pd.DataFrame,
                    storage_list: Optional[List[str]] = None,
                    allowed_parallelism: Optional[List[int]] = None,
                    multi_nodes: bool = True,
                    debug: bool = False,
                    workflow_name: Optional[str] = None) -> Dict[str, Any]:
    """Same path as analyzer after staging/aggregate: estimate rates → WFG+edges → SPM."""
    if storage_list is None:
        storage_list = STORAGE_LIST
    wf = df_prepared.copy()
    if allowed_parallelism is None:
        max_p = int(wf['parallelism'].max()) if 'parallelism' in wf.columns and len(wf['parallelism']) else 1
        allowed_parallelism = [1, max_p]
    wf = estimate_transfer_rates_for_workflow(
        wf, ior_df, storage_list, allowed_parallelism, multi_nodes=multi_nodes, debug=debug
    )
    from modules.workflow_spm_calculator import calculate_spm_for_edges
    WFG = calculate_spm_for_edges(wf, debug=debug, workflow_name=workflow_name)
    return calculate_spm_from_wfg(WFG, debug=debug)
