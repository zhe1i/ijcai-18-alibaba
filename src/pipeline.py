from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .feature_engineering import build_multi_value_cache
from .schema import DataSchema, add_missing_indicator_columns, add_time_columns
from .utils import ensure_dir


@dataclass
class PreparedFrame:
    df: pd.DataFrame
    missing_cols: list[str]
    match_cols: list[str]
    mv_len_cols: list[str]



def read_txt_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=" ")



def prepare_frame(
    df: pd.DataFrame,
    schema: DataSchema,
    cache_dir: str | Path,
    split_name: str,
    use_cache: bool = True,
) -> PreparedFrame:
    out = add_time_columns(df, schema)
    out, missing_cols = add_missing_indicator_columns(out, schema)
    out, _ = build_multi_value_cache(
        df=out,
        schema=schema,
        cache_dir=cache_dir,
        split_name=split_name,
        use_cache=use_cache,
    )
    match_cols = sorted([c for c in out.columns if c.startswith("match_")])
    mv_len_cols = sorted([c for c in out.columns if c.startswith("mv_len__")])
    return PreparedFrame(df=out, missing_cols=missing_cols, match_cols=match_cols, mv_len_cols=mv_len_cols)



def infer_stat_columns(schema: DataSchema, config_features: dict[str, Any]) -> tuple[list[str], list[str]]:
    cvr_group_cols = config_features.get("cvr_group_cols", [])
    freq_cols = config_features.get("freq_cols", [])

    if not cvr_group_cols:
        cvr_group_cols = [c for c in [schema.user_col, schema.item_col, schema.shop_col] if c is not None]
        if "item_brand_id" in schema.categorical_cols:
            cvr_group_cols.append("item_brand_id")

    if not freq_cols:
        freq_cols = [c for c in [schema.user_col, schema.item_col, schema.shop_col] if c is not None]

    cvr_group_cols = [c for c in cvr_group_cols if c]
    freq_cols = [c for c in freq_cols if c]
    return cvr_group_cols, freq_cols



def infer_gate_columns(schema: DataSchema, freq_cols: list[str]) -> list[str]:
    gate_cols = [schema.day_col, schema.hour_col, "drift_score"]
    for c in freq_cols:
        gate_cols.append(f"log_freq_{c}")
    return gate_cols



def build_dense_columns(
    schema: DataSchema,
    missing_cols: list[str],
    mv_len_cols: list[str],
    match_cols: list[str],
    stat_feature_cols: list[str],
    include_match: bool,
) -> list[str]:
    cols = []
    cols.extend([c for c in schema.numeric_cols if c not in [schema.day_col, schema.hour_col]])
    cols.extend([schema.day_col, schema.hour_col])
    cols.extend(missing_cols)
    cols.extend(mv_len_cols)
    if include_match:
        cols.extend(match_cols)
    cols.extend(stat_feature_cols)

    dedup = []
    seen = set()
    for c in cols:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup



def dump_pickle(obj: Any, path: str | Path) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("wb") as f:
        pickle.dump(obj, f)



def load_pickle(path: str | Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)
