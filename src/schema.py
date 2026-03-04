from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DataSchema:
    label_col: str | None
    instance_id_col: str
    time_col: str | None
    day_col: str
    hour_col: str
    multi_value_cols: list[str] = field(default_factory=list)
    numeric_cols: list[str] = field(default_factory=list)
    categorical_cols: list[str] = field(default_factory=list)
    missing_sentinel_cols: list[str] = field(default_factory=list)
    user_col: str | None = None
    item_col: str | None = None
    shop_col: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_col": self.label_col,
            "instance_id_col": self.instance_id_col,
            "time_col": self.time_col,
            "day_col": self.day_col,
            "hour_col": self.hour_col,
            "multi_value_cols": self.multi_value_cols,
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "missing_sentinel_cols": self.missing_sentinel_cols,
            "user_col": self.user_col,
            "item_col": self.item_col,
            "shop_col": self.shop_col,
        }



def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for key in candidates:
        if key.lower() in lowered:
            return lowered[key.lower()]
    for c in columns:
        cl = c.lower()
        if any(key.lower() in cl for key in candidates):
            return c
    return None



def _detect_time_col(df: pd.DataFrame) -> str | None:
    cols = df.columns.tolist()
    named = _pick_column(cols, ["context_timestamp", "timestamp", "ts", "time"])
    if named is not None:
        return named

    for c in cols:
        s = df[c]
        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            sample = pd.to_numeric(s.head(5000), errors="coerce").dropna()
            if sample.empty:
                continue
            med = float(sample.median())
            if 1e9 <= med <= 2e9:
                return c
    return None



def _detect_multi_value_cols(df: pd.DataFrame, sample_rows: int = 3000) -> list[str]:
    mv_cols: list[str] = []
    for c in df.columns:
        s = df[c].astype(str).head(sample_rows)
        if s.str.contains(";", regex=False).any():
            mv_cols.append(c)
    return mv_cols



def detect_schema(
    df: pd.DataFrame,
    label_col: str = "is_trade",
    instance_id_candidates: list[str] | None = None,
) -> DataSchema:
    if instance_id_candidates is None:
        instance_id_candidates = ["instance_id", "sample_id"]

    cols = df.columns.tolist()
    label = label_col if label_col in cols else None

    instance_id_col = _pick_column(cols, instance_id_candidates)
    if instance_id_col is None:
        raise ValueError("Could not detect instance id column.")

    time_col = _detect_time_col(df)

    mv_cols = _detect_multi_value_cols(df)
    if label and label in mv_cols:
        mv_cols.remove(label)

    user_col = _pick_column(cols, ["user_id", "uid", "user"])
    item_col = _pick_column(cols, ["item_id", "iid", "item"])
    shop_col = _pick_column(cols, ["shop_id", "seller_id", "shop"])

    excluded = {instance_id_col, label}
    if time_col:
        excluded.add(time_col)
    excluded = {c for c in excluded if c is not None}

    numeric_cols: list[str] = []
    categorical_cols: list[str] = []
    missing_cols: list[str] = []

    for c in cols:
        if c in excluded or c in mv_cols:
            continue
        s = df[c]
        if (pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s)) and (s == -1).any():
            missing_cols.append(c)

        if pd.api.types.is_float_dtype(s):
            numeric_cols.append(c)
            continue

        if pd.api.types.is_integer_dtype(s):
            nunique = int(s.nunique(dropna=True))
            c_low = c.lower()
            if (
                "_id" in c_low
                or c_low.endswith("_level")
                or c_low.endswith("_page")
                or nunique > 64
            ):
                categorical_cols.append(c)
            else:
                numeric_cols.append(c)
            continue

        categorical_cols.append(c)

    schema = DataSchema(
        label_col=label,
        instance_id_col=instance_id_col,
        time_col=time_col,
        day_col="day",
        hour_col="hour",
        multi_value_cols=mv_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        missing_sentinel_cols=missing_cols,
        user_col=user_col,
        item_col=item_col,
        shop_col=shop_col,
    )
    return schema



def add_time_columns(df: pd.DataFrame, schema: DataSchema) -> pd.DataFrame:
    out = df.copy()
    if schema.day_col in out.columns and schema.hour_col in out.columns:
        return out

    if schema.time_col and schema.time_col in out.columns:
        ts = pd.to_datetime(pd.to_numeric(out[schema.time_col], errors="coerce"), unit="s", errors="coerce")
        out[schema.day_col] = ts.dt.day.astype("float32").fillna(0).astype("int16")
        out[schema.hour_col] = ts.dt.hour.astype("float32").fillna(0).astype("int16")
    else:
        out[schema.day_col] = 0
        out[schema.hour_col] = 0
    return out



def add_missing_indicator_columns(df: pd.DataFrame, schema: DataSchema) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    added: list[str] = []
    for c in schema.missing_sentinel_cols:
        miss_col = f"is_missing__{c}"
        out[miss_col] = (pd.to_numeric(out[c], errors="coerce") == -1).astype("int8")
        added.append(miss_col)
    return out, added



def sanitize_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
