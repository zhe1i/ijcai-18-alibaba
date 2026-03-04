from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import DataSchema
from .utils import ensure_dir



def _hash_key(parts: list[str]) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:12]



def _safe_split(v: Any, sep: str = ";") -> list[str]:
    if pd.isna(v):
        return []
    s = str(v).strip()
    if not s:
        return []
    return [t for t in s.split(sep) if t and t != "nan"]



def parse_predict_category_property(v: Any) -> tuple[set[str], set[str]]:
    cats: set[str] = set()
    props: set[str] = set()
    for token in _safe_split(v, ";"):
        if ":" not in token:
            cats.add(token)
            continue
        cat, prop_str = token.split(":", 1)
        if cat:
            cats.add(cat)
        for p in prop_str.split(","):
            if p and p != "-1":
                props.add(p)
    return cats, props



def build_multi_value_cache(
    df: pd.DataFrame,
    schema: DataSchema,
    cache_dir: str | Path,
    split_name: str,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    cache_root = ensure_dir(cache_dir)
    key = _hash_key([split_name, str(len(df)), "|".join(schema.multi_value_cols)])
    cache_file = cache_root / f"mv_cache_{split_name}_{key}.pkl"
    if use_cache and cache_file.exists():
        with cache_file.open("rb") as f:
            payload = pickle.load(f)
        return payload["df"], payload["meta"]

    out = df.copy()
    meta: dict[str, str] = {}

    for col in schema.multi_value_cols:
        token_col = f"mv_tokens__{col}"
        len_col = f"mv_len__{col}"
        out[token_col] = out[col].map(_safe_split)
        out[len_col] = out[token_col].map(len).astype("int16")
        meta[col] = token_col

    cat_col = next((c for c in schema.multi_value_cols if "category" in c.lower() and "predict" not in c.lower()), None)
    prop_col = next((c for c in schema.multi_value_cols if "property" in c.lower() and "predict" not in c.lower()), None)
    pred_col = next((c for c in schema.multi_value_cols if "predict" in c.lower()), None)

    if cat_col and pred_col:
        item_cat_tokens = out[f"mv_tokens__{cat_col}"].tolist()
        pred_tokens = out[pred_col].tolist()
        cat_hit_cnt = []
        cat_jaccard = []
        cat_cover_item = []
        cat_cover_pred = []
        main_cat_hit = []
        for ic, pv in zip(item_cat_tokens, pred_tokens, strict=False):
            item_cats = set(ic)
            pred_cats, _ = parse_predict_category_property(pv)
            inter = item_cats & pred_cats
            union = item_cats | pred_cats
            cat_hit_cnt.append(float(len(inter)))
            cat_jaccard.append(float(len(inter) / len(union)) if union else 0.0)
            cat_cover_item.append(float(len(inter) / len(item_cats)) if item_cats else 0.0)
            cat_cover_pred.append(float(len(inter) / len(pred_cats)) if pred_cats else 0.0)
            main_cat = ic[0] if ic else None
            main_cat_hit.append(float(main_cat in pred_cats) if main_cat else 0.0)

        out["match_cat_hit_cnt"] = cat_hit_cnt
        out["match_cat_jaccard"] = cat_jaccard
        out["match_cat_cover_item"] = cat_cover_item
        out["match_cat_cover_pred"] = cat_cover_pred
        out["match_main_cat_hit"] = main_cat_hit

    if prop_col and pred_col:
        item_prop_tokens = out[f"mv_tokens__{prop_col}"].tolist()
        pred_tokens = out[pred_col].tolist()
        prop_hit_cnt = []
        prop_jaccard = []
        prop_cover_item = []
        prop_cover_pred = []
        for ip, pv in zip(item_prop_tokens, pred_tokens, strict=False):
            item_props = set(ip)
            _, pred_props = parse_predict_category_property(pv)
            inter = item_props & pred_props
            union = item_props | pred_props
            prop_hit_cnt.append(float(len(inter)))
            prop_jaccard.append(float(len(inter) / len(union)) if union else 0.0)
            prop_cover_item.append(float(len(inter) / len(item_props)) if item_props else 0.0)
            prop_cover_pred.append(float(len(inter) / len(pred_props)) if pred_props else 0.0)

        out["match_prop_hit_cnt"] = prop_hit_cnt
        out["match_prop_jaccard"] = prop_jaccard
        out["match_prop_cover_item"] = prop_cover_item
        out["match_prop_cover_pred"] = prop_cover_pred

    if use_cache:
        with cache_file.open("wb") as f:
            pickle.dump({"df": out, "meta": meta}, f)

    return out, meta


@dataclass
class StatFeatureBuilder:
    label_col: str
    cvr_group_cols: list[str]
    freq_cols: list[str]
    day_col: str
    prior_strength: float = 20.0
    drift_z_threshold: float = 1.0
    global_mean_: float = 0.0
    cvr_maps_: dict[str, dict[Any, float]] = field(default_factory=dict)
    freq_maps_: dict[str, dict[Any, float]] = field(default_factory=dict)
    day_drift_map_: dict[Any, float] = field(default_factory=dict)

    def fit(self, train_df: pd.DataFrame) -> None:
        y = pd.to_numeric(train_df[self.label_col], errors="coerce").fillna(0.0)
        self.global_mean_ = float(y.mean())
        alpha = self.global_mean_ * self.prior_strength
        beta = (1.0 - self.global_mean_) * self.prior_strength

        self.cvr_maps_.clear()
        for col in self.cvr_group_cols:
            if col not in train_df.columns:
                continue
            agg = (
                train_df.groupby(col, dropna=False)[self.label_col]
                .agg(["sum", "count"])
                .rename(columns={"sum": "sum_y", "count": "cnt"})
            )
            smooth = (agg["sum_y"] + alpha) / (agg["cnt"] + alpha + beta)
            self.cvr_maps_[col] = smooth.to_dict()

        self.freq_maps_.clear()
        for col in self.freq_cols:
            if col not in train_df.columns:
                continue
            cnt = train_df[col].value_counts(dropna=False)
            self.freq_maps_[col] = cnt.to_dict()

        if self.day_col in train_df.columns:
            day_stats = train_df.groupby(self.day_col)[self.label_col].mean()
            mean = float(day_stats.mean())
            std = float(day_stats.std()) if day_stats.size > 1 else 0.0
            if std <= 1e-8:
                std = 1.0
            self.day_drift_map_ = ((day_stats - mean) / std).to_dict()
        else:
            self.day_drift_map_ = {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        for col, mapping in self.cvr_maps_.items():
            out[f"cvr_{col}"] = (
                df[col].map(mapping).fillna(self.global_mean_).astype("float32")
                if col in df.columns
                else self.global_mean_
            )

        for col, mapping in self.freq_maps_.items():
            vals = df[col].map(mapping).fillna(0.0) if col in df.columns else 0.0
            out[f"log_freq_{col}"] = np.log1p(pd.to_numeric(vals, errors="coerce").fillna(0.0)).astype("float32")

        if self.day_col in df.columns:
            out["drift_score"] = df[self.day_col].map(self.day_drift_map_).fillna(0.0).astype("float32")
        else:
            out["drift_score"] = 0.0

        out["drift_scenario"] = np.where(
            out["drift_score"].abs() >= self.drift_z_threshold,
            "Drift",
            "Normal",
        )

        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_col": self.label_col,
            "cvr_group_cols": self.cvr_group_cols,
            "freq_cols": self.freq_cols,
            "day_col": self.day_col,
            "prior_strength": self.prior_strength,
            "drift_z_threshold": self.drift_z_threshold,
            "global_mean_": self.global_mean_,
            "cvr_maps_": self.cvr_maps_,
            "freq_maps_": self.freq_maps_,
            "day_drift_map_": self.day_drift_map_,
        }
