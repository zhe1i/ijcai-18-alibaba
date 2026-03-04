from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import safe_auc, safe_logloss

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


@dataclass
class BaselineFoldModel:
    categorical_mappings: dict[str, dict[str, int]]
    feature_cols: list[str]
    categorical_cols: list[str]
    booster: Any



def _fit_category_mapping(series: pd.Series, max_size: int = 300_000) -> dict[str, int]:
    vc = series.astype(str).value_counts(dropna=False)
    vc = vc.head(max_size)
    mapping = {tok: i for i, tok in enumerate(vc.index.tolist())}
    return mapping



def _encode_with_mapping(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    return series.astype(str).map(mapping).fillna(-1).astype(np.int32).to_numpy()



def build_baseline_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, int]], list[str], list[str]]:
    X_train = pd.DataFrame(index=train_df.index)
    X_valid = pd.DataFrame(index=valid_df.index)

    kept_numeric = [c for c in numeric_cols if c in train_df.columns]
    for c in kept_numeric:
        X_train[c] = pd.to_numeric(train_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        X_valid[c] = pd.to_numeric(valid_df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    cat_maps: dict[str, dict[str, int]] = {}
    kept_cats = [c for c in categorical_cols if c in train_df.columns]
    for c in kept_cats:
        mapping = _fit_category_mapping(train_df[c])
        cat_maps[c] = mapping
        X_train[c] = _encode_with_mapping(train_df[c], mapping)
        X_valid[c] = _encode_with_mapping(valid_df[c], mapping)

    feature_cols = X_train.columns.tolist()
    return X_train, X_valid, cat_maps, feature_cols, kept_cats



def fit_lgbm_binary(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_cols: list[str],
    params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> Any:
    if lgb is None:
        raise ImportError("lightgbm is not installed. Please install requirements first.")

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols, free_raw_data=False)
    valid_set = lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical_cols, free_raw_data=False)

    booster = lgb.train(
        params=params,
        train_set=train_set,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        num_boost_round=num_boost_round,
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return booster



def train_baseline_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    lgb_params: dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> tuple[BaselineFoldModel, np.ndarray, dict[str, float]]:
    X_train, X_valid, cat_maps, feature_cols, kept_cats = build_baseline_matrices(
        train_df=train_df,
        valid_df=valid_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
    )

    y_train = pd.to_numeric(train_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y_valid = pd.to_numeric(valid_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    booster = fit_lgbm_binary(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid,
        y_valid=y_valid,
        categorical_cols=kept_cats,
        params=lgb_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
    )

    pred_valid = booster.predict(X_valid, num_iteration=booster.best_iteration)
    metrics = {
        "auc": safe_auc(y_valid, pred_valid),
        "logloss": safe_logloss(y_valid, pred_valid),
    }
    model = BaselineFoldModel(
        categorical_mappings=cat_maps,
        feature_cols=feature_cols,
        categorical_cols=kept_cats,
        booster=booster,
    )
    return model, pred_valid, metrics



def predict_baseline(model: BaselineFoldModel, df: pd.DataFrame) -> np.ndarray:
    X = pd.DataFrame(index=df.index)
    for c in model.feature_cols:
        if c in model.categorical_cols:
            mapping = model.categorical_mappings[c]
            X[c] = _encode_with_mapping(df[c], mapping)
        else:
            X[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return model.booster.predict(X, num_iteration=model.booster.best_iteration)



def save_baseline_model(model: BaselineFoldModel, save_path: str | Path) -> None:
    with Path(save_path).open("wb") as f:
        pickle.dump(model, f)



def load_baseline_model(load_path: str | Path) -> BaselineFoldModel:
    with Path(load_path).open("rb") as f:
        return pickle.load(f)
