from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Fold:
    fold_id: int
    train_idx: np.ndarray
    valid_idx: np.ndarray
    valid_days: list[int]



def make_time_based_folds(
    df: pd.DataFrame,
    day_col: str,
    n_folds: int = 3,
    val_days: int = 2,
) -> list[Fold]:
    if day_col not in df.columns:
        raise ValueError(f"{day_col} not found for time split")

    unique_days = sorted(int(x) for x in pd.to_numeric(df[day_col], errors="coerce").dropna().unique())
    if len(unique_days) < val_days + 2:
        raise ValueError("Not enough unique days for time-based CV")

    start = max(1, len(unique_days) - n_folds * val_days)
    folds: list[Fold] = []
    fold_id = 0
    for i in range(n_folds):
        v_start = start + i * val_days
        v_end = min(v_start + val_days, len(unique_days))
        if v_start >= len(unique_days):
            break
        valid_days = unique_days[v_start:v_end]
        if not valid_days:
            continue
        train_days = unique_days[:v_start]
        if not train_days:
            continue

        train_idx = df.index[df[day_col].isin(train_days)].to_numpy()
        valid_idx = df.index[df[day_col].isin(valid_days)].to_numpy()
        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        folds.append(Fold(fold_id=fold_id, train_idx=train_idx, valid_idx=valid_idx, valid_days=valid_days))
        fold_id += 1

    if not folds:
        raise ValueError("No valid folds generated. Please adjust n_folds or val_days.")
    return folds



def make_random_split(
    df: pd.DataFrame,
    label_col: str,
    valid_size: float = 0.2,
    seed: int = 2026,
) -> tuple[np.ndarray, np.ndarray]:
    idx = df.index.to_numpy()
    y = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()
    train_idx, valid_idx = train_test_split(
        idx,
        test_size=valid_size,
        random_state=seed,
        stratify=y,
    )
    return np.array(train_idx), np.array(valid_idx)
