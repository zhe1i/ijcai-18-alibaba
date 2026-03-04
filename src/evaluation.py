from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score



def safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))



def safe_logloss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y = np.asarray(y_true)
    return float(log_loss(y, p))



def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true)
    p = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p, bins) - 1
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = bin_ids == i
        if not np.any(mask):
            continue
        acc = y[mask].mean()
        conf = p[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)



def plot_reliability_diagram(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    save_path: str | Path,
    n_bins: int = 10,
) -> dict[str, float]:
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    def bin_curve(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ids = np.digitize(probs, bins) - 1
        x, y = [], []
        for i in range(n_bins):
            mask = ids == i
            if not np.any(mask):
                continue
            x.append(probs[mask].mean())
            y.append(y_true[mask].mean())
        return np.array(x), np.array(y)

    xb, yb = bin_curve(np.asarray(prob_before))
    xa, ya = bin_curve(np.asarray(prob_after))

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    if len(xb):
        plt.plot(xb, yb, marker="o", label="Before calibration")
    if len(xa):
        plt.plot(xa, ya, marker="o", label="After calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed positive rate")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return {
        "ece_before": compute_ece(y_true, np.asarray(prob_before), n_bins=n_bins),
        "ece_after": compute_ece(y_true, np.asarray(prob_after), n_bins=n_bins),
    }



def group_metrics(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    group_col: str,
) -> pd.DataFrame:
    frame = df_meta.copy()
    frame["y_true"] = y_true
    frame["y_prob"] = y_prob

    rows = []
    for g, sub in frame.groupby(group_col):
        yt = sub["y_true"].to_numpy()
        yp = sub["y_prob"].to_numpy()
        rows.append(
            {
                group_col: g,
                "count": int(len(sub)),
                "pos_rate": float(sub["y_true"].mean()),
                "auc": safe_auc(yt, yp),
                "logloss": safe_logloss(yt, yp),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col).reset_index(drop=True)



def plot_day_curve(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    day_col: str,
    save_path: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    frame = df_meta.copy()
    frame["y_true"] = y_true
    frame["y_prob"] = y_prob
    day_stats = frame.groupby(day_col)[["y_true", "y_prob"]].mean().reset_index()

    plt.figure(figsize=(8, 4))
    plt.plot(day_stats[day_col], day_stats["y_true"], marker="o", label="Actual CVR")
    plt.plot(day_stats[day_col], day_stats["y_prob"], marker="o", label="Predicted CVR")
    plt.xlabel(day_col)
    plt.ylabel("CVR")
    plt.title("Day-level CVR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def format_metric_summary(prefix: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_auc": safe_auc(y_true, y_prob),
        f"{prefix}_logloss": safe_logloss(y_true, y_prob),
    }
