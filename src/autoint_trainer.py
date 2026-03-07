from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .autoint_data import AutoIntPreprocessor, CTRTorchDataset
from .calibration import apply_temperature, fit_temperature_scaling
from .evaluation import safe_auc, safe_logloss
from .models.autoint_moe import AutoIntMoEModel, FocalLoss, load_balancing_loss

logger = logging.getLogger(__name__)


@dataclass
class AutoIntFoldResult:
    model_state_dict: dict[str, Any]
    model_init_params: dict[str, Any]
    preprocessor: AutoIntPreprocessor
    temperature: float
    valid_logits: np.ndarray
    valid_prob: np.ndarray
    valid_prob_cal: np.ndarray
    valid_metrics: dict[str, float]
    gate_weights: np.ndarray | None



def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}



def _collect_predictions(
    model: AutoIntMoEModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    all_logits = []
    all_gates = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            out_logits, extras = model(batch)
            all_logits.append(out_logits.detach().cpu().numpy())
            if "gate_weights" in extras:
                all_gates.append(extras["gate_weights"].detach().cpu().numpy())
    logits = np.concatenate(all_logits)
    if all_gates:
        gates = np.concatenate(all_gates)
    else:
        gates = None
    return logits, gates



def train_autoint_fold(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str,
    single_cat_cols: list[str],
    dense_cols: list[str],
    gate_cols: list[str],
    multi_cols: list[str],
    preprocessor_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    device: torch.device,
    use_moe: bool,
    use_multivalue: bool,
    use_wide_deep: bool,
    use_calibration: bool,
) -> AutoIntFoldResult:
    y_train = pd.to_numeric(train_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    y_valid = pd.to_numeric(valid_df[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

    used_multi_cols = multi_cols if use_multivalue else []

    pre = AutoIntPreprocessor(
        single_cat_cols=single_cat_cols,
        dense_cols=dense_cols,
        gate_cols=gate_cols,
        multi_token_cols=used_multi_cols,
        min_freq=int(preprocessor_cfg.get("min_freq", 2)),
        max_vocab=int(preprocessor_cfg.get("max_vocab", 200_000)),
        multi_min_freq=int(preprocessor_cfg.get("multi_min_freq", 2)),
        multi_max_vocab=int(preprocessor_cfg.get("multi_max_vocab", 300_000)),
        max_seq_len=int(preprocessor_cfg.get("max_seq_len", 24)),
    )
    pre.fit(train_df)

    train_input = pre.transform(train_df, y_train)
    valid_input = pre.transform(valid_df, y_valid)

    train_ds = CTRTorchDataset(train_input)
    valid_ds = CTRTorchDataset(valid_input)

    batch_size = int(train_cfg.get("batch_size", 1024))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    model_init_params = {
        "cat_cardinalities": pre.cat_cardinalities(),
        "multi_cardinalities": pre.multi_cardinalities(),
        "num_dense": len(dense_cols),
        "num_gate": len(gate_cols),
        "embed_dim": int(model_cfg.get("embed_dim", 16)),
        "attn_layers": int(model_cfg.get("attn_layers", 2)),
        "num_heads": int(model_cfg.get("num_heads", 4)),
        "dropout": float(model_cfg.get("dropout", 0.1)),
        "shared_hidden": list(model_cfg.get("shared_hidden", [256, 128])),
        "num_experts": int(model_cfg.get("num_experts", 3)),
        "expert_hidden": list(model_cfg.get("expert_hidden", [64])),
        "dense_tower_hidden": list(model_cfg.get("dense_tower_hidden", [128, 64])),
        "use_moe": bool(use_moe),
        "use_multivalue": bool(use_multivalue),
        "use_wide_deep": bool(use_wide_deep),
    }

    model = AutoIntMoEModel(**model_init_params).to(device)
    pos_rate_train = float(np.clip(y_train.mean(), 1e-6, 1.0 - 1e-6))
    prior_logit = float(np.log(pos_rate_train / (1.0 - pos_rate_train)))
    model.init_output_bias(prior_logit)

    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_name = str(train_cfg.get("loss", "weighted_bce")).lower()
    if loss_name == "focal":
        criterion = FocalLoss(
            alpha=float(train_cfg.get("focal_alpha", 0.25)),
            gamma=float(train_cfg.get("focal_gamma", 2.0)),
        )
        pos_weight = None
    else:
        if bool(train_cfg.get("dynamic_pos_weight", True)):
            pos_count = max(float(y_train.sum()), 1.0)
            neg_count = max(float(len(y_train) - y_train.sum()), 1.0)
            pos_weight = neg_count / pos_count
        else:
            pos_weight = float(train_cfg.get("pos_weight", 20.0))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    logger.info(
        "Fold train label stats: pos_rate=%.6f prior_logit=%.6f pos_weight=%s",
        pos_rate_train,
        prior_logit,
        "focal" if pos_weight is None else f"{pos_weight:.6f}",
    )

    max_epochs = int(train_cfg.get("epochs", 5))
    patience = int(train_cfg.get("early_stop_patience", 2))
    clip_grad = float(train_cfg.get("clip_grad", 5.0))
    lb_weight = float(train_cfg.get("load_balance_weight", 0.01)) if use_moe else 0.0
    wide_l2 = float(train_cfg.get("wide_l2", 0.0)) if use_wide_deep else 0.0

    best_metric = float("inf")
    best_state: dict[str, Any] | None = None
    no_improve = 0
    train_eval_loader = DataLoader(train_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    pos_rate_valid = float(y_valid.mean()) if len(y_valid) else 0.0

    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            batch = _to_device(batch, device)
            y = batch["label"]
            optimizer.zero_grad()
            logits, extras = model(batch)
            loss = criterion(logits, y)
            if use_moe and "gate_weights" in extras and lb_weight > 0:
                loss = loss + lb_weight * load_balancing_loss(extras["gate_weights"])
            if wide_l2 > 0:
                loss = loss + wide_l2 * model.wide_l2_penalty()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        train_logits, _ = _collect_predictions(model, train_eval_loader, device)
        valid_logits, _ = _collect_predictions(model, valid_loader, device)
        train_prob = 1.0 / (1.0 + np.exp(-train_logits))
        valid_prob = 1.0 / (1.0 + np.exp(-valid_logits))
        valid_ll = safe_logloss(y_valid, valid_prob)
        logger.info(
            "Epoch %d/%d: mean_prob_train=%.6f mean_prob_val=%.6f pos_rate_train=%.6f pos_rate_val=%.6f mean_logit_train=%.6f valid_logloss=%.6f",
            epoch + 1,
            max_epochs,
            float(train_prob.mean()),
            float(valid_prob.mean()),
            pos_rate_train,
            pos_rate_valid,
            float(train_logits.mean()),
            valid_ll,
        )

        if valid_ll < best_metric:
            best_metric = valid_ll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    valid_logits, valid_gates = _collect_predictions(model, valid_loader, device)
    valid_prob = 1.0 / (1.0 + np.exp(-valid_logits))

    if use_calibration:
        temperature = fit_temperature_scaling(valid_logits, y_valid)
    else:
        temperature = 1.0
    valid_prob_cal = apply_temperature(valid_logits, temperature)

    metrics = {
        "auc_before": safe_auc(y_valid, valid_prob),
        "logloss_before": safe_logloss(y_valid, valid_prob),
        "auc_after": safe_auc(y_valid, valid_prob_cal),
        "logloss_after": safe_logloss(y_valid, valid_prob_cal),
        "temperature": float(temperature),
    }

    return AutoIntFoldResult(
        model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        model_init_params=model_init_params,
        preprocessor=pre,
        temperature=float(temperature),
        valid_logits=valid_logits,
        valid_prob=valid_prob,
        valid_prob_cal=valid_prob_cal,
        valid_metrics=metrics,
        gate_weights=valid_gates,
    )



def predict_autoint(
    model_state_dict: dict[str, Any],
    model_init_params: dict[str, Any],
    preprocessor: AutoIntPreprocessor,
    df: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> np.ndarray:
    model = AutoIntMoEModel(**model_init_params).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    batch_input = preprocessor.transform(df, labels=None)
    dataset = CTRTorchDataset(batch_input)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logits_list: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits, _ = model(batch)
            logits_list.append(logits.detach().cpu().numpy())

    logits = np.concatenate(logits_list)
    probs = apply_temperature(logits, temperature)
    return probs
