from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F



def fit_temperature_scaling(logits: np.ndarray, labels: np.ndarray, max_iter: int = 80) -> float:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.reshape(-1)
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)

    temperature = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=max_iter)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temp = torch.clamp(temperature, min=1e-3)
        loss = F.binary_cross_entropy_with_logits(logits_t / temp, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(torch.clamp(temperature.detach(), min=1e-3).item())
    return t



def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = max(temperature, 1e-3)
    x = logits / t
    return 1.0 / (1.0 + np.exp(-x))
