from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class AutoIntBatchInput:
    single_cat: np.ndarray
    dense: np.ndarray
    gate: np.ndarray
    labels: np.ndarray | None
    multi_inputs: dict[str, np.ndarray]



def _build_vocab_from_series(
    values: pd.Series,
    min_freq: int,
    max_vocab: int,
) -> dict[str, int]:
    vc = values.astype(str).value_counts(dropna=False)
    if min_freq > 1:
        vc = vc[vc >= min_freq]
    if max_vocab > 0:
        vc = vc.head(max_vocab)

    items = sorted(vc.items(), key=lambda x: (-int(x[1]), x[0]))
    vocab = {tok: i + 2 for i, (tok, _) in enumerate(items)}
    return vocab



def _build_vocab_from_token_lists(
    token_lists: list[list[str]],
    min_freq: int,
    max_vocab: int,
) -> dict[str, int]:
    counter: dict[str, int] = {}
    for tokens in token_lists:
        for t in tokens:
            counter[t] = counter.get(t, 0) + 1

    items = [(k, v) for k, v in counter.items() if v >= min_freq]
    items.sort(key=lambda x: (-x[1], x[0]))
    if max_vocab > 0:
        items = items[:max_vocab]

    vocab = {tok: i + 2 for i, (tok, _) in enumerate(items)}
    return vocab


class AutoIntPreprocessor:
    def __init__(
        self,
        single_cat_cols: list[str],
        dense_cols: list[str],
        gate_cols: list[str],
        multi_token_cols: list[str],
        min_freq: int = 2,
        max_vocab: int = 200_000,
        multi_min_freq: int = 2,
        multi_max_vocab: int = 300_000,
        max_seq_len: int = 24,
    ) -> None:
        self.single_cat_cols = single_cat_cols
        self.dense_cols = dense_cols
        self.gate_cols = gate_cols
        self.multi_token_cols = multi_token_cols

        self.min_freq = min_freq
        self.max_vocab = max_vocab
        self.multi_min_freq = multi_min_freq
        self.multi_max_vocab = multi_max_vocab
        self.max_seq_len = max_seq_len

        self.single_vocab: dict[str, dict[str, int]] = {}
        self.multi_vocab: dict[str, dict[str, int]] = {}

        self.dense_mean: np.ndarray | None = None
        self.dense_std: np.ndarray | None = None
        self.gate_mean: np.ndarray | None = None
        self.gate_std: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> None:
        self.single_vocab = {}
        for c in self.single_cat_cols:
            self.single_vocab[c] = _build_vocab_from_series(df[c], self.min_freq, self.max_vocab)

        self.multi_vocab = {}
        for c in self.multi_token_cols:
            token_col = f"mv_tokens__{c}"
            if token_col not in df.columns:
                raise ValueError(f"Missing token column {token_col}; run multi-value parser first.")
            token_lists = df[token_col].tolist()
            self.multi_vocab[c] = _build_vocab_from_token_lists(
                token_lists=token_lists,
                min_freq=self.multi_min_freq,
                max_vocab=self.multi_max_vocab,
            )

        dense_vals = (
            df[self.dense_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            if self.dense_cols
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        self.dense_mean = dense_vals.mean(axis=0) if dense_vals.shape[1] else np.array([], dtype=np.float32)
        self.dense_std = dense_vals.std(axis=0) if dense_vals.shape[1] else np.array([], dtype=np.float32)
        self.dense_std = np.where(self.dense_std < 1e-6, 1.0, self.dense_std)

        gate_vals = (
            df[self.gate_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
            if self.gate_cols
            else np.zeros((len(df), 0), dtype=np.float32)
        )
        self.gate_mean = gate_vals.mean(axis=0) if gate_vals.shape[1] else np.array([], dtype=np.float32)
        self.gate_std = gate_vals.std(axis=0) if gate_vals.shape[1] else np.array([], dtype=np.float32)
        self.gate_std = np.where(self.gate_std < 1e-6, 1.0, self.gate_std)

    def _encode_single_cats(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        m = len(self.single_cat_cols)
        arr = np.zeros((n, m), dtype=np.int32)
        for j, c in enumerate(self.single_cat_cols):
            vocab = self.single_vocab[c]
            vals = df[c].astype(str).map(vocab).fillna(1).astype(np.int32)
            arr[:, j] = vals.to_numpy()
        return arr

    def _encode_multi_tokens(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for c in self.multi_token_cols:
            vocab = self.multi_vocab[c]
            token_col = f"mv_tokens__{c}"
            seq = np.zeros((len(df), self.max_seq_len), dtype=np.int32)
            mask = np.zeros((len(df), self.max_seq_len), dtype=np.float32)
            token_lists = df[token_col].tolist()
            for i, toks in enumerate(token_lists):
                if not toks:
                    continue
                ids = [vocab.get(str(t), 1) for t in toks[: self.max_seq_len]]
                l = len(ids)
                if l:
                    seq[i, :l] = np.asarray(ids, dtype=np.int32)
                    mask[i, :l] = 1.0
            out[c] = seq
            out[f"{c}__mask"] = mask
        return out

    def _transform_dense(self, df: pd.DataFrame) -> np.ndarray:
        if not self.dense_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        vals = (
            df[self.dense_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        return ((vals - self.dense_mean) / self.dense_std).astype(np.float32)

    def _transform_gate(self, df: pd.DataFrame) -> np.ndarray:
        if not self.gate_cols:
            return np.zeros((len(df), 0), dtype=np.float32)
        vals = (
            df[self.gate_cols]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        return ((vals - self.gate_mean) / self.gate_std).astype(np.float32)

    def transform(self, df: pd.DataFrame, labels: np.ndarray | None = None) -> AutoIntBatchInput:
        single_cat = self._encode_single_cats(df)
        dense = self._transform_dense(df)
        gate = self._transform_gate(df)
        multi = self._encode_multi_tokens(df)

        y = None if labels is None else labels.astype(np.float32)
        return AutoIntBatchInput(
            single_cat=single_cat,
            dense=dense,
            gate=gate,
            labels=y,
            multi_inputs=multi,
        )

    def cat_cardinalities(self) -> dict[str, int]:
        return {c: len(v) + 2 for c, v in self.single_vocab.items()}

    def multi_cardinalities(self) -> dict[str, int]:
        return {c: len(v) + 2 for c, v in self.multi_vocab.items()}


class CTRTorchDataset(Dataset):
    def __init__(self, batch_input: AutoIntBatchInput) -> None:
        self.single_cat = torch.from_numpy(batch_input.single_cat).long()
        self.dense = torch.from_numpy(batch_input.dense).float()
        self.gate = torch.from_numpy(batch_input.gate).float()
        self.multi_inputs = {
            k: torch.from_numpy(v).long() if not k.endswith("__mask") else torch.from_numpy(v).float()
            for k, v in batch_input.multi_inputs.items()
        }
        self.labels = None if batch_input.labels is None else torch.from_numpy(batch_input.labels).float()

    def __len__(self) -> int:
        return self.single_cat.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = {
            "single_cat": self.single_cat[idx],
            "dense": self.dense[idx],
            "gate": self.gate[idx],
        }
        for k, v in self.multi_inputs.items():
            out[k] = v[idx]
        if self.labels is not None:
            out["label"] = self.labels[idx]
        return out
