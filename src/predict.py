from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .baseline_lgbm import predict_baseline
from .config import load_config
from .pipeline import prepare_frame, read_txt_table
from .schema import DataSchema
from .settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_A_FILE,
    DEFAULT_TEST_B_FILE,
    resolve_path,
)
from .utils import ensure_dir, seed_everything



def _resolve_data_file(v: str, data_dir: Path, project_root: Path) -> Path:
    p = Path(v)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return (data_dir / p).resolve()
    return resolve_path(p, project_root)



def _load_schema(exp_dir: Path) -> DataSchema:
    schema_path = exp_dir / "schema.json"
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return DataSchema(**payload)



def run_predict(config_path: str, split: str) -> None:
    config = load_config(config_path)
    project_root = Path.cwd()

    seed_everything(int(config.get("seed", 2026)))

    paths_cfg = config.get("paths", {})
    data_dir = resolve_path(paths_cfg.get("data_dir", str(DEFAULT_DATA_DIR)), project_root)
    output_dir = resolve_path(paths_cfg.get("output_dir", str(DEFAULT_OUTPUT_DIR)), project_root)
    cache_dir = resolve_path(paths_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)), project_root)

    test_a_path = _resolve_data_file(paths_cfg.get("test_a_file", DEFAULT_TEST_A_FILE), data_dir, project_root)
    test_b_path = _resolve_data_file(paths_cfg.get("test_b_file", DEFAULT_TEST_B_FILE), data_dir, project_root)

    pred_cfg = config.get("prediction", {})
    experiment_name = pred_cfg.get("experiment_name", "autoint_moe_calibrated")

    exp_dir = output_dir / "experiments" / experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")

    schema = _load_schema(exp_dir)

    split_path = test_a_path if split == "test_a" else test_b_path
    test_raw = read_txt_table(split_path)
    test_df = prepare_frame(
        test_raw,
        schema,
        cache_dir=cache_dir,
        split_name=split,
        use_cache=bool(config.get("use_cache", True)),
    ).df

    fold_dirs = sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
    if not fold_dirs:
        raise ValueError(f"No fold artifacts found in {exp_dir}")

    preds = []
    model_type = None
    torch_mod = None
    predict_autoint_fn = None
    device = "cpu"

    for fd in fold_dirs:
        fold_meta = json.loads((fd / "fold_meta.json").read_text(encoding="utf-8"))
        model_type = fold_meta.get("model_type", "autoint")
        sb = pd.read_pickle(fd / "stat_builder.pkl")
        test_stat = sb.transform(test_df)
        test_aug = pd.concat([test_df, test_stat], axis=1)

        if model_type == "baseline":
            model = pd.read_pickle(fd / "model.pkl")
            pred = predict_baseline(model, test_aug)
        else:
            if torch_mod is None or predict_autoint_fn is None:
                try:
                    import torch as _torch
                    from .autoint_trainer import predict_autoint as _predict_autoint

                    torch_mod = _torch
                    predict_autoint_fn = _predict_autoint
                    auto_cfg = config.get("autoint", {})
                    use_gpu = bool(auto_cfg.get("use_gpu", True))
                    device = torch_mod.device("cuda" if use_gpu and torch_mod.cuda.is_available() else "cpu")
                except Exception as e:
                    raise RuntimeError(
                        "This experiment uses AutoInt artifacts, but PyTorch is unavailable."
                    ) from e

            bundle = torch_mod.load(fd / "model.pt", map_location=device)
            pre = pd.read_pickle(fd / "preprocessor.pkl")
            pred = predict_autoint_fn(
                model_state_dict=bundle["model_state_dict"],
                model_init_params=bundle["model_init_params"],
                preprocessor=pre,
                df=test_aug,
                batch_size=int(pred_cfg.get("batch_size", 2048)),
                device=device,
                temperature=float(bundle.get("temperature", 1.0)),
            )
        preds.append(pred)

    final_pred = np.mean(np.vstack(preds), axis=0)
    out = pd.DataFrame(
        {
            schema.instance_id_col: test_df[schema.instance_id_col].to_numpy(),
            "predicted_score": final_pred,
        }
    )

    ensure_dir(output_dir)
    out_path = output_dir / f"pred_{split}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")



def main() -> None:
    parser = argparse.ArgumentParser(description="IJCAI-18 prediction pipeline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["test_a", "test_b"])
    args = parser.parse_args()
    run_predict(args.config, args.split)


if __name__ == "__main__":
    main()
