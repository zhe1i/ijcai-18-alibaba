from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .baseline_lgbm import predict_baseline, train_baseline_fold
from .config import load_config
from .evaluation import (
    format_metric_summary,
    group_metrics,
    plot_day_curve,
    plot_reliability_diagram,
)
from .feature_engineering import StatFeatureBuilder
from .pipeline import (
    build_dense_columns,
    dump_pickle,
    infer_gate_columns,
    infer_stat_columns,
    prepare_frame,
    read_txt_table,
)
from .schema import detect_schema
from .settings import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEST_A_FILE,
    DEFAULT_TEST_B_FILE,
    DEFAULT_TRAIN_FILE,
    resolve_path,
)
from .splits import make_random_split, make_time_based_folds
from .utils import ensure_dir, save_json, seed_everything, setup_logger



def _resolve_data_file(v: str, data_dir: Path, project_root: Path) -> Path:
    p = Path(v)
    if p.is_absolute():
        return p
    if p.parent == Path("."):
        return (data_dir / p).resolve()
    return resolve_path(p, project_root)



def _save_expert_weight_plot(df: pd.DataFrame, day_col: str, save_path: Path) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return
    agg = df.groupby(day_col).mean(numeric_only=True).reset_index()
    plt.figure(figsize=(8, 4))
    for c in [x for x in agg.columns if x.startswith("expert_")]:
        plt.plot(agg[day_col], agg[c], marker="o", label=c)
    plt.xlabel(day_col)
    plt.ylabel("Average gate weight")
    plt.title("Expert Weights by Day")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()



def _build_stat_builder(
    label_col: str,
    day_col: str,
    cvr_group_cols: list[str],
    freq_cols: list[str],
    prior_strength: float,
    drift_z_threshold: float,
) -> StatFeatureBuilder:
    return StatFeatureBuilder(
        label_col=label_col,
        cvr_group_cols=cvr_group_cols,
        freq_cols=freq_cols,
        day_col=day_col,
        prior_strength=prior_strength,
        drift_z_threshold=drift_z_threshold,
    )



def _prepare_experiment_frame(
    train_fold: pd.DataFrame,
    valid_fold: pd.DataFrame,
    stat_builder: StatFeatureBuilder,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    stat_builder.fit(train_fold)
    tr_stat = stat_builder.transform(train_fold)
    va_stat = stat_builder.transform(valid_fold)

    stat_cols = [c for c in tr_stat.columns if c != "drift_scenario"]

    tr = pd.concat([train_fold, tr_stat], axis=1)
    va = pd.concat([valid_fold, va_stat], axis=1)
    return tr, va, stat_cols



def run_train(config_path: str) -> None:
    config = load_config(config_path)
    project_root = Path.cwd()

    paths_cfg = config.get("paths", {})
    data_dir = resolve_path(paths_cfg.get("data_dir", str(DEFAULT_DATA_DIR)), project_root)
    output_dir = resolve_path(paths_cfg.get("output_dir", str(DEFAULT_OUTPUT_DIR)), project_root)
    cache_dir = resolve_path(paths_cfg.get("cache_dir", str(DEFAULT_CACHE_DIR)), project_root)

    train_path = _resolve_data_file(paths_cfg.get("train_file", DEFAULT_TRAIN_FILE), data_dir, project_root)
    test_a_path = _resolve_data_file(paths_cfg.get("test_a_file", DEFAULT_TEST_A_FILE), data_dir, project_root)
    test_b_path = _resolve_data_file(paths_cfg.get("test_b_file", DEFAULT_TEST_B_FILE), data_dir, project_root)

    ensure_dir(output_dir)
    ensure_dir(cache_dir)
    reports_dir = ensure_dir(output_dir / "reports")
    experiments_root = ensure_dir(output_dir / "experiments")

    logger = setup_logger(output_dir / "train.log")
    seed = int(config.get("seed", 2026))
    seed_everything(seed)

    logger.info("Loading datasets...")
    train_raw = read_txt_table(train_path)
    test_a_raw = read_txt_table(test_a_path) if test_a_path.exists() else None
    test_b_raw = read_txt_table(test_b_path) if test_b_path.exists() else None

    label_col = config.get("task", {}).get("label_col", "is_trade")
    schema = detect_schema(train_raw, label_col=label_col)

    logger.info("Detected schema: %s", json.dumps(schema.to_dict(), ensure_ascii=False))

    train_prep = prepare_frame(train_raw, schema, cache_dir, split_name="train", use_cache=bool(config.get("use_cache", True)))
    train_df = train_prep.df
    test_a_df = None
    test_b_df = None
    if test_a_raw is not None:
        test_a_df = prepare_frame(test_a_raw, schema, cache_dir, split_name="test_a", use_cache=bool(config.get("use_cache", True))).df
    if test_b_raw is not None:
        test_b_df = prepare_frame(test_b_raw, schema, cache_dir, split_name="test_b", use_cache=bool(config.get("use_cache", True))).df

    save_json(schema.to_dict(), output_dir / "schema.json")

    cv_cfg = config.get("time_cv", {})
    folds = make_time_based_folds(
        train_df,
        day_col=schema.day_col,
        n_folds=int(cv_cfg.get("n_folds", 3)),
        val_days=int(cv_cfg.get("val_days", 2)),
    )
    logger.info("Time CV folds=%d", len(folds))

    features_cfg = config.get("features", {})
    cvr_group_cols, freq_cols = infer_stat_columns(schema, features_cfg)
    logger.info("CVR group cols=%s, freq cols=%s", cvr_group_cols, freq_cols)

    random_train_idx, random_valid_idx = make_random_split(
        train_df,
        label_col=label_col,
        valid_size=float(cv_cfg.get("random_valid_size", 0.2)),
        seed=seed,
    )

    random_sb = _build_stat_builder(
        label_col=label_col,
        day_col=schema.day_col,
        cvr_group_cols=cvr_group_cols,
        freq_cols=freq_cols,
        prior_strength=float(features_cfg.get("prior_strength", 20.0)),
        drift_z_threshold=float(config.get("drift", {}).get("zscore_threshold", 1.0)),
    )
    rand_train_df = train_df.loc[random_train_idx]
    rand_valid_df = train_df.loc[random_valid_idx]
    rand_train_aug, rand_valid_aug, rand_stat_cols = _prepare_experiment_frame(rand_train_df, rand_valid_df, random_sb)

    dense_cols_rand = build_dense_columns(
        schema=schema,
        missing_cols=train_prep.missing_cols,
        mv_len_cols=train_prep.mv_len_cols,
        match_cols=train_prep.match_cols,
        stat_feature_cols=rand_stat_cols,
        include_match=True,
    )

    lgb_cfg = config.get("baseline", {})
    random_model, random_pred, random_metrics = train_baseline_fold(
        train_df=rand_train_aug,
        valid_df=rand_valid_aug,
        label_col=label_col,
        numeric_cols=dense_cols_rand,
        categorical_cols=schema.categorical_cols,
        lgb_params=lgb_cfg.get("params", {}),
        num_boost_round=int(lgb_cfg.get("num_boost_round", 500)),
        early_stopping_rounds=int(lgb_cfg.get("early_stopping_rounds", 50)),
    )
    del random_model
    random_summary = {
        "split": "random",
        "auc": float(random_metrics["auc"]),
        "logloss": float(random_metrics["logloss"]),
    }
    save_json(random_summary, reports_dir / "random_split_metrics.json")
    logger.info("Random split baseline: %s", random_summary)

    experiments = config.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments configured.")

    ablation_rows: list[dict[str, Any]] = []
    summary_for_submission: dict[str, Any] = {}

    submission_experiment = config.get("submission_experiment", "autoint_moe_calibrated")
    requires_autoint = any(exp.get("model_type", "autoint") != "baseline" for exp in experiments)
    torch_mod = None
    train_autoint_fold_fn = None
    predict_autoint_fn = None
    device = "cpu"
    if requires_autoint:
        try:
            import torch as _torch
            from .autoint_trainer import predict_autoint as _predict_autoint
            from .autoint_trainer import train_autoint_fold as _train_autoint_fold

            torch_mod = _torch
            train_autoint_fold_fn = _train_autoint_fold
            predict_autoint_fn = _predict_autoint
            device_cfg = config.get("autoint", {}).get("use_gpu", True)
            device = torch_mod.device("cuda" if device_cfg and torch_mod.cuda.is_available() else "cpu")
            logger.info("Using device=%s", device)
        except Exception as e:
            raise RuntimeError(
                "AutoInt experiments are configured but PyTorch is unavailable. "
                "Please install/verify torch runtime."
            ) from e

    for exp in experiments:
        exp_name = exp["name"]
        model_type = exp.get("model_type", "autoint")
        exp_dir = ensure_dir(experiments_root / exp_name)
        logger.info("Running experiment=%s type=%s", exp_name, model_type)

        oof_before = np.zeros(len(train_df), dtype=np.float32)
        oof_after = np.zeros(len(train_df), dtype=np.float32)
        oof_fold = np.full(len(train_df), -1, dtype=np.int16)
        fold_metrics: list[dict[str, Any]] = []
        gate_rows: list[pd.DataFrame] = []

        for fold in folds:
            fold_dir = ensure_dir(exp_dir / f"fold_{fold.fold_id}")
            tr_fold = train_df.loc[fold.train_idx].copy()
            va_fold = train_df.loc[fold.valid_idx].copy()

            sb = _build_stat_builder(
                label_col=label_col,
                day_col=schema.day_col,
                cvr_group_cols=cvr_group_cols,
                freq_cols=freq_cols,
                prior_strength=float(features_cfg.get("prior_strength", 20.0)),
                drift_z_threshold=float(config.get("drift", {}).get("zscore_threshold", 1.0)),
            )
            tr_aug, va_aug, stat_cols = _prepare_experiment_frame(tr_fold, va_fold, sb)

            include_match = bool(exp.get("include_match_features", True))
            dense_cols = build_dense_columns(
                schema=schema,
                missing_cols=train_prep.missing_cols,
                mv_len_cols=train_prep.mv_len_cols,
                match_cols=train_prep.match_cols,
                stat_feature_cols=stat_cols,
                include_match=include_match,
            )

            if bool(exp.get("disable_long_tail_features", False)):
                dense_cols = [c for c in dense_cols if not c.startswith("log_freq_")]

            gate_cols = infer_gate_columns(schema, freq_cols)
            if bool(exp.get("disable_long_tail_features", False)):
                gate_cols = [c for c in gate_cols if not c.startswith("log_freq_")]

            y_valid = pd.to_numeric(va_aug[label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)

            if model_type == "baseline":
                model, pred_valid, metrics = train_baseline_fold(
                    train_df=tr_aug,
                    valid_df=va_aug,
                    label_col=label_col,
                    numeric_cols=dense_cols,
                    categorical_cols=schema.categorical_cols,
                    lgb_params=lgb_cfg.get("params", {}),
                    num_boost_round=int(lgb_cfg.get("num_boost_round", 500)),
                    early_stopping_rounds=int(lgb_cfg.get("early_stopping_rounds", 50)),
                )
                pred_valid_cal = pred_valid.copy()

                dump_pickle(model, fold_dir / "model.pkl")
                dump_pickle(sb, fold_dir / "stat_builder.pkl")
                save_json(
                    {
                        "dense_cols": dense_cols,
                        "gate_cols": gate_cols,
                        "categorical_cols": schema.categorical_cols,
                        "multi_value_cols": schema.multi_value_cols,
                        "include_match_features": include_match,
                        "model_type": "baseline",
                    },
                    fold_dir / "fold_meta.json",
                )

                fold_metric = {
                    "fold": fold.fold_id,
                    "auc_before": float(metrics["auc"]),
                    "logloss_before": float(metrics["logloss"]),
                    "auc_after": float(metrics["auc"]),
                    "logloss_after": float(metrics["logloss"]),
                    "temperature": 1.0,
                }
            else:
                autoint_cfg = config.get("autoint", {})
                use_moe = bool(exp.get("use_moe", True))
                use_multivalue = bool(exp.get("use_multivalue_attention", True))
                use_wide_deep = bool(exp.get("use_wide_deep", False))
                use_calibration = bool(exp.get("use_calibration", True))

                if train_autoint_fold_fn is None or torch_mod is None:
                    raise RuntimeError("AutoInt trainer is not initialized.")

                result = train_autoint_fold_fn(
                    train_df=tr_aug,
                    valid_df=va_aug,
                    label_col=label_col,
                    single_cat_cols=schema.categorical_cols,
                    dense_cols=dense_cols,
                    gate_cols=gate_cols,
                    multi_cols=schema.multi_value_cols,
                    preprocessor_cfg=autoint_cfg.get("preprocessor", {}),
                    model_cfg=autoint_cfg.get("model", {}),
                    train_cfg=autoint_cfg.get("train", {}),
                    device=device,
                    use_moe=use_moe,
                    use_multivalue=use_multivalue,
                    use_wide_deep=use_wide_deep,
                    use_calibration=use_calibration,
                )

                torch_mod.save(
                    {
                        "model_state_dict": result.model_state_dict,
                        "model_init_params": result.model_init_params,
                        "temperature": result.temperature,
                    },
                    fold_dir / "model.pt",
                )
                dump_pickle(result.preprocessor, fold_dir / "preprocessor.pkl")
                dump_pickle(sb, fold_dir / "stat_builder.pkl")
                save_json(
                    {
                        "dense_cols": dense_cols,
                        "gate_cols": gate_cols,
                        "categorical_cols": schema.categorical_cols,
                        "multi_value_cols": schema.multi_value_cols,
                        "include_match_features": include_match,
                        "use_moe": use_moe,
                        "use_multivalue_attention": use_multivalue,
                        "use_wide_deep": use_wide_deep,
                        "use_calibration": use_calibration,
                        "model_type": "autoint",
                    },
                    fold_dir / "fold_meta.json",
                )

                pred_valid = result.valid_prob
                pred_valid_cal = result.valid_prob_cal
                fold_metric = {"fold": fold.fold_id, **result.valid_metrics}

                if result.gate_weights is not None:
                    g = pd.DataFrame(result.gate_weights, columns=[f"expert_{i}" for i in range(result.gate_weights.shape[1])])
                    g[schema.day_col] = va_aug[schema.day_col].to_numpy()
                    g["fold"] = fold.fold_id
                    gate_rows.append(g)
                    avg_weights = result.gate_weights.mean(axis=0)
                    fold_metric.update({f"avg_expert_{i}": float(w) for i, w in enumerate(avg_weights)})

            oof_before[fold.valid_idx] = pred_valid
            oof_after[fold.valid_idx] = pred_valid_cal
            oof_fold[fold.valid_idx] = fold.fold_id

            meta = pd.DataFrame(
                {
                    schema.day_col: va_aug[schema.day_col].to_numpy(),
                    "drift_scenario": sb.transform(va_fold)["drift_scenario"].to_numpy(),
                },
                index=va_fold.index,
            )
            day_report = group_metrics(meta, y_valid, pred_valid_cal, schema.day_col)
            scenario_report = group_metrics(meta, y_valid, pred_valid_cal, "drift_scenario")
            day_report.to_csv(fold_dir / "metrics_by_day.csv", index=False)
            scenario_report.to_csv(fold_dir / "metrics_by_scenario.csv", index=False)

            fold_metrics.append(fold_metric)
            logger.info("Exp=%s fold=%d metrics=%s", exp_name, fold.fold_id, fold_metric)

        valid_mask = oof_fold >= 0
        y_all = pd.to_numeric(train_df.loc[valid_mask, label_col], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        before_all = oof_before[valid_mask]
        after_all = oof_after[valid_mask]

        oof_path = exp_dir / "oof.csv"
        oof_df = pd.DataFrame(
            {
                schema.instance_id_col: train_df.loc[valid_mask, schema.instance_id_col].to_numpy(),
                "label": y_all,
                "fold": oof_fold[valid_mask],
                "pred_before_calibration": before_all,
                "pred_after_calibration": after_all,
            }
        )
        oof_df.to_csv(oof_path, index=False)

        metrics_summary = {}
        metrics_summary.update(format_metric_summary("oof_before", y_all, before_all))
        metrics_summary.update(format_metric_summary("oof_after", y_all, after_all))
        metrics_summary["fold_metrics"] = fold_metrics

        rel_fig = exp_dir / "reliability_diagram.png"
        rel_stats = plot_reliability_diagram(
            y_true=y_all,
            prob_before=before_all,
            prob_after=after_all,
            save_path=rel_fig,
            n_bins=int(config.get("evaluation", {}).get("ece_bins", 10)),
        )
        metrics_summary.update(rel_stats)

        meta_all = pd.DataFrame(
            {
                schema.day_col: train_df.loc[valid_mask, schema.day_col].to_numpy(),
                "drift_scenario": "Unknown",
            }
        )
        # Approximate scenario using global train drift map for OOF-level report.
        global_sb = _build_stat_builder(
            label_col=label_col,
            day_col=schema.day_col,
            cvr_group_cols=cvr_group_cols,
            freq_cols=freq_cols,
            prior_strength=float(features_cfg.get("prior_strength", 20.0)),
            drift_z_threshold=float(config.get("drift", {}).get("zscore_threshold", 1.0)),
        )
        global_sb.fit(train_df)
        meta_all["drift_scenario"] = global_sb.transform(train_df.loc[valid_mask])["drift_scenario"].to_numpy()

        day_metrics = group_metrics(meta_all, y_all, after_all, schema.day_col)
        scenario_metrics = group_metrics(meta_all, y_all, after_all, "drift_scenario")
        day_metrics.to_csv(exp_dir / "metrics_by_day_oof.csv", index=False)
        scenario_metrics.to_csv(exp_dir / "metrics_by_scenario_oof.csv", index=False)

        plot_day_curve(
            df_meta=meta_all,
            y_true=y_all,
            y_prob=after_all,
            day_col=schema.day_col,
            save_path=exp_dir / "day_cvr_curve.png",
        )

        if gate_rows:
            gate_df = pd.concat(gate_rows, axis=0).reset_index(drop=True)
            gate_df.to_csv(exp_dir / "expert_weights.csv", index=False)
            _save_expert_weight_plot(gate_df, schema.day_col, exp_dir / "expert_weight_by_day.png")

        save_json(metrics_summary, exp_dir / "metrics.json")

        ablation_row = {
            "experiment": exp_name,
            "model_type": model_type,
            "oof_auc_before": metrics_summary["oof_before_auc"],
            "oof_logloss_before": metrics_summary["oof_before_logloss"],
            "oof_auc_after": metrics_summary["oof_after_auc"],
            "oof_logloss_after": metrics_summary["oof_after_logloss"],
            "ece_before": metrics_summary["ece_before"],
            "ece_after": metrics_summary["ece_after"],
        }
        ablation_rows.append(ablation_row)

        save_json(schema.to_dict(), exp_dir / "schema.json")
        save_json(
            {
                "dense_cols": dense_cols,
                "categorical_cols": schema.categorical_cols,
                "cvr_group_cols": cvr_group_cols,
                "freq_cols": freq_cols,
                "model_type": model_type,
                "experiment": exp_name,
            },
            exp_dir / "experiment_meta.json",
        )

        summary_for_submission[exp_name] = {
            "exp_dir": str(exp_dir),
            "model_type": model_type,
            "oof_logloss_after": metrics_summary["oof_after_logloss"],
        }

    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(reports_dir / "ablation_results.csv", index=False)
    try:
        ablation_df.to_markdown(reports_dir / "ablation_results.md", index=False)
    except Exception:
        pass

    save_json(
        {
            "random_split": random_summary,
            "time_cv_experiments": summary_for_submission,
        },
        reports_dir / "overall_summary.json",
    )

    if submission_experiment not in summary_for_submission:
        raise ValueError(f"submission_experiment={submission_experiment} not found in experiments")

    sub_meta = summary_for_submission[submission_experiment]
    sub_dir = Path(sub_meta["exp_dir"])
    model_type = sub_meta["model_type"]

    def _predict_test(test_df: pd.DataFrame, split_name: str) -> None:
        fold_dirs = sorted([p for p in sub_dir.iterdir() if p.is_dir() and p.name.startswith("fold_")])
        preds = []
        for fd in fold_dirs:
            sb = pd.read_pickle(fd / "stat_builder.pkl")
            fold_meta = json.loads((fd / "fold_meta.json").read_text(encoding="utf-8"))

            test_stat = sb.transform(test_df)
            test_aug = pd.concat([test_df, test_stat], axis=1)

            if model_type == "baseline":
                model = pd.read_pickle(fd / "model.pkl")
                p = predict_baseline(model, test_aug)
            else:
                if torch_mod is None or predict_autoint_fn is None:
                    raise RuntimeError("AutoInt predictor is not initialized.")
                bundle = torch_mod.load(fd / "model.pt", map_location=device)
                pre = pd.read_pickle(fd / "preprocessor.pkl")
                p = predict_autoint_fn(
                    model_state_dict=bundle["model_state_dict"],
                    model_init_params=bundle["model_init_params"],
                    preprocessor=pre,
                    df=test_aug,
                    batch_size=int(config.get("predict", {}).get("batch_size", 2048)),
                    device=device,
                    temperature=float(bundle.get("temperature", 1.0)),
                )
            preds.append(p)

        pred = np.mean(np.vstack(preds), axis=0)
        out = pd.DataFrame(
            {
                schema.instance_id_col: test_df[schema.instance_id_col].to_numpy(),
                "predicted_score": pred,
            }
        )
        out_path = output_dir / f"pred_{split_name}.csv"
        out.to_csv(out_path, index=False)

    if test_a_df is not None:
        _predict_test(test_a_df, "test_a")
    if test_b_df is not None:
        _predict_test(test_b_df, "test_b")

    logger.info("Training finished. outputs=%s", output_dir)



def main() -> None:
    parser = argparse.ArgumentParser(description="IJCAI-18 training pipeline")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    run_train(args.config)


if __name__ == "__main__":
    main()
