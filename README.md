# IJCAI-18 Alimama pCVR End-to-End Project

## 1. Task
- Objective: binary pCVR prediction `P(is_trade=1 | x)`.
- Primary metric: Logloss; secondary: AUC.
- Extra focus: probability calibration (Reliability Diagram + ECE).
- Validation: strict time-based CV with random split as leakage-risk contrast.

## 2. Project Structure
- `/Users/zheli/code/ijcai18/src` training/inference/evaluation code
- `/Users/zheli/code/ijcai18/configs` experiment configs
- `/Users/zheli/code/ijcai18/outputs` models/logs/reports/predictions
- `/Users/zheli/code/ijcai18/data` dataset files

## 3. Install
```bash
cd /Users/zheli/code/ijcai18
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Train
```bash
python -m src.train --config configs/train.yaml
```

After training, outputs include:
- `outputs/experiments/<experiment>/fold_*/` (weights + preprocessors + fold reports)
- `outputs/experiments/<experiment>/oof.csv`
- `outputs/experiments/<experiment>/metrics.json`
- `outputs/reports/ablation_results.csv`
- `outputs/pred_test_a.csv` and `outputs/pred_test_b.csv` (for `submission_experiment`)

## 5. Predict (Standalone)
```bash
python -m src.predict --config configs/predict.yaml --split test_a
python -m src.predict --config configs/predict.yaml --split test_b
```

Submission format:
- columns: `instance_id`, `predicted_score`
- files: `outputs/pred_test_a.csv`, `outputs/pred_test_b.csv`

## 6. Implemented Methods
### 6.1 Baseline (LightGBM)
- Handcrafted matching features from multi-value fields:
  - category/property hit count
  - coverage rate
  - Jaccard
  - main category hit
- Smoothed CVR statistics (Beta-Binomial) on configurable groups
- Strict time-based CV metrics by fold

### 6.2 Main Model
`AutoInt backbone + Multi-value set attention pooling + Drift-aware MoE + temperature scaling`

- Field embeddings for high-cardinality discrete fields
- `-1` kept as valid category + explicit missing indicators
- Multi-value parsing (`;`) and attention pooling
- AutoInt self-attention layers for feature interaction learning
- MoE experts:
  - `Normal`
  - `Event(Drift)`
  - `Long-tail`
- Gating inputs include:
  - `day/hour`
  - `drift_score` (day-level CVR z-score)
  - `log_freq_user/item/shop`
- Load-balancing auxiliary loss to avoid expert starvation
- Imbalance handling: weighted BCE (optional focal loss)
- Fold-level temperature scaling, no leakage

## 7. Validation Strategy
- Time-based rolling CV (`configs/train.yaml`: `time_cv.n_folds`, `time_cv.val_days`)
- Random split baseline logged at `outputs/reports/random_split_metrics.json`

## 8. Automatic Reports
For each experiment:
- overall AUC/logloss (before/after calibration)
- `metrics_by_day_oof.csv`
- `metrics_by_scenario_oof.csv` (`Normal` vs `Drift`)
- `reliability.png`
- `day_cvr_curve.png`
- MoE runs also produce `expert_weights.csv` and `expert_weight_by_day.png`

## 9. Ablation Setup (Config Switches)
Defined in `configs/train.yaml`:
1. `baseline_lgbm`
2. `autoint_without_moe`
3. `autoint_multivalue_no_moe`
4. `autoint_moe_no_calibration`
5. `autoint_moe_calibrated`

Optional switches supported:
- `include_match_features`
- `disable_long_tail_features`
- loss switch: weighted BCE / focal

## 10. Reproducibility
- Fixed seed (`seed` in config)
- Complete artifact save: fold model, preprocessor, stat maps, metrics, OOF
- Cached parsing/feature preprocessing under `outputs/cache`

## 11. Interview Talking Points
- Why time-based CV is mandatory for leakage prevention.
- Why severe imbalance needs weighted loss/focal and why calibration is still needed.
- AutoInt intuition: learn high-order feature interactions through attention over fields.
- Why multi-value attention pooling is better than mean pooling:
  - query-conditioned weights highlight intent-relevant category/property tokens.
- Why MoE for mixed normal/drift distributions:
  - gating leverages temporal drift and long-tail density signals.
- Long-tail expert role:
  - reduce overfitting on head patterns and improve rare-ID behavior.
- Ablation interpretation:
  - quantify gains from multi-value attention, MoE, and calibration.

## 12. Notes
- Column names are not hardcoded for parsing multi-value fields; code detects semicolon-separated columns from headers.
- Day/hour are auto-derived from detected timestamp column when needed.
