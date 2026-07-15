# Predict Customer Churn - Docker + Jupyter + Kaggle Baseline

This project sets up a reproducible local environment for the 2026 Kaggle Playground churn competition and provides a first CatBoost baseline.

## What is included
- Dockerized JupyterLab environment (`python:3.11-slim`)
- Baseline training script (`src/train_baseline.py`)
- Improved CV blend script (`src/train_blend_v2.py`)
- Multi-seed blend script (`src/train_blend_v3_multiseed.py`)
- GPU-oriented blend script (`src/train_blend_v4_gpu.py`)
- Hybrid TE + LR + XGB + Torch MLP script (`src/train_blend_v5_te_lr_xgb_torch.py`)
- Super GPU blend script (`src/train_blend_v6_super_gpu.py`)
- Rank-aware post-blend script (`src/blend_ranked_submissions.py`)
- Data research / EDA script (`src/research_data_profile.py`)
- End-to-end train/validate/submit helper (`src/predict_and_submit.py`)
- Baseline notebook (`notebooks/01_catboost_baseline.ipynb`)
- Kaggle submission script (`scripts/submit_kaggle.ps1`)
- Kaggle kernel push script (`scripts/push_kernel.ps1`)

## Prerequisites
1. Docker Desktop installed and running.
2. Kaggle API token at `%USERPROFILE%\.kaggle\kaggle.json`.
3. Competition data in `data/` (`train.csv`, `test.csv`, `sample_submission.csv`).

## Quick start
1. Create `.env` from template:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Edit `.env` and set:
   - `JUPYTER_TOKEN`
   - `KAGGLE_COMPETITION`
3. Build and start the container:
   ```powershell
   docker compose up -d --build
   ```
4. Open JupyterLab:
   - `http://localhost:8888` (or your `JUPYTER_PORT`)

Optional GPU container (separate service/profile):
```powershell
docker compose --profile gpu up -d --build ml-gpu
```

## Workflow Docs
- Reusable playbook: [SKILLS.md](SKILLS.md)
- Competition log: [DAILY_TRAINING_RECORD.md](DAILY_TRAINING_RECORD.md)

Use `SKILLS.md` for general methods you can reuse in future tabular ML challenges.
Use `DAILY_TRAINING_RECORD.md` for competition-specific progress, scores, and decisions.

## Data Research (EDA + Feature Ideas)
Run this to generate a full profile and feature-engineering recommendations from `train.csv`/`test.csv`:
```powershell
python src/research_data_profile.py --train data/train.csv --test data/test.csv --out-json outputs/data_research_report.json --out-md outputs/data_research_report.md
```

Expected outputs:
- `outputs/data_research_report.json`
- `outputs/data_research_report.md`

## Train baseline and create submission
Run inside the container:
```powershell
docker compose exec -T ml python src/train_baseline.py --train data/train.csv --test data/test.csv --submission outputs/submission_baseline.csv --metrics outputs/metrics_baseline.json --model artifacts/catboost_baseline.cbm
```

Expected outputs:
- `outputs/submission_baseline.csv`
- `outputs/metrics_baseline.json`
- `artifacts/catboost_baseline.cbm`

## Train improved blend v2 (recommended)
Run inside the container:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v2.py --train data/train.csv --test data/test.csv --submission outputs/submission_blend_v2.csv --metrics outputs/metrics_blend_v2.json --n-splits 5 --device-mode auto
```

Fast test run (fewer folds):
```powershell
docker compose exec -T ml-gpu python src/train_blend_v2.py --n-splits 3 --device-mode auto
```

Expected outputs:
- `outputs/submission_blend_v2.csv`
- `outputs/metrics_blend_v2.json`

Best leaderboard candidate (from full CV):
- `outputs/submission_blend_v2_5fold.csv`
- `outputs/metrics_blend_v2_5fold.json`

## Train blend v3 multi-seed (recommended)
Run inside the container:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v3_multiseed.py --train data/train.csv --test data/test.csv --submission outputs/submission_blend_v3_multiseed.csv --metrics outputs/metrics_blend_v3_multiseed.json --n-splits 3 --seeds 42,2024,3407 --device-mode auto
```

Higher-accuracy variant (slower):
```powershell
docker compose exec -T ml-gpu python src/train_blend_v3_multiseed.py --n-splits 5 --seeds 42,2024,3407 --device-mode auto
```

Expected outputs:
- `outputs/submission_blend_v3_multiseed.csv`
- `outputs/metrics_blend_v3_multiseed.json`

Current best local candidate:
- `outputs/submission_blend_v3_multiseed.csv`

## Train blend v4 GPU-oriented (recommended when GPU is available)
Run in the GPU service:
```powershell
docker compose --profile gpu up -d ml-gpu
docker compose exec -T ml-gpu python src/train_blend_v4_gpu.py --train data/train.csv --test data/test.csv --submission outputs/submission_blend_v4_gpu.csv --metrics outputs/metrics_blend_v4_gpu.json --n-splits 3 --seeds 42,2024 --device-mode auto
```

Force GPU and fail if GPU is not exposed:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v4_gpu.py --device-mode gpu
```

Fast smoke test:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v4_gpu.py --device-mode auto --n-splits 2 --seeds 42 --max-train-rows 120000
```

Expected outputs:
- `outputs/submission_blend_v4_gpu.csv`
- `outputs/metrics_blend_v4_gpu.json`

## Train blend v5 hybrid (TargetEncoder + Logistic + XGBoost + Torch MLP)
First-time GPU setup for Torch in `ml-gpu`:
```powershell
docker compose exec -T ml-gpu pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Run full training in GPU service:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v5_te_lr_xgb_torch.py --train data/train.csv --test data/test.csv --submission outputs/submission_blend_v5_te_lr_xgb_torch.csv --metrics outputs/metrics_blend_v5_te_lr_xgb_torch.json --n-splits 3 --device-mode auto
```

Fast smoke run:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v5_te_lr_xgb_torch.py --n-splits 2 --max-train-rows 120000 --xgb-estimators 1400 --mlp-epochs 6 --device-mode auto --submission outputs/submission_blend_v5_te_lr_xgb_torch_smoke.csv --metrics outputs/metrics_blend_v5_te_lr_xgb_torch_smoke.json
```

## Train blend v6 super GPU (CatBoost + LightGBM + XGBoost + LR + Torch MLP)
Full run:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v6_super_gpu.py --train data/train.csv --test data/test.csv --submission outputs/submission_blend_v6_super_gpu.csv --metrics outputs/metrics_blend_v6_super_gpu.json --n-splits 3 --device-mode auto
```

Fast smoke run:
```powershell
docker compose exec -T ml-gpu python src/train_blend_v6_super_gpu.py --n-splits 2 --max-train-rows 140000 --cat-iterations 1800 --lgb-estimators 2400 --xgb-estimators 1600 --mlp-epochs 5 --blend-samples 2000 --device-mode auto --submission outputs/submission_blend_v6_super_gpu_smoke.csv --metrics outputs/metrics_blend_v6_super_gpu_smoke.json
```

## Post-blend existing submissions (rank-aware)
Use this when you want to blend already-generated submission CSVs without retraining models.

Order-adjusted blend (closest to the community h-blend style):
```powershell
docker compose exec -T ml python src/blend_ranked_submissions.py `
  --inputs outputs/submission_blend_v2_5fold.csv,outputs/submission_blend_v3_multiseed.csv,outputs/submission_blend_v4_gpu_3seeds.csv `
  --method order `
  --weights 0.33,0.33,0.34 `
  --rank-adjustments=-0.08,-0.02,-0.06 `
  --asc-weight 0.30 --desc-weight 0.70 `
  --output outputs/submission_rank_order_today.csv `
  --report outputs/metrics_rank_order_today.json
```

Simple weighted probability blend:
```powershell
docker compose exec -T ml python src/blend_ranked_submissions.py `
  --inputs outputs/submission_blend_v2_5fold.csv,outputs/submission_blend_v3_multiseed.csv `
  --method weighted `
  --weights 0.65,0.35 `
  --output outputs/submission_weighted_v2_v3.csv
```

## Optional: one-command train + submit flow
```powershell
docker compose exec -T ml python src/predict_and_submit.py --submit --competition $env:KAGGLE_COMPETITION --message "catboost baseline v1"
```

Dry-run auth check without submitting:
```powershell
docker compose exec -T ml python src/predict_and_submit.py --skip-train --dry-run --competition $env:KAGGLE_COMPETITION
```

## Submit with helper script
```powershell
./scripts/submit_kaggle.ps1 -Competition $env:KAGGLE_COMPETITION -File outputs/submission_blend_v3_multiseed.csv -Message "blend v3 multiseed"
```

## Upload notebook to Kaggle
1. Update `kaggle/kernel-metadata.json` with your Kaggle username in `id`.
2. Push notebook:
   ```powershell
   ./scripts/push_kernel.ps1
   ```

Note: Kaggle competitions accept prediction files (CSV) for scoring. Notebook upload is for sharing/running code, not a direct competition submission.

## Baseline model details
- Target mapping: `Churn` `{No: 0, Yes: 1}`
- Features dropped: only `id`
- Validation: stratified holdout (`test_size=0.2`, `random_state=42`)
- Metric: ROC-AUC
- Model: CatBoost CPU (`iterations=2000`, `learning_rate=0.05`, `depth=8`, `early_stopping_rounds=200`)

## Blend v2 details
- Feature engineering: tenure/charge ratios, churn-risk tenure bands, service-count features, and categorical interaction features.
- Validation: Stratified K-Fold CV.
- Models: CatBoost + LightGBM.
- Ensembling: CV OOF-based weighted blend.

## Blend v3 details
- Multi-seed CV: trains full CV pipelines across multiple seeds and averages predictions.
- Models per seed: CatBoost + LightGBM.
- Final ensembling: compares two strategies and keeps the one with higher OOF AUC.

## Blend v4 details
- Multi-seed CV with GPU-capable models (CatBoost + XGBoost).
- `device-mode auto` tries GPU first and falls back to CPU per model if unavailable.
- Includes optional row-capping for quick iteration (`--max-train-rows`).

## Troubleshooting
- If Docker commands fail, start Docker Desktop first.
- If Kaggle commands fail with auth errors, confirm `%USERPROFILE%\.kaggle\kaggle.json` exists and is valid.
- If kernel push fails, verify `kaggle/kernel-metadata.json` fields are valid and unique.


