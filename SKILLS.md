# SKILLS - Tabular ML Competition Playbook

This file is a reusable playbook for tabular machine learning competitions.
It is intentionally general so it can be reused in future projects.

## Challenge Type Definition
Use this playbook for:
- Supervised tabular prediction tasks.
- Hidden-label test sets with leaderboard feedback.
- Binary or multiclass classification where rank metrics matter (for example ROC-AUC).

Typical constraints:
- Limited submission count per day.
- Need for fast local iteration and stable validation.
- Public leaderboard may not match private leaderboard.

## When To Use This Playbook
Use this playbook when you need:
- A reliable baseline quickly.
- A repeatable experiment loop (features, CV, blend, submit, review).
- A balance between model quality and runtime.
- Documentation that supports team collaboration.

Do not use this playbook as-is for:
- Time series with strict temporal leakage rules.
- Heavy unstructured data tasks (raw text, audio, images) without tabular features.

## End-to-End Workflow
1. Confirm task contract.
2. Build a deterministic baseline.
3. Add strong validation.
4. Add feature engineering in small batches.
5. Add model diversity.
6. Blend models using out-of-fold predictions.
7. Submit controlled candidates.
8. Review leaderboard drift versus local validation.
9. Log decisions and next steps in a daily record.

## Validation Strategy Rules
1. Start with one holdout split for smoke testing.
2. Move to stratified K-fold CV for ranking decisions.
3. Use fixed random seeds for comparability.
4. Use multi-seed CV only after single-seed CV is stable.
5. Treat CV as primary decision signal and public LB as secondary signal.
6. Track CV-LB disagreement explicitly in the daily log.

## Feature Engineering Patterns
Good default patterns for tabular competitions:
- Ratio features between related numeric columns.
- Difference and gap features.
- Binary service or status flags.
- Count features across related binary columns.
- Categorical interaction features (`A__B` style).
- Low-risk bins for monotonic numeric behavior.

Rules:
- Add features in small sets.
- Keep one-source-of-truth feature code per pipeline.
- Validate schema and value ranges before training.

## Model Family Selection Matrix
| Model Family | Best Use Case | Pros | Risks |
|---|---|---|---|
| CatBoost | Mixed categorical + numeric tables | Strong baseline, low preprocessing | Can be slow on CPU with large CV |
| LightGBM | Large tabular data, fast iteration | Fast, flexible, strong with tuned params | Categorical handling needs care |
| XGBoost | High-quality tree boosting, GPU support | Strong performance, mature tooling | Encoding and memory choices matter |
| Linear/LogReg | Sanity baseline | Very fast and interpretable | Usually lower ceiling |
| PyTorch MLP | Dense encoded tabular features with GPU | Adds functional-form diversity and can train fast on GPU | Sensitive to scaling and regularization |

## Ensembling Patterns
1. In-family multi-seed average.
2. Cross-family weighted blend using OOF predictions.
3. Compare blending methods:
- Optimize weights after model averaging.
- Average per-seed blends.
4. Choose blend strategy by OOF metric, then validate with submissions.

Practical rule:
- Keep at least one conservative blend candidate and one aggressive candidate.

## Generalization-First Blend Rules
1. Keep one leaderboard-proven anchor submission unchanged in every merge cycle.
2. Add new high-diversity models with small initial weight (start around 2% to 10%).
3. Increase a new model above 10% to 15% only after at least one real leaderboard gain.
4. Use probability-space weighted blends as default and treat rank-order blends as exploratory.
5. If OOF improves but public LB drops in repeated submissions, freeze that model family as a minor diversity component only.
6. Before adding a model, measure both prediction correlation and mean absolute prediction difference versus the anchor.

## Public-LB Promotion Rules
1. Treat smoke submissions (reduced rows/folds) as pipeline checks, not final ranking candidates.
2. Promote a new method to the primary lane only after public LB meets or beats the current anchor.
3. If a new method is consistently below anchor on public LB, demote it to exploratory or archive.
4. When two candidates are effectively tied, prefer the simpler and more reproducible one.

## Dynamic and Reweighting Rules
1. For dynamic row-wise blends, always compare against the static-anchor baseline in the same run.
2. For second-stage correction branches, run an explicit no-correction ablation (`correction_scale = 0`) and keep correction disabled unless it wins.
3. Use adversarial train/test reweighting conservatively when train-vs-test separability is weak (near-random).
4. Keep weight clipping bounded and normalized to avoid unstable shifts from noisy domain signals.

## Encoding Rules For Mixed Families
1. Use fold-aware TargetEncoder to avoid leakage.
2. Keep one shared encoded feature schema for LR, XGBoost, and MLP branches.
3. Keep at least one native-categorical tree branch (for example CatBoost) as a non-encoded anchor.
4. Treat TargetEncoder branches as diversity sources when their standalone score is lower but blend contribution is positive.

## GPU/CPU Decision Rules
Use GPU when:
- CV runtime blocks iteration speed.
- GPU is available in the training container.
- Model family supports stable GPU mode.

Keep CPU path when:
- GPU runtime causes instability.
- Reproducibility is more important than speed.
- GPU model underperforms in OOF and submissions.

Operational rule:
- Keep a script-level device mode (`auto`, `gpu`, `cpu`) with explicit fallback behavior.
- Before forcing `gpu` mode, validate CUDA dependency availability in the exact runtime target (for example the Docker service used for execution).

## Submission Strategy
1. Define a short candidate queue before submitting.
2. Submit no-op duplicates only when checking pipeline integrity.
3. Name outputs with model family and config signature.
4. Use submission budget for meaningful deltas.
5. Preserve one best-known stable candidate at all times.
6. In low-budget days, use a two-lane queue: one stable candidate and one exploratory candidate.

## Failure Modes + Debug Checklist
Common failure modes:
- Wrong target mapping.
- Train/test feature mismatch.
- Submission schema mismatch (`id`, target column).
- Probability values outside `[0, 1]`.
- Leakage through target-dependent features.
- Public leaderboard overfitting.

Quick checklist:
1. Validate columns and row counts.
2. Validate probability bounds.
3. Check fold stability and variance.
4. Compare OOF against previous run.
5. Record decisions before next submission.

## Reusable Command Patterns
CPU training pattern:
```powershell
docker compose exec -T ml python <train_script>.py <args>
```

GPU training pattern:
```powershell
docker compose --profile gpu up -d ml-gpu
docker compose exec -T ml-gpu python <train_script>.py <args>
```

Submission pattern:
```powershell
./scripts/submit_kaggle.ps1 -Competition <slug> -File <submission.csv> -Message "<tag>"
```

## Promotion Rule
Promote an item from daily notes into this file only if:
1. It is reusable across multiple tabular competitions.
2. It is robust to dataset changes.
3. It improves decision quality, not just one score snapshot.

Keep competition-specific numbers and one-off findings in the daily training record.

## Transferability Check
Before adding any new rule into `SKILLS.md`, ask:
1. Does this advice depend on this competition's synthetic distribution?
2. If yes, should this stay in `DAILY_TRAINING_RECORD.md` instead?
