# DAILY TRAINING RECORD - Predict Customer Churn

## Competition Header
- Competition: Kaggle Playground Series Season 6 Episode 3 - Predict Customer Churn
- Primary task type: Tabular binary classification
- Evaluation metric: ROC-AUC
- Logging timezone default: America/Tijuana
- Secondary timestamp: UTC
- Public score source: Kaggle submissions page screenshot from this session
- Local metric source: `outputs/metrics_*.json`

## How To Read This Log
- This is an append-only running record.
- New entries go at the top of the `Daily Entries` section.
- `Public Score` tracks Kaggle leaderboard feedback.
- `Local Metric` tracks validation/OOF quality from local training.
- Decision notes explain why we changed direction.

## Session Entry Template
```markdown
### Session: <Month Day, Year (America/Tijuana)> | <Month Day, Year (UTC)>

Objective:
- <What we tried to improve>

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| <submission.csv> | <score> | <comment> |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| <name> | <metric> | <time/cost> | <result> |

What Worked:
- <point>

What Did Not Work:
- <point>

Decisions Taken:
- <decision>

Next Actions:
- <action>
```

## Daily Entries

### Session: March 4, 2026 (America/Tijuana) | March 5, 2026 (UTC) - Public LB Results: Merge Variants + V7 Trials

Objective:
- Validate public leaderboard behavior for notebook-style merges and V7 dynamic/adversarial experiments.
- Decide which lane stays primary for the next cycle.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_merge_v2v3v5_tight_76_09_15.csv` | 0.91402 | Matched current best public score |
| `submission_merge_best_plus_v6_97_03.csv` | 0.91401 | Near-best, conservative v6 injection remains stable |
| `submission_median_calibrated_anchor_v2v3_v6.csv` | 0.91399 | Better than raw median/rank, but below anchor |
| `submission_median_anchor_v2v3_v6.csv` | 0.91398 | Slightly below calibrated median |
| `submission_rank_diverse_top3_from5.csv` | 0.91396 | Below anchor; diversity rank did not transfer |
| `submission_blend_v7_dynamic_adv_nocorr_trial120k.csv` | 0.91169 | Strong local trial but weak public transfer |
| `submission_blend_v7_dynamic_adv_error_trial120k.csv` | 0.91167 | Slightly below no-correction variant |
| `submission_blend_v7_dynamic_adv_error_smoke.csv` | 0.90986 | Smoke-only scale underperforms as expected |
| `submission_blend_v7_dynamic_adv_error_gpu_smoke.csv` | 0.90913 | Smoke-only scale underperforms as expected |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| Tight anchor sweep (`76/09/15`) | N/A | Fast merge-only | Publicly robust; ties best |
| Conservative `v6` injection (`97/03`) | N/A | Fast merge-only | Still safe near best |
| Median vs calibrated-median merge | N/A | Fast merge-only | Calibration helped slightly, not enough to beat anchor |
| V7 dynamic+adv no-correction (`120k`) | OOF AUC `0.913360` | GPU training | Local gains did not transfer to public LB |
| V7 error-correction branch (`120k`) | OOF AUC `0.913343` | GPU training | Both local and public slightly worse than no-correction |

What Worked:
- Anchor-centered weighted merges remain the strongest public lane.
- Small `v6` injection remains a safe secondary option.

What Did Not Work:
- Notebook-style median/rank blends did not beat the anchor.
- V7 branch (dynamic/adversarial/error-correction) is currently below anchor on public LB.

Decisions Taken:
- Keep `v2/v3/v5` anchor-weight lane as primary submit path.
- Keep `v6` in small-weight injection only.
- Freeze V7 branch as exploratory until stronger local and public validation at larger scale.

Next Actions:
- Continue narrow anchor sweeps around top-performing weighted merges.
- If revisiting V7, run larger/full-row training and compare against anchor under same submission-day conditions.
- Use one stable anchor candidate and one exploratory candidate per day.

### Session: March 4, 2026 (America/Tijuana) | March 5, 2026 (UTC) - V7 Dynamic + Adversarial + Error-Correction Trial

Objective:
- Implement and test three new techniques together:
- `1)` row-wise dynamic blending,
- `3)` adversarial train/test reweighting,
- `6)` error-focused second-stage correction model.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_merge_v2v3v5_explore_75_10_15.csv` | 0.91402 | Current public anchor |
| `submission_blend_v7_dynamic_adv_error_trial120k.csv` | _(pending)_ | New GPU trial with all three techniques |
| `submission_blend_v7_dynamic_adv_nocorr_trial120k.csv` | _(pending)_ | GPU ablation (`error-corr-scale=0`) |
| `submission_blend_v7_dynamic_adv_error_gpu_smoke.csv` | _(not submitted)_ | GPU smoke validation only |
| `submission_blend_v7_dynamic_adv_error_smoke.csv` | _(not submitted)_ | CPU smoke validation only |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| V7 CPU smoke (`30k`, 2-fold) | Final OOF AUC `0.912140` | ~49s | End-to-end pipeline validated |
| V7 GPU smoke (`20k`, 2-fold) | Final OOF AUC `0.911751` | ~67s | GPU mode works for CatBoost/XGBoost in container |
| V7 GPU trial (`120k`, 3-fold) | Anchor OOF AUC `0.913239` | ~154s | Stronger baseline with larger subset |
| Dynamic blend gain (120k trial) | `0.913239 -> 0.913357` | Same run | Row-wise blend improved over static anchor |
| Error-correction effect (120k trial) | `0.913357 -> 0.913343` | Same run | Slight negative delta at current settings |
| Ablation (`error-corr-scale=0`) | Final OOF AUC `0.913360` | ~146s | Best local result; confirms correction branch should be disabled for now |
| Adversarial weighting signal (120k trial) | Adv AUC `0.507711` | Same run | Shift signal exists but is weak/moderate |

What Worked:
- Successfully built a single script implementing all three requested techniques.
- Dynamic blending produced a consistent positive OOF delta in the larger GPU trial.
- GPU path is now functional for this workflow in `ml-gpu`.

What Did Not Work:
- Error-focused correction model slightly reduced OOF in the current configuration.
- Adversarial classifier had weak separation (`~0.508` AUC), so reweighting impact is likely limited.

Decisions Taken:
- Keep dynamic blend as the main promoted component from this round.
- Disable correction branch for next candidate (`error-corr-scale=0`) unless future tuning shows gain.
- Keep adversarial weights but with conservative clipping due weak shift signal.

Next Actions:
- Submit `submission_blend_v7_dynamic_adv_nocorr_trial120k.csv` first as exploratory candidate.
- Submit `submission_blend_v7_dynamic_adv_error_trial120k.csv` second only as fallback comparison.
- If dynamic+adv gains on public LB, scale up from `120k` subset toward larger/full-row training.

### Session: March 4, 2026 (America/Tijuana) | March 5, 2026 (UTC) - Notebook-Style Merge Trials (Median + Diverse Rank)

Objective:
- Reproduce top public notebook merge ideas locally using existing strong submissions.
- Test whether median-calibrated and diversity-filtered rank blending can improve public LB over anchor.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_merge_v2v3v5_explore_75_10_15.csv` | 0.91402 | Current public anchor |
| `submission_median_calibrated_anchor_v2v3_v6.csv` | _(pending)_ | Median + monotonic calibration across 4 inputs |
| `submission_median_anchor_v2v3_v6.csv` | _(pending)_ | Raw median blend across 4 inputs |
| `submission_rank_diverse_top3_from5.csv` | _(pending)_ | Greedy diverse rank blend selected anchor + v6 |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| Blender extension: `median` + `median_calibrated` methods | N/A | Script update only | Public-notebook median logic now reproducible locally |
| Blender extension: greedy correlation selection | N/A | Script update only | Auto-picked low-correlation subset before blend |
| `median_calibrated` on anchor+v2+v3+v6 | N/A | ~4.7s in `ml-gpu` container | Stable distribution; ready for LB check |
| Diverse rank from 5 inputs (`max_models=3`, `min_corr=0.999`) | N/A | ~4.7s in `ml-gpu` container | Selected 2 models: anchor + `v6` |

What Worked:
- Successfully replicated high-scoring notebook blend styles in local pipeline.
- Diversity selector behaved as expected and rejected highly correlated near-duplicates.

What Did Not Work:
- `ml-gpu` runtime still lacks `torch`, so compute path stayed CPU for this round.
- No public-LB feedback yet; these are generation-only results so far.

Decisions Taken:
- Keep anchor unchanged while testing notebook-style variants as exploratory lane.
- Prioritize one median-calibrated candidate and one diverse-rank candidate for next submissions.

Next Actions:
- Submit `submission_median_calibrated_anchor_v2v3_v6.csv` and `submission_rank_diverse_top3_from5.csv`.
- Compare against anchor (`0.91402`) and keep only methods with repeatable LB gain.

### Session: March 4, 2026 (America/Tijuana) | March 5, 2026 (UTC) - Model-Merging Sweep + GPU Merge Runtime Prep

Objective:
- Focus the day on model merging around the strongest public anchor (`v2/v3/v5 = 75/10/15`).
- Enable GPU-capable merge execution path while keeping reliable CPU fallback.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_merge_v2v3v5_explore_75_10_15.csv` | 0.91402 | Current public anchor retained |
| `submission_merge_v2v3v5_tight_76_09_15.csv` | _(pending)_ | Tight exploratory shift around anchor |
| `submission_merge_v2v3v5_tight_74_11_15.csv` | _(pending)_ | Counter-shift exploratory variant |
| `submission_merge_best_plus_v6_97_03.csv` | _(pending)_ | Conservative `v6` injection (3%) |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| `blend_ranked_submissions.py` GPU backend upgrade | N/A | Script-level change only | Added `--compute-device auto/gpu/cpu` with CUDA path + fallback |
| Tight anchor merge (`76/09/15`) | N/A | ~4.5s in `ml-gpu` container | Candidate generated for controlled LB test |
| Tight anchor merge (`74/11/15`) | N/A | ~4.6s in `ml-gpu` container | Candidate generated for controlled LB test |
| `anchor + v6` conservative merge (`97/03`) | N/A | ~4.6s in `ml-gpu` container | Keeps `v6` inside small-weight safety zone |

What Worked:
- Kept merge exploration narrow and aligned with prior public-LB evidence.
- Added compute-backend logging into merge diagnostics for reproducibility.
- Generated new merge files and metrics reports without retraining.

What Did Not Work:
- Forced GPU merge mode could not run in current `ml-gpu` image because `torch` is not installed there.
- Today’s generated candidates were produced with CPU fallback in container.

Decisions Taken:
- Keep `submission_merge_v2v3v5_explore_75_10_15.csv` as stable lane.
- Use only one or two exploratory merge submissions from the new compact sweep.
- Keep `v6` injection capped to very small weights until real LB gain appears.

Next Actions:
- Install/verify `torch` in `ml-gpu`, then rerun merge script with `--compute-device gpu`.
- Submit in low-risk order: `97/03` conservative injection first, then one tight anchor variant.
- Continue narrow weight sweeps only if public LB shows positive signal.

### Session: March 3, 2026 (America/Tijuana) | March 4, 2026 (UTC) - Public LB Validation of v6 Blends

Objective:
- Validate whether `v6` diversity injection improves public LB versus the `v2/v3/v5` anchor blend.
- Convert this round into reusable generalization rules.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_merge_v2v3v5_explore_75_10_15.csv` | 0.91402 | Current best public score remains unchanged |
| `submission_merge_best_plus_v6_95_05.csv` | 0.91401 | Very small drop (-0.00001) versus best |
| `submission_merge_v2v3v5_stable_85_10_05.csv` | 0.91400 | Stable checkpoint |
| `submission_v2_stable80_fe20.csv` | 0.91400 | Stable checkpoint |
| `submission_merge_v2v3v5v6_60_10_15_15.csv` | 0.91399 | Lower than best; v6 likely overweighted |
| `submission_rank_order_v2v3v5v6.csv` | 0.91399 | Rank-order merge did not help |
| `submission_rank_order_with_fe.csv` | 0.91396 | Remains below benchmark |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| `v6` super blend reference | OOF AUC 0.916293 | ~30.64 min GPU | Strong local metric but no public gain in this round |
| `best + v6` small injection (`95/05`) | Public 0.91401 | No retrain | Small v6 weight is near-neutral and safe |
| `v2v3v5v6` heavier injection (`60/10/15/15`) | Public 0.91399 | No retrain | Larger v6 share hurt generalization |
| Rank-order blend with v6 | Public 0.91399 | No retrain | Probability-weight blends remain safer default |

What Worked:
- The best known anchor (`v2v3v5` explore) stayed robust at `0.91402`.
- Small diversity injection from `v6` was nearly neutral and did not collapse score.

What Did Not Work:
- `v6`-enhanced submissions did not beat the current best public score.
- Aggressive `v6` weighting and rank-order merging both underperformed.

Decisions Taken:
- Keep `submission_merge_v2v3v5_explore_75_10_15.csv` as the public anchor.
- Cap `v6` usage to a small optional injection range (about `0%` to `5%`) until new evidence appears.
- Prioritize narrow weight sweeps around `v2/v3/v5` rather than broad new merge structures.
- Promote generalized lessons into `SKILLS.md`.

Next Actions:
- Run a compact sweep around the anchor blend (for example around `75/10/15` with small step size).
- Keep one stable and one exploratory submission in each daily queue.
- Revisit `v6` only after improving standalone branch quality or calibration.

### Session: March 3, 2026 (America/Tijuana) | March 4, 2026 (UTC) - GPU Acceleration + Hybrid Ensemble

Objective:
- Reduce iteration time using GPU on strongest existing pipelines.
- Test community-inspired tri-family ensemble (ML + GBDT + DL) with TargetEncoder.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_blend_v2_5fold.csv` | 0.91400 | Current public benchmark (unchanged) |
| `submission_v2_stable80_fe20.csv` | 0.91400 | Matched benchmark |
| `submission_rank_order_with_fe.csv` | 0.91396 | Lower than benchmark |
| `submission_blend_v5_te_lr_xgb_torch.csv` | _(pending)_ | Hybrid GPU candidate |
| `submission_blend_v6_super_gpu.csv` | _(pending)_ | New 5-model super blend candidate |
| `submission_merge_v2v3v5_stable_85_10_05.csv` | _(pending)_ | Conservative merge including v5 |
| `submission_merge_v2v3v5_explore_75_10_15.csv` | _(pending)_ | Experimental merge including v5 |
| `submission_merge_best_plus_v6_95_05.csv` | _(pending)_ | Best current submit with 5% v6 injection |
| `submission_merge_v2v3v5v6_60_10_15_15.csv` | _(pending)_ | Weighted 4-model merge with v6 |
| `submission_rank_order_v2v3v5v6.csv` | _(pending)_ | Aggressive rank-order merge with v6 |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| `v2` GPU smoke (2-fold) | OOF AUC 0.915749 | ~3.66 min | CatBoost GPU path works and is much faster |
| `v3` GPU smoke (2-fold, 1 seed) | OOF AUC 0.915724 | ~3.71 min | Multi-seed script GPU path works |
| `v5` full (TE + LR + XGB + Torch MLP, 3-fold) | OOF AUC 0.916190 | ~1.99 min | Very fast on GPU, but below `v2/v3` OOF |
| `v6` super blend full (Cat+LGB+XGB+LR+MLP, 3-fold) | OOF AUC 0.916293 | ~30.64 min | Better than v5 local OOF, but still below `v2/v3` best |
| v5 diversity check vs v2/v3 | mean abs diff ~0.0073 vs v2 | N/A | More diverse predictions than v2-v3 pair (~0.00254) |
| v6 diversity check vs v2/v3 | mean abs diff ~0.0698 vs v2/v3 | N/A | Very high diversity signal, useful for small-weight blending |

What Worked:
- Added GPU auto/fallback to `v2` and `v3` CatBoost without changing core model family.
- Installed Torch CUDA in `ml-gpu` and validated `torch.cuda.is_available=True`.
- Built and ran a new hybrid ensemble pipeline quickly end-to-end.

What Did Not Work:
- Hybrid model did not beat current local OOF leaders (`v2/v3`), and blend optimizer gave near-zero weight to LR/MLP.

Decisions Taken:
- Keep `v2_5fold` as primary stable anchor.
- Use `v5` as a diversity component only (small blend weight), not as standalone mainline model.
- Use `v6` primarily as controlled diversity injection into best-known blends.
- Submit conservative and exploratory merges with limited `v5/v6` contribution.

Next Actions:
- Submit in order: `submission_merge_best_plus_v6_95_05.csv`, `submission_merge_v2v3v5v6_60_10_15_15.csv`, then `submission_rank_order_v2v3v5v6.csv`.
- If first candidate improves LB, run a narrow ±3% weight sweep around that blend.
- If all drop, keep v6 frozen and focus on v2/v3 anchor blends only.

### Session: March 3, 2026 (America/Tijuana) | March 4, 2026 (UTC)

Objective:
- Run a full EDA pass on train/test before new model submissions.
- Convert data findings into actionable feature engineering.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| _(none yet)_ | - | Research + feature design session, no new submission sent |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| Full data profile (`research_data_profile.py`) | N/A | ~11s local CSV scan | No missing values, no schema integrity violations, low train/test drift |
| Risk segmentation by tenure/payment/contract | N/A | Included in report | Early-tenure month-to-month and electronic-check segments are highest-risk |
| `v2` with EDA features (`submission_blend_v2_fe_eda.csv`) | OOF AUC 0.916441 | Full 5-fold run in Docker | Lower than prior `v2_5fold` (0.916547) |
| `v3` with EDA features (`submission_blend_v3_multiseed_fe_eda.csv`) | OOF AUC 0.916527 | 3-seed x 3-fold run in Docker | Lower than prior `v3` (0.916612) |
| Hedge blend (`submission_v2_stable80_fe20.csv`) | N/A | Fast post-blend | Keeps strong legacy signal with 20% new-feature signal |
| Order blend (`submission_rank_order_with_fe.csv`) | N/A | Fast post-blend | Experimental non-linear blend across old/new submissions |

What Worked:
- EDA identified stable risk structure with clear high-lift segments.
- Train/test distribution shift is small (low PSI/TV distance), so feature transfer risk is moderate.
- New features were integrated consistently across CPU and GPU pipelines.

What Did Not Work:
- EDA-driven feature expansion did not improve local OOF on first pass (`v2` and `v3` both decreased).
- New and old submissions remain highly correlated, limiting blend upside from local-only variants.

Decisions Taken:
- Add reusable data-research script outputs to `outputs/data_research_report.{json,md}`.
- Expand feature set with payment, contract, tenure-bin, and interaction features across all blend scripts.
- Keep legacy `v2_5fold` and `v3` as primary checkpoints until LB confirms otherwise.
- Keep two new hedge candidates for controlled public-LB testing.

Next Actions:
- Submit two controlled candidates: `submission_v2_stable80_fe20.csv` (stable) and `submission_rank_order_with_fe.csv` (experimental).
- Compare public LB against `submission_blend_v2_5fold.csv` baseline.
- If LB also drops, run feature ablation to isolate harmful new features before next retrain.

### Session: March 3, 2026 (America/Tijuana) | March 3, 2026 (UTC)

Objective:
- Build a stronger blend pipeline beyond baseline.
- Compare CPU multiseed versus GPU-oriented models.
- Improve public score while keeping workflows reproducible.

Submission Snapshot:
| File | Public Score | Notes |
|---|---:|---|
| `submission_blend_v2_5fold.csv` | 0.91400 | Best public score today |
| `submission_blend_v3_multiseed.csv` | 0.91393 | Strong local OOF, weaker public generalization today |
| `submission_blend_v3_v4mix_90_10.csv` | 0.91393 | Blend hedge between v3 and v4 |
| `submission_blend_v3_v4mix_80_20.csv` | 0.91392 | Blend hedge |
| `submission_blend_v2.csv` | 0.91391 | Faster v2 variant |
| `submission_blend_v3_v4mix_70_30.csv` | 0.91391 | Blend hedge |
| `submission_blend_v4_gpu.csv` | 0.91375 | Faster GPU path, lower public score |
| `submission_baseline.csv` | 0.91357 | First valid baseline |
| `submission_blend_v4_gpu_smoke.csv` | 0.91139 | Smoke test only, row-capped |

Key Local Experiments:
| Experiment | Local Metric | Runtime Note | Takeaway |
|---|---:|---|---|
| Baseline CatBoost holdout | ROC-AUC 0.916525 | Moderate CPU runtime | Good starting point |
| Blend v2 (CatBoost + LightGBM, 5-fold) | OOF AUC 0.916547 | Slower CPU | Best public score today |
| Blend v3 multiseed (CatBoost + LightGBM) | OOF AUC 0.916612 | Slower CPU than v2 | Best local OOF today |
| Blend v4 GPU 3-seed (CatBoost + XGBoost) | OOF AUC 0.916448 | Much faster with GPU | Faster loop, lower public score today |

What Worked:
- Cross-family blending improved over baseline.
- Multi-seed workflow improved local stability.
- GPU container setup worked and reduced runtime significantly.
- Submission naming and output tracking stayed organized.

What Did Not Work:
- Higher local OOF (v3) did not beat v2 on public score today.
- GPU v4 improved speed but underperformed CPU v2 on public LB.
- Multiple near-identical blends gave little public score separation.

Decisions Taken:
- Keep `v2 5-fold` as today's best public checkpoint.
- Keep `v3 multiseed` as best local-quality reference.
- Keep GPU pipeline active for rapid iteration, but not primary submit path yet.
- Use mixed v3/v4 blends as low-cost hedge submissions, not main line.

Next Actions:
- Run targeted parameter sweeps around v2/v3 settings instead of broad model changes.
- Add calibration and rank-based blend checks to test public generalization.
- Keep one stable submission candidate and one experimental candidate per day.

## Decision Register
| Date | Decision | Why | Status |
|---|---|---|---|
| March 3, 2026 | Separate reusable playbook from daily log | Prevent overfitting process docs to one competition | Active |
| March 3, 2026 | Keep v2 5-fold as public benchmark | Best public score (0.91400) | Active |
| March 3, 2026 | Keep v3 as local reference model | Best local OOF signal | Active |
| March 3, 2026 | Keep GPU pipeline for speed | Faster iteration for future experiments | Active |
| March 3, 2026 | Keep `v2v3v5 explore 75/10/15` as public anchor | Best public score after v6 validation (0.91402) | Active |
| March 3, 2026 | Cap `v6` injection to small weights | Heavier v6 and rank-order variants regressed to 0.91399 | Active |
| March 4, 2026 | Keep anchor-centered weighted merge lane as primary | New `76/09/15` sweep matched best public score (0.91402) | Active |
| March 4, 2026 | Keep `v6` as small injection only | `97/03` remained near-best (0.91401) but did not beat anchor | Active |
| March 4, 2026 | Freeze V7 dynamic/adversarial branch to exploratory | Public scores (0.91167 to 0.91169) below anchor lane | Active |

## Next Session Queue
1. Run a narrow `v2/v3/v5` weight sweep centered on `75/10/15`.
2. Keep daily submission queue to one stable anchor and one exploratory variant.
3. Re-check diversity metrics before increasing any new model branch above small injection weight.
4. Improve standalone branch quality first (calibration/feature cleanup) before adding more blend complexity.
