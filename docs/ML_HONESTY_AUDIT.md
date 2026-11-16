# ML Honesty Audit — invented values removed

Scope: every file in `model/`. A value counts as "invented" when it changes a number shown to a student or a decision the engine takes, and it isn't fitted to data, taken from a cited source, or derived mathematically.

## 1. Fixed

| # | File | What was there | What it is now | Effect |
|---|---|---|---|---|
| 1 | `monte_carlo_conformal_neet.py` | Chapters with no calibrated items drew random a/b/c (`uniform(0.95,1.45)`, `uniform(-0.45,0.55)`, `uniform(0.18,0.24)`) | Draw from the pooled calibrated items; counts reported in `item_parameter_sources` | No random item parameters |
| 2 | same | Item difficulty shrunk: `b × 0.55` | Calibrated `b` used as is | Removes upward score bias |
| 3 | same | Ability boosted by mastery: `θ + 0.35·(mastery − 0.40)` | Chapter θ only (mastery is already reflected in θ) | Removes double counting |
| 4 | same | "Trap penalty" `0.52 + 1.80·p` when p < 0.26 | Removed; plain 3PL probability | No unfitted distractor effect |
| 5 | same | Probability clipped to [0.08, 0.995] | 3PL bounds only (≥ c) | |
| 6 | same | Skip rule at P < 0.23, applied to each simulated outcome (oracle) | P < 0.20 (the +4/−1 break-even), decided on the point-estimate θ | Skip advice is achievable by a real student |
| 7 | same | Called "split conformal" | `interval_method = monte_carlo_empirical_quantile_not_conformal_calibrated`; API key `conformal_90_interval` kept | Label matches method |
| 8 | same, leverage ranking | Hidden constants (0.89, 0.70, 0.45, 0.85) and fallback a/b/c (1.15, 0.10, 0.22) | Named constants, returned in `leverage_ranking_method` and labelled heuristic; fallback uses pooled calibrated items | Transparent |
| 9 | `national_rank_simulator.py` | `TOTAL_NEET_CANDIDATES = 2,400,000` | 1,999,895 (NEET UG 2026 appeared); 2024–2026 facts in `NEET_YEAR_FACTS` | Ranks ~17% lower for the same percentile |
| 10 | same | Qualifying cutoff fixed at 165 | 2026 General/EWS cutoff 213, with source | |
| 11 | same | Percentile curve and college thresholds shown as facts | Kept, but flagged `approximate` in `model_assumptions`; each college row carries `threshold_status` | Honest labelling |
| 12 | `continuous_retraining_engine.py` | Default "current" metrics PSI 0.02, Brier 0.0078, ECE 0.0011 (invented, always "stable") | Reads live drift monitor; fewer than 30 attempts → `INSUFFICIENT_DATA` | No fake "healthy" status |
| 13 | same | Thresholds Brier 0.05 / ECE 0.04, set for the leaky model | 25% above the validation-report baseline (OOF Brier 0.298, ECE 0.037) | Thresholds match the real model |
| 14 | same | Checked `verif["all_verified"]`, a key the verifier never returns, so verification always read FAILED | Checks `integrity_status == "VERIFIED"` | Bug fixed |
| 15 | `feature_store_and_drift.py` | Response-time drift compared against invented `[65, 85, 110]` and N(85, 25) | Compared against the corpus `ideal_time_sec` distribution, labelled as formula-derived, not observed | |
| 16 | same | Missing-file fallback tier mix (0.3448 / 0.4974 / 0.1578) | Uniform fallback; empty state reports `metrics_available: false` | |
| 17 | `streaming_online_calibration.py` | Called an "ELBO" | Renamed log joint (log-likelihood + log-prior); old method name kept as an alias | Label matches maths |
| 18 | `backend/tests/test_v2_engine.py` | Expected a strong student to score > 460, a value that relied on items 2–4 | > 400. The student's θ maps to the 82nd percentile, which the anchors put near 380–430 | Test no longer encodes the bias |

## 2. Kept, and labelled as design choices (not fitted)

| File | Values | Why kept |
|---|---|---|
| `study_plan_and_coverage_engine.py` | Priority weights, spaced-retest intervals | Product policy, not a prediction |
| `linucb_bandit.py` | Reward shaping, α | Policy; warm start is a simulation sanity check (README §4) |
| `dina_cognitive_diagnosis.py` | Response-time weights, slip/guess priors | Priors until student responses exist |
| BKT telemetry modulation | Hesitation / switch multipliers | Heuristic rules |
| `evaluation_harness` personas | Simulated students | Simulations, labelled as such |

## 3. Models improved

| Model | Before | After | Evaluation |
|---|---|---|---|
| Difficulty (verified sample tiers) | v2 structural RF: 78.5% acc, macro F1 0.72, ECE 0.082 | **v2.1 TF-IDF word + char logistic regression: 82.4% acc, macro F1 0.77, ECE 0.019** | GroupKFold(5) by paper; no paper in both train and test |
| Same, duplicate-free rows (89% of test rows) | v2 78.3%, F1 0.72 | v2.1 81.4%, F1 0.76 | Rows whose nearest training row has cosine < 0.80 |
| Chapter resolver | Word TF-IDF LR | Word + char TF-IDF tried: +2–3 pts coverage at 90% precision | Not promoted: the gain is too small to justify re-ingesting 9,448 rows; the 2,633 quarantined rows need human relabelling with top-3 suggestions |

v2.1 is registered in the ML extension registry as a candidate. The production classifier is unchanged.

## 4. What still cannot be made honest without student data

IRT a/b/c, BKT parameters, DINA slip/guess, bandit rewards, score-interval calibration and drift baselines all need real responses. Trigger: about 300 students completing the 180-question initial assessment.
