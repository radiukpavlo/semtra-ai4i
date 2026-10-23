# Experimental protocol: SEMTRA audit of a neural failure-mode classifier on AI4I 2020

Status: FROZEN on 2026-10-23, before any evaluation on test rows. Everything below was fixed using training and
validation rows only. Test results are reported as obtained; any analysis added after this point is labelled exploratory.

## 1. Data

- UCI AI4I 2020 Predictive Maintenance Dataset (DOI 10.24432/C5HS5C, CC BY 4.0), file `ai4i2020.csv`
  (SHA-256 `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`), 10,000 rows.
- Audit of the CSV (`outputs/data_audit.json`): L/M/H = 6000/2997/1003; Machine failure 339; TWF 46, HDF 115, PWF 95,
  OSF 98, RNF 19; 24 rows carry more than one mode. The HDF, PWF and OSF labels agree with the documented generating rules
  on every row (115/115, 95/95, 98/98, no false positives). TWF is partly random: 790 rows have tool wear in 200-240 min,
  43 of them are labelled TWF, and 3 TWF rows lie outside that window.
- Labels (decision 13): none, HDF, PWF, OSF, TWF. Dropped: the 19 RNF rows and 23 further rows with more than one mode.
  Kept 9,958 rows: none 9652, HDF 106, PWF 80, OSF 78, TWF 42.
- Splits: stratified 60/20/20 train/validation/test (`train_test_split` twice, `random_state = seed`), seeds 42-46.
  Per seed: 5,974 train, 1,992 validation, 1,992 test rows.

## 2. Concepts (decision 15)

Rule-relevant: ΔT = process − air temperature [K], rotational speed [rpm], torque [Nm], power = torque · speed · 2π/60 [W],
tool wear [min], strain = tool wear · torque [minNm], product type (L/M/H coded 0/1/2). Distractors: raw air temperature [K]
and raw process temperature [K] (components of ΔT that appear in no rule by themselves) and a pure-noise sensor
N(50, 10²) drawn once with seed 2020.

Documented ground truth: HDF iff ΔT < 8.6 K and speed < 1380 rpm; PWF iff power < 3500 W or > 9000 W; OSF iff
strain > 11000 / 12000 / 13000 minNm for L / M / H; TWF: tool replaced or failed at a random wear time in 200-240 min.

## 3. Pipeline (all settings fixed unless marked "selected on validation")

| Stage | Setting |
|-------|---------|
| Inputs | 5 sensor readings + one-hot product type, standardised on training rows |
| MLP | scikit-learn `MLPClassifier`, hidden (64, 64), ReLU, Adam, batch 256, learning rate 1e-3, max 300 epochs, `random_state = seed`; failure modes randomly oversampled (training rows only) to 20% of the 'none' count; L2 alpha **selected on validation** from {1e-4, 1e-3, 1e-2} by macro-F1 |
| Embedding | last hidden layer (64 ReLU units), computed from the fitted weights |
| Transition | MAKE `fit_transition` unchanged: `StandardScaler(with_mean=False)`, `TruncatedSVD(n_components=64 → 63, n_iter=2)`, MinMax-scaled concept targets clipped to [0, 1], ridge α chosen by validation MAE from {0.01, 0.1, 1, 10, 100} |
| Discretisation | MAKE WEDD `fit_discretizers` unchanged: λ_H = 0.65, max depth 2 (≤ 3 cuts per concept), minimum support max(10, s_min) = 11 |
| s_min | MAKE runner formula max(8, min(18, ⌊0.002 · n_train⌋)) = 11 |
| MAKE rulebook | MAKE `induce_rules` + conflict-aware inference unchanged: τ = 0.84, s_min = 11, ≤ 4 rules per class, prototype rules allowed, fallback Hamming distance 0.45, conflict margin 0.05 |
| Class-reduct rulebook (new) | Same WEDD states. Per failure mode c (one-vs-rest): greedy class-specific reduct maximising the lower-approximation recall of c (share of c rows in granules with c-purity ≥ τ and support ≥ s_min); add the best single concept, or the best pair when no single concept gains ≥ 0.01; stop at gain < 0.01 or 3 concepts. Rules = granules on the reduct with c-purity ≥ 0.84 and support ≥ 11. Inference: fired rules vote conf · ln(1 + support) / length; no fired rule → default class 'none'; two modes within the 0.05 margin → abstain |
| Upper bound | Both rulebooks also fitted on the true concept values (same code), to measure what the transition loses |
| CART | `DecisionTreeClassifier` on the same predicted concepts, `min_samples_leaf = 11`; max depth **selected on validation** from {3, ..., 8} by macro-F1 |

Software: Python 3.13.15, numpy 2.5.4, pandas 3.1.0, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2.
`src/semtra_core.py` is the MAKE module `scripts/core/paper1_core.py` (SHA-256 of the original
`466a031dbff9ef1694d3952b74860fc8e45ff4d929c76a435abb00723b70805a`) with a two-line header added.

## 4. Endpoints

Primary
1. **Concept recovery** per failure mode: Jaccard between the concepts used by the mode's rules and the documented set;
   reported for the class-reduct rulebook on predicted and true concepts, the MAKE rulebook and CART.
2. **Threshold recovery**: for every documented threshold, the nearest learned cut on that concept, absolute error in
   physical units (WEDD cuts are shared by both rulebooks; CART cuts reported separately).
3. **Error triage** for the MLP: AUROC and AURC of the class-reduct disagreement score (1 − share of rule votes for the MLP's
   class; silence supports 'none') against the MLP's own doubt (1 − max probability); budget-matched comparison: rows the
   rule flags (doubt ≥ 0.5) vs the same number of top-doubt rows by max probability (errors caught by each, by rules only,
   by either).

Secondary: macro-F1 with abstention counted wrong, per-mode recall, coverage, covered fidelity to the MLP, rule count,
conditions per rule, rules using distractors, distractor cuts; per-concept transition R² and MAE; the hybrid score
(mean of the two rank-normalised doubts); the MAKE-rulebook triage scores; error recall at 5% and 10% review budgets;
confident errors (MLP errors with max probability ≥ 0.90) flagged by rules. All reported as mean ± SD over 5 seeds.

## 5. Validation summary (seeds 42-46; for reference, not used for test claims)

| Method | Macro-F1 | HDF | PWF | OSF | TWF recall |
|--------|---------:|----:|----:|----:|-----------:|
| MLP | 0.714 | 0.838 | 0.812 | 0.870 | 0.139 |
| Class-reduct rulebook, predicted concepts | 0.451 | 0.162 | 0.425 | 0.402 | 0.000 |
| Class-reduct rulebook, true concepts | 0.708 | 0.733 | 0.662 | 0.843 | 0.000 |
| MAKE rulebook, predicted concepts | 0.317 | 0.295 | 0.162 | 0.238 | 0.000 |
| MAKE rulebook, true concepts | 0.262 | 0.314 | 0.038 | 0.064 | 0.075 |
| CART, predicted concepts | 0.661 | 0.743 | 0.775 | 0.623 | 0.000 |

Class reducts on true concepts were exactly the documented sets in all five seeds (HDF {ΔT, speed}, PWF {power},
OSF {strain, product type}). Triage: AUROC 0.951 (max probability), 0.776 (class-reduct disagreement), 0.955 (hybrid);
the rule flags covered about 52 rows per seed and caught 20.4 of 35.6 errors, against 17.8 for max probability at the
same budget (7.0 caught only by the rules).

Selected on validation: MLP alpha 1e-3 (seed 42) and 1e-2 (seeds 43-46); CART depth 6, 7, 5, 5, 5; ridge α 0.01
(0.1 for seed 46).

## 6. Design history (validation rows only)

1. First run with the MAKE rulebook settings (seed 42): rules for failure modes were prototype rules; inference was
   dominated by high-support 'none' rules.
2. `src/explore_validation.py` compared the three rulebook settings the MAKE code uses on its own datasets (AwA2 G2
   0.84/formula, Derm7pt 0.60/5, synthetic 0.86/25) on seeds 42-43: validation macro-F1 0.24-0.34 on predicted concepts.
3. The class-reduct rulebook was added; a first version without pair lookahead missed HDF, whose rule needs two concepts
   jointly; the pair lookahead was added. The author approved reporting both rulebooks (decision 32).
4. The budget-matched triage comparison was added before the protocol was frozen, because over half of the MLP's errors
   were confident ones.

## 7. Integrity

SHA-256 at freezing time:

| File | SHA-256 |
|------|---------|
| src/ai4i_data.py | 4ea6eca1af1050dd49d1e8cc0c4f8ece4cdb3a269b2d199e7240a0961d5af145 |
| src/pipeline.py | 85ae7cffc8e47b696a971fc6b93620a708a4a933b9231148514d80489992a063 |
| src/run_experiment.py | 8e73c8281f1c0ddaaf8eb8cf6a052b3b4a1cc32514b8b93f12f69948e98484d3 |
| src/semtra_core.py | 9a1814755a738bd0c68d112d986df489c9886938960b3d7a8930c8813e6e6f23 |
| src/check_data.py | 1d6fb207cf809380efa033cac2dd493f0ae0487d0336b188027e016130e120c9 |
| src/explore_validation.py | c91255bfa532ef0f60975ad29923591982d910a91ae254c3ec3d832192047c83 |
| data/raw/ai4i2020.csv | dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e |

Run the test stage with: `python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage test`
