# Results memo: SEMTRA audit of a neural failure-mode classifier on AI4I 2020

Test-stage results under the frozen protocol (`PROTOCOL.md`, frozen 2026-10-23), 5 seeds (42-46), mean ± SD.
Every number below is traced to a file in `outputs/`. This memo states facts for the authors; it is not paper text.

## 1. Key findings

1. **With faithful concepts, the class-reduct rulebook recovers the generating physics.** Fitted on the true concept values,
   its rules use exactly the documented concept sets for HDF {ΔT, speed}, PWF {power} and OSF {strain, product type} in
   5/5 seeds, with no distractor. The WEDD cuts sit at ΔT 8.600 K (documented 8.6), speed 1379.4 rpm (1380), power
   9005.0 W (9000) and strain 11045 minNm (11000 for type L). With 5.2 rules it reaches macro-F1 0.710 ± 0.020, on par with
   the MLP (0.705 ± 0.044).
2. **Through the MLP's embedding, recovery degrades although the transition looks excellent.** Transition R² is 0.974-0.992
   for every physical concept, yet the rulebook on predicted concepts recovers HDF's concept set in 1/5 seeds, PWF's in 4/5,
   OSF's in 2/5, and reaches macro-F1 0.457 ± 0.061. A high average transition fidelity does not guarantee that narrow
   decision regions (HDF: the lowest ~1 K of ΔT and the lowest ~200 rpm of speed) survive the mapping.
3. **The MAKE rulebook as published collapses at 97:3 class imbalance.** Macro-F1 0.280 ± 0.054 (predicted concepts) and
   0.266 ± 0.034 (true concepts): support-weighted voting lets broad 'none' rules win, and full-signature granules fragment
   the rare modes into prototype rules.
4. **Rule disagreement is complementary to the MLP's confidence, not a replacement.** AUROC for detecting MLP errors:
   0.938 ± 0.010 (max-probability) vs 0.782 ± 0.047 (class-reduct disagreement); the rank-averaged hybrid reaches
   0.944 ± 0.008 and is higher than max-probability in 5/5 seeds (by 0.002-0.010). At the rulebook's own operating point
   (54.4 ± 10.0 flagged rows, 2.7% of decisions) the rule flags catch 23.2 ± 9.3 errors against 19.8 ± 6.1 for
   max-probability with the same budget (more in 3 seeds, equal in 1, fewer in 1); 8.0 ± 3.0 errors per seed are caught
   only by the rules.
5. **Confident errors are common.** 22.4 ± 6.9 of the MLP's 38.8 ± 10.0 test errors per seed (58%) have max-probability
   ≥ 0.90; the class-reduct rules flag 10.2 ± 5.0 of them.
6. **TWF is not recoverable, as designed.** The dataset generates TWF at a random tool-wear time; no method recovers a TWF
   rule, and the MLP's TWF recall is 0.10 ± 0.10.

## 2. Data facts (`outputs/data_audit.json`, `outputs/test/run_info.json`)

- CSV: 10,000 rows; L/M/H 6000/2997/1003; Machine failure 339; TWF 46, HDF 115, PWF 95, OSF 98, RNF 19; 24 rows with several
  modes; 9 machine failures without a mode label.
- Documented rules vs labels: HDF 115/115, PWF 95/95, OSF 98/98, no false positives. TWF: 790 rows in the 200-240 min
  window, 43 labelled TWF; 3 TWF rows outside the window.
- PWF splits into 31 low-power (< 3500 W) and 64 high-power (> 9000 W) cases; OSF is mostly type L (L 87, M 9, H 2).
- Kept 9,958 rows after dropping 19 RNF and 23 multi-mode rows: none 9652, HDF 106, PWF 80, OSF 78, TWF 42.
  Per seed: 5,974 train, 1,992 validation, 1,992 test.

## 3. Transition fidelity (`outputs/tables/T1_concepts.*`, Fig. F4)

Test R²: ΔT 0.974 ± 0.007, speed 0.992 ± 0.003, torque 0.989 ± 0.002, power 0.980 ± 0.005, tool wear 0.992 ± 0.003,
strain 0.990 ± 0.003, product type 0.992 ± 0.002; distractors: air temperature 0.991 ± 0.002, process temperature
0.990 ± 0.002, noise sensor −0.017 ± 0.003. Test MAE: ΔT 0.119 K, speed 12.3 rpm, power 110 W, strain 204 minNm.
The raw temperatures are encoded as well as the physical concepts, so ignoring them is a property of the rules, not of the
transition. Transition: 63 SVD components (64-unit embedding; explained variance 1.0), ridge α 0.01 (0.1 in seed 46).

## 4. Physics recovery (`outputs/tables/T2a_concept_recovery.*`, `T2b_threshold_recovery.*`, Fig. F2)

Exact concept set, seeds out of 5 (mean Jaccard):

| Mode | Class-reduct, true | Class-reduct, predicted | MAKE, predicted | CART, predicted |
|------|-------------------|-------------------------|-----------------|-----------------|
| HDF | 5/5 (1.00) | 1/5 (0.20) | 3/5 (0.77) | 0/5 (0.48) |
| PWF | 5/5 (1.00) | 4/5 (0.80) | 4/5 (0.85) | 0/5 (0.30) |
| OSF | 5/5 (1.00) | 2/5 (0.67) | 0/5 (0.47) | 1/5 (0.68) |
| TWF | 0/5 (0.00) | 0/5 (0.00) | 0/5 (0.20) | 0/5 (0.00) |

Predicted-concept deviations: seed 45 used torque instead of power for PWF (torque ≥ 66.2 Nm, a proxy); seed 46 paired
strain with the noise sensor for OSF; seeds 42 and 45 used strain without product type for OSF.

Nearest learned cut to each documented threshold (physical units; |error| in brackets):

| Threshold | WEDD, true concepts | WEDD, predicted concepts | CART |
|-----------|---------------------|--------------------------|------|
| ΔT 8.6 K | 8.600 (0.000) | 8.6 ± 0.1 (0.04) | 8.6 ± 0.1 (0.04) |
| Speed 1380 rpm | 1379.4 ± 0.5 (0.6) | 1388.0 ± 10.4 (8.5) | 1376.8 ± 10.0 (8.4) |
| Power 3500 W | 3834 ± 35 (334) | 5308 ± 1998 (1808) | 8976 ± 65 (5476) |
| Power 9000 W | 9005.0 ± 1.7 (5.0) | 8999.5 ± 148.7 (99.6) | 9010.0 ± 22.3 (19.7) |
| Strain 11000 (L) | 11045 ± 40 (47) | 11084 ± 244 (188) | 11045 ± 176 (143) |
| Strain 12000 (M) | 11933 ± 44 (67) | 11244 ± 227 (756) | 11185 ± 168 (815) |
| Strain 13000 (H) | 11933 ± 44 (1067) | 11244 ± 227 (1756) | 11185 ± 168 (1815) |
| Wear 200 min | 198.1 ± 6.3 (4.5) | 200.9 ± 5.3 (4.2) | 213.8 ± 9.6 (13.8) |

Misses explained by the method, not by chance: WEDD allows at most 3 cuts per concept (max depth 2), so the H-type
strain threshold (2 OSF rows in the whole dataset) and, often, the low-power threshold are not cut. The class-reduct rules
on true concepts cover only L-type OSF and high-power PWF.

Representative rules, seed 42 (`outputs/test/rules_per_seed.csv`, column `physical`):
- True concepts: HDF ← ΔT < 8.3 AND 1294 ≤ speed < 1379 (support 29, confidence 1.00); HDF ← 8.3 ≤ ΔT < 8.6 AND
  1294 ≤ speed < 1379 (23, 1.00); PWF ← power ≥ 9004 (30, 1.00); OSF ← 10996 ≤ strain < 11959 AND type L (32, 1.00);
  OSF ← strain ≥ 11959 AND type L (12, 1.00).
- Predicted concepts: PWF ← power ≥ 9042 (23, 0.96); OSF ← strain ≥ 11260 (35, 0.89).
- Seed 44, the only seed with an HDF rule on predicted concepts: HDF ← ΔT < 8.412 AND 1317 ≤ speed < 1381 (51, 0.90).

## 5. Method comparison (`outputs/tables/T3_method_comparison.*`)

| Method | Macro-F1 | HDF | PWF | OSF | TWF | Rules | Cond./rule | Rules with distractors |
|--------|---------:|----:|----:|----:|----:|------:|-----------:|-----------------------:|
| MLP | 0.705 ± 0.044 | 0.88 | 0.76 | 0.88 | 0.10 | - | - | - |
| Class-reduct, predicted | 0.457 ± 0.061 | 0.13 | 0.33 | 0.55 | 0.00 | 2.2 | 1.33 | 0.2 |
| Class-reduct, true | 0.710 ± 0.020 | 0.85 | 0.57 | 0.85 | 0.00 | 5.2 | 1.80 | 0.0 |
| MAKE, predicted | 0.280 ± 0.054 | 0.24 | 0.09 | 0.20 | 0.00 | 8.0 | 1.68 | 1.0 |
| MAKE, true | 0.266 ± 0.034 | 0.41 | 0.01 | 0.01 | 0.14 | 8.0 | 1.60 | 2.2 |
| CART, predicted | 0.669 ± 0.040 | 0.74 | 0.76 | 0.70 | 0.00 | 12.4 | 4.29 | 2.4 |

Macro-F1 counts abstentions as errors; recall columns are per-mode recall. Coverage is ≥ 0.997 for every method and the
covered fidelity to the MLP is 0.948-0.981, dominated by the 'none' class; use per-mode recall, not fidelity, to compare.
CART's leaves are counted as rules. Per-seed macro-F1: `outputs/test/metrics_per_seed.csv`.

## 6. Error triage (`outputs/tables/T4_triage.*`, Fig. F3)

| Score | AUROC | Error recall @5% | @10% |
|-------|------:|-----------------:|-----:|
| MLP max-probability | 0.938 ± 0.010 | 0.70 | 0.81 |
| Class-reduct disagreement | 0.782 ± 0.047 | 0.58 | 0.62 |
| Hybrid (MLP + class-reduct) | 0.944 ± 0.008 | 0.73 | 0.81 |
| MAKE disagreement | 0.808 ± 0.059 | 0.51 | 0.69 |
| Hybrid (MLP + MAKE) | 0.931 ± 0.030 | 0.64 | 0.76 |

Budget-matched (class-reduct flags vs max-probability, same number of flagged rows), per seed 42/43/44/45/46: flagged
68/50/46/46/62; caught by rules 34/13/23/15/31; by max-probability 28/13/18/16/24; rules only 11/5/8/5/11; total errors
50/26/39/32/47. The MAKE flags cover twice as many rows (108.6 ± 50.4) and catch fewer errors than max-probability at that
budget (22.4 vs 26.2).
Caveat: the disagreement scores are nearly binary, so their AURC depends on how ties are ordered; compare them at their
operating point (Fig. F3a markers, the budget-matched numbers), not by AURC.

## 7. What the figures show (facts for captions and alt text, to be written by the authors)

- **F2** (`outputs/figures/F2_thresholds_vs_physics.pdf/.svg`), three panels over all 9,958 kept rows.
  (a) ΔT vs rotational speed; HDF rows in vermillion inside the shaded documented region (ΔT < 8.6 K, speed < 1380 rpm);
  vertical and horizontal lines are the WEDD cuts nearest to 1380 rpm and 8.6 K in each seed, blue for true concepts,
  orange for predicted concepts; both cluster on the documented boundaries. (b) Histogram of power (log scale), PWF rows in
  vermillion at both tails; dashed lines at 3500 and 9000 W; learned cuts coincide with 9000 W but sit above 3500 W.
  (c) Strain vs product type (jittered); OSF rows mostly in type L beyond about 11000 minNm; dashed segments mark the
  type-specific thresholds; learned cuts cluster near 11000 minNm.
- **F3** (`outputs/figures/F3_triage.pdf/.svg`). (a) MLP error rate among accepted decisions against the share accepted
  (coverage 0.90-1.00), averaged over seeds: max-probability (black) and hybrid (teal dashed) curves; markers with SD bars
  show the class-reduct rule-flag operating point (coverage 0.973, error rate 0.81%), below the max-probability curve at
  the same coverage (about 1.0%), and the MAKE flag point (coverage 0.946, error rate 0.87%), above it (about 0.6%).
  (b) Per seed: errors caught by the rule flags, by max-probability at the same budget, and by either; the dash marks all
  MLP errors.
- **F4** (`outputs/figures/F4_concept_fidelity.pdf/.svg`). Test R² of the transition per concept (mean ± SD over seeds):
  every physical concept and both raw temperatures between 0.974 and 0.992; the noise sensor at about 0.

## 8. Component list for F1 (the authors draw it)

1. AI4I 2020 machine record: air temperature, process temperature, rotational speed, torque, tool wear, product type.
2. MLP failure-mode classifier (2 × 64 ReLU) → predicted mode (none, HDF, PWF, OSF, TWF) and max-probability.
3. Embedding: last hidden layer (64 units).
4. Transition T (scaling, truncated SVD, ridge) → predicted concept values.
5. Concept dictionary: 7 physical concepts (ΔT, speed, torque, power, tool wear, strain, product type) + 3 distractors.
6. WEDD discretisation: ≤ 3 cuts per concept → concept states.
7. Rulebooks on the states: (a) MAKE rulebook as published; (b) class-reduct rulebook: one reduct and rule set per failure
   mode, 'none' as the default, abstain on ties between modes.
8. Audit outputs: (i) physics check, concept sets and cuts vs the documented failure rules; (ii) triage, rule–MLP
   disagreement flags a decision for human review.
9. Reference path (dashed): the same discretisation and rulebooks on the true concept values computed from the record,
   giving the upper bound that separates transition loss from rule-induction loss.

## 9. Claims the evidence supports, and claims it does not

Supported: physics recovery with faithful concepts (finding 1); transition loss concentrated in narrow regions (finding 2);
the MAKE rulebook's imbalance failure (finding 3); a small, consistent hybrid gain and complementary error catches at a
matched budget (finding 4).
Not supported: that rules outperform the MLP on predicted concepts; any safety guarantee; validity on real plant data
(AI4I is synthetic, Matzka 2020); statistical significance (5 seeds, no hypothesis tests run; each test set holds about
21 HDF, 16 PWF, 16 OSF and 8 TWF rows, so per-mode recalls vary widely, e.g. HDF 0.13 ± 0.30 for the predicted-concept
rulebook).

## 10. Limitations to state

Synthetic data with deterministic labels for three modes; one MLP architecture; WEDD depth limits the number of cuts; a
greedy reduct (with pair lookahead) can miss interactions of three concepts; MAKE settings were reused, not tuned, for the
rulebooks; triage evaluated on an MLP with about 2% test error.

## 11. Reproduce

```
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
python -I src/check_data.py data/raw/ai4i2020.csv outputs/data_audit.json
python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage validation
python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage test
python -I src/make_report.py --csv data/raw/ai4i2020.csv --run outputs/test --out outputs
```
About 50 s per seed on a laptop CPU. The test stage needs `PROTOCOL.md` with "Status: FROZEN".

## 12. Post hoc analyses after the reviews (2026-11-17)

Run after the AIT&AIS'2026 reviews with the frozen code unchanged; scripts in `src/revision/`, outputs in
`outputs/revision/`, log in `CHANGES_AFTER_FREEZE.md`. Labelled post hoc in the paper.

- **Reproduction** (`reproduction_check.json`): the re-run of the test stage reproduced `outputs/test` exactly (objects,
  metrics, triage and cuts; largest absolute difference 0.0). The full check (`reproduction_check_full.json`, from
  `reproduction_full.py`) compares all eight per-seed outputs (metrics, concepts, recovery, cuts, rules, triage,
  risk-coverage curves, objects) and `selection.json` without its wall-clock seconds: identical, largest absolute
  difference 0.0.
- **Bootstrap** (`bootstrap_ci.csv`, `bootstrap_ci_per_seed.csv`; 2,000 class-stratified resamples of the test rows per
  seed, statistic averaged over seeds, 95% percentile interval):
  - macro-F1 difference, class-reduct on true concepts minus MLP: 0.0055 (-0.0304 to 0.0347);
  - AUROC: max-probability 0.938 (0.922-0.950); class-reduct rule doubt 0.782 (0.748-0.809); hybrid 0.944 (0.928-0.957);
  - AUROC gain of the hybrid over max-probability: 0.0065 (0.0031-0.0102); per seed, the interval excludes zero in
    seeds 42, 44 and 46; as-published hybrid: -0.0071 (-0.0191 to 0.0034);
  - recall, MLP: HDF 0.88 (0.81-0.93), PWF 0.76 (0.66-0.85), OSF 0.88 (0.81-0.95);
    class-reduct, true concepts: HDF 0.85 (0.78-0.91), PWF 0.57 (0.46-0.68), OSF 0.85 (0.77-0.92);
    class-reduct, predicted concepts: HDF 0.13 (0.09-0.17);
  - paired recall difference, MLP minus class-reduct on true concepts: HDF 0.029 (-0.048 to 0.114), PWF 0.188
    (0.063-0.313), OSF 0.038 (-0.052 to 0.127), TWF 0.097 (0.025-0.192).
- **Timing** (`timing_summary.csv`, `timing_calls.csv`, `run_info.json`; AMD Ryzen 7 2700, medians over 5 seeds):
  transition fit 0.09 s per seed; WEDD on ten concepts 0.14 s per call; class-reduct search 0.59 s per failure mode
  (0.28-0.83 s); all stages except the MLP grid (transition, WEDD, the four rulebooks, CART) 6.2 s per seed
  (5.4-6.4 s), of which the rulebooks and CART 5.6 s; grid of three MLPs 52.7 s (42.0-55.3 s); whole seed
  58.3 s (47.7-61.8 s).
- **Threshold sensitivity** (`sensitivity_tau_smin.csv`; predicted concepts, WEDD cuts fixed, training rows for the concept
  sets, validation rows for the metrics). Frozen setting tau 0.84, s_min 11: exact HDF / PWF / OSF sets in 1 / 4 / 2
  seeds, validation macro-F1 0.451, HDF recall 0.162 (identical to the frozen concept sets). tau 0.60, s_min 11:
  4 / 4 / 0 seeds, no distractor, macro-F1 0.566, HDF recall 0.610. tau 0.70, s_min 11: 3 / 5 / 1 seeds, macro-F1 0.534.
  At tau 0.84, s_min 5 or 8 changes the sets little (1 / 3-4 / 2), and s_min 5 admits distractors. CART on validation
  rows: macro-F1 0.661.
