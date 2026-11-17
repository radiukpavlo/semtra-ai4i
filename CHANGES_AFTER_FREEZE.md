# Changes after the protocol freeze

`PROTOCOL.md` was frozen on 2026-10-23, before any test evaluation, and `outputs/protocol_freeze.sha256` records the
checksums of the frozen files. No frozen code, setting or output has changed since. A comment in one frozen file was
masked for the double-blind review and restored afterwards, and analyses requested by the reviewers were added later
in separate files.

## Frozen files

| Date | File | Change | Reason |
|------|------|--------|--------|
| 2026-10-23 | `src/semtra_core.py`, line 1 | The upstream commit hash in the provenance comment was replaced with "[masked for double-blind review]" | Double-blind review |
| 2026-11-17 | `src/semtra_core.py`, line 1 | The original hash `263c3225d6b8e5e1ae962a51b2ab14d7a1d23a3d` was restored | Camera-ready version after acceptance |

Consequence: the file is again byte-identical to the frozen file, and `sha256sum -c outputs/protocol_freeze.sha256`
passes for every file (the CSV after it is downloaded as described in `README.md`). Between 2026-10-23 and 2026-11-17
the check reported `src/semtra_core.py: FAILED`; the code below the two comment lines never changed:

```
tail -n +3 src/semtra_core.py | sha256sum
466a031dbff9ef1694d3952b74860fc8e45ff4d929c76a435abb00723b70805a
```

| `src/semtra_core.py` | SHA-256 of the whole file |
|----------------------|---------------------------|
| Frozen (in `outputs/protocol_freeze.sha256`) and since 2026-11-17 | `9a1814755a738bd0c68d112d986df489c9886938960b3d7a8930c8813e6e6f23` |
| 2026-10-23 to 2026-11-17 (masked) | `df482ee87ddda24219614b62944e4c40a5af8b5cce093e955fd5f7ee401ed5aa` |

Files are stored byte for byte (`.gitattributes` turns off line-ending conversion), so the checksums hold on any checkout.

## Figures (not frozen)

On 2026-10-24 the branch `figures-paper` restyled F2-F4 and split F2 and F3 into the panel files that the paper prints
as Figures 3 and 4 (`src/make_report.py`, `src/figure_style.py`, `outputs/figures/`). It was merged into `main` on
2026-11-17. Appearance only: no number changed.

## Post hoc analyses after the reviews (2026-11-17)

The reviewers of the AIT&AIS'2026 workshop asked for confidence intervals, the cost of the audit and the effect of
relaxing the rule thresholds. The analyses below were run after the test stage. They are labelled post hoc in the paper,
call the frozen functions unchanged and write only to `outputs/revision/`.

| Script | What it does | Outputs |
|--------|--------------|---------|
| `src/revision/rerun_time_sensitivity.py` | (1) re-runs the frozen test stage (`run_experiment.run_seed`) and compares it with `outputs/test`; it also saves the per-row predictions of the true-concept rulebooks, which the frozen run did not save; (2) times each stage with wrappers around the frozen functions; (3) refits the class-reduct rulebook on predicted concepts with the rule purity tau in {0.60, 0.70, 0.84} and the support s_min in {5, 8, 11}, with the WEDD cuts fixed at the frozen setting, on training and validation rows only | `reproduction_check.json`, `objects_full.csv`, `timing_calls.csv`, `timing_summary.csv`, `sensitivity_tau_smin.csv`, `run_info.json` |
| `src/revision/bootstrap_ci.py` | 95% percentile bootstrap: 2,000 resamples of the test rows within each seed, stratified by class, each statistic averaged over the five seeds; paired for differences (macro-F1, per-mode recall of the MLP minus the true-concept rulebook, AUROC gains); scores fixed at their full-test values | `bootstrap_ci.csv`, `bootstrap_ci_per_seed.csv`, `bootstrap_info.json` |
| `src/revision/reproduction_full.py` | re-runs the frozen test stage again and compares all eight per-seed outputs of `outputs/test` and `selection.json` (without seconds) with the frozen files; written after `rerun_time_sensitivity.py` compared only four of them | `reproduction_check_full.json` |

```
.venv/Scripts/python.exe -I src/revision/rerun_time_sensitivity.py --csv data/raw/ai4i2020.csv --out outputs/revision
.venv/Scripts/python.exe -I src/revision/bootstrap_ci.py --objects outputs/revision/objects_full.csv --out outputs/revision
.venv/Scripts/python.exe -I src/revision/reproduction_full.py --csv data/raw/ai4i2020.csv --out outputs/revision
```
