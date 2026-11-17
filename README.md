# Auditing a neural failure-mode classifier with semantic transition and rough-set rules (AI4I 2020)

Code and results for a companion study to the SEMTRA method (global semantic transition matrix, WEDD discretisation,
rough-set rules). A small MLP classifies failure modes on the UCI AI4I 2020 predictive-maintenance benchmark; its hidden
layer is mapped to physical concepts, and the extracted rules are checked against the dataset's documented failure rules
and used to flag the MLP's likely errors for human review.

- `PROTOCOL.md`: the experimental protocol, frozen before any test evaluation, with file checksums.
- `CHANGES_AFTER_FREEZE.md`: the comment masked for the double-blind review and restored after it, the figure restyle, and the
  post hoc analyses added after the reviews (`src/revision/`, `outputs/revision/`).
- `RESULTS.md`: results memo with every number traced to `outputs/`.
- `outputs/`: data audit, validation and test outputs, tables (`outputs/tables/`) and figures (`outputs/figures/`).

## Paper

P. Radiuk, O. Barmak, T. Skrypnyk, S. Petrovskyi, I. Krak, M. Kvassay, "Do extracted rules recover the physics? Auditing a
neural predictive-maintenance classifier", AIT&AIS'2026: International Scientific Workshop on Applied Information Technologies and
Artificial Intelligence Systems, Chernivtsi, Ukraine, 8-9 December 2026, CEUR Workshop Proceedings (to appear).

## Layout

| Path | Contents |
|------|----------|
| `src/semtra_core.py` | SEMTRA reference implementation, vendored unchanged (transition, WEDD, rough-set rules, inference) |
| `src/ai4i_data.py` | AI4I loading, failure-mode labels, concept dictionary (7 physical concepts + 3 distractors) |
| `src/pipeline.py` | MLP and embedding, rulebooks (as published, and the class-reduct variant), CART baseline, metrics, triage, physics recovery |
| `src/run_experiment.py` | Validation and test stages over 5 seeds |
| `src/make_report.py` | Tables T1-T4 and figures F2-F4 from the test outputs |
| `src/check_data.py` | Audit of the CSV against the documented failure rules |
| `src/explore_validation.py` | Validation-only comparison of rulebook settings (design history, see `PROTOCOL.md` §6) |
| `src/figure_style.py` | Figure style |
| `src/revision/` | Post hoc analyses after the reviews: reproduction re-run, timing, threshold sensitivity, bootstrap intervals |

## Data

Download the AI4I 2020 Predictive Maintenance Dataset (S. Matzka, UCI Machine Learning Repository,
DOI 10.24432/C5HS5C, CC BY 4.0) into an empty folder:

```
mkdir -p data/raw
curl -L -o data/raw/ai4i2020.zip "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
unzip data/raw/ai4i2020.zip -d data/raw
```

Expected SHA-256 of `ai4i2020.csv`: `dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`.

## Run

Python 3.13; packages in `requirements.txt`.

```
uv venv --python 3.13 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
python -I src/check_data.py data/raw/ai4i2020.csv outputs/data_audit.json
python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage validation
python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage test
python -I src/make_report.py --csv data/raw/ai4i2020.csv --run outputs/test --out outputs
```

The test stage refuses to run unless `PROTOCOL.md` contains "Status: FROZEN". Each seed takes about a minute on a laptop CPU.

## License

MIT (see `LICENSE`). The AI4I dataset is CC BY 4.0 and is not redistributed here.
