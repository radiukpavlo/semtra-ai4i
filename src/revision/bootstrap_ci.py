"""Post hoc bootstrap confidence intervals for the AIT&AIS'2026 revision (Reviewer 2, item 2), run after the reviews.

Input: outputs/revision/objects_full.csv, written by rerun_time_sensitivity.py. That script re-ran the frozen test stage,
checked it against outputs/test and also saved the true-concept rulebooks' predictions.

Percentile bootstrap. B = 2000 resamples of the test rows within each seed, stratified by the true class, so each
resample keeps the seed's class counts. Every statistic is computed per seed and averaged over the five seeds, as in
the paper's tables. The 2.5th and 97.5th percentiles of these averages give the 95% interval. Differences are paired:
both terms are computed on the same resample. Scores keep the values computed on the full test split (the hybrid's
ranks are not recomputed). The intervals therefore describe the sampling variability of the test rows for fixed
trained models; the SD over seeds in the paper describes resplitting and retraining.

    .venv/Scripts/python.exe -I src/revision/bootstrap_ci.py --objects outputs/revision/objects_full.csv --out outputs/revision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

B = 2000
RNG_SEED = 20261117
CLASSES = ["none", "HDF", "PWF", "OSF", "TWF"]
METHODS = {"MLP": "mlp_pred", "class-reduct, true": "cr_true_pred", "class-reduct, predicted": "cr_pred",
           "as-published, predicted": "make_pred", "as-published, true": "make_true_pred", "CART, predicted": "cart_pred"}
SCORES = {"max-probability": "doubt_mlp_max_prob", "class-reduct rule doubt": "doubt_class_reduct_disagreement",
          "hybrid, class-reduct": "doubt_class_reduct_hybrid", "as-published rule doubt": "doubt_make_disagreement",
          "hybrid, as-published": "doubt_make_hybrid"}


def f1_recall(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class F1 and recall over CLASSES; an abstention (-1) is a miss, as in pipeline.predictive_metrics."""
    f1, rec = np.zeros(len(CLASSES)), np.zeros(len(CLASSES))
    for k in range(len(CLASSES)):
        tp = np.sum((y == k) & (p == k))
        fp = np.sum((y != k) & (p == k))
        fn = np.sum((y == k) & (p != k))
        f1[k] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        rec[k] = tp / (tp + fn) if (tp + fn) else 0.0
    return f1, rec


def auroc(err: np.ndarray, score: np.ndarray) -> float:
    n1 = err.sum()
    n0 = len(err) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[err == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stats(d: pd.DataFrame) -> dict:
    y = d["y"].to_numpy(int)
    out = {}
    for name, col in METHODS.items():
        f1, rec = f1_recall(y, d[col].to_numpy(int))
        out[("macro-F1", name)] = f1.mean()
        out[("mean failure-mode recall", name)] = rec[1:].mean()
        for k in range(1, len(CLASSES)):
            out[(f"recall, {CLASSES[k]}", name)] = rec[k]
    err = (d["mlp_pred"].to_numpy(int) != y).astype(int)
    for name, col in SCORES.items():
        out[("AUROC", name)] = auroc(err, d[col].to_numpy(float))
    out[("macro-F1 difference", "class-reduct, true minus MLP")] = out[("macro-F1", "class-reduct, true")] - out[("macro-F1", "MLP")]
    for k in range(1, len(CLASSES)):
        out[("recall difference", f"{CLASSES[k]}: MLP minus class-reduct, true")] = (
            out[(f"recall, {CLASSES[k]}", "MLP")] - out[(f"recall, {CLASSES[k]}", "class-reduct, true")])
    out[("AUROC gain", "hybrid, class-reduct minus max-probability")] = out[("AUROC", "hybrid, class-reduct")] - out[("AUROC", "max-probability")]
    out[("AUROC gain", "hybrid, as-published minus max-probability")] = out[("AUROC", "hybrid, as-published")] - out[("AUROC", "max-probability")]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    obj = pd.read_csv(args.objects)
    rng = np.random.default_rng(RNG_SEED)
    seeds = sorted(obj["seed"].unique())

    point_seed = {s: stats(obj[obj["seed"] == s]) for s in seeds}
    keys = list(point_seed[seeds[0]].keys())
    point = {k: float(np.mean([point_seed[s][k] for s in seeds])) for k in keys}

    boot = {s: np.empty((B, len(keys))) for s in seeds}
    for s in seeds:
        d = obj[obj["seed"] == s].reset_index(drop=True)
        groups = [np.flatnonzero(d["y"].to_numpy(int) == k) for k in range(len(CLASSES))]
        for b in range(B):
            idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g)])
            st = stats(d.iloc[idx])
            boot[s][b] = [st[k] for k in keys]
    mean_boot = np.mean([boot[s] for s in seeds], axis=0)
    lo, hi = np.nanpercentile(mean_boot, 2.5, axis=0), np.nanpercentile(mean_boot, 97.5, axis=0)

    rows = [{"statistic": k[0], "method": k[1], "point_mean_over_seeds": point[k], "ci95_low": lo[i], "ci95_high": hi[i]}
            for i, k in enumerate(keys)]
    pd.DataFrame(rows).to_csv(Path(args.out) / "bootstrap_ci.csv", index=False)

    per_seed = []
    for s in seeds:
        for i, k in enumerate(keys):
            if k[0] in ("AUROC gain", "macro-F1 difference", "recall difference"):
                per_seed.append({"seed": s, "statistic": k[0], "method": k[1], "point": point_seed[s][k],
                                 "ci95_low": float(np.nanpercentile(boot[s][:, i], 2.5)),
                                 "ci95_high": float(np.nanpercentile(boot[s][:, i], 97.5))})
    pd.DataFrame(per_seed).to_csv(Path(args.out) / "bootstrap_ci_per_seed.csv", index=False)
    (Path(args.out) / "bootstrap_info.json").write_text(json.dumps(
        {"B": B, "rng_seed": RNG_SEED, "resampling": "test rows within each seed, stratified by true class",
         "aggregation": "statistic per seed, mean over seeds, percentile interval of the mean",
         "scores": "fixed at their full-test-split values"}, indent=2), encoding="utf-8")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(pd.DataFrame(rows).round(4).to_string(index=False))
        print(pd.DataFrame(per_seed).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
