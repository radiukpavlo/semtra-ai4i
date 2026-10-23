"""Validation-only exploration of the rulebook settings the MAKE code already uses on its datasets.

Candidates (from the MAKE runners): AwA2 G2 (tau 0.84, s_min by formula, WEDD support max(10, s_min)),
Derm7pt (tau 0.60, s_min 5, WEDD support 5), synthetic (tau 0.86, s_min 25, WEDD support 25).
Evaluation rows are the validation split; test rows are never used.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
from ai4i_data import load_ai4i  # noqa: E402

CANDIDATES = {
    "AwA2-G2 (0.84, formula)": dict(min_confidence=0.84, s_min=None, wedd_min_support=None),
    "Derm7pt (0.60, 5)": dict(min_confidence=0.60, s_min=5, wedd_min_support=5),
    "synthetic (0.86, 25)": dict(min_confidence=0.86, s_min=25, wedd_min_support=25),
}


def main(csv: str, out: str, seeds: list[int]) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    data = load_ai4i(csv)
    rows = []
    for seed in seeds:
        train, val, _ = P.split(data.y, seed)
        y_tr, y_val = data.y[train], data.y[val]
        mlp = P.fit_mlp(data.X[train], y_tr, data.X[val], y_val, seed)
        A_tr, A_val = P.embed(mlp, data.X[train]), P.embed(mlp, data.X[val])
        mlp_val = mlp.model.predict(mlp.scaler.transform(data.X[val])).astype(int)
        tr = P.fit_transition(A_tr, A_val, A_val, data.B[train], data.B[val], data.B[val], n_components=P.N_COMPONENTS, seed=seed)
        for name, kw in CANDIDATES.items():
            for space, Ytr, Yev in [("predicted", tr.Yhat_train, tr.Yhat_val), ("true", tr.Y_train, tr.Y_val)]:
                rb = P.fit_rulebook(Ytr, y_tr, data.concept_names, data.class_names, **kw)
                pred = P.apply_rulebook(rb, Yev)["prediction"].to_numpy(int)
                m = P.predictive_metrics(y_val, pred, mlp_val, data.class_names)
                src = rb.rules["source"].value_counts().to_dict()
                rows.append({"seed": seed, "setting": name, "space": space, "rules": len(rb.rules),
                             "non_prototype_failure_rules": int(((rb.rules["consequent"] > 0) & (rb.rules["source"] != "semantic_prototype")).sum()),
                             "prototype_rules": int(src.get("semantic_prototype", 0)), **{k: m[k] for k in (
                                 "coverage", "covered_fidelity", "macro_f1_abstain_as_wrong", "failure_mode_recall_mean",
                                 "recall_HDF", "recall_PWF", "recall_OSF", "recall_TWF")}})
        for space, Ytr, Yev in [("predicted", tr.Yhat_train, tr.Yhat_val), ("true", tr.Y_train, tr.Y_val)]:
            rb = P.fit_mode_rulebook(Ytr, y_tr, data.concept_names, data.class_names)
            pred = P.apply_mode_rulebook(rb, Yev)["prediction"].to_numpy(int)
            m = P.predictive_metrics(y_val, pred, mlp_val, data.class_names)
            rows.append({"seed": seed, "setting": "class reducts (0.84, formula)", "space": space, "rules": len(rb.rules),
                         "non_prototype_failure_rules": len(rb.rules), "prototype_rules": 0, **{k: m[k] for k in (
                             "coverage", "covered_fidelity", "macro_f1_abstain_as_wrong", "failure_mode_recall_mean",
                             "recall_HDF", "recall_PWF", "recall_OSF", "recall_TWF")}})
            if len(rb.rules):
                print(seed, space, rb.rules.groupby("consequent_name")["reduct"].first().to_dict(), flush=True)
    df = pd.DataFrame(rows)
    Path(out).mkdir(parents=True, exist_ok=True)
    df.to_csv(Path(out) / "rule_settings_validation.csv", index=False)
    print(df.groupby(["space", "setting"]).mean(numeric_only=True).drop(columns="seed").round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]] or [42, 43])
