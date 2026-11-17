"""Post hoc analyses for the AIT&AIS'2026 revision (17 Nov 2026), run after the reviews.

They are not part of the frozen protocol (PROTOCOL.md) and change none of its code or outputs. Each one wraps or calls
the frozen functions unchanged and writes only to a new folder (default outputs/revision):

1. Reproduction. Re-runs the frozen test stage (run_experiment.run_seed) and compares it with outputs/test. The re-run
   also saves the per-row predictions of the true-concept rulebooks, which the frozen run did not save
   (objects_full.csv, the input of bootstrap_ci.py). Reviewer 2, item 2.
2. Timing. Wall-clock time of each stage of the audit, measured by wrappers around the frozen functions. Reviewer 1,
   item 5.
3. Sensitivity. The class-reduct rulebook on predicted concepts with the rule purity tau in {0.60, 0.70, 0.84} and the
   rule support s_min in {5, 8, 11}. The WEDD cuts stay at the frozen setting, and only training and validation rows
   are used. Reviewer 2, item 3.

    .venv/Scripts/python.exe -I src/revision/rerun_time_sensitivity.py --csv data/raw/ai4i2020.csv --out outputs/revision
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
import pipeline as P  # noqa: E402
import run_experiment as RE  # noqa: E402
from ai4i_data import load_ai4i  # noqa: E402

TAUS = (0.60, 0.70, 0.84)
SUPPORTS = (5, 8, 11)
DOCUMENTED = {"HDF": {"delta_t_K", "speed_rpm"}, "PWF": {"power_W"}, "OSF": {"strain_minNm", "product_type"},
              "TWF": {"tool_wear_min"}}

TIMES: list = []
CAPTURE: dict = {}
STATE = {"seed": None}


def timed(stage: str, fn, capture: str | None = None):
    def wrapper(*a, **k):
        t = time.perf_counter()
        out = fn(*a, **k)
        TIMES.append({"seed": STATE["seed"], "stage": stage, "seconds": time.perf_counter() - t})
        if capture:
            CAPTURE.setdefault(capture, []).append(out)
        return out
    return wrapper


def cpu_name() -> str:
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command",
                              "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"],
                             capture_output=True, text=True, timeout=60)
        return out.stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        return platform.processor()


def compare(frozen: pd.DataFrame, new: pd.DataFrame, keys: list) -> dict:
    """Largest absolute difference per numeric column, and exact equality of the other columns, after aligning on keys."""
    f = frozen.sort_values(keys).reset_index(drop=True)
    n = new.sort_values(keys).reset_index(drop=True)
    out = {"rows_frozen": int(len(f)), "rows_rerun": int(len(n)), "max_abs_diff": {}, "unequal_columns": []}
    if len(f) != len(n):
        out["unequal_columns"].append("<row count>")
        return out
    for c in f.columns:
        if c not in n.columns:
            out["unequal_columns"].append(f"{c} (missing)")
            continue
        a, b = f[c], n[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(float), b.to_numpy(float)
            both_nan = np.isnan(av) & np.isnan(bv)
            d = np.where(both_nan, 0.0, np.abs(av - bv))
            out["max_abs_diff"][c] = float(np.nanmax(d)) if len(d) else 0.0
            if np.isnan(d).any():
                out["unequal_columns"].append(f"{c} (NaN mismatch)")
        elif not (a.astype(str).fillna("") == b.astype(str).fillna("")).all():
            out["unequal_columns"].append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=P.SEEDS)
    args = ap.parse_args()
    protocol = ROOT / "PROTOCOL.md"
    if "Status: FROZEN" not in protocol.read_text(encoding="utf-8"):
        sys.exit("PROTOCOL.md is not frozen.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frozen_dir = ROOT / "outputs" / "test"

    originals = {k: getattr(P, k) for k in ("fit_mlp", "fit_transition", "fit_discretizers", "class_reduct",
                                            "induce_rules", "fit_cart", "apply_mode_rulebook", "apply_rulebook")}
    data = load_ai4i(args.csv)
    names, classes = data.concept_names, data.class_names
    objects, metrics, triage, cuts, sens = [], [], [], [], []

    for seed in args.seeds:
        STATE["seed"] = seed
        CAPTURE.clear()
        P.fit_mlp = timed("mlp_grid_3_networks", originals["fit_mlp"])
        P.fit_transition = timed("transition_fit", originals["fit_transition"], capture="tr")
        P.fit_discretizers = timed("wedd_10_concepts", originals["fit_discretizers"])
        P.class_reduct = timed("class_reduct_search_one_mode", originals["class_reduct"])
        P.induce_rules = timed("as_published_rule_induction", originals["induce_rules"])
        P.fit_cart = timed("cart_grid_6_depths", originals["fit_cart"])
        P.apply_mode_rulebook = timed("class_reduct_inference", originals["apply_mode_rulebook"], capture="mode_out")
        P.apply_rulebook = timed("as_published_inference", originals["apply_rulebook"], capture="make_out")
        t0 = time.perf_counter()
        res = RE.run_seed(data, seed, "test")
        TIMES.append({"seed": seed, "stage": "whole_seed", "seconds": time.perf_counter() - t0})
        for k, fn in originals.items():
            setattr(P, k, fn)

        # 1. per-row outputs, including the true-concept rulebooks (calls 2 of apply_mode_rulebook / apply_rulebook)
        obj = res["objects"].copy()
        obj.insert(obj.columns.get_loc("cr_pred") + 1, "cr_true_pred", CAPTURE["mode_out"][1]["prediction"].to_numpy(int))
        obj.insert(obj.columns.get_loc("make_pred") + 1, "make_true_pred", CAPTURE["make_out"][1]["prediction"].to_numpy(int))
        objects.append(obj)
        metrics += res["metrics"]
        triage += res["triage"]
        cuts += res["cuts"]

        # 3. sensitivity on training and validation rows only, WEDD cuts fixed at the frozen setting
        tr = CAPTURE["tr"][0]
        train, val, _ = P.split(data.y, seed)
        y_tr, y_val = data.y[train], data.y[val]
        s_frozen = P.rule_min_support(len(y_tr))
        cached = originals["fit_discretizers"](tr.Yhat_train, y_tr, list(range(len(names))), alpha=P.WEDD_LAMBDA_H,
                                               max_depth=P.WEDD_MAX_DEPTH, min_support=max(10, s_frozen))
        P.fit_discretizers = lambda *a, **k: cached
        try:
            for tau in TAUS:
                for s in SUPPORTS:
                    rb = P.fit_mode_rulebook(tr.Yhat_train, y_tr, names, classes, min_confidence=tau, s_min=s)
                    used = P.rulebook_concepts_by_class(rb.rules, names, classes) if len(rb.rules) else {}
                    pred = P.apply_mode_rulebook(rb, tr.Yhat_val)["prediction"].to_numpy(int)
                    m = P.predictive_metrics(y_val, pred, None, classes)
                    row = {"seed": seed, "tau": tau, "s_min": s, "rules": int(len(rb.rules)),
                           "val_macro_f1": m["macro_f1_abstain_as_wrong"]}
                    for mode, doc in DOCUMENTED.items():
                        u = set(used.get(mode, set()))
                        row[f"{mode}_concepts"] = ";".join(sorted(u))
                        row[f"{mode}_exact"] = int(u == doc)
                        row[f"{mode}_rules"] = int((rb.rules["consequent_name"] == mode).sum()) if len(rb.rules) else 0
                        row[f"val_recall_{mode}"] = m[f"recall_{mode}"]
                    sens.append(row)
        finally:
            P.fit_discretizers = originals["fit_discretizers"]
        print(f"seed {seed}: done in {TIMES[-1]['seconds']:.1f} s", flush=True)

    objects = pd.concat(objects, ignore_index=True)
    objects.to_csv(out / "objects_full.csv", index=False)
    pd.DataFrame(sens).to_csv(out / "sensitivity_tau_smin.csv", index=False)
    times = pd.DataFrame(TIMES)
    times.to_csv(out / "timing_calls.csv", index=False)

    # 1. reproduction check against the frozen test outputs
    m_new, t_new, c_new = pd.DataFrame(metrics), pd.DataFrame(triage), pd.DataFrame(cuts)
    frozen_obj = pd.read_csv(frozen_dir / "objects.csv")
    new_obj = objects.drop(columns=["cr_true_pred", "make_true_pred"])
    for c in ("cr_activated_rules",):
        frozen_obj[c] = frozen_obj[c].fillna("").astype(str)
        new_obj[c] = new_obj[c].fillna("").astype(str)
    check = {
        "objects": compare(frozen_obj, new_obj, ["seed", "udi"]),
        "metrics": compare(pd.read_csv(frozen_dir / "metrics_per_seed.csv"), m_new, ["seed", "method"]),
        "triage": compare(pd.read_csv(frozen_dir / "triage_per_seed.csv"), t_new, ["seed", "score"]),
        "cuts": compare(pd.read_csv(frozen_dir / "cuts_per_seed.csv"), c_new, ["seed", "method", "concept", "cut_physical"]),
    }
    worst = max((v for part in check.values() for v in part["max_abs_diff"].values()), default=0.0)
    check["reproduced"] = bool(worst < 1e-9 and not any(part["unequal_columns"] for part in check.values()))
    check["largest_abs_diff"] = worst

    # 2. timing summary: per seed, then median over seeds
    per_seed = times.groupby(["seed", "stage"])["seconds"].agg(["sum", "count", "median"]).reset_index()
    summary = (per_seed.groupby("stage")
               .agg(total_per_seed_median=("sum", "median"), total_per_seed_min=("sum", "min"),
                    total_per_seed_max=("sum", "max"), calls_per_seed=("count", "median"),
                    per_call_median=("median", "median"))
               .reset_index())
    summary.to_csv(out / "timing_summary.csv", index=False)
    info = {"date": time.strftime("%Y-%m-%d %H:%M"), "cpu": cpu_name(), "logical_cpus": os.cpu_count(),
            "python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "note": "Single process; wall-clock time from time.perf_counter around the frozen functions."}
    (out / "reproduction_check.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps({"reproduced": check["reproduced"], "largest_abs_diff": worst}, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
