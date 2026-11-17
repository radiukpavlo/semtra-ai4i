"""Full reproduction check for the AIT&AIS'2026 revision (post hoc, 17 Nov 2026).

Re-runs the frozen test stage (run_experiment.run_seed, unchanged) and compares EVERY per-seed output that the frozen
run saved in outputs/test: metrics, concepts, recovery, cuts, rules, triage, risk-coverage curves, objects and the
selection record (without its wall-clock seconds). rerun_time_sensitivity.py compared only objects, metrics, triage and
cuts; this script extends the check to the outputs behind Tables 2 and 3. Writes outputs/revision/reproduction_check_full.json.

    .venv/Scripts/python.exe -I src/revision/reproduction_full.py --csv data/raw/ai4i2020.csv --out outputs/revision
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
import run_experiment as RE  # noqa: E402
from ai4i_data import load_ai4i  # noqa: E402
from rerun_time_sensitivity import compare  # noqa: E402

KEYS = {"metrics_per_seed.csv": ["seed", "method"], "concepts_per_seed.csv": ["seed", "concept"],
        "recovery_per_seed.csv": ["seed", "method", "mode", "concept", "documented"],
        "cuts_per_seed.csv": ["seed", "method", "concept", "cut_physical"],
        "rules_per_seed.csv": ["seed", "method", "rule_id"], "triage_per_seed.csv": ["seed", "score"],
        "risk_coverage_curves.csv": ["seed", "score", "coverage", "risk"], "objects.csv": ["seed", "udi"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = load_ai4i(args.csv)
    acc = {k: [] for k in ["metrics", "concepts", "recovery", "cuts", "rules", "triage", "curves", "objects", "selection"]}
    for seed in P.SEEDS:
        res = RE.run_seed(data, seed, "test")
        for k, v in res.items():
            acc[k].append(v) if k in ("rules", "objects", "selection") else acc[k].extend(v)
        print(f"seed {seed} done", flush=True)
    tmp = Path(args.out) / "_rerun_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    new = {"metrics_per_seed.csv": pd.DataFrame(acc["metrics"]), "concepts_per_seed.csv": pd.DataFrame(acc["concepts"]),
           "recovery_per_seed.csv": pd.DataFrame(acc["recovery"]), "cuts_per_seed.csv": pd.DataFrame(acc["cuts"]),
           "rules_per_seed.csv": pd.concat(acc["rules"], ignore_index=True).drop(columns=["covered_indices_sample"]),
           "triage_per_seed.csv": pd.DataFrame(acc["triage"]), "risk_coverage_curves.csv": pd.DataFrame(acc["curves"]),
           "objects.csv": pd.concat(acc["objects"], ignore_index=True)}
    frozen_dir = ROOT / "outputs" / "test"
    check = {}
    for name, df in new.items():
        df.to_csv(tmp / name, index=False)          # same CSV round trip as the frozen files
        fresh = pd.read_csv(tmp / name)
        frozen = pd.read_csv(frozen_dir / name)
        check[name] = compare(frozen, fresh, KEYS[name])
    sel_new = [{k: v for k, v in s.items() if k != "seconds"} for s in acc["selection"]]
    sel_old = [{k: v for k, v in s.items() if k != "seconds"}
               for s in json.loads((frozen_dir / "selection.json").read_text(encoding="utf-8"))]
    check["selection.json (without seconds)"] = {"identical": json.loads(json.dumps(sel_new)) == sel_old}
    worst = max((v for part in check.values() for v in part.get("max_abs_diff", {}).values()), default=0.0)
    check["reproduced"] = bool(worst < 1e-9 and check["selection.json (without seconds)"]["identical"]
                               and not any(part.get("unequal_columns") for part in check.values()))
    check["largest_abs_diff"] = worst
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    (Path(args.out) / "reproduction_check_full.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "max_abs_diff"})
                      for k, v in check.items()}, indent=1))


if __name__ == "__main__":
    main()
