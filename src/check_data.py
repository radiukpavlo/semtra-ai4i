"""Audit the AI4I 2020 CSV: counts, label overlaps, and agreement with the documented failure rules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ai4i_data import load_raw, physical_quantities  # noqa: E402


def main(csv: str, out: str) -> None:
    df = load_raw(csv)
    q = physical_quantities(df)
    modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    n_modes = df[modes].sum(axis=1)

    doc = {
        "HDF": (q["delta_t_K"] < 8.6) & (q["speed_rpm"] < 1380),
        "PWF": (q["power_W"] < 3500) | (q["power_W"] > 9000),
        "OSF": q["strain_minNm"] > df["Type"].map({"L": 11000, "M": 12000, "H": 13000}),
        "TWF": (q["tool_wear_min"] >= 200) & (q["tool_wear_min"] <= 240),
    }
    agreement = {}
    for m, rule in doc.items():
        lab = df[m].astype(bool)
        agreement[m] = {
            "label_count": int(lab.sum()),
            "rule_count": int(rule.sum()),
            "label_and_rule": int((lab & rule).sum()),
            "label_not_rule": int((lab & ~rule).sum()),
            "rule_not_label": int((rule & ~lab).sum()),
        }

    report = {
        "rows": int(len(df)),
        "type_counts": df["Type"].value_counts().to_dict(),
        "machine_failure": int(df["Machine failure"].sum()),
        "mode_counts": {m: int(df[m].sum()) for m in modes},
        "rows_with_any_mode": int((n_modes > 0).sum()),
        "rows_with_multiple_modes": int((n_modes > 1).sum()),
        "machine_failure_without_mode": int(((df["Machine failure"] == 1) & (n_modes == 0)).sum()),
        "mode_without_machine_failure": int(((df["Machine failure"] == 0) & (n_modes > 0)).sum()),
        "rnf_only_rows": int(((df["RNF"] == 1) & (n_modes == 1)).sum()),
        "documented_rule_agreement": agreement,
        "quantity_ranges": {c: [float(q[c].min()), float(q[c].max())] for c in q.columns},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
