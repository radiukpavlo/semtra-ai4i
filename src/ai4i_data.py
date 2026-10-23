"""AI4I 2020 loading, failure-mode labels and the concept dictionary (decisions 13 and 15)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

MODES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
CLASS_NAMES = ["none", "HDF", "PWF", "OSF", "TWF"]
TYPE_CODE = {"L": 0, "M": 1, "H": 2}

# Concepts the documented failure rules use, then distractors (decision 15).
RULE_CONCEPTS = ["delta_t_K", "speed_rpm", "torque_Nm", "power_W", "tool_wear_min", "strain_minNm", "product_type"]
DISTRACTORS = ["air_temp_K", "process_temp_K", "noise_sensor"]
CONCEPTS = RULE_CONCEPTS + DISTRACTORS

# MLP inputs: the raw sensor readings plus a one-hot product type.
INPUT_COLUMNS = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]


def load_raw(csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv, encoding="utf-8-sig")
    expected = {"UDI", "Type", "Machine failure", *INPUT_COLUMNS, *MODES}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"AI4I CSV is missing columns: {sorted(missing)}")
    return df


def physical_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Derived quantities in physical units, as defined on the UCI dataset page."""
    speed = df["Rotational speed [rpm]"].astype(float)
    torque = df["Torque [Nm]"].astype(float)
    wear = df["Tool wear [min]"].astype(float)
    return pd.DataFrame({
        "delta_t_K": df["Process temperature [K]"].astype(float) - df["Air temperature [K]"].astype(float),
        "speed_rpm": speed,
        "torque_Nm": torque,
        "power_W": torque * speed * 2.0 * np.pi / 60.0,
        "tool_wear_min": wear,
        "strain_minNm": wear * torque,
        "product_type": df["Type"].map(TYPE_CODE).astype(float),
        "air_temp_K": df["Air temperature [K]"].astype(float),
        "process_temp_K": df["Process temperature [K]"].astype(float),
    })


@dataclass
class AI4IData:
    X: np.ndarray            # MLP inputs (unscaled): 5 sensors + 3 one-hot type columns
    B: np.ndarray            # concept matrix in physical units, columns = CONCEPTS
    y: np.ndarray            # failure-mode class index into CLASS_NAMES
    udi: np.ndarray
    product_type: np.ndarray  # "L" / "M" / "H"
    input_names: List[str]
    concept_names: List[str] = field(default_factory=lambda: list(CONCEPTS))
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    dropped: Dict[str, int] = field(default_factory=dict)


def load_ai4i(csv: str | Path, noise_seed: int = 2020) -> AI4IData:
    """Failure-mode dataset: drop RNF rows and rows with more than one failure mode (decision 13)."""
    df = load_raw(csv)
    n_modes = df[MODES].sum(axis=1)
    rnf = df["RNF"] == 1
    multi = (n_modes > 1) & ~rnf
    keep = ~rnf & ~multi
    dropped = {"rnf_rows": int(rnf.sum()), "multi_mode_rows_without_rnf": int(multi.sum())}
    df = df.loc[keep].reset_index(drop=True)

    y = np.zeros(len(df), dtype=int)
    for k, name in enumerate(CLASS_NAMES[1:], start=1):
        y[df[name].to_numpy() == 1] = k

    q = physical_quantities(df)
    # A pure-noise "sensor" with a plausible scale; fixed seed so every run sees the same column.
    rng = np.random.default_rng(noise_seed)
    q["noise_sensor"] = rng.normal(50.0, 10.0, size=len(df))
    B = q[CONCEPTS].to_numpy(dtype=float)

    onehot = pd.get_dummies(df["Type"]).reindex(columns=["L", "M", "H"], fill_value=0).astype(float)
    X = np.hstack([df[INPUT_COLUMNS].to_numpy(dtype=float), onehot.to_numpy()])
    input_names = INPUT_COLUMNS + ["type_L", "type_M", "type_H"]
    return AI4IData(X=X, B=B, y=y, udi=df["UDI"].to_numpy(), product_type=df["Type"].to_numpy(),
                    input_names=input_names, dropped=dropped)
