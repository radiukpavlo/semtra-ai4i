"""SEMTRA on AI4I 2020: MLP embedding -> concept transition -> WEDD -> rough-set rules -> audit and triage.

Builds on the vendored `semtra_core` (unchanged). Everything AI4I-specific or new lives here.
"""
from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from semtra_core import (
    class_mode_signatures,
    fit_discretizers,
    fit_transition,
    induce_rules,
    match_rule,
    parse_antecedent,
    quantize,
)

# ----------------------------------------------------------------------------
# Frozen settings (see PROTOCOL.md). MAKE G2 rulebook settings unless noted.
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]
MLP_HIDDEN = (64, 64)
MLP_ALPHA_GRID = (1e-4, 1e-3, 1e-2)       # chosen per seed by validation macro-F1
MLP_MAX_ITER = 300
OVERSAMPLE_FRACTION = 0.20                # each failure mode up to 20% of the 'none' count (train only)
N_COMPONENTS = 64                         # MAKE setting; becomes 63 on a 64-unit embedding
WEDD_LAMBDA_H = 0.65
WEDD_MAX_DEPTH = 2
RULE_MIN_CONFIDENCE = 0.84
MAX_RULES_PER_CLASS = 4
FALLBACK_MAX_DISTANCE = 0.45
CONFLICT_MARGIN = 0.05
CART_DEPTH_GRID = (3, 4, 5, 6, 7, 8)      # chosen per seed by validation macro-F1
REVIEW_BUDGETS = (0.05, 0.10)

DOCUMENTED_RULES: Dict[str, Dict[str, Any]] = {
    "HDF": {"concepts": {"delta_t_K", "speed_rpm"}, "thresholds": {"delta_t_K": [8.6], "speed_rpm": [1380.0]},
            "text": "ΔT < 8.6 K and speed < 1380 rpm"},
    "PWF": {"concepts": {"power_W"}, "thresholds": {"power_W": [3500.0, 9000.0]},
            "text": "power < 3500 W or > 9000 W"},
    "OSF": {"concepts": {"strain_minNm", "product_type"}, "thresholds": {"strain_minNm": [11000.0, 12000.0, 13000.0]},
            "text": "wear × torque > 11000 / 12000 / 13000 minNm (L / M / H)"},
    "TWF": {"concepts": {"tool_wear_min"}, "thresholds": {"tool_wear_min": [200.0, 240.0]},
            "text": "random replacement or failure at wear 200-240 min"},
}


def rule_min_support(n_train: int) -> int:
    """MAKE runner formula (run_revision_v1.py): max(8, min(18, int(0.002 * n_train)))."""
    return max(8, min(18, int(0.002 * n_train)))


def split(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20, as `protocol_a_split` in the MAKE runner."""
    idx = np.arange(len(y))
    train, temp = train_test_split(idx, test_size=0.40, random_state=seed, stratify=y)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=y[temp])
    return train, val, test


# ----------------------------------------------------------------------------
# MLP and embedding
# ----------------------------------------------------------------------------


def oversample(X: np.ndarray, y: np.ndarray, seed: int, fraction: float = OVERSAMPLE_FRACTION) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_major = int(np.sum(y == 0))
    target = int(round(fraction * n_major))
    parts_X, parts_y = [X], [y]
    for c in np.unique(y):
        if c == 0:
            continue
        idx = np.where(y == c)[0]
        extra = target - len(idx)
        if extra > 0:
            pick = rng.choice(idx, size=extra, replace=True)
            parts_X.append(X[pick])
            parts_y.append(y[pick])
    Xo, yo = np.vstack(parts_X), np.concatenate(parts_y)
    order = rng.permutation(len(yo))
    return Xo[order], yo[order]


@dataclass
class MLPResult:
    model: MLPClassifier
    scaler: StandardScaler
    alpha: float
    n_iter: int
    selection: List[Dict[str, Any]]


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int) -> MLPResult:
    scaler = StandardScaler().fit(X_train)
    Xs_train, Xs_val = scaler.transform(X_train), scaler.transform(X_val)
    Xo, yo = oversample(Xs_train, y_train, seed)
    labels = sorted(np.unique(y_train).tolist())
    best, rows = None, []
    for alpha in MLP_ALPHA_GRID:
        model = MLPClassifier(hidden_layer_sizes=MLP_HIDDEN, activation="relu", alpha=alpha, batch_size=256,
                              learning_rate_init=1e-3, max_iter=MLP_MAX_ITER, random_state=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Xo, yo)
        f1 = float(f1_score(y_val, model.predict(Xs_val), labels=labels, average="macro", zero_division=0))
        rows.append({"alpha": alpha, "val_macro_f1": f1, "n_iter": int(model.n_iter_)})
        if best is None or f1 > best[0]:
            best = (f1, alpha, model)
    _, alpha, model = best
    return MLPResult(model=model, scaler=scaler, alpha=alpha, n_iter=int(model.n_iter_), selection=rows)


def embed(mlp: MLPResult, X: np.ndarray) -> np.ndarray:
    """Last hidden layer activations, computed from the fitted weights."""
    h = mlp.scaler.transform(X)
    for W, b in zip(mlp.model.coefs_[:-1], mlp.model.intercepts_[:-1]):
        h = np.maximum(h @ W + b, 0.0)
    return h


# ----------------------------------------------------------------------------
# Rulebook: WEDD + rough-set rules + scored conflict-aware inference
# ----------------------------------------------------------------------------


def infer_rules_scored(Z: np.ndarray, rules_df: pd.DataFrame, prototypes: Dict[int, np.ndarray],
                       fallback_max_distance: float = FALLBACK_MAX_DISTANCE,
                       conflict_margin: float = CONFLICT_MARGIN) -> pd.DataFrame:
    """`semtra_core.infer_rules` with the same decisions, plus the vote table needed for triage scores."""
    rules = []
    for _, r in rules_df.iterrows():
        rr = r.to_dict()
        rr["antecedent_parsed"] = parse_antecedent(rr)
        rules.append(rr)
    proto_labels = [int(k) for k in prototypes]
    proto_mat = np.vstack([np.asarray(v, dtype=int) for v in prototypes.values()]) if prototypes else None
    out = []
    for i, z in enumerate(Z):
        votes: Dict[int, float] = defaultdict(float)
        activated = []
        for r in rules:
            if match_rule(z, r["antecedent_parsed"]):
                votes[int(r["consequent"])] += float(r["confidence"]) * math.log1p(float(r["support"])) / max(1, int(r["antecedent_length"]))
                activated.append(r["rule_id"])
        mode, pred, abstained, conflict, reason, fb_dist = ("exact" if activated else "fallback"), -1, False, False, "", float("nan")
        if votes:
            ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            pred = int(ranked[0][0])
            if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < conflict_margin * max(1.0, ranked[0][1]):
                conflict, abstained, pred, reason = True, True, -1, "conflict"
        elif proto_mat is None:
            abstained, reason = True, "no_rule"
        else:
            d = np.mean(proto_mat != z[None, :], axis=1)
            j = int(np.argmin(d))
            fb_dist = float(d[j])
            if fb_dist <= fallback_max_distance:
                pred = proto_labels[j]
            else:
                abstained, reason = True, "far_fallback"
        out.append({"row_index": i, "prediction": int(pred), "mode": mode, "abstained": abstained, "conflict": conflict,
                    "abstain_reason": reason, "fallback_distance": fb_dist, "n_activated": len(activated),
                    "activated_rules": ";".join(activated[:10]), "votes": dict(votes)})
    return pd.DataFrame(out)


@dataclass
class Rulebook:
    thresholds: Dict[int, List[float]]   # MinMax units of the concept space
    threshold_df: pd.DataFrame
    rules: pd.DataFrame
    prototypes: Dict[int, np.ndarray]
    min_support: int


def fit_rulebook(Ytrain: np.ndarray, y_train: np.ndarray, concept_names: Sequence[str], class_names: Sequence[str],
                 min_confidence: float = RULE_MIN_CONFIDENCE, s_min: Optional[int] = None,
                 wedd_min_support: Optional[int] = None) -> Rulebook:
    attrs = list(range(Ytrain.shape[1]))   # keep every concept: the rules decide whether distractors matter
    s_min = rule_min_support(len(y_train)) if s_min is None else s_min
    wedd_min_support = max(10, s_min) if wedd_min_support is None else wedd_min_support
    thresholds, threshold_df, _ = fit_discretizers(Ytrain, y_train, attrs, alpha=WEDD_LAMBDA_H,
                                                   max_depth=WEDD_MAX_DEPTH, min_support=wedd_min_support)
    Z = quantize(Ytrain, attrs, thresholds)
    attr_scores = np.array([len(thresholds.get(j, [])) + np.std(Z[:, j]) for j in attrs], dtype=float)
    rules, _ = induce_rules(Z, y_train, list(concept_names), list(class_names), min_confidence=min_confidence,
                            min_support=s_min, max_rules_per_class=MAX_RULES_PER_CLASS, attr_scores=attr_scores,
                            allow_prototypes=True, reduce=True)
    prototypes = class_mode_signatures(Z, y_train, sorted(np.unique(y_train).tolist()))
    return Rulebook(thresholds, threshold_df, rules, prototypes, s_min)


def apply_rulebook(rb: Rulebook, Y: np.ndarray) -> pd.DataFrame:
    Z = quantize(Y, list(range(Y.shape[1])), rb.thresholds)
    return infer_rules_scored(Z, rb.rules, rb.prototypes)


# ----------------------------------------------------------------------------
# Imbalance-aware variant: class-specific reducts per failure mode, 'none' as the default class
# ----------------------------------------------------------------------------
REDUCT_MIN_GAIN = 0.01     # stop adding concepts when lower-approximation recall rises by less than this
REDUCT_MAX_LEN = 3


def lower_approx_recall(Z: np.ndarray, pos: np.ndarray, attrs: Sequence[int], min_confidence: float, min_support: int) -> float:
    """Share of positive rows lying in granules (on `attrs`) whose positive purity >= tau and support >= s_min."""
    if not attrs:
        return 0.0
    keys = pd.Series(list(map(tuple, Z[:, list(attrs)])))
    g = pd.DataFrame({"k": keys, "p": pos.astype(int)}).groupby("k")["p"].agg(["sum", "count"])
    ok = (g["count"] >= min_support) & (g["sum"] / g["count"] >= min_confidence)
    return float(g.loc[ok, "sum"].sum() / max(1, pos.sum()))


def class_reduct(Z: np.ndarray, pos: np.ndarray, min_confidence: float, min_support: int) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Greedy (QuickReduct-style) class-specific reduct maximising the lower-approximation recall of one class.

    When no single concept raises the recall, pairs are tried, so conjunctive conditions (both concepts needed,
    neither sufficient alone) are not missed by the one-step greedy search.
    """
    chosen: List[int] = []
    trace, current = [], 0.0
    free = lambda: [a for a in range(Z.shape[1]) if a not in chosen]
    while len(chosen) < REDUCT_MAX_LEN:
        score, add = max(((lower_approx_recall(Z, pos, chosen + [a], min_confidence, min_support), [a]) for a in free()),
                         default=(0.0, []))
        if score - current < REDUCT_MIN_GAIN and len(chosen) + 2 <= REDUCT_MAX_LEN:
            pairs = [[a, b] for i, a in enumerate(free()) for b in free()[i + 1:]]
            score, add = max(((lower_approx_recall(Z, pos, chosen + p, min_confidence, min_support), p) for p in pairs),
                             default=(0.0, []))
        if not add or score - current < REDUCT_MIN_GAIN:
            break
        current = score
        chosen += [int(a) for a in add]
        trace.append({"added": [int(a) for a in add], "lower_approx_recall": current})
    return chosen, trace


def fit_mode_rulebook(Ytrain: np.ndarray, y_train: np.ndarray, concept_names: Sequence[str], class_names: Sequence[str],
                      min_confidence: float = RULE_MIN_CONFIDENCE, s_min: Optional[int] = None) -> Rulebook:
    """Same WEDD cuts as the MAKE rulebook; rules per failure mode on its class-specific reduct."""
    attrs = list(range(Ytrain.shape[1]))
    s_min = rule_min_support(len(y_train)) if s_min is None else s_min
    thresholds, threshold_df, _ = fit_discretizers(Ytrain, y_train, attrs, alpha=WEDD_LAMBDA_H,
                                                   max_depth=WEDD_MAX_DEPTH, min_support=max(10, s_min))
    Z = quantize(Ytrain, attrs, thresholds)
    rows, rid = [], 1
    for c in sorted(set(np.unique(y_train).tolist()) - {0}):
        pos = y_train == c
        red, trace = class_reduct(Z, pos, min_confidence, s_min)
        if not red:
            continue
        keys = pd.Series(list(map(tuple, Z[:, red])))
        g = pd.DataFrame({"k": keys, "p": pos.astype(int)}).groupby("k")["p"].agg(["sum", "count"])
        g = g[(g["count"] >= s_min) & (g["sum"] / g["count"] >= min_confidence)]
        for sig, r in g.iterrows():
            ant = {int(a): int(s) for a, s in zip(red, sig)}
            rows.append({"rule_id": f"R{rid:04d}", "antecedent": {str(a): s for a, s in ant.items()},
                         "antecedent_text": " AND ".join(f"{concept_names[a]}=s{s}" for a, s in sorted(ant.items())),
                         "consequent": int(c), "consequent_name": class_names[int(c)], "support": int(r["count"]),
                         "confidence": float(r["sum"] / r["count"]), "coverage": float(r["count"] / len(y_train)),
                         "conflict_count": int(r["count"] - r["sum"]), "antecedent_length": len(ant),
                         "original_antecedent_length": len(ant), "source": "class_reduct",
                         "reduct": ";".join(concept_names[a] for a in red), "covered_indices_sample": []})
            rid += 1
    return Rulebook(thresholds, threshold_df, pd.DataFrame(rows), {}, s_min)


def apply_mode_rulebook(rb: Rulebook, Y: np.ndarray) -> pd.DataFrame:
    """Failure-mode rules vote; no fired rule means the default class 'none' (0); near-ties between modes abstain."""
    Z = quantize(Y, list(range(Y.shape[1])), rb.thresholds)
    out = infer_rules_scored(Z, rb.rules, prototypes={})
    no_rule = out["n_activated"] == 0
    out.loc[no_rule, ["prediction", "abstained", "abstain_reason", "mode"]] = [0, False, "", "default"]
    return out


# ----------------------------------------------------------------------------
# CART baseline
# ----------------------------------------------------------------------------


def fit_cart(Ytrain, y_train, Yval, y_val, seed: int, min_leaf: int) -> Tuple[DecisionTreeClassifier, List[Dict[str, Any]]]:
    labels = sorted(np.unique(y_train).tolist())
    best, rows = None, []
    for depth in CART_DEPTH_GRID:
        tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_leaf, random_state=seed).fit(Ytrain, y_train)
        f1 = float(f1_score(y_val, tree.predict(Yval), labels=labels, average="macro", zero_division=0))
        rows.append({"max_depth": depth, "val_macro_f1": f1, "leaves": int(tree.get_n_leaves())})
        if best is None or f1 > best[0]:
            best = (f1, tree)
    return best[1], rows


def cart_paths(tree: DecisionTreeClassifier) -> List[Dict[str, Any]]:
    """Leaf rules: list of (feature, op, threshold) conditions and the predicted class."""
    t = tree.tree_
    out = []

    def rec(node: int, conds: List[Tuple[int, str, float]]):
        if t.children_left[node] == t.children_right[node]:
            out.append({"conditions": conds, "class": int(tree.classes_[int(np.argmax(t.value[node][0]))]),
                        "n": int(t.n_node_samples[node])})
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        rec(int(t.children_left[node]), conds + [(f, "<=", thr)])
        rec(int(t.children_right[node]), conds + [(f, ">", thr)])

    rec(0, [])
    return out


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------


def predictive_metrics(y: np.ndarray, pred: np.ndarray, reference: Optional[np.ndarray], class_names: Sequence[str]) -> Dict[str, Any]:
    """`pred` uses -1 for abstention. Per-mode recall counts abstentions as misses."""
    covered = pred >= 0
    labels = list(range(len(class_names)))
    m: Dict[str, Any] = {
        "coverage": float(covered.mean()),
        "covered_accuracy": float(np.mean(pred[covered] == y[covered])) if covered.any() else float("nan"),
        "covered_macro_f1": float(f1_score(y[covered], pred[covered], labels=labels, average="macro", zero_division=0)) if covered.any() else float("nan"),
        "macro_f1_abstain_as_wrong": float(f1_score(y, np.where(covered, pred, -1), labels=labels, average="macro", zero_division=0)),
    }
    rec = recall_score(y, np.where(covered, pred, -1), labels=labels, average=None, zero_division=0)
    for k, name in enumerate(class_names):
        m[f"recall_{name}"] = float(rec[k])
    m["failure_mode_recall_mean"] = float(np.mean(rec[1:]))
    if reference is not None:
        m["covered_fidelity"] = float(np.mean(pred[covered] == reference[covered])) if covered.any() else float("nan")
    return m


def risk_coverage(error: np.ndarray, doubt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Accept rows from least to most doubtful; return (coverage, risk) for every prefix."""
    order = np.argsort(doubt, kind="stable")
    errs = np.cumsum(error[order].astype(float))
    k = np.arange(1, len(error) + 1)
    return k / len(error), errs / k


def triage_metrics(error: np.ndarray, doubt: np.ndarray) -> Dict[str, float]:
    cov, risk = risk_coverage(error, doubt)
    m = {"aurc": float(np.mean(risk)),
         "auroc": float(roc_auc_score(error, doubt)) if 0 < error.sum() < len(error) else float("nan")}
    order = np.argsort(-doubt, kind="stable")
    for b in REVIEW_BUDGETS:
        n = max(1, int(round(b * len(error))))
        m[f"error_recall_at_{int(b * 100)}pct"] = float(error[order[:n]].sum() / max(1, error.sum()))
    return m


def rule_doubt(rule_pred: pd.DataFrame, mlp_pred: np.ndarray, default_class: Optional[int] = None) -> np.ndarray:
    """1 - rule support for the MLP's class (higher = more suspicious).

    Support is the class's share of the fired-rule votes. When no rule fires it is 0, except for a rulebook with
    a default class: then silence fully supports a default-class decision (no alarm) and contradicts any other.
    """
    support = np.zeros(len(mlp_pred))
    for i, votes in enumerate(rule_pred["votes"]):
        if votes:
            support[i] = votes.get(int(mlp_pred[i]), 0.0) / max(sum(votes.values()), 1e-12)
        elif default_class is not None and int(mlp_pred[i]) == default_class:
            support[i] = 1.0
    return 1.0 - support


def budget_matched(error: np.ndarray, rule_doubt_: np.ndarray, mlp_doubt: np.ndarray) -> Dict[str, float]:
    """Rule flags (doubt >= 0.5) against the same number of rows flagged by MLP max-probability."""
    flag_rule = rule_doubt_ >= 0.5
    k = int(flag_rule.sum())
    flag_mlp = np.zeros(len(error), dtype=bool)
    if k:
        flag_mlp[np.argsort(-mlp_doubt, kind="stable")[:k]] = True
    err = error.astype(bool)
    return {"flagged_rows": k, "errors": int(err.sum()),
            "caught_by_rule_flags": int((err & flag_rule).sum()),
            "caught_by_mlp_same_budget": int((err & flag_mlp).sum()),
            "caught_by_rule_only": int((err & flag_rule & ~flag_mlp).sum()),
            "caught_by_either": int((err & (flag_rule | flag_mlp)).sum())}


def rank01(v: np.ndarray) -> np.ndarray:
    return pd.Series(v).rank(method="average").to_numpy() / len(v)


# ----------------------------------------------------------------------------
# Physics recovery
# ----------------------------------------------------------------------------


def to_physical(j: int, t: float, scaler_B) -> float:
    return float(t * scaler_B.data_range_[j] + scaler_B.data_min_[j])


def state_interval(cuts: List[float], state: int) -> Tuple[float, float]:
    lo = -np.inf if state == 0 else cuts[state - 1]
    hi = np.inf if state >= len(cuts) else cuts[state]
    return lo, hi


def describe_rule(antecedent: Dict[int, int], cuts_phys: Dict[int, List[float]], names: Sequence[str]) -> str:
    parts = []
    for j, s in sorted(antecedent.items()):
        lo, hi = state_interval(cuts_phys.get(j, []), s)
        if np.isinf(lo) and np.isinf(hi):
            parts.append(f"{names[j]} any")
        elif np.isinf(lo):
            parts.append(f"{names[j]} < {hi:.4g}")
        elif np.isinf(hi):
            parts.append(f"{names[j]} ≥ {lo:.4g}")
        else:
            parts.append(f"{lo:.4g} ≤ {names[j]} < {hi:.4g}")
    return " AND ".join(parts)


def recovery_from_cuts(cuts_phys: Dict[int, List[float]], used: Dict[str, set], names: Sequence[str],
                       ranges: Dict[int, float]) -> List[Dict[str, Any]]:
    """Per documented failure mode: concept Jaccard and nearest-cut threshold errors (physical units)."""
    rows = []
    idx = {n: j for j, n in enumerate(names)}
    for mode, spec in DOCUMENTED_RULES.items():
        got = used.get(mode, set())
        jacc = len(got & spec["concepts"]) / max(1, len(got | spec["concepts"]))
        for concept, thetas in spec["thresholds"].items():
            j = idx[concept]
            cuts = cuts_phys.get(j, [])
            for theta in thetas:
                err = min(abs(c - theta) for c in cuts) if cuts else float("nan")
                rows.append({"mode": mode, "concept": concept, "documented": theta,
                             "nearest_cut": (min(cuts, key=lambda c: abs(c - theta)) if cuts else float("nan")),
                             "abs_error": err, "rel_error": err / ranges[j] if cuts else float("nan"),
                             "concepts_used": ";".join(sorted(got)), "concept_jaccard": jacc})
    return rows


def rulebook_concepts_by_class(rules: pd.DataFrame, names: Sequence[str], class_names: Sequence[str],
                               sources: Optional[set] = None) -> Dict[str, set]:
    used: Dict[str, set] = defaultdict(set)
    for _, r in rules.iterrows():
        if sources is not None and r["source"] not in sources:
            continue
        for j in parse_antecedent(r):
            used[class_names[int(r["consequent"])]].add(names[j])
    return used


def cart_concepts_by_class(paths: List[Dict[str, Any]], names: Sequence[str], class_names: Sequence[str]) -> Dict[str, set]:
    used: Dict[str, set] = defaultdict(set)
    for p in paths:
        for f, _, _ in p["conditions"]:
            used[class_names[p["class"]]].add(names[f])
    return used
