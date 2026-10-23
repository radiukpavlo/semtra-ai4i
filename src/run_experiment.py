"""Run the AI4I experiment.

    python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage validation
    python -I src/run_experiment.py --csv data/raw/ai4i2020.csv --out outputs --stage test

The validation stage never touches test rows. The test stage refuses to run until PROTOCOL.md says "Status: FROZEN".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
from ai4i_data import DISTRACTORS, load_ai4i  # noqa: E402
from semtra_core import parse_antecedent  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

CR_PRED, CR_TRUE = "Class-reduct rulebook (predicted concepts)", "Class-reduct rulebook (true concepts)"
MAKE_PRED, MAKE_TRUE = "MAKE rulebook (predicted concepts)", "MAKE rulebook (true concepts)"
CART_PRED = "CART (predicted concepts)"
CONFIDENT = 0.90   # an MLP error with max-probability >= 0.90 counts as a confident error


def concept_metrics(tr, names: List[str], seed: int) -> List[Dict[str, Any]]:
    rows = []
    for j, name in enumerate(names):
        y, yh = tr.Y_test[:, j], tr.Yhat_test[:, j]
        ss_res, ss_tot = float(np.sum((y - yh) ** 2)), float(np.sum((y - y.mean()) ** 2))
        rows.append({"seed": seed, "concept": name, "distractor": name in DISTRACTORS,
                     "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
                     "mae_minmax": float(np.mean(np.abs(y - yh))),
                     "mae_physical": float(np.mean(np.abs(y - yh)) * tr.scaler_B.data_range_[j]),
                     "pearson_r": float(np.corrcoef(y, yh)[0, 1]) if np.std(yh) > 0 else float("nan")})
    return rows


def rulebook_report(rb: P.Rulebook, tr, names, class_names, method: str, seed: int):
    cuts_phys = {j: [P.to_physical(j, t, tr.scaler_B) for t in ts] for j, ts in rb.thresholds.items()}
    ranges = {j: float(tr.scaler_B.data_range_[j]) for j in range(len(names))}
    rules = rb.rules.copy()
    rules["physical"] = [P.describe_rule(parse_antecedent(r), cuts_phys, names) for _, r in rules.iterrows()]
    rules["uses_distractor"] = [any(names[j] in DISTRACTORS for j in parse_antecedent(r)) for _, r in rules.iterrows()]
    rules.insert(0, "method", method)
    rules.insert(0, "seed", seed)
    used = P.rulebook_concepts_by_class(rb.rules, names, class_names)
    recovery = P.recovery_from_cuts(cuts_phys, used, names, ranges)
    cut_rows = [{"seed": seed, "method": method, "concept": names[j], "cut_minmax": t, "cut_physical": P.to_physical(j, t, tr.scaler_B)}
                for j, ts in rb.thresholds.items() for t in ts]
    summary = {"rule_count": int(len(rules)),
               "mean_conditions": float(rules["antecedent_length"].mean()) if len(rules) else float("nan"),
               "rules_using_distractor": int(rules["uses_distractor"].sum()),
               "distractor_cuts": int(sum(len(rb.thresholds.get(j, [])) for j, n in enumerate(names) if n in DISTRACTORS)),
               **{f"source_{s}": int((rules["source"] == s).sum()) for s in ("strict_lower", "boundary_high_confidence", "semantic_prototype")}}
    return rules, [dict(seed=seed, method=method, **r) for r in recovery], cut_rows, summary


def cart_report(tree, tr, names, class_names, seed: int):
    paths = P.cart_paths(tree)
    cuts_phys: Dict[int, List[float]] = {}
    for p in paths:
        for f, _, thr in p["conditions"]:
            cuts_phys.setdefault(f, set()).add(round(P.to_physical(f, thr, tr.scaler_B), 6))
    cuts_phys = {j: sorted(v) for j, v in cuts_phys.items()}
    ranges = {j: float(tr.scaler_B.data_range_[j]) for j in range(len(names))}
    used = P.cart_concepts_by_class(paths, names, class_names)
    recovery = P.recovery_from_cuts(cuts_phys, used, names, ranges)
    lengths = [len(p["conditions"]) for p in paths]
    summary = {"rule_count": int(len(paths)), "mean_conditions": float(np.mean(lengths)),
               "rules_using_distractor": int(sum(any(names[f] in DISTRACTORS for f, _, _ in p["conditions"]) for p in paths)),
               "distractor_cuts": int(sum(len(v) for j, v in cuts_phys.items() if names[j] in DISTRACTORS)),
               "max_depth": int(tree.get_depth())}
    cut_rows = [{"seed": seed, "method": "CART", "concept": names[j], "cut_minmax": float("nan"), "cut_physical": c}
                for j, cs in cuts_phys.items() for c in cs]
    return [dict(seed=seed, method="CART", **r) for r in recovery], cut_rows, summary


def run_seed(data, seed: int, stage: str) -> Dict[str, Any]:
    t0 = time.time()
    names, classes = data.concept_names, data.class_names
    train, val, test = P.split(data.y, seed)
    evalset = val if stage == "validation" else test
    y_tr, y_ev = data.y[train], data.y[evalset]

    mlp = P.fit_mlp(data.X[train], y_tr, data.X[val], data.y[val], seed)
    A_tr, A_val, A_ev = (P.embed(mlp, data.X[i]) for i in (train, val, evalset))
    Xs_ev = mlp.scaler.transform(data.X[evalset])
    mlp_pred = mlp.model.predict(Xs_ev).astype(int)
    mlp_proba = mlp.model.predict_proba(Xs_ev)

    # In the validation stage the "test" arguments are the validation rows, so test rows are never touched.
    tr = P.fit_transition(A_tr, A_val, A_ev, data.B[train], data.B[val], data.B[evalset], n_components=P.N_COMPONENTS, seed=seed)

    # name -> (fitted rulebook, inference outputs on the evaluation rows)
    books = {}
    for name, fit, apply, Ytr, Yev in [
        (CR_PRED, P.fit_mode_rulebook, P.apply_mode_rulebook, tr.Yhat_train, tr.Yhat_test),
        (CR_TRUE, P.fit_mode_rulebook, P.apply_mode_rulebook, tr.Y_train, tr.Y_test),
        (MAKE_PRED, P.fit_rulebook, P.apply_rulebook, tr.Yhat_train, tr.Yhat_test),
        (MAKE_TRUE, P.fit_rulebook, P.apply_rulebook, tr.Y_train, tr.Y_test),
    ]:
        rb = fit(Ytr, y_tr, names, classes)
        books[name] = (rb, apply(rb, Yev))
    s_min = books[CR_PRED][0].min_support
    cart, cart_sel = P.fit_cart(tr.Yhat_train, y_tr, tr.Yhat_val, data.y[val], seed, min_leaf=s_min)
    cart_pred = cart.predict(tr.Yhat_test).astype(int)

    metrics = [{"seed": seed, "method": "MLP", **P.predictive_metrics(y_ev, mlp_pred, None, classes)}]
    rules_all, recovery, cuts = [], [], []
    for name, (rb, out) in books.items():
        m = {"seed": seed, "method": name, **P.predictive_metrics(y_ev, out["prediction"].to_numpy(int), mlp_pred, classes),
             "conflict_rate": float(out["conflict"].mean()), "fallback_rate": float((out["mode"] == "fallback").mean()),
             "far_fallback_abstention_rate": float((out["abstain_reason"] == "far_fallback").mean())}
        rules, rec, cut_rows, summary = rulebook_report(rb, tr, names, classes, name, seed)
        metrics.append({**m, **summary})
        rules_all.append(rules)
        recovery += rec
        cuts += cut_rows if name in (CR_PRED, CR_TRUE) else []   # WEDD cuts are shared by both rulebooks
    rec_c, cuts_c, sum_c = cart_report(cart, tr, names, classes, seed)
    metrics.append({"seed": seed, "method": CART_PRED, **P.predictive_metrics(y_ev, cart_pred, mlp_pred, classes), **sum_c})
    recovery += rec_c
    cuts += cuts_c

    # Triage: does rule evidence flag the MLP's errors? Primary: class-reduct rulebook; secondary: MAKE rulebook.
    error = (mlp_pred != y_ev).astype(int)
    mlp_doubt = 1.0 - mlp_proba.max(axis=1)
    cr_doubt = P.rule_doubt(books[CR_PRED][1], mlp_pred, default_class=0)
    make_doubt = P.rule_doubt(books[MAKE_PRED][1], mlp_pred)
    scores = {"mlp_max_prob": mlp_doubt,
              "class_reduct_disagreement": cr_doubt,
              "class_reduct_hybrid": 0.5 * P.rank01(mlp_doubt) + 0.5 * P.rank01(cr_doubt),
              "make_disagreement": make_doubt,
              "make_hybrid": 0.5 * P.rank01(mlp_doubt) + 0.5 * P.rank01(make_doubt)}
    confident_errors = (error == 1) & (mlp_proba.max(axis=1) >= CONFIDENT)
    triage, curves = [], []
    for name, s in scores.items():
        row = {"seed": seed, "score": name, "mlp_errors": int(error.sum()), "n": int(len(error)), **P.triage_metrics(error, s),
               "confident_errors": int(confident_errors.sum()),
               "confident_errors_flagged": int((confident_errors & (s >= 0.5)).sum()) if name.endswith("disagreement") else np.nan}
        if name.endswith("disagreement"):
            row.update({f"bm_{k}": v for k, v in P.budget_matched(error, s, mlp_doubt).items()})
        triage.append(row)
        cov, risk = P.risk_coverage(error, s)
        keep = np.unique(np.linspace(0, len(cov) - 1, 200).astype(int))
        curves += [{"seed": seed, "score": name, "coverage": float(cov[k]), "risk": float(risk[k])} for k in keep]

    cr_out = books[CR_PRED][1]
    objects = pd.DataFrame({"seed": seed, "udi": data.udi[evalset], "y": y_ev, "mlp_pred": mlp_pred,
                            "mlp_max_prob": mlp_proba.max(axis=1), "cr_pred": cr_out["prediction"],
                            "cr_activated_rules": cr_out["activated_rules"], "make_pred": books[MAKE_PRED][1]["prediction"],
                            "cart_pred": cart_pred, **{f"doubt_{k}": v for k, v in scores.items()}})
    reducts = {name: rb.rules.groupby("consequent_name")["reduct"].first().to_dict() if len(rb.rules) else {}
               for name, (rb, _) in books.items() if name in (CR_PRED, CR_TRUE)}
    selection = {"seed": seed, "mlp_alpha": mlp.alpha, "mlp_n_iter": mlp.n_iter, "mlp_grid": mlp.selection,
                 "cart_max_depth": int(cart.get_depth()), "cart_grid": cart_sel, "rule_min_support": s_min,
                 "class_reducts": reducts, "transition_n_components": int(tr.n_components),
                 "transition_alpha": float(tr.alpha), "transition_explained_variance": float(tr.explained_variance),
                 "n_train": int(len(train)), "n_val": int(len(val)), "n_eval": int(len(evalset)),
                 "seconds": round(time.time() - t0, 1)}
    return {"metrics": metrics, "concepts": concept_metrics(tr, names, seed), "recovery": recovery, "cuts": cuts,
            "rules": pd.concat(rules_all, ignore_index=True), "triage": triage, "curves": curves,
            "objects": objects, "selection": selection}


def summarize(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    num = [c for c in df.columns if c not in by + ["seed"] and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(by, sort=False)[num]
    mean, sd = g.mean(), g.std(ddof=1)
    out = mean.copy()
    for c in num:
        out[c + "_sd"] = sd[c]
    return out.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", choices=["validation", "test"], required=True)
    ap.add_argument("--seeds", type=int, nargs="*", default=P.SEEDS)
    args = ap.parse_args()
    if args.stage == "test":
        protocol = ROOT / "PROTOCOL.md"
        if not protocol.exists() or "Status: FROZEN" not in protocol.read_text(encoding="utf-8"):
            sys.exit("Refusing to touch test rows: PROTOCOL.md is missing or not frozen.")

    data = load_ai4i(args.csv)
    out = Path(args.out) / args.stage
    out.mkdir(parents=True, exist_ok=True)
    print(f"rows kept {len(data.y)}, dropped {data.dropped}, class counts {np.bincount(data.y).tolist()}", flush=True)

    acc: Dict[str, list] = {k: [] for k in ["metrics", "concepts", "recovery", "cuts", "rules", "triage", "curves", "objects", "selection"]}
    for seed in args.seeds:
        res = run_seed(data, seed, args.stage)
        for k, v in res.items():
            acc[k].append(v) if k in ("rules", "objects", "selection") else acc[k].extend(v)
        m = {r["method"]: r for r in res["metrics"]}
        print(f"seed {seed}: macro-F1 MLP {m['MLP']['macro_f1_abstain_as_wrong']:.3f} | "
              f"CR pred {m[CR_PRED]['macro_f1_abstain_as_wrong']:.3f} true {m[CR_TRUE]['macro_f1_abstain_as_wrong']:.3f} | "
              f"MAKE pred {m[MAKE_PRED]['macro_f1_abstain_as_wrong']:.3f} | CART {m[CART_PRED]['macro_f1_abstain_as_wrong']:.3f} | "
              f"reducts {res['selection']['class_reducts']} | {res['selection']['seconds']} s", flush=True)

    metrics = pd.DataFrame(acc["metrics"])
    concepts = pd.DataFrame(acc["concepts"])
    recovery = pd.DataFrame(acc["recovery"])
    triage = pd.DataFrame(acc["triage"])
    metrics.to_csv(out / "metrics_per_seed.csv", index=False)
    concepts.to_csv(out / "concepts_per_seed.csv", index=False)
    recovery.to_csv(out / "recovery_per_seed.csv", index=False)
    pd.DataFrame(acc["cuts"]).to_csv(out / "cuts_per_seed.csv", index=False)
    pd.concat(acc["rules"], ignore_index=True).drop(columns=["covered_indices_sample"]).to_csv(out / "rules_per_seed.csv", index=False)
    triage.to_csv(out / "triage_per_seed.csv", index=False)
    pd.DataFrame(acc["curves"]).to_csv(out / "risk_coverage_curves.csv", index=False)
    pd.concat(acc["objects"], ignore_index=True).to_csv(out / "objects.csv", index=False)
    (out / "selection.json").write_text(json.dumps(acc["selection"], indent=2), encoding="utf-8")

    summarize(metrics, ["method"]).to_csv(out / "metrics_summary.csv", index=False)
    summarize(concepts.drop(columns=["distractor"]), ["concept"]).to_csv(out / "concepts_summary.csv", index=False)
    summarize(recovery.drop(columns=["concepts_used"]), ["method", "mode", "concept", "documented"]).to_csv(out / "recovery_summary.csv", index=False)
    summarize(triage, ["score"]).to_csv(out / "triage_summary.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"stage": args.stage, "seeds": args.seeds, "rows_kept": int(len(data.y)),
                                                    "dropped": data.dropped, "class_counts": np.bincount(data.y).tolist(),
                                                    "class_names": data.class_names, "concepts": data.concept_names},
                                                   indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
