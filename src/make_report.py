"""Build table data (T1-T4) and data plots (F2-F4) from the frozen test-stage outputs.

    python -I src/make_report.py --csv data/raw/ai4i2020.csv --run outputs/test --out outputs
    python -I src/make_report.py --csv data/raw/ai4i2020.csv --run outputs/test --out outputs --figures-only

Reads only the frozen run directory (and the raw CSV for the scatter plots); it never trains or re-runs a stage.
The figures use the paper style of figure_style.py (15.8 cm wide, Libertinus Sans 7.5 pt, PDF + SVG + PNG).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P  # noqa: E402
from ai4i_data import CLASS_NAMES, load_ai4i  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

from figure_style import (PALETTE, add_panel_label, apply_paper_style, paper_figure, save_figure,  # noqa: E402
                          style_axis, thousands_formatter)

CR_PRED, CR_TRUE = "Class-reduct rulebook (predicted concepts)", "Class-reduct rulebook (true concepts)"
MAKE_PRED, MAKE_TRUE = "MAKE rulebook (predicted concepts)", "MAKE rulebook (true concepts)"
CART_PRED = "CART (predicted concepts)"

CONCEPT_INFO = {
    "delta_t_K": ("ΔT = process − air temperature", "K", "rule (HDF)"),
    "speed_rpm": ("Rotational speed", "rpm", "rule (HDF)"),
    "torque_Nm": ("Torque", "Nm", "physical, in no rule by itself"),
    "power_W": ("Power = torque · speed · 2π/60", "W", "rule (PWF)"),
    "tool_wear_min": ("Tool wear", "min", "rule (TWF, random)"),
    "strain_minNm": ("Strain = tool wear · torque", "minNm", "rule (OSF)"),
    "product_type": ("Product type L/M/H coded 0/1/2", "-", "rule (OSF)"),
    "air_temp_K": ("Air temperature", "K", "distractor"),
    "process_temp_K": ("Process temperature", "K", "distractor"),
    "noise_sensor": ("Noise sensor, N(50, 10²)", "-", "distractor"),
}
SHORT = {"delta_t_K": "ΔT", "speed_rpm": "Speed", "torque_Nm": "Torque", "power_W": "Power", "tool_wear_min": "Tool wear",
         "strain_minNm": "Strain", "product_type": "Product type", "air_temp_K": "Air temp.", "process_temp_K": "Process temp.",
         "noise_sensor": "Noise sensor"}
# Concept names on figure axes (tables keep SHORT); spelled out because the figures are full width.
FIG_LABEL = {**SHORT, "air_temp_K": "Air temperature", "process_temp_K": "Process temperature"}
# Figure heights in inches at the full text width (15.8 cm).
H_THRESHOLDS, H_TRIAGE, H_FIDELITY = 2.45, 2.45, 2.15


def pm(mean: float, sd: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "-"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if not pd.isna(sd) else f"{mean:.{digits}f}"


def write_table(df: pd.DataFrame, out: Path, name: str) -> None:
    df.to_csv(out / f"{name}.csv", index=False)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_concepts(run: Path, out: Path) -> None:
    c = pd.read_csv(run / "concepts_summary.csv").set_index("concept")
    rows = []
    for name, (definition, unit, role) in CONCEPT_INFO.items():
        r = c.loc[name]
        rows.append({"Concept": SHORT[name], "Definition": definition, "Unit": unit, "Role": role,
                     "Test R²": pm(r["r2"], r["r2_sd"]), "Test MAE (unit)": pm(r["mae_physical"], r["mae_physical_sd"], 3)})
    write_table(pd.DataFrame(rows), out, "T1_concepts")


def table_recovery(run: Path, out: Path) -> None:
    rec = pd.read_csv(run / "recovery_per_seed.csv")
    rows = []
    for mode, spec in P.DOCUMENTED_RULES.items():
        sub = rec[rec["mode"] == mode]
        jac = sub.groupby(["method", "seed"])["concept_jaccard"].first()
        row = {"Mode": mode, "Documented rule": spec["text"]}
        for label, method in [("Class-reduct, true concepts", CR_TRUE), ("Class-reduct, predicted concepts", CR_PRED),
                              ("MAKE rulebook, predicted concepts", MAKE_PRED), ("CART, predicted concepts", "CART")]:
            j = jac.loc[method]
            row[f"{label}: exact concept set (seeds)"] = f"{int((j == 1.0).sum())}/5"
            row[f"{label}: mean Jaccard"] = f"{j.mean():.2f}"
        rows.append(row)
    write_table(pd.DataFrame(rows), out, "T2a_concept_recovery")

    rows = []
    for (mode, concept, theta), sub in rec.groupby(["mode", "concept", "documented"], sort=False):
        row = {"Mode": mode, "Concept": SHORT[concept], "Documented threshold": f"{theta:g}"}
        for label, method in [("WEDD cut, true concepts", CR_TRUE), ("WEDD cut, predicted concepts", CR_PRED), ("CART cut", "CART")]:
            s = sub[sub["method"] == method]
            row[label] = pm(s["nearest_cut"].mean(), s["nearest_cut"].std(ddof=1), 1)
            row[label + " |error|"] = pm(s["abs_error"].mean(), s["abs_error"].std(ddof=1), 1)
        rows.append(row)
    write_table(pd.DataFrame(rows), out, "T2b_threshold_recovery")


def table_methods(run: Path, out: Path) -> None:
    m = pd.read_csv(run / "metrics_summary.csv").set_index("method")
    order = ["MLP", CR_PRED, CR_TRUE, MAKE_PRED, MAKE_TRUE, CART_PRED]
    rows = []
    for method in order:
        r = m.loc[method]
        rows.append({"Method": method, "Macro-F1": pm(r["macro_f1_abstain_as_wrong"], r["macro_f1_abstain_as_wrong_sd"]),
                     "Mean failure-mode recall": pm(r["failure_mode_recall_mean"], r["failure_mode_recall_mean_sd"]),
                     **{f"Recall {c}": pm(r[f"recall_{c}"], r[f"recall_{c}_sd"], 2) for c in CLASS_NAMES[1:]},
                     "Coverage": pm(r["coverage"], r["coverage_sd"]),
                     "Covered fidelity to MLP": pm(r.get("covered_fidelity"), r.get("covered_fidelity_sd")),
                     "Rules": pm(r.get("rule_count"), r.get("rule_count_sd"), 1),
                     "Conditions per rule": pm(r.get("mean_conditions"), r.get("mean_conditions_sd"), 2),
                     "Rules using distractors": pm(r.get("rules_using_distractor"), r.get("rules_using_distractor_sd"), 1)})
    write_table(pd.DataFrame(rows), out, "T3_method_comparison")


def table_triage(run: Path, out: Path) -> None:
    t = pd.read_csv(run / "triage_summary.csv").set_index("score")
    labels = {"mlp_max_prob": "MLP max-probability", "class_reduct_disagreement": "Class-reduct rule disagreement",
              "class_reduct_hybrid": "Hybrid (MLP + class-reduct)", "make_disagreement": "MAKE rule disagreement",
              "make_hybrid": "Hybrid (MLP + MAKE)"}
    rows = []
    for key, label in labels.items():
        r = t.loc[key]
        row = {"Score": label, "AUROC": pm(r["auroc"], r["auroc_sd"]), "AURC (×10³)": pm(r["aurc"] * 1e3, r["aurc_sd"] * 1e3, 2),
               "Error recall @5%": pm(r["error_recall_at_5pct"], r["error_recall_at_5pct_sd"], 2),
               "Error recall @10%": pm(r["error_recall_at_10pct"], r["error_recall_at_10pct_sd"], 2)}
        for col, name in [("bm_flagged_rows", "Rows flagged"), ("bm_caught_by_rule_flags", "Errors caught by rule flags"),
                          ("bm_caught_by_mlp_same_budget", "Errors caught by max-probability, same budget"),
                          ("bm_caught_by_rule_only", "Caught by rules only"), ("bm_caught_by_either", "Caught by either")]:
            row[name] = pm(r.get(col), r.get(col + "_sd"), 1)
        rows.append(row)
    write_table(pd.DataFrame(rows), out, "T4_triage")
    total = t.loc["mlp_max_prob"]
    (out / "T4_triage_note.md").write_text(
        f"MLP errors per seed: {pm(total['mlp_errors'], total['mlp_errors_sd'], 1)} of {int(total['n'])} test rows; "
        f"confident errors (max-probability ≥ 0.90): {pm(total['confident_errors'], total['confident_errors_sd'], 1)}; "
        f"flagged by class-reduct rules: {pm(t.loc['class_reduct_disagreement', 'confident_errors_flagged'], t.loc['class_reduct_disagreement', 'confident_errors_flagged_sd'], 1)}.\n",
        encoding="utf-8")


def nearest_cuts(cuts: pd.DataFrame, method: str, concept: str, theta: float) -> list[float]:
    vals = []
    for _, s in cuts[(cuts["method"] == method) & (cuts["concept"] == concept)].groupby("seed"):
        c = s["cut_physical"].to_numpy()
        vals.append(float(c[np.argmin(np.abs(c - theta))]))
    return vals


def figure_thresholds(data, run: Path, out: Path) -> None:
    cuts = pd.read_csv(run / "cuts_per_seed.csv")
    B = pd.DataFrame(data.B, columns=data.concept_names)
    y = data.y
    fig, axes = paper_figure(H_THRESHOLDS, 3)
    # (method, color, legend label, line width). True-concept cuts are drawn wider, so they stay visible where a
    # predicted-concept cut lies on top of them.
    lines = [(CR_TRUE, PALETTE["blue"], "WEDD cut, true concepts", 1.0),
             (CR_PRED, PALETTE["orange"], "WEDD cut, predicted concepts", 0.6)]

    ax = axes[0]
    none = y == 0
    ax.scatter(B.loc[none, "speed_rpm"], B.loc[none, "delta_t_K"], s=1.2, c=PALETTE["light"], lw=0, rasterized=True, label="other rows")
    hdf = y == CLASS_NAMES.index("HDF")
    ax.scatter(B.loc[hdf, "speed_rpm"], B.loc[hdf, "delta_t_K"], s=4, c=PALETTE["vermillion"], lw=0, rasterized=True, label="HDF")
    ax.add_patch(plt.Rectangle((1150, 7.5), 1380 - 1150, 8.6 - 7.5, fc=PALETTE["pale_orange"], ls="--", lw=0.8,
                               ec=PALETTE["black"], zorder=0))
    for method, color, _, lw in lines:
        for v in nearest_cuts(cuts, method, "speed_rpm", 1380):
            ax.axvline(v, color=color, lw=lw, alpha=0.75)
        for v in nearest_cuts(cuts, method, "delta_t_K", 8.6):
            ax.axhline(v, color=color, lw=lw, alpha=0.75)
    ax.set(xlim=(1150, 1700), ylim=(7.5, 10.5), xlabel="Rotational speed (rpm)", ylabel="ΔT (K)")
    ax.xaxis.set_major_locator(MultipleLocator(100))

    ax = axes[1]
    bins = np.linspace(1000, 10600, 97)
    ax.hist(B.loc[none, "power_W"], bins=bins, color=PALETTE["light"], label="other rows")
    ax.hist(B.loc[y == CLASS_NAMES.index("PWF"), "power_W"], bins=bins, color=PALETTE["vermillion"], label="PWF")
    for theta in (3500, 9000):
        ax.axvline(theta, color=PALETTE["black"], ls="--", lw=0.8)
        for method, color, _, lw in lines:
            for v in nearest_cuts(cuts, method, "power_W", theta):
                ax.axvline(v, color=color, lw=lw, alpha=0.75)
    ax.set(yscale="log", xlabel="Power (W)", ylabel="Rows")
    ax.xaxis.set_major_locator(MultipleLocator(3000))
    ax.xaxis.set_major_formatter(thousands_formatter())
    ax.yaxis.set_major_formatter(thousands_formatter())

    ax = axes[2]
    rng = np.random.default_rng(0)
    jitter = data.B[:, data.concept_names.index("product_type")] + rng.uniform(-0.25, 0.25, len(y))
    osf = y == CLASS_NAMES.index("OSF")
    ax.scatter(B.loc[~osf, "strain_minNm"], jitter[~osf], s=1.2, c=PALETTE["light"], lw=0, rasterized=True)
    ax.scatter(B.loc[osf, "strain_minNm"], jitter[osf], s=4, c=PALETTE["vermillion"], lw=0, rasterized=True, label="OSF")
    for k, theta in enumerate((11000, 12000, 13000)):
        ax.plot([theta, theta], [k - 0.4, k + 0.4], color=PALETTE["black"], ls="--", lw=0.8)
    for method, color, _, lw in lines:
        for v in nearest_cuts(cuts, method, "strain_minNm", 11000):
            ax.axvline(v, color=color, lw=lw, alpha=0.75)
    ax.set(xlim=(6000, 16500), yticks=[0, 1, 2], yticklabels=["L", "M", "H"],
           xlabel="Strain = tool wear × torque (min·Nm)", ylabel="Product type")
    ax.xaxis.set_major_locator(MultipleLocator(3000))
    ax.xaxis.set_major_formatter(thousands_formatter())
    for a, lab in zip(axes, "abc"):
        style_axis(a)
        add_panel_label(a, lab)
    handles = [plt.Line2D([], [], color=PALETTE["black"], ls="--", lw=0.8, label="documented rule")]
    handles += [plt.Line2D([], [], color=c, lw=1.0, label=l) for _, c, l, _ in lines]
    handles += [plt.Line2D([], [], ls="", marker="o", ms=3, mew=0, color=PALETTE["vermillion"], label="HDF, PWF, OSF rows"),
                plt.Line2D([], [], ls="", marker="o", ms=3, mew=0, color=PALETTE["light"], label="other rows")]
    fig.legend(handles=handles, loc="outside upper center", ncol=5)
    save_figure(fig, out / "F2_thresholds_vs_physics")


def figure_triage(run: Path, out: Path) -> None:
    curves = pd.read_csv(run / "risk_coverage_curves.csv")
    trg = pd.read_csv(run / "triage_per_seed.csv")
    fig, axes = paper_figure(H_TRIAGE, 2, gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    # Continuous scores as curves; the near-binary rule flags as operating points (their curves only reflect tie order).
    for score, color, ls, label in [("mlp_max_prob", PALETTE["black"], "-", "MLP max-probability"),
                                    ("class_reduct_hybrid", PALETTE["teal"], "--", "Hybrid (MLP + class-reduct)")]:
        c = curves[curves["score"] == score].groupby("coverage")["risk"].mean()
        ax.plot(c.index, c.to_numpy() * 100, color=color, ls=ls, lw=1.0, label=label)
    for score, color, marker, label in [("class_reduct_disagreement", PALETTE["blue"], "o", "Class-reduct rule flags"),
                                        ("make_disagreement", PALETTE["purple"], "s", "As-published rule flags")]:
        s = trg[trg["score"] == score]
        cov = 1 - s["bm_flagged_rows"] / s["n"]
        risk = 100 * (s["mlp_errors"] - s["bm_caught_by_rule_flags"]) / (s["n"] - s["bm_flagged_rows"])
        ax.errorbar(cov.mean(), risk.mean(), xerr=cov.std(ddof=1), yerr=risk.std(ddof=1), fmt=marker, ms=4, color=color,
                    lw=0.7, capsize=1.5, label=label)
    ax.set(xlim=(0.9, 1.0), xlabel="Coverage (share of MLP decisions accepted)", ylabel="MLP error rate on accepted rows (%)")
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.legend(loc="upper left")
    style_axis(ax, "y")
    add_panel_label(ax, "a")

    ax = axes[1]
    s = trg[trg["score"] == "class_reduct_disagreement"].sort_values("seed")
    x = np.arange(len(s))
    w = 0.26
    ax.bar(x - w, s["bm_caught_by_rule_flags"], w, color=PALETTE["blue"], label="Class-reduct rule flags")
    ax.bar(x, s["bm_caught_by_mlp_same_budget"], w, color=PALETTE["mid"], label="Max-probability, same budget")
    # White hatching keeps "either" apart from the blue bars in greyscale print.
    ax.bar(x + w, s["bm_caught_by_either"], w, color=PALETTE["teal"], label="Either", hatch="////", edgecolor="white", lw=0)
    ax.scatter(x, s["mlp_errors"], marker="_", s=120, color=PALETTE["black"], label="All MLP errors", zorder=3)
    ax.set(xticks=x, xticklabels=[str(v) for v in s["seed"]], ylim=(0, 72), xlabel="Seed", ylabel="MLP errors caught")
    ax.legend(loc="upper center", ncol=2, handlelength=1.1, columnspacing=0.8, markerscale=0.75)
    style_axis(ax, "y")
    add_panel_label(ax, "b")
    save_figure(fig, out / "F3_triage")


def figure_fidelity(run: Path, out: Path) -> None:
    c = pd.read_csv(run / "concepts_summary.csv").set_index("concept").loc[list(CONCEPT_INFO)]
    fig, ax = paper_figure(H_FIDELITY)
    yy = np.arange(len(c))[::-1]
    # Distractors in mid grey (not dark grey), so that they differ from the blue bars when printed in greyscale.
    colors = [PALETTE["mid"] if CONCEPT_INFO[n][2] == "distractor" else PALETTE["blue"] for n in c.index]
    ax.barh(yy, c["r2"].clip(lower=0), xerr=c["r2_sd"], color=colors, height=0.65, error_kw={"lw": 0.6, "capsize": 1.5})
    for yv, r2 in zip(yy, c["r2"]):
        ax.text(max(r2, 0) + 0.02, yv, f"{r2:.3f}".replace("-", "−"), va="center", fontsize=6.5)
    ax.set(yticks=yy, yticklabels=[FIG_LABEL[n] for n in c.index], xlim=(0, 1.15),
           xlabel="Test R² of the transition (mean ± SD over five seeds)")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PALETTE["blue"], label="Physical concept"),
                       plt.Rectangle((0, 0), 1, 1, color=PALETTE["mid"], label="Distractor")],
              loc="lower right", ncol=2)
    style_axis(ax, "x")
    save_figure(fig, out / "F4_concept_fidelity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures-only", action="store_true", help="re-plot the figures; leave the tables untouched")
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    tables, figures = out / "tables", out / "figures"
    apply_paper_style()
    if not args.figures_only:
        tables.mkdir(parents=True, exist_ok=True)
        table_concepts(run, tables)
        table_recovery(run, tables)
        table_methods(run, tables)
        table_triage(run, tables)
    data = load_ai4i(args.csv)
    figure_thresholds(data, run, figures)
    figure_triage(run, figures)
    figure_fidelity(run, figures)
    print(f"wrote {figures}" if args.figures_only else f"wrote {tables} and {figures}")


if __name__ == "__main__":
    main()
