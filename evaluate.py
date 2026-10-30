"""Evaluate the trained selectors on the held-out TEST programs.

For every rule budget it compares, on programs never used for training:

  none            no optimisation
  textbook        the rules in their natural order (CF, AS, SR, CP, ...), first B of them
  random          a random pipeline of B rules (mean of 20)
  best fixed      the single best pipeline, chosen on the training programs
  ML-selected     the pipeline predicted for each program by the trained model
  oracle          the best pipeline found by exhaustive search for each program
  worst           the worst pipeline for each program
and, for reference, the "repeat the textbook order until nothing changes" baseline.

It also re-runs every ML-chosen pipeline and checks with the interpreter that
the optimised program behaves exactly like the original.

Run:  python evaluate.py --budgets 1 2 3 4 5 6 7 8
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from cost import cost
from engine import apply_ordering, clear_caches
from generator import generate
from interpreter import equivalent
from tac import to_text
from train import predict_choice

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")

# validated reference palette (light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def bootstrap_ci(diff, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def pipeline_text(o, names):
    return " > ".join(names[r] for r in o)


def evaluate_budget(b, check_equivalence=True, write_examples=False):
    d = np.load(os.path.join(DATA, f"dataset_b{b}.npz"))
    with open(os.path.join(DATA, f"portfolio_b{b}.json")) as f:
        meta = json.load(f)
    bundle = joblib.load(os.path.join(MODELS, f"best_model_b{b}_full.joblib"))
    names = meta["rule_names"]
    X, C = d["Xte"][:, : bundle["n_features"]], d["Cte"]
    choice = predict_choice(bundle["kind"], bundle["model"], X, C.shape[1])
    ml = C[np.arange(len(C)), choice]
    static, oracle, base = d["static_te"], d["oracle_te"], d["bte"]
    n = len(base)

    diff = static - ml                         # positive = ML better than the fixed pipeline
    lo, hi = bootstrap_ci(diff)
    headroom = static.mean() - oracle.mean()
    out = {
        "budget": b,
        "model": bundle["name"],
        "n_test": int(n),
        "mean_cost": {
            "none": float(base.mean()),
            "textbook": float(d["textbook_te"].mean()),
            "random": float(d["random_te"].mean()),
            "best_fixed": float(static.mean()),
            "ml": float(ml.mean()),
            "portfolio_oracle": float(C.min(axis=1).mean()),
            "oracle": float(oracle.mean()),
            "worst": float(d["worst_te"].mean()),
            "fixpoint_textbook": float(d["fixpoint_te"].mean()),
        },
        "fixpoint_mean_rule_executions": float(d["fixpoint_effort_te"].mean()),
        "reduction_pct": {k: float(100 * (1 - v / base.mean()))
                          for k, v in {"best_fixed": static.mean(), "ml": ml.mean(), "oracle": oracle.mean(),
                                       "textbook": d["textbook_te"].mean()}.items()},
        "ml_vs_fixed": {
            "mean_gain": float(diff.mean()), "ci95": [lo, hi],
            "better": int((diff > 0).sum()), "same": int((diff == 0).sum()), "worse": int((diff < 0).sum()),
        },
        "headroom_fixed_vs_oracle": float(headroom),
        "headroom_recovered_pct": float(100 * diff.mean() / headroom) if headroom > 1e-9 else None,
        "ml_optimal_rate": float((ml == oracle).mean()),
        "fixed_optimal_rate": float((static == oracle).mean()),
        "static_pipeline": pipeline_text(meta["static_pipeline"], names),
    }

    seeds = d["seeds_te"]
    if check_equivalence:
        bad = []
        for i, s in enumerate(seeds):
            code = generate(int(s))
            clear_caches()
            opt = apply_ordering(code, meta["portfolio"][int(choice[i])])
            if cost(opt) != ml[i] or not equivalent(code, opt, trials=6, seed=int(s)):
                bad.append(int(s))
        out["equivalence_failures"] = len(bad)
        out["equivalence_checked"] = int(n)

    if write_examples:
        gap = static - oracle
        order = np.argsort(-gap)[:4]
        lines = [f"Phase-ordering examples, budget {b}: programs where the best fixed pipeline is "
                 f"furthest from the per-program optimum.\n"
                 f"Best fixed pipeline: {out['static_pipeline']}\n"]
        for i in order:
            code = generate(int(seeds[i]))
            clear_caches()
            best = tuple(meta["oracle_orders_te"][i])
            fixed = tuple(meta["static_pipeline"])
            lines.append("=" * 70)
            lines.append(f"test seed {int(seeds[i])}: unoptimised cost {int(base[i])}")
            lines.append(f"  fixed pipeline  [{pipeline_text(fixed, names)}] -> cost {int(static[i])}")
            lines.append(f"  best pipeline   [{pipeline_text(best, names)}] -> cost {int(oracle[i])}")
            lines.append(f"  ML pipeline     [{pipeline_text(meta['portfolio'][int(choice[i])], names)}] -> cost {int(ml[i])}")
            lines.append("\n--- original ---\n" + to_text(code))
            lines.append("--- after fixed pipeline ---\n" + to_text(apply_ordering(code, fixed)))
            lines.append("--- after best pipeline ---\n" + to_text(apply_ordering(code, best)))
        with open(os.path.join(RESULTS, f"phase_ordering_examples_b{b}.txt"), "w") as f:
            f.write("\n".join(lines))
    return out


# ------------------------------------------------------------------ plots
def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_budget_curve(all_results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["budget"] for r in all_results]
    # draw the oracle first and a bit larger, so the ML line stays visible where they coincide
    series = [
        ("Textbook order", "textbook", YELLOW, 6),
        ("Best fixed pipeline", "best_fixed", ORANGE, 6),
        ("Exhaustive oracle", "oracle", AQUA, 10),
        ("ML-selected pipeline", "ml", BLUE, 6),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    for label, key, col, ms in series:
        ys = [r["mean_cost"][key] for r in all_results]
        ax.plot(bs, ys, color=col, linewidth=2, marker="o", markersize=ms, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
    fx = all_results[-1]["mean_cost"]["fixpoint_textbook"]
    ex = all_results[-1]["fixpoint_mean_rule_executions"]
    ax.axhline(fx, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(42, 108)
    ax.text(1.0, fx - 0.9, f"Textbook order repeated to a fixpoint: {fx:.1f}, using about {ex:.0f} rule executions",
            color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("Rule budget (rules a pipeline may run)", color=INK2, fontsize=10)
    ax.set_ylabel("Average cost of optimised program (lower is better)", color=INK2, fontsize=10)
    ax.set_xticks(bs)
    ax.set_title("A learned selector helps most when the rule budget is small", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_headroom(all_results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bs = [r["budget"] for r in all_results]
    head = [r["headroom_fixed_vs_oracle"] for r in all_results]
    gain = [r["ml_vs_fixed"]["mean_gain"] for r in all_results]
    w = 0.38
    xs = np.arange(len(bs))
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.bar(xs - w / 2 - 0.01, head, w - 0.02, color=AQUA, label="Available: fixed pipeline vs oracle")
    ax.bar(xs + w / 2 + 0.01, gain, w - 0.02, color=BLUE, label="Achieved: ML vs fixed pipeline")
    ax.set_xticks(xs)
    ax.set_xticklabels(bs)
    ax.set_xlabel("Rule budget", color=INK2, fontsize=10)
    ax.set_ylabel("Cost units saved per program", color=INK2, fontsize=10)
    ax.set_title("How much of the available improvement the model captures", loc="left",
                 color=INK, fontsize=12, fontweight="bold")
    for x, v in zip(xs + w / 2 + 0.01, gain):
        ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    for x, v in zip(xs - w / 2 - 0.01, head):
        ax.text(x, v + 0.12, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_models(b, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(RESULTS, f"model_comparison_b{b}_full.json")) as f:
        comp = json.load(f)
    rows = sorted(((k, v["mean_cost"]) for k, v in comp["cv"].items()), key=lambda kv: kv[1])
    labels = [k for k, _ in rows][::-1]
    vals = [v for _, v in rows][::-1]
    fixed = comp["cv"]["Best fixed pipeline"]["mean_cost"]
    best = comp["portfolio_best_mean_cost"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    colours = [ORANGE if l == "Best fixed pipeline" else BLUE for l in labels]
    ax.barh(labels, vals, color=colours, height=0.62)
    lo = min(vals + [best]) - 1.0
    ax.set_xlim(lo, max(vals) + 0.9)
    ax.axvline(best, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(best + 0.05, len(labels) - 0.45, "best possible in portfolio", color=INK2, fontsize=8.5, va="top")
    for y, v in enumerate(vals):
        ax.text(v + 0.06, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Cross-validated average cost (lower is better)", color=INK2, fontsize=10)
    ax.set_title(f"Model comparison at rule budget {b}", loc="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_table(all_results, path):
    lines = [
        "| Budget | None | Textbook | Best fixed | ML-selected | Oracle | ML vs fixed (95% CI) | Headroom recovered | Behaviour check |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in all_results:
        m = r["mean_cost"]
        g = r["ml_vs_fixed"]
        rec = r["headroom_recovered_pct"]
        eq = r.get("equivalence_failures")
        lines.append(
            f"| {r['budget']} | {m['none']:.1f} | {m['textbook']:.1f} | {m['best_fixed']:.2f} | {m['ml']:.2f} | "
            f"{m['oracle']:.2f} | {g['mean_gain']:+.2f} [{g['ci95'][0]:+.2f}, {g['ci95'][1]:+.2f}] | "
            f"{'n/a' if rec is None else f'{rec:.0f}%'} | "
            f"{'not run' if eq is None else f'{eq} failures / {r['equivalence_checked']}'} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=list(range(1, 9)))
    ap.add_argument("--no-equivalence", action="store_true")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    all_results = []
    for b in args.budgets:
        r = evaluate_budget(b, check_equivalence=not args.no_equivalence, write_examples=b in (4, 8))
        all_results.append(r)
        m = r["mean_cost"]
        print(f"B={b} [{r['model']}] none {m['none']:.1f} | textbook {m['textbook']:.1f} | fixed {m['best_fixed']:.2f} | "
              f"ML {m['ml']:.2f} | oracle {m['oracle']:.2f} | ML vs fixed {r['ml_vs_fixed']['mean_gain']:+.2f} "
              f"CI [{r['ml_vs_fixed']['ci95'][0]:+.2f}, {r['ml_vs_fixed']['ci95'][1]:+.2f}] "
              f"(better {r['ml_vs_fixed']['better']}/same {r['ml_vs_fixed']['same']}/worse {r['ml_vs_fixed']['worse']}) "
              f"| equivalence failures {r.get('equivalence_failures', 'n/a')}")
    with open(os.path.join(RESULTS, "summary_all.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    write_table(all_results, os.path.join(RESULTS, "results_table.md"))
    if len(all_results) > 1:
        plot_budget_curve(all_results, os.path.join(RESULTS, "cost_vs_budget.png"))
        plot_headroom(all_results, os.path.join(RESULTS, "headroom_captured.png"))
    for b in (4, 8):
        if b in args.budgets and os.path.exists(os.path.join(RESULTS, f"model_comparison_b{b}_full.json")):
            plot_models(b, os.path.join(RESULTS, f"model_comparison_b{b}.png"))


if __name__ == "__main__":
    main()
