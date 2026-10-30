"""Train and compare models that choose a rule pipeline for a program.

Two ways to frame the choice (both are compared):

* classification - predict *which* portfolio pipeline is best for the program;
* regression     - predict the *regret* (extra cost compared with the best
                   pipeline) of every portfolio pipeline and pick the smallest.
                   Regression copes better with ties, because many pipelines
                   are often equally good.

Models are compared with 5-fold cross-validation on the TRAINING programs
only.  The test programs are not touched here.

Metric (lower is better): mean final cost of the chosen pipeline, and its
"regret" = extra cost compared with the best pipeline in the portfolio.

Run:  python train.py --budget 8
"""
from __future__ import annotations

import argparse
import json
import os
import time

import joblib
import numpy as np
from sklearn.ensemble import (HistGradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(HERE, "models")
SEED = 0


def model_zoo():
    """name -> (kind, factory)."""
    return {
        "DecisionTree (cls)": ("cls", lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=3, random_state=SEED)),
        "RandomForest (cls)": ("cls", lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=SEED)),
        "HistGradBoost (cls)": ("cls", lambda: HistGradientBoostingClassifier(max_iter=100, learning_rate=0.1, max_depth=5, random_state=SEED)),
        "LogisticReg (cls)": ("cls", lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))),
        "kNN (cls)": ("cls", lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15, weights="distance"))),
        "Ridge (reg)": ("reg", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "DecisionTree (reg)": ("reg", lambda: DecisionTreeRegressor(max_depth=10, min_samples_leaf=3, random_state=SEED)),
        "RandomForest (reg)": ("reg", lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=2, n_jobs=2, random_state=SEED)),
        "kNN (reg)": ("reg", lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance"))),
    }


# ------------------------------------------------------------ targets
def regret_matrix(C):
    return C - C.min(axis=1, keepdims=True)


def class_labels(C):
    """Best pipeline per program; ties go to the pipeline that is optimal most often."""
    opt = C == C.min(axis=1, keepdims=True)
    popularity = opt.sum(axis=0)
    score = np.where(opt, popularity[None, :] + 1e-3 * np.arange(C.shape[1])[::-1] / C.shape[1], -1)
    return score.argmax(axis=1)


def fit_model(kind, factory, X, C):
    m = factory()
    if kind == "cls":
        m.fit(X, class_labels(C))
    else:
        m.fit(X, regret_matrix(C))
    return m


def predict_choice(kind, model, X, n_pipelines):
    """Index of the chosen portfolio pipeline for every row of X."""
    if kind == "cls":
        return np.asarray(model.predict(X), dtype=int)
    return np.asarray(model.predict(X)).argmin(axis=1)


# ------------------------------------------------------------ scoring
def score(choice, C, base, static_cost):
    picked = C[np.arange(len(C)), choice]
    best = C.min(axis=1)
    return {
        "mean_cost": float(picked.mean()),
        "mean_regret": float((picked - best).mean()),
        "optimal_rate": float((picked == best).mean()),
        "reduction_pct": float(100 * (1 - picked.mean() / base.mean())),
        "better_than_static": float((picked < static_cost).mean()),
        "worse_than_static": float((picked > static_cost).mean()),
        "gain_vs_static": float((static_cost - picked).mean()),
    }


def cross_validate(kind, factory, X, C, base, static_cost, folds=5):
    kf = KFold(folds, shuffle=True, random_state=SEED)
    choice = np.zeros(len(X), dtype=int)
    for tr, va in kf.split(X):
        m = fit_model(kind, factory, X[tr], C[tr])
        choice[va] = predict_choice(kind, m, X[va], C.shape[1])
    return choice


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--features", choices=["full", "structural"], default="full")
    ap.add_argument("--quick", action="store_true", help="skip the slow gradient boosting model")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)

    d = np.load(os.path.join(DATA, f"dataset_b{args.budget}.npz"))
    with open(os.path.join(DATA, f"portfolio_b{args.budget}.json")) as f:
        meta = json.load(f)
    nf = meta["n_struct"] if args.features == "structural" else d["Xtr"].shape[1]
    X, C, base, static_cost = d["Xtr"][:, :nf], d["Ctr"], d["btr"], d["static_tr"]
    K = C.shape[1]
    print(f"budget {args.budget}: {len(X)} training programs, {nf} features, {K} pipelines in the portfolio")
    print(f"portfolio-best mean cost {C.min(axis=1).mean():.2f}, best fixed pipeline {static_cost.mean():.2f}, "
          f"unoptimised {base.mean():.2f}")

    results = {}
    results["Best fixed pipeline"] = score(np.zeros(len(X), dtype=int), C, base, static_cost)
    results["Best fixed pipeline"]["note"] = "portfolio entry 0 = best single pipeline over all candidates"
    zoo = model_zoo()
    for name, (kind, factory) in zoo.items():
        if args.quick and "HistGrad" in name:
            continue
        t = time.time()
        choice = cross_validate(kind, factory, X, C, base, static_cost)
        results[name] = score(choice, C, base, static_cost)
        results[name]["seconds"] = round(time.time() - t, 1)
        r = results[name]
        print(f"  {name:22s} cost {r['mean_cost']:.2f}  regret {r['mean_regret']:.3f}  "
              f"optimal {100 * r['optimal_rate']:.1f}%  vs fixed: better {100 * r['better_than_static']:.0f}% "
              f"worse {100 * r['worse_than_static']:.0f}%  ({r['seconds']}s)")

    # pick the winner by cross-validated mean cost
    contenders = {k: v for k, v in results.items() if k != "Best fixed pipeline"}
    winner = min(contenders, key=lambda k: contenders[k]["mean_cost"])
    print("winner:", winner)

    kind, factory = zoo[winner]
    model = fit_model(kind, factory, X, C)
    joblib.dump({"model": model, "kind": kind, "n_features": nf, "name": winner,
                 "portfolio": meta["portfolio"], "feature_names": meta["feature_names"][:nf],
                 "budget": args.budget, "rule_names": meta["rule_names"]},
                os.path.join(MODELS, f"best_model_b{args.budget}_{args.features}.joblib"), compress=3)

    # an interpretable shallow tree, for the report
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED)
    tree_choice = cross_validate("cls", lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED),
                                 X, C, base, static_cost)
    tree_score = score(tree_choice, C, base, static_cost)
    tree.fit(X, class_labels(C))
    names = meta["feature_names"][:nf]
    pipelines = [" > ".join(meta["rule_names"][r] for r in o) for o in meta["portfolio"]]
    txt = export_text(tree, feature_names=names)
    with open(os.path.join(RESULTS, f"decision_tree_b{args.budget}_{args.features}.txt"), "w") as f:
        f.write(f"Depth-4 decision tree (budget {args.budget}). Leaves are portfolio indices.\n")
        f.write(f"Cross-validated mean cost {tree_score['mean_cost']:.2f} "
                f"(best fixed pipeline {results['Best fixed pipeline']['mean_cost']:.2f})\n\n")
        f.write("Portfolio:\n" + "\n".join(f"  {i:2d}: {p}" for i, p in enumerate(pipelines)) + "\n\n")
        f.write(txt)

    # feature-set ablation for the winner
    ablation = {}
    for label, cols in [("structural only", meta["n_struct"]), ("structural + probes", d["Xtr"].shape[1])]:
        ch = cross_validate(kind, factory, d["Xtr"][:, :cols], C, base, static_cost)
        ablation[label] = score(ch, C, base, static_cost)
        print(f"  ablation [{label}] cost {ablation[label]['mean_cost']:.2f}")

    # permutation importance would be costly; use model's own importances when available
    importances = None
    inner = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
        order = np.argsort(imp)[::-1][:12]
        importances = [(names[i], float(imp[i])) for i in order]

    out = {"budget": args.budget, "features": args.features, "winner": winner, "cv": results,
           "shallow_tree_cv": tree_score, "ablation_winner": ablation, "top_features": importances,
           "portfolio_best_mean_cost": float(C.min(axis=1).mean()), "base_mean_cost": float(base.mean())}
    with open(os.path.join(RESULTS, f"model_comparison_b{args.budget}_{args.features}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved model and comparison")


if __name__ == "__main__":
    main()
