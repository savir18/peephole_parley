"""Command-line optimiser.

    python main.py examples/demo_chain.tac                      # ML-selected pipeline, budget 4
    python main.py examples/demo_large.tac --budget 3
    python main.py examples/demo_large.tac --strategy all       # compare every strategy
    python main.py prog.tac --strategy oracle -o prog_opt.tac   # exhaustive search, save result

Strategies
  ml        the trained model picks a pipeline of ``--budget`` rules for this program
  fixed     the single best pipeline found on the training programs
  oracle    exhaustive search over every sequence of ``--budget`` rules
  textbook  the first ``--budget`` rules in their natural order
  fixpoint  repeat all eight rules (textbook order) until nothing changes
  all       run all of the above and print a comparison
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import joblib

from cost import cost
from engine import apply_fixpoint_counted, apply_ordering, explore, ordering_names
from features import TEXTBOOK_ORDER, extract
from interpreter import equivalent
from tac import parse, to_text
from train import predict_choice

HERE = os.path.dirname(os.path.abspath(__file__))


def load_meta(budget):
    path = os.path.join(HERE, "data", f"portfolio_b{budget}.json")
    if not os.path.exists(path):
        sys.exit(f"No data for budget {budget}. Run:  python dataset.py --budget {budget}  and  python train.py --budget {budget}")
    with open(path) as f:
        return json.load(f)


def run_strategy(name, code, budget, meta, bundle):
    if name == "ml":
        x = extract(code)[: bundle["n_features"]].reshape(1, -1)
        idx = int(predict_choice(bundle["kind"], bundle["model"], x, len(meta["portfolio"]))[0])
        pipe = tuple(meta["portfolio"][idx])
    elif name == "fixed":
        pipe = tuple(meta["static_pipeline"])
    elif name == "oracle":
        pipe = explore(code, budget)[1]
    elif name == "textbook":
        pipe = TEXTBOOK_ORDER[:budget]
    elif name == "fixpoint":
        out, effort = apply_fixpoint_counted(code, TEXTBOOK_ORDER)
        return out, f"{ordering_names(TEXTBOOK_ORDER)} (repeated), {effort} rule executions", effort
    else:
        raise ValueError(name)
    return apply_ordering(code, pipe), ordering_names(pipe), len(pipe)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="TAC source file")
    ap.add_argument("--budget", type=int, default=4, help="rules per pipeline, 1..8 (default 4)")
    ap.add_argument("--strategy", default="ml",
                    choices=["ml", "fixed", "oracle", "textbook", "fixpoint", "all"])
    ap.add_argument("-o", "--output", help="write the optimised code here")
    ap.add_argument("--no-verify", action="store_true", help="skip the behaviour check")
    args = ap.parse_args()

    try:
        with open(args.file) as f:
            code = parse(f.read())
    except FileNotFoundError:
        sys.exit(f"error: file not found: {args.file}")
    except ValueError as e:
        sys.exit(f"error: invalid TAC input: {e}")
    if not code or cost(code) == 0:
        sys.exit("error: program is empty (nothing to optimise)")
    meta = load_meta(args.budget)
    bundle = joblib.load(os.path.join(HERE, "models", f"best_model_b{args.budget}_full.joblib"))
    base = cost(code)

    names = ["ml", "fixed", "oracle", "textbook", "fixpoint"] if args.strategy == "all" else [args.strategy]
    rows = []
    last = None
    for name in names:
        out, pipe, effort = run_strategy(name, code, args.budget, meta, bundle)
        ok = True if args.no_verify else equivalent(code, out, trials=10)
        rows.append((name, cost(out), effort, ok, pipe))
        last = out

    if args.strategy != "all":
        print(to_text(last))
    print(f"unoptimised cost: {base}")
    print(f"{'strategy':10s} {'cost':>6s} {'saved':>7s} {'rule runs':>10s}  same behaviour  pipeline")
    for name, c, effort, ok, pipe in rows:
        print(f"{name:10s} {c:6d} {100 * (base - c) / base:6.1f}% {effort:10d}  {'yes' if ok else 'NO!':14s}  {pipe}")
    if args.output and args.strategy != "all":
        with open(args.output, "w") as f:
            f.write(to_text(last))
        print(f"written to {args.output}")


if __name__ == "__main__":
    main()
