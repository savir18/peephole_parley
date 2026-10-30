"""Build the training / test data for a given rule *budget*.

A *pipeline* is a sequence of ``budget`` distinct rules applied one after the
other (each rule is run to its own fixed point).  ``budget = 8`` means every
rule is used once, so the only question is the order; a smaller budget also
asks *which* rules to skip, i.e. adaptive rule selection under a compile-time
limit.

Steps
1. generate random programs (train and test seeds never overlap);
2. evaluate every candidate pipeline on every **training** program and pick
   * the best single fixed pipeline (the static baseline), and
   * a *portfolio* of K pipelines, greedily, so that choosing the best member
     per program loses almost nothing (these are the classes the ML model
     chooses between);
3. for every program record its features and the cost it reaches under each
   portfolio pipeline;
4. for the **test** programs also record the exhaustive oracle (best over *all*
   sequences, found by dynamic programming) and the other baselines.

Run:  python dataset.py --budget 8
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import random
import time

import numpy as np

from cost import cost
from engine import apply_fixpoint_counted, apply_ordering, clear_caches, explore
from features import FEATURE_NAMES, N_STRUCT, TEXTBOOK_ORDER, extract
from generator import generate
from rules import N_RULES, RULE_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

TRAIN_SEED0, N_TRAIN = 100_000, 1600
TEST_SEED0, N_TEST = 200_000, 400
MAX_CANDIDATES = 2500       # at most this many pipelines are evaluated per budget
PORTFOLIO_SIZE = 16
N_RANDOM_BASELINE = 20


# ------------------------------------------------------------ candidates
def n_sequences(budget: int) -> int:
    n = 1
    for i in range(budget):
        n *= N_RULES - i
    return n


def candidate_pipelines(budget: int, seed: int = 0):
    """All sequences of ``budget`` distinct rules, or a random sample of them."""
    textbook = TEXTBOOK_ORDER[:budget]
    if n_sequences(budget) <= MAX_CANDIDATES:
        rest = [p for p in itertools.permutations(range(N_RULES), budget) if p != textbook]
        return [textbook] + rest
    rng = random.Random(seed)
    seen = {textbook}
    out = [textbook]
    while len(out) < MAX_CANDIDATES:
        p = tuple(rng.sample(range(N_RULES), budget))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def greedy_portfolio(M: np.ndarray, k: int):
    """Pick k rows of M (candidates x programs) greedily to minimise the per-program best."""
    k = min(k, M.shape[0])
    chosen = []
    cur = np.full(M.shape[1], np.inf)
    for _ in range(k):
        total = np.minimum(M, cur).sum(axis=1)
        total[chosen] = np.inf
        best = int(np.argmin(total))
        chosen.append(best)
        cur = np.minimum(cur, M[best])
    return chosen


# ------------------------------------------------------------- workers
_G = {}


def _init_worker(cands, portfolio, static, budget):
    _G.update(cands=cands, portfolio=portfolio, static=static, budget=budget)


def _train_candidate_costs(seed):
    code = generate(seed)
    clear_caches()
    return np.array([cost(apply_ordering(code, o)) for o in _G["cands"]], dtype=np.int32)


def _program_row(args):
    seed, is_test = args
    code = generate(seed)
    clear_caches()
    x = extract(code)
    portfolio_costs = [cost(apply_ordering(code, o)) for o in _G["portfolio"]]
    row = {"x": x, "C": portfolio_costs, "base": cost(code), "n": len(code),
           "static": cost(apply_ordering(code, _G["static"]))}
    if is_test:
        b = _G["budget"]
        best, order, worst = explore(code, b)
        row["oracle"], row["oracle_order"], row["worst"] = best, order, worst
        row["textbook"] = cost(apply_ordering(code, TEXTBOOK_ORDER[:b]))
        fx_code, fx_effort = apply_fixpoint_counted(code, TEXTBOOK_ORDER)
        row["fixpoint"], row["fixpoint_effort"] = cost(fx_code), fx_effort
        rng = random.Random(seed)
        rnd = [cost(apply_ordering(code, tuple(rng.sample(range(N_RULES), b))))
               for _ in range(N_RANDOM_BASELINE)]
        row["random"] = float(np.mean(rnd))
    return row


def _stack(rows, key, dtype=float):
    return np.array([r[key] for r in rows], dtype=dtype)


# ---------------------------------------------------------------- main
def build(budget: int, quiet: bool = False):
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    log = (lambda *a: None) if quiet else print
    train_seeds = list(range(TRAIN_SEED0, TRAIN_SEED0 + N_TRAIN))
    test_seeds = list(range(TEST_SEED0, TEST_SEED0 + N_TEST))
    assert not set(train_seeds) & set(test_seeds)

    cands = candidate_pipelines(budget)
    log(f"budget {budget}: {len(cands)} candidate pipelines "
        f"({'all' if n_sequences(budget) <= MAX_CANDIDATES else 'sampled from ' + str(n_sequences(budget))})")

    ncpu = os.cpu_count() or 2
    with mp.Pool(ncpu, initializer=_init_worker, initargs=(cands, None, None, budget)) as pool:
        M = np.array(pool.map(_train_candidate_costs, train_seeds, chunksize=8)).T   # cands x programs
    static_idx = int(np.argmin(M.sum(axis=1)))
    static = cands[static_idx]
    chosen = greedy_portfolio(M, PORTFOLIO_SIZE)
    portfolio = [cands[i] for i in chosen]
    log(f"candidates evaluated in {time.time() - t0:.1f}s")
    log("best fixed pipeline:", " > ".join(RULE_NAMES[r] for r in static))
    for i, o in enumerate(portfolio):
        log(f"  portfolio {i:2d}: {' > '.join(RULE_NAMES[r] for r in o)}")

    with mp.Pool(ncpu, initializer=_init_worker, initargs=(cands, portfolio, static, budget)) as pool:
        tr = pool.map(_program_row, [(s, False) for s in train_seeds], chunksize=16)
        te = pool.map(_program_row, [(s, True) for s in test_seeds], chunksize=4)

    out = dict(
        Xtr=np.array([r["x"] for r in tr]), Ctr=_stack(tr, "C"), btr=_stack(tr, "base"),
        ntr=_stack(tr, "n"), static_tr=_stack(tr, "static"), seeds_tr=np.array(train_seeds),
        Xte=np.array([r["x"] for r in te]), Cte=_stack(te, "C"), bte=_stack(te, "base"),
        nte=_stack(te, "n"), static_te=_stack(te, "static"), seeds_te=np.array(test_seeds),
        oracle_te=_stack(te, "oracle"), worst_te=_stack(te, "worst"),
        textbook_te=_stack(te, "textbook"), fixpoint_te=_stack(te, "fixpoint"),
        fixpoint_effort_te=_stack(te, "fixpoint_effort"), random_te=_stack(te, "random"),
    )
    np.savez_compressed(os.path.join(DATA, f"dataset_b{budget}.npz"), **out)
    meta = {
        "budget": budget,
        "portfolio": [list(o) for o in portfolio],
        "static_pipeline": list(static),
        "textbook_pipeline": list(TEXTBOOK_ORDER[:budget]),
        "rule_names": list(RULE_NAMES),
        "feature_names": FEATURE_NAMES,
        "n_struct": N_STRUCT,
        "n_candidates": len(cands),
        "candidates_exhaustive": n_sequences(budget) <= MAX_CANDIDATES,
        "train_seeds": [TRAIN_SEED0, N_TRAIN],
        "test_seeds": [TEST_SEED0, N_TEST],
        "oracle_orders_te": [list(r["oracle_order"]) for r in te],
    }
    with open(os.path.join(DATA, f"portfolio_b{budget}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"done: train {out['Xtr'].shape}, test {out['Xte'].shape} in {time.time() - t0:.1f}s")
    return out, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=N_RULES, help="rules per pipeline (1..8)")
    args = ap.parse_args()
    assert 1 <= args.budget <= N_RULES
    build(args.budget)
