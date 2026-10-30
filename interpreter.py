"""A tiny TAC interpreter used to check that optimisations preserve behaviour.

The observable behaviour of a program is the list of values it prints.
Variables that are read before being written take their value from the
initial environment (default 0).
"""
from __future__ import annotations

import random

from tac import eval_op, eval_rel, is_var, variables


class StepLimit(RuntimeError):
    pass


def run(code, env=None, max_steps: int = 20000):
    """Execute ``code`` and return the list of printed values."""
    env = dict(env or {})
    out = []
    labels = {ins.target: i for i, ins in enumerate(code) if ins.kind == "label"}

    def val(x):
        return env.get(x, 0) if is_var(x) else x

    pc = 0
    steps = 0
    n = len(code)
    while pc < n:
        steps += 1
        if steps > max_steps:
            raise StepLimit("step limit exceeded")
        ins = code[pc]
        k = ins.kind
        if k == "asg":
            env[ins.dst] = val(ins.a)
        elif k == "bin":
            env[ins.dst] = eval_op(ins.op, val(ins.a), val(ins.b))
        elif k == "print":
            out.append(val(ins.a))
        elif k == "goto":
            pc = labels[ins.target]
            continue
        elif k == "if":
            if eval_rel(ins.op, val(ins.a), val(ins.b)):
                pc = labels[ins.target]
                continue
        # labels are no-ops
        pc += 1
    return out


def random_env(names, rng: random.Random, lo: int = -20, hi: int = 20):
    """Random initial values for *every* variable (temporaries included).

    Giving temporaries random start values is deliberate: it exposes any
    optimisation that makes the program read a temporary before writing it.
    """
    return {n: rng.randint(lo, hi) for n in sorted(names)}


def run_safe(code, env):
    """Run and return ('ok', output) or ('err', exception-name)."""
    try:
        return ("ok", run(code, env))
    except (ZeroDivisionError, ValueError, StepLimit, KeyError) as exc:
        return ("err", type(exc).__name__)


def equivalent(code_a, code_b, trials: int = 6, seed: int = 0) -> bool:
    """Compare observable behaviour on random inputs."""
    rng = random.Random(seed)
    names = variables(code_a) | variables(code_b)
    for _ in range(trials):
        env = random_env(names, rng)
        if run_safe(code_a, env) != run_safe(code_b, env):
            return False
    return True
