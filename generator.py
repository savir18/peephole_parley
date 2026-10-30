"""Random TAC program generator.

Each program is built from small *pattern templates* (constant expressions,
multiplications by powers of two, repeated sub-expressions, copy chains,
dead temporaries, ...).  Every program draws its own random mixture of
patterns, so some programs are dominated by (say) repeated expressions and
others by multiplication chains.  That variety is what makes the best rule
ordering differ from program to program.

Guarantees (needed so that optimisations are provably safe to test):
* jumps only go forward, so every program terminates;
* temporaries (t1, t2, ...) are created and used inside a single pattern,
  so they never live across a label or a jump;
* divisors are non-zero constants.
"""
from __future__ import annotations

import random

from tac import Asg, Bin, If, Label, Print

INPUTS = ["a", "b", "c", "d"]
LOCALS = ["x", "y", "z", "w", "u", "v", "p", "q", "r", "s", "m", "n"]
POW2 = [2, 4, 8, 16, 32, 64]
OTHER = [3, 5, 6, 7, 9, 10, 12, 15, 100]
RELS = ["<", "<=", ">", ">=", "==", "!="]

PATTERN_NAMES = [
    "const_expr", "const_prop", "mul_pow2", "div_pow2", "mod_pow2", "mul_other",
    "identity", "copy_chain", "cse", "cse_hidden", "reassoc_add", "reassoc_mul",
    "dead_temp", "dead_store", "redundant_copy", "plain",
]


class _Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.ntemp = 0
        self.defined = list(INPUTS)

    # ---- names
    def temp(self):
        self.ntemp += 1
        return f"t{self.ntemp}"

    def local(self, avoid=()):
        choices = [v for v in LOCALS if v not in avoid]
        v = self.rng.choice(choices)
        if v not in self.defined:
            self.defined.append(v)
        return v

    def var(self, avoid=()):
        pool = [v for v in self.defined if v not in avoid]
        if not pool:
            pool = list(INPUTS)
        # lean towards the inputs so that expressions repeat more often
        if self.rng.random() < 0.45:
            inp = [v for v in INPUTS if v not in avoid]
            if inp:
                return self.rng.choice(inp)
        return self.rng.choice(pool)

    def gap(self):
        return self.rng.choice([0, 0, 1, 2, 3, 5, 8])

    def filler(self, k, avoid):
        out = []
        for _ in range(k):
            tgt = self.local(avoid)
            op = self.rng.choice(["+", "-", "*", "&"])
            out.append(Bin(tgt, op, self.var(avoid), self.var(avoid)))
        return out


# ------------------------------------------------------------- patterns
def p_const_expr(g):
    t = g.temp()
    x = g.local()
    v = g.var({x})
    return [
        Bin(t, g.rng.choice(["+", "-", "*"]), g.rng.randint(1, 20), g.rng.randint(1, 20)),
        Bin(x, g.rng.choice(["+", "-", "*"]), v, t),
    ]


def p_const_prop(g):
    r = g.rng
    k = g.temp() if r.random() < 0.6 else g.local()
    x = g.local({k})
    v = g.var({k, x})
    op = r.choice(["*", "*", "+", "-", "/", "%", "<<"])
    if op in ("/", "%"):
        c = r.choice(POW2 + OTHER)
    elif op == "<<":
        c = r.randint(0, 5)
    else:
        c = r.choice(POW2 + OTHER + [0, 1, 2])
    return [Asg(k, c)] + g.filler(g.gap(), {k, x, v}) + [Bin(x, op, v, k)]


def p_mul_pow2(g):
    x = g.local()
    v = g.var({x})
    c = g.rng.choice(POW2)
    return [Bin(x, "*", v, c) if g.rng.random() < 0.6 else Bin(x, "*", c, v)]


def p_div_pow2(g):
    x = g.local()
    return [Bin(x, "/", g.var({x}), g.rng.choice(POW2))]


def p_mod_pow2(g):
    x = g.local()
    return [Bin(x, "%", g.var({x}), g.rng.choice(POW2))]


def p_mul_other(g):
    x = g.local()
    return [Bin(x, "*", g.var({x}), g.rng.choice(OTHER))]


def p_identity(g):
    r = g.rng
    x = g.local()
    v = g.var({x})
    form = r.randint(0, 6)
    if form == 0:
        return [Bin(x, "*", v, 1)]
    if form == 1:
        return [Bin(x, "+", v, 0)]
    if form == 2:
        return [Bin(x, "-", v, 0)]
    if form == 3:
        return [Bin(x, "-", v, v)]
    if form == 4:
        return [Bin(x, "*", 1, v)]
    if form == 5:
        return [Bin(x, "/", v, 1)]
    return [Bin(x, "*", v, 0)]


def p_copy_chain(g):
    r = g.rng
    length = r.randint(1, 3)
    names = [g.temp() for _ in range(length)]
    x = g.local()
    v = g.var({x})
    w = g.var({x})
    out = [Asg(names[0], v)]
    for a, b in zip(names, names[1:]):
        out.append(Asg(b, a))
    out += g.filler(g.gap(), {x, v, w} | set(names))
    out.append(Bin(x, r.choice(["+", "-", "*"]), names[-1], w))
    return out


def _cse(g, hidden: bool):
    r = g.rng
    op = r.choice(["+", "*", "-", "&"])
    v, w = g.var(), g.var()
    t1 = g.temp()
    x, y = g.local({v, w}), g.local({v, w})
    avoid = {v, w, x, y}
    out = [Bin(t1, op, v, w)]
    out += g.filler(g.gap(), avoid)
    out.append(Bin(x, "+", t1, r.randint(1, 9)))
    out += g.filler(g.gap(), avoid)
    second_a, second_b = v, w
    pre = []
    if hidden:
        # the repeat is only visible after copy propagation
        c = g.temp()
        pre = [Asg(c, v)]
        second_a = c
    if op in ("+", "*", "&") and r.random() < 0.5:
        second_a, second_b = second_b, second_a
    t2 = g.temp()
    out += pre + [Bin(t2, op, second_a, second_b), Bin(y, "+", t2, r.randint(1, 9))]
    return out


def p_cse(g):
    return _cse(g, False)


def p_cse_hidden(g):
    return _cse(g, True)


def p_reassoc_add(g):
    r = g.rng
    t1, t2 = g.temp(), g.temp()
    x = g.local()
    v = g.var({x})
    return [
        Bin(t1, "+", v, r.randint(1, 20)),
        Bin(t2, r.choice(["+", "-"]), t1, r.randint(1, 20)),
        Bin(x, r.choice(["+", "-"]), t2, r.randint(1, 20)),
    ]


def p_reassoc_mul(g):
    r = g.rng
    pool = POW2 if r.random() < 0.6 else OTHER
    t1 = g.temp()
    x = g.local()
    v = g.var({x})
    out = [Bin(t1, "*", v, r.choice(pool))]
    if r.random() < 0.5:
        t2 = g.temp()
        out += [Bin(t2, "*", t1, r.choice(pool)), Bin(x, "+", t2, 0 if r.random() < 0.3 else 1)]
    else:
        out += [Bin(x, "*", t1, r.choice(pool))]
    return out


def p_dead_temp(g):
    r = g.rng
    t1 = g.temp()
    out = [Bin(t1, r.choice(["+", "*", "-"]), g.var(), g.var())]
    if r.random() < 0.5:
        t2 = g.temp()
        out.append(Bin(t2, r.choice(["*", "+"]), t1, r.choice([2, 3, 4, 8])))
    return out


def p_dead_store(g):
    r = g.rng
    x = g.local()
    v, w = g.var({x}), g.var({x})
    return [Bin(x, r.choice(["+", "*", "-"]), v, w)] + g.filler(g.gap() % 3, {x, v, w}) + [
        Bin(x, r.choice(["+", "-", "*"]), g.var({x}), r.randint(1, 9))
    ]


def p_redundant_copy(g):
    r = g.rng
    x = g.local()
    y = g.var({x})
    if y in INPUTS and r.random() < 0.5:
        y = g.local({x})
    form = r.randint(0, 2)
    if form == 0:
        return [Asg(x, y), Asg(y, x)]
    if form == 1:
        return [Asg(x, x)]
    return [Asg(x, y)] + g.filler(g.gap() % 3, {x, y}) + [Asg(x, y)]


def p_plain(g):
    r = g.rng
    x = g.local()
    op = r.choice(["+", "-", "*", "+", "&"])
    b = g.var({x}) if r.random() < 0.6 else r.randint(1, 30)
    return [Bin(x, op, g.var({x}), b)]


PATTERNS = {
    "const_expr": p_const_expr, "const_prop": p_const_prop, "mul_pow2": p_mul_pow2,
    "div_pow2": p_div_pow2, "mod_pow2": p_mod_pow2, "mul_other": p_mul_other,
    "identity": p_identity, "copy_chain": p_copy_chain, "cse": p_cse,
    "cse_hidden": p_cse_hidden, "reassoc_add": p_reassoc_add,
    "reassoc_mul": p_reassoc_mul, "dead_temp": p_dead_temp,
    "dead_store": p_dead_store, "redundant_copy": p_redundant_copy, "plain": p_plain,
}


# ------------------------------------------------------------- programs
def _cond_jump(g, target):
    """A forward conditional jump: variable test, or a constant test through a temp."""
    r = g.rng
    kind = r.random()
    rel = r.choice(RELS)
    if kind < 0.45:
        return [If(rel, g.var(), g.var() if r.random() < 0.5 else r.randint(0, 20), target)]
    if kind < 0.8:
        t = g.temp()
        return [Asg(t, r.randint(0, 9)), If(rel, t, r.randint(0, 9), target)]
    return [If(rel, r.randint(0, 9), r.randint(0, 9), target)]


def generate(seed: int):
    """Generate one program (a tuple of instructions) from a seed."""
    rng = random.Random(seed)
    g = _Gen(rng)
    weights = {name: rng.gammavariate(0.55, 1.0) + 0.02 for name in PATTERNS}
    names = list(PATTERNS)
    w = [weights[n] for n in names]

    nblocks = rng.randint(1, 4)
    labels = [None] + [f"L{i}" for i in range(1, nblocks)] + ["Lend"]
    code = []
    for b in range(nblocks):
        if labels[b] is not None:
            code.append(Label(labels[b]))
        npat = rng.randint(2, 7)
        jump_at = rng.randrange(npat + 1) if rng.random() < 0.55 and b < nblocks else None
        for i in range(npat + 1):
            if jump_at == i:
                target = rng.choice(labels[b + 1:])
                code += _cond_jump(g, target)
            if i < npat:
                name = rng.choices(names, weights=w)[0]
                code += PATTERNS[name](g)
    code.append(Label("Lend"))
    outs = [v for v in g.defined if v not in INPUTS]
    rng.shuffle(outs)
    outs = outs[: rng.randint(2, 6)] or ["x"]
    for v in outs:
        code.append(Print(v))
    return tuple(code)


def generate_many(seeds):
    return [generate(s) for s in seeds]


# ------------------------------------------------------------- fuzzing
def generate_fuzz(seed: int):
    """Dense random code over a tiny variable pool (used only for testing).

    With few variables, definitions and uses collide constantly, which
    stresses the safety conditions of every rule (copy source redefined,
    operand overwritten between two equal expressions, ...).  Temporaries
    are only read after being written in the same straight-line segment.
    """
    rng = random.Random(seed)
    pool = ["a", "b", "x", "y", "z"]
    consts = [0, 1, 2, 3, 4, 5, 8, 16, -1, -3]
    code = []
    live_temps = set()
    pending = [f"F{i}" for i in range(rng.randint(1, 3))]   # labels still to be placed
    ntemp = 0

    def operand():
        r = rng.random()
        if r < 0.55:
            return rng.choice(pool)
        if r < 0.7 and live_temps:
            return rng.choice(sorted(live_temps))
        return rng.choice(consts)

    def dest():
        nonlocal ntemp
        if rng.random() < 0.3:
            ntemp += 1
            return f"t{ntemp}", True
        return rng.choice(pool), False

    for _ in range(rng.randint(6, 28)):
        r = rng.random()
        if r < 0.08 and pending:
            code.append(Label(pending.pop(0)))
            live_temps.clear()
        elif r < 0.16 and pending:
            code.append(If(rng.choice(RELS), operand(), operand(), rng.choice(pending)))
            live_temps.clear()
        elif r < 0.30:
            d, is_t = dest()
            code.append(Asg(d, operand()))
            if is_t:
                live_temps.add(d)
        else:
            op = rng.choice(["+", "-", "*", "&", "<<", ">>", "/", "%", "+", "*"])
            a = operand()
            if op in ("<<", ">>"):
                b = rng.randint(0, 4)
            elif op in ("/", "%"):
                b = rng.choice([1, 2, 3, 4, 8, 16])
            else:
                b = operand()
            d, is_t = dest()
            code.append(Bin(d, op, a, b))
            if is_t:
                live_temps.add(d)
    for lab in pending:
        code.append(Label(lab))
    live_temps.clear()
    for v in ("x", "y", "z", "a", "b"):
        code.append(Print(v))
    return tuple(code)
