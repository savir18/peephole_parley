"""The eight peephole rules.

Every rule takes a tuple of instructions and returns a new tuple (one pass
over the code).  ``engine.apply_rule`` repeats a rule until nothing changes.

Rules that need to look ahead (copy propagation, reassociation, CSE and
redundant-instruction removal) only look at the next ``WINDOW``
instructions and never look past a label or a jump.  That is what makes
them *peephole* rules.

Index  Name                 What it does
-----  -------------------  ------------------------------------------------
  0    CF   const folding   t = 2 + 3          ->  t = 5      (also if 1 < 2 goto L)
  1    AS   algebraic       x * 1, x + 0, x - x ->  copy / constant
  2    SR   strength red.   x * 8 -> x << 3,  x / 8 -> x >> 3,  x % 8 -> x & 7
  3    CP   copy/const prop y = x; z = y + 1   ->  z = x + 1
  4    RA   reassociation   t = x + 3; u = t + 5  ->  u = x + 8   (also * chains)
  5    CSE  common subexpr  t = a + b; ...; u = a + b  ->  u = t
  6    RS   redundant stmts x = x, a = b; b = a, goto next, repeated statements
  7    DCE  dead code       unused temporaries, overwritten stores, unreachable code
"""
from __future__ import annotations

from tac import (
    COMMUTATIVE, Asg, Bin, Goto, eval_op, eval_rel, is_const, is_jump, is_pow2,
    is_temp, is_var, segments, substitute, uses, defs,
)

WINDOW = 6


# ------------------------------------------------------------------- 0: CF
def constant_folding(code):
    out = []
    for ins in code:
        if ins.kind == "bin" and is_const(ins.a) and is_const(ins.b):
            try:
                v = eval_op(ins.op, ins.a, ins.b)
            except (ZeroDivisionError, ValueError):
                out.append(ins)
                continue
            out.append(Asg(ins.dst, v))
        elif ins.kind == "if" and is_const(ins.a) and is_const(ins.b):
            if eval_rel(ins.op, ins.a, ins.b):
                out.append(Goto(ins.target))
            # a branch that is never taken simply disappears
        else:
            out.append(ins)
    return tuple(out)


# ------------------------------------------------------------------- 1: AS
def algebraic_simplification(code):
    out = []
    for ins in code:
        if ins.kind != "bin":
            out.append(ins)
            continue
        op, a, b, d = ins.op, ins.a, ins.b, ins.dst
        new = ins
        if op == "+":
            if a == 0:
                new = Asg(d, b)
            elif b == 0:
                new = Asg(d, a)
        elif op == "-":
            if b == 0:
                new = Asg(d, a)
            elif is_var(a) and a == b:
                new = Asg(d, 0)
        elif op == "*":
            if a == 1:
                new = Asg(d, b)
            elif b == 1:
                new = Asg(d, a)
            elif a == 0 or b == 0:
                new = Asg(d, 0)
        elif op == "/":
            if b == 1:
                new = Asg(d, a)
        elif op == "%":
            if b == 1:
                new = Asg(d, 0)
        elif op in ("<<", ">>"):
            if b == 0:
                new = Asg(d, a)
            elif a == 0:
                new = Asg(d, 0)
        elif op == "&":
            if a == 0 or b == 0:
                new = Asg(d, 0)
            elif is_var(a) and a == b:
                new = Asg(d, a)
        out.append(new)
    return tuple(out)


# ------------------------------------------------------------------- 2: SR
def strength_reduction(code):
    out = []
    for ins in code:
        new = ins
        if ins.kind == "bin":
            op, a, b, d = ins.op, ins.a, ins.b, ins.dst
            if op == "*":
                if is_var(a) and is_pow2(b):
                    new = Bin(d, "<<", a, b.bit_length() - 1)
                elif is_pow2(a) and is_var(b):
                    new = Bin(d, "<<", b, a.bit_length() - 1)
            elif op == "/":
                if is_var(a) and is_pow2(b):
                    new = Bin(d, ">>", a, b.bit_length() - 1)
            elif op == "%":
                if is_var(a) and is_pow2(b):
                    new = Bin(d, "&", a, b - 1)
        out.append(new)
    return tuple(out)


# ------------------------------------------------------------------- 3: CP
def copy_propagation(code):
    code = list(code)
    n = len(code)
    for i in range(n):
        ins = code[i]
        if ins.kind != "asg" or ins.dst == ins.a:
            continue
        d, s = ins.dst, ins.a
        for j in range(i + 1, min(n, i + 1 + WINDOW)):
            cj = code[j]
            if cj.kind == "label":
                break
            if d in uses(cj):
                code[j] = cj = substitute(cj, d, s)
            w = defs(cj)
            if w == d or (is_var(s) and w == s):
                break
            if is_jump(cj):
                break
    return tuple(code)


# ------------------------------------------------------------------- 4: RA
def _chain_form(ins):
    """(x, family, c) when ins is  dst = x + c,  x - c  or  x * c."""
    if ins.kind != "bin":
        return None
    op, a, b = ins.op, ins.a, ins.b
    if op == "+":
        if is_var(a) and is_const(b):
            return (a, "add", b)
        if is_const(a) and is_var(b):
            return (b, "add", a)
    elif op == "-":
        if is_var(a) and is_const(b):
            return (a, "add", -b)
    elif op == "*":
        if is_var(a) and is_const(b):
            return (a, "mul", b)
        if is_const(a) and is_var(b):
            return (b, "mul", a)
    return None


def _make_chain(dst, x, family, k):
    if family == "add":
        if k == 0:
            return Asg(dst, x)
        return Bin(dst, "+", x, k) if k > 0 else Bin(dst, "-", x, -k)
    if k == 1:
        return Asg(dst, x)
    if k == 0:
        return Asg(dst, 0)
    return Bin(dst, "*", x, k)


def reassociation(code):
    code = list(code)
    n = len(code)
    for i in range(n):
        form = _chain_form(code[i])
        if form is None:
            continue
        x, fam, c1 = form
        t = code[i].dst
        if x == t:
            continue
        for j in range(i + 1, min(n, i + 1 + WINDOW)):
            cj = code[j]
            if cj.kind == "label":
                break
            f2 = _chain_form(cj)
            if f2 is not None and f2[0] == t and f2[1] == fam:
                k = c1 + f2[2] if fam == "add" else c1 * f2[2]
                code[j] = cj = _make_chain(cj.dst, x, fam, k)
            w = defs(cj)
            if w == t or w == x:
                break
            if is_jump(cj):
                break
    return tuple(code)


# ------------------------------------------------------------------ 5: CSE
def _expr_key(ins):
    op, a, b = ins.op, ins.a, ins.b
    if op in COMMUTATIVE and (isinstance(a, int), str(a)) > (isinstance(b, int), str(b)):
        a, b = b, a
    return (op, a, b)


def common_subexpression(code):
    code = list(code)
    n = len(code)
    for i in range(n):
        ci = code[i]
        if ci.kind != "bin" or (is_const(ci.a) and is_const(ci.b)):
            continue
        d = ci.dst
        if d == ci.a or d == ci.b:
            continue
        key = _expr_key(ci)
        watch = {d}
        watch.update(v for v in (ci.a, ci.b) if is_var(v))
        for j in range(i + 1, min(n, i + 1 + WINDOW)):
            cj = code[j]
            if cj.kind == "label":
                break
            if cj.kind == "bin" and _expr_key(cj) == key:
                code[j] = cj = Asg(cj.dst, d)
            if defs(cj) in watch:
                break
            if is_jump(cj):
                break
    return tuple(code)


# ------------------------------------------------------------------- 6: RS
def redundant_statements(code):
    code = list(code)
    n = len(code)
    drop = set()

    # self assignments and jumps to the very next label
    for i, ins in enumerate(code):
        if ins.kind == "asg" and ins.dst == ins.a:
            drop.add(i)
        elif ins.kind in ("goto", "if"):
            k = i + 1
            while k < n and code[k].kind == "label":
                if code[k].target == ins.target:
                    drop.add(i)
                    break
                k += 1

    # mirrored copies (a = b ... b = a) and exact repeats within the window
    for i in range(n):
        if i in drop:
            continue
        ci = code[i]
        if ci.kind not in ("asg", "bin"):
            continue
        d = ci.dst
        ops = uses(ci)
        if d in ops:
            continue
        watch = set(ops)
        watch.add(d)
        for j in range(i + 1, min(n, i + 1 + WINDOW)):
            if j in drop:
                continue
            cj = code[j]
            if cj.kind == "label":
                break
            if cj == ci:
                drop.add(j)
                continue
            if (ci.kind == "asg" and is_var(ci.a) and cj.kind == "asg"
                    and cj.dst == ci.a and cj.a == d):
                drop.add(j)
                continue
            if defs(cj) in watch:
                break
            if is_jump(cj):
                break
    return tuple(ins for i, ins in enumerate(code) if i not in drop)


# ------------------------------------------------------------------ 7: DCE
def dead_code_elimination(code):
    # 1. unreachable code: everything after an unconditional goto up to the next label
    reach = []
    dead = False
    for ins in code:
        if ins.kind == "label":
            dead = False
        if not dead:
            reach.append(ins)
        if ins.kind == "goto":
            dead = True
    code = reach

    # 2. dead assignments, one straight-line segment at a time (backwards).
    #    Temporaries are dead at the end of their segment; ordinary variables
    #    are assumed live at the end of a segment (conservative).
    keep = [True] * len(code)
    for start, end in segments(code):
        live_temps = set()
        killed = set()          # ordinary variables overwritten later in the segment
        for i in range(end - 1, start - 1, -1):
            ins = code[i]
            d = defs(ins)
            if d is not None:
                live = (d in live_temps) if is_temp(d) else (d not in killed)
                if not live:
                    keep[i] = False
                    continue
                if is_temp(d):
                    live_temps.discard(d)
                else:
                    killed.add(d)
            for u in uses(ins):
                if is_temp(u):
                    live_temps.add(u)
                else:
                    killed.discard(u)
    return tuple(ins for ins, k in zip(code, keep) if k)


# --------------------------------------------------------------- registry
RULES = (
    ("CF", constant_folding),
    ("AS", algebraic_simplification),
    ("SR", strength_reduction),
    ("CP", copy_propagation),
    ("RA", reassociation),
    ("CSE", common_subexpression),
    ("RS", redundant_statements),
    ("DCE", dead_code_elimination),
)
RULE_NAMES = tuple(n for n, _ in RULES)
N_RULES = len(RULES)
