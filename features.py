"""Feature extraction: turn a TAC program into a numeric vector.

Two groups of features:

* **structural** - cheap syntactic counts (how many multiplications, how many
  constant assignments, how many repeated expressions, ...).
* **probe** - what happens if each rule is applied *alone* to the code
  (cost saved and instructions removed).  This is the kind of quick probe an
  auto-tuning compiler can afford, because each rule is a single linear scan.
"""
from __future__ import annotations

import numpy as np

from cost import cost
from engine import apply_rule, apply_ordering
from rules import N_RULES, RULE_NAMES, WINDOW, _chain_form, _expr_key
from tac import defs, is_const, is_jump, is_pow2, is_temp, is_var, segments, uses

STRUCT_NAMES = [
    "n_instr", "n_segments", "n_labels", "n_if", "n_goto", "n_print",
    "n_asg_const", "n_asg_copy",
    "n_add", "n_sub", "n_mul", "n_div", "n_mod", "n_shl", "n_shr", "n_and",
    "n_bin_const_const", "n_bin_var_const", "n_bin_var_var",
    "n_mul_pow2", "n_div_pow2", "n_mod_pow2", "n_mul_other_const",
    "n_identity", "n_self_assign", "n_if_const", "n_temps",
    "n_chain_pairs", "n_cse_pairs", "n_copy_used", "n_dead_temps",
    "base_cost", "frac_mul_div",
]
PROBE_NAMES = (
    [f"gain_{n}" for n in RULE_NAMES]
    + [f"shrink_{n}" for n in RULE_NAMES]
    + [f"gain_{n}_then_DCE" for n in RULE_NAMES[:-1]]
    + ["gain_textbook", "gain_textbook_frac"]
)
FEATURE_NAMES = STRUCT_NAMES + PROBE_NAMES
N_STRUCT = len(STRUCT_NAMES)

TEXTBOOK_ORDER = tuple(range(N_RULES))     # CF, AS, SR, CP, RA, CSE, RS, DCE


def _is_identity(ins) -> bool:
    op, a, b = ins.op, ins.a, ins.b
    if op == "+":
        return a == 0 or b == 0
    if op == "-":
        return b == 0 or (is_var(a) and a == b)
    if op == "*":
        return a in (0, 1) or b in (0, 1)
    if op == "/":
        return b == 1
    if op == "%":
        return b == 1
    if op in ("<<", ">>"):
        return b == 0 or a == 0
    if op == "&":
        return a == 0 or b == 0 or (is_var(a) and a == b)
    return False


def structural(code):
    c = {k: 0 for k in STRUCT_NAMES}
    temps = set()
    n = len(code)
    c["n_instr"] = n
    c["n_segments"] = len(segments(code))
    c["base_cost"] = cost(code)
    op_key = {"+": "n_add", "-": "n_sub", "*": "n_mul", "/": "n_div", "%": "n_mod",
              "<<": "n_shl", ">>": "n_shr", "&": "n_and"}
    for i, ins in enumerate(code):
        k = ins.kind
        d = defs(ins)
        if d is not None and is_temp(d):
            temps.add(d)
        if k == "label":
            c["n_labels"] += 1
        elif k == "if":
            c["n_if"] += 1
            if is_const(ins.a) and is_const(ins.b):
                c["n_if_const"] += 1
        elif k == "goto":
            c["n_goto"] += 1
        elif k == "print":
            c["n_print"] += 1
        elif k == "asg":
            if is_const(ins.a):
                c["n_asg_const"] += 1
            else:
                c["n_asg_copy"] += 1
                if ins.a == ins.dst:
                    c["n_self_assign"] += 1
        elif k == "bin":
            c[op_key[ins.op]] += 1
            ca, cb = is_const(ins.a), is_const(ins.b)
            if ca and cb:
                c["n_bin_const_const"] += 1
            elif ca or cb:
                c["n_bin_var_const"] += 1
            else:
                c["n_bin_var_var"] += 1
            other = ins.b if ca else ins.a
            const = ins.a if ca else ins.b
            if ins.op == "*" and (ca != cb):
                if is_pow2(const):
                    c["n_mul_pow2"] += 1
                else:
                    c["n_mul_other_const"] += 1
            elif ins.op == "/" and cb and not ca and is_pow2(ins.b):
                c["n_div_pow2"] += 1
            elif ins.op == "%" and cb and not ca and is_pow2(ins.b):
                c["n_mod_pow2"] += 1
            if _is_identity(ins):
                c["n_identity"] += 1

        # opportunities inside the peephole window
        if k in ("asg", "bin"):
            d = ins.dst
            form = _chain_form(ins)
            key = _expr_key(ins) if k == "bin" and not (is_const(ins.a) and is_const(ins.b)) else None
            used = False
            dead = True
            for j in range(i + 1, n):
                cj = code[j]
                if cj.kind == "label":
                    break
                inwin = j <= i + WINDOW
                u = uses(cj)
                if d in u:
                    dead = False
                    if inwin and k == "asg":
                        used = True
                    if inwin and form is not None and d != form[0]:
                        f2 = _chain_form(cj)
                        if f2 is not None and f2[0] == d and f2[1] == form[1]:
                            c["n_chain_pairs"] += 1
                if inwin and key is not None and cj.kind == "bin" and _expr_key(cj) == key:
                    c["n_cse_pairs"] += 1
                if defs(cj) == d:
                    break
                if is_jump(cj):
                    break
            if used:
                c["n_copy_used"] += 1
            if dead and is_temp(d):
                c["n_dead_temps"] += 1
    c["n_temps"] = len(temps)
    c["frac_mul_div"] = (c["n_mul"] + c["n_div"] + c["n_mod"]) / max(1, c["n_instr"])
    return [float(c[k]) for k in STRUCT_NAMES]


def probe(code):
    base = cost(code)
    n0 = len(code)
    gains, shrinks, with_dce = [], [], []
    for r in range(N_RULES):
        out = apply_rule(code, r)
        gains.append(float(base - cost(out)))
        shrinks.append(float(n0 - len(out)))
        if r < N_RULES - 1:
            with_dce.append(float(base - cost(apply_rule(out, N_RULES - 1))))
    tb = base - cost(apply_ordering(code, TEXTBOOK_ORDER))
    return gains + shrinks + with_dce + [float(tb), tb / base if base else 0.0]


def extract(code, with_probe: bool = True):
    s = structural(code)
    return np.array(s + probe(code) if with_probe else s, dtype=float)
