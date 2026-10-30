"""Rule engine: apply rules and rule orderings, and search all orderings.

* ``apply_rule``      runs one rule repeatedly until the code stops changing
* ``apply_ordering``  runs the rules once each, in the given order (a "pass pipeline")
* ``apply_fixpoint``  repeats a whole pipeline until nothing changes
* ``explore``         dynamic programme over *all* sequences of B distinct rules (B = 8 gives
                      all 8! = 40320 orderings); returns the cheapest and the most expensive
                      result (the "oracle")
"""
from __future__ import annotations

from cost import cost
from rules import N_RULES, RULES, RULE_NAMES

_MAX_ROUNDS = 30
_CACHE_LIMIT = 400_000
_cache = [dict() for _ in range(N_RULES)]


def clear_caches():
    for c in _cache:
        c.clear()


def apply_rule(code, r: int):
    """Apply rule ``r`` repeatedly until a fixed point (memoised)."""
    c = _cache[r]
    hit = c.get(code)
    if hit is not None:
        return hit
    fn = RULES[r][1]
    cur = tuple(code)
    for _ in range(_MAX_ROUNDS):
        nxt = fn(cur)
        if nxt == cur:
            break
        cur = nxt
    if len(c) > _CACHE_LIMIT:
        c.clear()
    c[code] = cur
    return cur


def apply_ordering(code, ordering):
    """Apply each rule in ``ordering`` once (each run to its own fixed point)."""
    code = tuple(code)
    for r in ordering:
        code = apply_rule(code, r)
    return code


def apply_fixpoint(code, ordering, max_rounds: int = 10):
    """Repeat the whole pipeline ``ordering`` until the code stops changing."""
    code = tuple(code)
    for _ in range(max_rounds):
        nxt = apply_ordering(code, ordering)
        if nxt == code:
            break
        code = nxt
    return code


def ordering_cost(code, ordering) -> int:
    return cost(apply_ordering(code, ordering))


def apply_fixpoint_counted(code, ordering, max_rounds: int = 10):
    """Like ``apply_fixpoint`` but also returns the number of rule executions.

    One execution = one rule run to its own fixed point.  The final round that
    detects "nothing changed" is counted, because a compiler has to run it.
    """
    code = tuple(code)
    rounds = 0
    for _ in range(max_rounds):
        rounds += 1
        nxt = apply_ordering(code, ordering)
        if nxt == code:
            break
        code = nxt
    return code, rounds * len(ordering)


def explore(code, budget: int = N_RULES):
    """Search every sequence of ``budget`` distinct rules.

    With ``budget = 8`` this is all 8! = 40320 orderings.  Returns
    ``(best_cost, best_sequence, worst_cost)``.  Many sequences lead to the
    same code, so the search is a dynamic programme over
    (set of rules already applied, current code).
    """
    code = tuple(code)
    memo = {}

    def go(mask, cur, left):
        if left == 0:
            c = cost(cur)
            return (c, (), c)
        key = (mask, cur)
        hit = memo.get(key)
        if hit is not None:
            return hit
        best_c, best_o, worst_c = None, None, None
        for r in range(N_RULES):
            if mask >> r & 1:
                continue
            sub_c, sub_o, sub_w = go(mask | (1 << r), apply_rule(cur, r), left - 1)
            if best_c is None or sub_c < best_c:
                best_c, best_o = sub_c, (r,) + sub_o
            if worst_c is None or sub_w > worst_c:
                worst_c = sub_w
        res = (best_c, best_o, worst_c)
        memo[key] = res
        return res

    return go(0, code, budget)


def ordering_names(ordering) -> str:
    return " > ".join(RULE_NAMES[r] for r in ordering)
