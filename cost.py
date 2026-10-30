"""Static cost model.

The cost of a program is the sum of per-instruction weights.  The weights
mimic a simple machine in which multiplication and division are much more
expensive than additions and shifts.
"""
from __future__ import annotations

OP_COST = {
    "+": 2,
    "-": 2,
    "&": 2,
    "<<": 1,
    ">>": 1,
    "*": 4,
    "/": 8,
    "%": 8,
}

KIND_COST = {
    "asg": 1,     # constant load or register copy
    "label": 0,
    "goto": 1,
    "if": 2,
    "print": 1,
}


def ins_cost(ins) -> int:
    if ins.kind == "bin":
        return OP_COST[ins.op]
    return KIND_COST[ins.kind]


def cost(code) -> int:
    return sum(ins_cost(i) for i in code)
