"""Three-address code (TAC): instruction model, parser and printer.

Text format (one instruction per line, '#' starts a comment):

    x = 5              constant assignment
    y = x              copy
    t1 = a * 4         binary operation  (+ - * / % << >> &)
    L1:                label
    goto L1            unconditional jump
    if a < b goto L1   conditional jump   (< <= > >= == !=)
    print x            observable output

Operands are either integer constants (Python ``int``) or variable names
(Python ``str``).  Names of the form ``t<digits>`` are *temporaries*: by
convention they are local to a straight-line segment (they are never read
after a label or a jump), which is what lets dead-code elimination remove
unused temporaries safely.

Semantics follow Python integers: ``/`` is floor division, ``%`` is the
floor modulo, ``>>`` is an arithmetic shift.
"""
from __future__ import annotations

import re
from collections import namedtuple

# kind: 'asg'   dst = a
#       'bin'   dst = a op b
#       'label' target:
#       'goto'  goto target
#       'if'    if a op b goto target
#       'print' print a
Ins = namedtuple("Ins", "kind dst op a b target")

BIN_OPS = ("+", "-", "*", "/", "%", "<<", ">>", "&")
REL_OPS = ("<", "<=", ">", ">=", "==", "!=")
COMMUTATIVE = frozenset({"+", "*", "&"})

_TEMP_RE = re.compile(r"t\d+")


# ----------------------------------------------------------------- builders
def Asg(dst, a):
    return Ins("asg", dst, None, a, None, None)


def Bin(dst, op, a, b):
    return Ins("bin", dst, op, a, b, None)


def Label(name):
    return Ins("label", None, None, None, None, name)


def Goto(name):
    return Ins("goto", None, None, None, None, name)


def If(op, a, b, name):
    return Ins("if", None, op, a, b, name)


def Print(a):
    return Ins("print", None, None, a, None, None)


# ------------------------------------------------------------------ helpers
def is_const(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def is_var(x) -> bool:
    return isinstance(x, str)


def is_temp(name: str) -> bool:
    return _TEMP_RE.fullmatch(name) is not None


def is_pow2(x) -> bool:
    return is_const(x) and x >= 2 and (x & (x - 1)) == 0


def uses(ins: Ins):
    """Variables read by an instruction."""
    k = ins.kind
    if k == "asg" or k == "print":
        return (ins.a,) if is_var(ins.a) else ()
    if k == "bin" or k == "if":
        return tuple(x for x in (ins.a, ins.b) if is_var(x))
    return ()


def defs(ins: Ins):
    """Variable written by an instruction (or None)."""
    return ins.dst if ins.kind in ("asg", "bin") else None


def is_jump(ins: Ins) -> bool:
    return ins.kind in ("goto", "if")


def substitute(ins: Ins, old: str, new) -> Ins:
    """Replace every *read* of variable ``old`` by operand ``new``."""
    k = ins.kind
    if k == "asg" or k == "print":
        return ins._replace(a=new) if ins.a == old else ins
    if k == "bin" or k == "if":
        a = new if ins.a == old else ins.a
        b = new if ins.b == old else ins.b
        return ins._replace(a=a, b=b)
    return ins


def eval_op(op: str, a: int, b: int) -> int:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a // b          # ZeroDivisionError on b == 0
    if op == "%":
        return a % b
    if op == "<<":
        if b < 0 or b > 64:
            raise ValueError("bad shift")
        return a << b
    if op == ">>":
        if b < 0 or b > 64:
            raise ValueError("bad shift")
        return a >> b
    if op == "&":
        return a & b
    raise ValueError(f"unknown operator {op}")


def eval_rel(op: str, a: int, b: int) -> bool:
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    raise ValueError(f"unknown relation {op}")


# ------------------------------------------------------------ segmentation
def segments(code):
    """Split code into straight-line segments.

    A segment ends after a jump and a new one starts at every label.
    Returns a list of (start, end) half-open index ranges.
    """
    out = []
    start = 0
    for i, ins in enumerate(code):
        if ins.kind == "label" and i > start:
            out.append((start, i))
            start = i
        if ins.kind in ("goto", "if"):
            out.append((start, i + 1))
            start = i + 1
    if start < len(code):
        out.append((start, len(code)))
    return out


# ------------------------------------------------------------ print / parse
def fmt_operand(x) -> str:
    return str(x)


def fmt(ins: Ins) -> str:
    k = ins.kind
    if k == "asg":
        return f"{ins.dst} = {fmt_operand(ins.a)}"
    if k == "bin":
        return f"{ins.dst} = {fmt_operand(ins.a)} {ins.op} {fmt_operand(ins.b)}"
    if k == "label":
        return f"{ins.target}:"
    if k == "goto":
        return f"goto {ins.target}"
    if k == "if":
        return f"if {fmt_operand(ins.a)} {ins.op} {fmt_operand(ins.b)} goto {ins.target}"
    if k == "print":
        return f"print {fmt_operand(ins.a)}"
    raise ValueError(k)


def to_text(code) -> str:
    lines = []
    for ins in code:
        s = fmt(ins)
        lines.append(s if ins.kind == "label" else "    " + s)
    return "\n".join(lines) + ("\n" if lines else "")


_NAME = r"[A-Za-z_]\w*"
_NUM = r"-?\d+"
_OPERAND = rf"(?:{_NAME}|{_NUM})"
_RE_LABEL = re.compile(rf"^({_NAME}):$")
_RE_GOTO = re.compile(rf"^goto\s+({_NAME})$")
_RE_IF = re.compile(rf"^if\s+({_OPERAND})\s*(<=|>=|==|!=|<|>)\s*({_OPERAND})\s+goto\s+({_NAME})$")
_RE_PRINT = re.compile(rf"^print\s+({_OPERAND})$")
_RE_BIN = re.compile(rf"^({_NAME})\s*=\s*({_OPERAND})\s+(<<|>>|\+|-|\*|/|%|&)\s+({_OPERAND})$")
_RE_ASG = re.compile(rf"^({_NAME})\s*=\s*({_OPERAND})$")


def _operand(tok: str):
    return int(tok) if re.fullmatch(_NUM, tok) else tok


def parse(text: str):
    """Parse TAC text into a tuple of instructions."""
    code = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _RE_LABEL.match(line)
        if m:
            code.append(Label(m.group(1)))
            continue
        m = _RE_GOTO.match(line)
        if m:
            code.append(Goto(m.group(1)))
            continue
        m = _RE_IF.match(line)
        if m:
            code.append(If(m.group(2), _operand(m.group(1)), _operand(m.group(3)), m.group(4)))
            continue
        m = _RE_PRINT.match(line)
        if m:
            code.append(Print(_operand(m.group(1))))
            continue
        m = _RE_BIN.match(line)
        if m:
            code.append(Bin(m.group(1), m.group(3), _operand(m.group(2)), _operand(m.group(4))))
            continue
        m = _RE_ASG.match(line)
        if m:
            code.append(Asg(m.group(1), _operand(m.group(2))))
            continue
        raise ValueError(f"line {lineno}: cannot parse {raw!r}")
    return tuple(code)


def variables(code):
    """All variable names mentioned in the code."""
    names = set()
    for ins in code:
        names.update(uses(ins))
        d = defs(ins)
        if d is not None:
            names.add(d)
    return names
