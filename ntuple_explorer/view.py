"""
Lazy column view over the loaded DataFrame.

Names resolve, in order, through
  aliases : name -> column name or callable(view)   (set by "alias" controls, e.g. which weight to use)
  derived : name -> callable(view)                   (preset-defined, e.g. a classification from several columns)
  df      : loaded columns
and are memoised for one Streamlit rerun, so a derived column is computed at most once per interaction.

Expressions (View.eval) use pandas.eval + numexpr; what it cannot handle (the FUNCS helpers such as max(a, b),
C-style && || !, `x if c else y`) goes through a small numpy evaluator of a whitelisted Python AST (np_eval).
"""
import ast
import io
import operator
import re
import tokenize
from functools import reduce

import numpy as np
import pandas as pd

_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()")
_KEYWORDS = {"and", "or", "not", "in", "if", "else", "True", "False", "None", "abs"}


class View:
    def __init__(self, df, aliases=None, derived=None):
        self.df = df
        self.aliases = dict(aliases or {})
        self.derived = dict(derived or {})
        self._memo = {}
        self._stack = []

    def __len__(self):
        return len(self.df)

    def names(self):
        return list(dict.fromkeys(list(self.df.columns) + list(self.derived) + list(self.aliases)))

    def __contains__(self, name):
        return name in self.aliases or name in self.derived or name in self.df.columns

    def __getitem__(self, name):
        if name in self._memo:
            return self._memo[name]
        if name in self._stack:
            raise RecursionError(f"circular column definition: {' -> '.join(self._stack + [name])}")
        self._stack.append(name)
        try:
            if name in self.aliases:
                src = self.aliases[name]
                val = src(self) if callable(src) else (self.df[src].to_numpy() if src == name else self[src])
            elif name in self.derived:
                val = self.derived[name](self)
            elif name in self.df.columns:
                val = self.df[name].to_numpy()
            else:
                raise KeyError(f"unknown column {name!r}")
        finally:
            self._stack.pop()
        val = np.asarray(val)
        self._memo[name] = val
        return val

    def eval(self, expr):
        """Boolean / numeric numpy expression over view columns (pandas.eval syntax: & | ~, and/or/not, ==, abs)."""
        if expr is None or (isinstance(expr, str) and not expr.strip()):
            return None
        if callable(expr):
            return np.asarray(expr(self))
        names = {n for n in _IDENT.findall(expr) if n not in _KEYWORDS}
        unknown = [n for n in names if n not in self]
        if unknown:
            raise KeyError(f"unknown column(s) in {expr!r}: {', '.join(sorted(unknown))}")
        local = {n: _ne_safe(self[n]) for n in names}
        try:
            out = pd.eval(expr, local_dict=local, engine="numexpr" if _has_numexpr() else "python")
        except Exception:       # noqa: BLE001 - helpers / C-style operators: numpy evaluator (its error if it fails too)
            out = np_eval(expr, self)
        out = np.asarray(out)
        if out.shape == ():
            out = np.full(len(self), out.item())
        return out


# ----------------------------------------------------------------- numpy evaluator
def _count(*conds):
    return reduce(operator.add, (np.asarray(c).astype(bool).astype(np.int32) for c in conds))


def _isin(x, *values):
    return np.isin(x, values)


FUNCS = {
    "max": lambda *a: reduce(np.maximum, a), "min": lambda *a: reduce(np.minimum, a),
    "where": np.where, "count": _count, "isin": _isin,
    "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "log10": np.log10, "sin": np.sin, "cos": np.cos,
    "tan": np.tan, "arctan2": np.arctan2, "hypot": np.hypot, "floor": np.floor, "ceil": np.ceil,
}
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
        ast.GtE: operator.ge}


def _python_syntax(expr):
    """C-style && || ! -> and / or / not; & | -> and / or, so they bind looser than comparisons (as in pandas.eval:
    `a > 0 | b > 0` means `(a > 0) | (b > 0)`)."""
    expr = re.sub(r"&&", " and ", expr)
    expr = re.sub(r"\|\|", " or ", expr)
    expr = re.sub(r"!(?!=)", " not ", expr)
    out = []
    for tok in tokenize.generate_tokens(io.StringIO(expr).readline):
        if tok.type == tokenize.OP and tok.string in ("&", "|"):
            tok = tok._replace(type=tokenize.NAME, string={"&": "and", "|": "or"}[tok.string])
        out.append((tok.type, tok.string))
    return tokenize.untokenize(out)


def np_eval(expr, columns):
    """Evaluate an expression over columns[name] arrays with numpy: arithmetic, comparisons (chained), and / or /
    not (also & | ~ && || !), `a if cond else b`, numbers and the FUNCS helpers. Nothing else is allowed."""
    tree = ast.parse(_python_syntax(expr).strip(), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float, bool)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in ("true", "false"):
                return n.id == "true"
            return columns[n.id]
        if isinstance(n, ast.BoolOp):
            f = np.logical_and if isinstance(n.op, ast.And) else np.logical_or
            return reduce(f, (np.asarray(ev(v)).astype(bool) for v in n.values))
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            if isinstance(n.op, ast.Not):
                return np.logical_not(np.asarray(v).astype(bool))
            if isinstance(n.op, ast.Invert):
                v = np.asarray(v)
                return np.logical_not(v) if v.dtype.kind == "b" else np.invert(v)
            return -v if isinstance(n.op, ast.USub) else +v
        if isinstance(n, ast.BinOp) and type(n.op) in _BIN:
            return _BIN[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.Compare) and all(type(o) in _CMP for o in n.ops):
            left, out = ev(n.left), None
            for op, c in zip(n.ops, n.comparators):
                right = ev(c)
                r = _CMP[type(op)](left, right)
                out = r if out is None else np.logical_and(out, r)
                left = right
            return out
        if isinstance(n, ast.IfExp):
            return np.where(np.asarray(ev(n.test)).astype(bool), ev(n.body), ev(n.orelse))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in FUNCS and not n.keywords:
            return FUNCS[n.func.id](*[ev(a) for a in n.args])
        what = n.func.id + "()" if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) else type(n).__name__
        raise ValueError(f"not allowed in an expression: {what} (functions: {', '.join(FUNCS)})")

    return ev(tree)


_NE = None


def _ne_safe(a):
    # numexpr has no int8 / int16 / uint types
    if a.dtype.kind in "iu" and a.dtype.itemsize < 4:
        return a.astype(np.int32)
    if a.dtype == np.uint32:
        return a.astype(np.int64)
    if a.dtype == np.uint64:
        return a.astype(np.float64)
    return a


def _has_numexpr():
    global _NE
    if _NE is None:
        try:
            import numexpr  # noqa: F401
            _NE = True
        except ImportError:
            _NE = False
    return _NE
