"""
Lazy column view over the loaded DataFrame.

Names resolve, in order, through
  aliases : name -> column name or callable(view)   (set by "alias" controls, e.g. which weight to use)
  derived : name -> callable(view)                   (preset-defined, e.g. a classification from several columns)
  df      : loaded columns
and are memoised for one Streamlit rerun, so a derived column is computed at most once per interaction.
"""
import re

import numpy as np
import pandas as pd

_IDENT = re.compile(r"(?<![\w.])([A-Za-z_]\w*)\b(?!\s*\()")
_KEYWORDS = {"and", "or", "not", "in", "True", "False", "None", "abs"}


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
        out = pd.eval(expr, local_dict=local, engine="numexpr" if _has_numexpr() else "python")
        out = np.asarray(out)
        if out.shape == ():
            out = np.full(len(self), out.item())
        return out


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
