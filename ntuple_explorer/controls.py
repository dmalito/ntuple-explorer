"""
Selection controls: spec dict -> Streamlit widget -> boolean mask (or aliases).

Kinds (spec["kind"]):
  alias       options {label: {target: column | callable(view)}}  - switches which column a name points to
  select      options {label: expr | None}                         - one of several cuts (None = no cut)
  multiselect column, labels {value: label} (optional), default [values]  - keep the chosen values
  checkboxes  options {label: expr}, default [labels], combine "or"|"and" - e.g. channels
  range       column, limits (lo, hi) (optional), default (lo, hi), step - slider; at a limit = open-ended
  toggle      expr, default bool, expr_off (optional)                - cut on / off
  expr        default str                                            - free pandas.eval expression
Expressions are pandas.eval strings over view columns (& | ~ and or not == < abs) or callables(view) -> bool array.
"""
import numpy as np
import streamlit as st

MAX_CATEGORIES = 20


# ----------------------------------------------------------------- column statistics (cached per dataset)
def stats(view, name, cache):
    if name in cache:
        return cache[name]
    a = view[name]
    s = dict(dtype=a.dtype.kind)
    finite = a[np.isfinite(a)] if a.dtype.kind == "f" else a
    if a.dtype.kind in "biu" or (a.dtype.kind == "f" and len(finite) and np.all(np.mod(finite[:100000], 1) == 0)):
        u, c = np.unique(finite, return_counts=True)
        if len(u) <= MAX_CATEGORIES:
            s.update(kind="cat", values=u.tolist(), counts=c.tolist())
            cache[name] = s
            return s
    if len(finite):
        lo, hi = np.quantile(finite, [0.001, 0.999]) if len(finite) > 1000 else (finite.min(), finite.max())
        s.update(kind="num", lo=float(lo), hi=float(hi), min=float(finite.min()), max=float(finite.max()),
                 integer=a.dtype.kind in "iu")
    else:
        s.update(kind="num", lo=0.0, hi=1.0, min=0.0, max=1.0, integer=False)
    cache[name] = s
    return s


def auto_spec(view, name, cache):
    """Control spec for a column from its values (generic mode / extra cuts)."""
    s = stats(view, name, cache)
    if s["kind"] == "cat":
        return dict(kind="multiselect", column=name, label=name)
    return dict(kind="range", column=name, label=name)


def _fmt(v):
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


def _and(a, b):
    return b if a is None else (a if b is None else a & b)


# ----------------------------------------------------------------- rendering
def render(spec, view, key, cache, container=st):
    """Draw the widget; return dict(label, mask (bool array | None), desc (str | None), column, aliases)."""
    kind = spec["kind"]
    label = spec.get("label") or spec.get("column") or kind
    help_ = spec.get("help")
    out = dict(label=label, mask=None, desc=None, column=spec.get("column"), aliases=None, kind=kind)

    if kind == "alias":
        opts = list(spec["options"])
        default = spec.get("default", opts[0])
        w = container.radio if len(opts) <= 4 and not spec.get("dropdown") else container.selectbox
        choice = w(label, opts, index=opts.index(default), key=key, help=help_,
                   **({"horizontal": True} if w is container.radio and spec.get("horizontal", True) else {}))
        out["aliases"] = spec["options"][choice]
        out["desc"] = f"[{label}: {choice}]"
        out["choice"] = choice
        return out

    if kind == "select":
        opts = list(spec["options"])
        default = spec.get("default", opts[0])
        w = container.radio if len(opts) <= 4 and not spec.get("dropdown") else container.selectbox
        choice = w(label, opts, index=opts.index(default), key=key, help=help_)
        e = spec["options"][choice]
        if e is not None:
            out["mask"] = view.eval(e).astype(bool)
            out["desc"] = e if isinstance(e, str) else f"[{label}: {choice}]"
        return out

    if kind == "multiselect":
        col = spec["column"]
        s = stats(view, col, cache)
        values = spec.get("values") or (s["values"] if s["kind"] == "cat" else [])
        labels = spec.get("labels") or {}
        if isinstance(labels, (list, tuple)):
            labels = dict(enumerate(labels))
        names = {v: str(labels.get(v, _fmt(v))) for v in values}
        default = spec.get("default")
        default = values if default is None else [v for v in values if v in default or names[v] in default]
        chosen = container.multiselect(label, values, default=default, format_func=lambda v: names[v], key=key,
                                       help=help_)
        if len(chosen) < len(values):
            out["mask"] = np.isin(view[col], chosen)
            out["desc"] = f"{col} in [{', '.join(names[v] for v in chosen)}]"
        return out

    if kind == "checkboxes":
        opts = spec["options"]
        default = spec.get("default", list(opts))
        combine = spec.get("combine", "or")
        container.markdown(f"**{label}**", help=help_)
        cols = container.columns(min(len(opts), 3))
        on = [lab for i, lab in enumerate(opts)
              if cols[i % len(cols)].checkbox(lab, value=lab in default, key=f"{key}__{i}")]
        if len(on) < len(opts) or combine == "and":
            m = None
            for lab in on:
                a = view.eval(opts[lab]).astype(bool)
                m = a if m is None else (m | a if combine == "or" else m & a)
            if m is None:
                m = np.zeros(len(view), bool) if combine == "or" else None
            out["mask"] = m
            j = " || " if combine == "or" else " && "
            out["desc"] = "(" + j.join(str(opts[lab]) for lab in on) + ")" if on else "false"
        return out

    if kind == "range":
        col = spec["column"]
        s = stats(view, col, cache)
        if s["kind"] == "cat":
            lo, hi = min(s["values"]), max(s["values"])
            integer = True
        else:
            lo, hi = s["lo"], s["hi"]
            integer = s.get("integer", False)
        lo, hi = spec.get("limits", (lo, hi))
        step = spec.get("step") or (1 if integer else None)
        if integer or (step is not None and float(step).is_integer()):
            lo, hi, step = int(np.floor(lo)), int(np.ceil(hi)), int(step or 1)
        else:
            lo, hi = float(lo), float(hi)
            step = float(step or _nice_step(lo, hi))
            lo, hi = np.floor(lo / step) * step, np.ceil(hi / step) * step
            lo, hi = float(round(lo, 10)), float(round(hi, 10))
        if hi <= lo:
            hi = lo + step
        d = spec.get("default", (lo, hi))
        d = (max(lo, min(hi, d[0] if d[0] is not None else lo)), max(lo, min(hi, d[1] if d[1] is not None else hi)))
        if isinstance(lo, int):
            d = (int(d[0]), int(d[1]))
        else:
            d = (float(d[0]), float(d[1]))
        a, b = container.slider(label, min_value=lo, max_value=hi, value=d, step=step, key=key, help=help_)
        # a handle at the slider limit leaves that side open (values beyond the limit are kept)
        x = view[col]
        m, parts = None, []
        if a > lo:
            m = _and(m, x >= a)
            parts.append(f"{col} >= {_fmt(a)}")
        if b < hi:
            m = _and(m, x <= b)
            parts.append(f"{col} <= {_fmt(b)}")
        out["mask"] = m
        out["desc"] = " && ".join(parts) or None
        return out

    if kind == "toggle":
        on = container.toggle(label, value=spec.get("default", True), key=key, help=help_)
        e = spec["expr"] if on else spec.get("expr_off")
        if e is not None:
            out["mask"] = view.eval(e).astype(bool)
            out["desc"] = e if isinstance(e, str) else f"[{label}]"
        return out

    if kind == "expr":
        txt = container.text_input(label, value=spec.get("default", ""), key=key,
                                   help=help_ or "pandas.eval syntax over loaded columns, e.g. "
                                                 "`(nsm == 1) & (abs(q_lep) == 1)`")
        if txt.strip():
            try:
                out["mask"] = view.eval(txt).astype(bool)
                out["desc"] = txt
            except Exception as e:      # noqa: BLE001 - shown to the user, selection ignored
                container.error(f"{type(e).__name__}: {e}")
        return out

    raise ValueError(f"unknown control kind {kind!r}")


def _nice_step(lo, hi):
    span = max(hi - lo, 1e-12) / 200
    p = 10 ** np.floor(np.log10(span))
    for m in (1, 2, 5, 10):
        if span <= m * p:
            return m * p
    return 10 * p


def combine(results, skip=None):
    """AND of the masks of all results (except index `skip`); None if nothing cuts."""
    m = None
    for i, r in enumerate(results):
        if i != skip and r["mask"] is not None:
            m = _and(m, r["mask"])
    return m
