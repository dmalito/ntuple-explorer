"""
Selection controls: spec dict -> Streamlit widget -> boolean mask (or aliases).

Kinds (spec["kind"]):
  alias       options {label: {target: column | callable(view)}}  - switches which column a name points to
  select      options {label: expr | None}                         - one of several cuts (None = no cut)
  multiselect column, labels {value: label} (optional), default [values]  - keep the chosen values; two values ->
              a 3-way switch (all | a | b); on the page (place="main") clickable chips instead of the box
  checkboxes  options {label: expr}, default [labels], combine "or"|"and" - e.g. channels
  range       column, limits (lo, hi) (optional), default (lo, hi), step - slider; at a limit = open-ended
  toggle      expr, default bool, expr_off (optional)                - cut on / off
  at_least    columns [k-th largest value, k = 0..], label, what, default (N, X) - "at least N objects with
              value >= X": N (any = no cut) and a threshold slider; at its lower end only the count is required
  expr        default str                                            - free expression
Expressions are strings over view columns (View.eval) or callables(view) -> bool array.
"""
import numpy as np
import streamlit as st

MAX_CATEGORIES = 20
ALL = "all"             # first option of the 3-way switch of two-valued columns (no cut)
# default range of sliders without preset limits (app setting): "quantiles" = 0.1-99.9 % quantiles; "snapped" = the
# same, but an end moves to the true min / max when that is within SNAP of the range (hard thresholds such as
# pT > 20 GeV stay reachable, far tails do not stretch the slider); "full" = min-max
RANGE_MODES = ("snapped", "quantiles", "full")
RANGE_MODE = "snapped"
SNAP = 0.05


# ----------------------------------------------------------------- column statistics (cached per dataset)
def stats(view, name, cache):
    """Column summary (kind cat / num, values or quantiles, min / max), cached in `cache`. A cache with a
    rows(name) method restricts it to those rows (e.g. a matched dataset: only the events of the column's level)."""
    if name in cache:
        return cache[name]
    a = view[name]
    rows = cache.rows(name) if hasattr(cache, "rows") else None
    if rows is not None:
        a = a[rows]
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
        s.update(kind="num", lo=0.0, hi=1.0, min=0.0, max=1.0, integer=False, empty=True)
    cache[name] = s
    return s


def slider_limits(s, mode=None):
    """(lo, hi) of a numeric column's slider from its stats() and the range mode (RANGE_MODE by default)."""
    mode = mode or RANGE_MODE
    if mode == "full":
        return s["min"], s["max"]
    lo, hi = s["lo"], s["hi"]
    if mode == "snapped":
        span = (hi - lo) or 1.0
        lo = s["min"] if lo - s["min"] <= SNAP * span else lo
        hi = s["max"] if s["max"] - hi <= SNAP * span else hi
    return lo, hi


def auto_spec(view, name, cache, labels=None, int_slider=True):
    """Control spec for a column from its values (generic mode / extra cuts): a multiselect for labelled columns and
    flags (≤ 2 values), a range slider for numbers; integer-like columns with few values (counts such as n jets) get a
    slider when int_slider, else a multiselect."""
    s = stats(view, name, cache)
    if s["kind"] == "cat" and (labels or len(s["values"]) <= 2 or not int_slider):
        return dict(kind="multiselect", column=name, label=name, **({"labels": labels} if labels else {}))
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
        if len(values) == 2:
            # two values (0/1, -1/1, two classes): a 3-way switch  all | first | second. A horizontal radio, not
            # st.segmented_control: AppTest cannot drive single-choice button groups
            opts = [ALL] + [names[v] for v in values]
            pick = container.radio(label, opts, index=opts.index(names[default[0]]) if len(default) == 1 else 0,
                                   horizontal=True, key=f"{key}__sw", help=help_)
            chosen = values if pick == ALL else [v for v in values if names[v] == pick]
        elif spec.get("place") == "main":
            # on the page: clickable chips (the multiselect box stays in the sidebar)
            chosen = container.pills(label, values, selection_mode="multi", default=default,
                                     format_func=lambda v: names[v], key=f"{key}__pills", help=help_) or []
        else:
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
            lo, hi = slider_limits(s)
            integer = s.get("integer", False)
        lo, hi = spec.get("limits", (lo, hi))
        lo, hi, step = _slider_grid(lo, hi, spec.get("step"), integer)
        d = spec.get("default", (lo, hi))
        d = (d[0] if d[0] is not None else lo, d[1] if d[1] is not None else hi)
        a, b = _carry_slider(container, label, lo, hi, d, step, key, help_=help_)
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

    if kind == "at_least":
        cols = spec["columns"]
        noun, what = label, spec.get("what", "value")
        n0, x0 = spec.get("default", (0, None))
        opts = ["any"] + [str(i) for i in range(1, len(cols) + 1)]
        n = container.radio(f"{noun}: at least N", opts, index=min(int(n0), len(cols)), horizontal=True,
                            key=f"{key}__n", help=help_)
        n = 0 if n == "any" else int(n)
        sts = [stats(view, c, cache) for c in cols]
        lims = [lh for lh in map(_limits, sts) if lh is not None] or [(0.0, 1.0)]
        lo, hi = min(lh[0] for lh in lims), max(lh[1] for lh in lims)
        integer = all(s_["kind"] == "cat" for s_ in sts if not s_.get("empty"))
        lo, hi, step = _slider_grid(lo, hi, spec.get("step"), integer)
        x = _carry_slider(container, f"{what} ≥ (for those N)", lo, hi, x0 if x0 is not None else lo, step,
                          f"{key}__x", single=True)
        if n:
            v = view[cols[n - 1]]
            if x > lo:
                out["mask"] = v >= x
                out["desc"] = f"at least {n} {noun} with {what} >= {_fmt(x)}"
            else:
                out["mask"] = np.isfinite(v) if v.dtype.kind == "f" else np.ones(len(v), bool)
                out["desc"] = f"at least {n} {noun}"
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


def _limits(s):
    """(lo, hi) slider limits from stats(), None for a column without finite values."""
    if s.get("empty"):
        return None
    if s["kind"] == "cat":
        return (min(s["values"]), max(s["values"])) if s["values"] else None
    return slider_limits(s)


def _slider_grid(lo, hi, step, integer):
    """(lo, hi, step) on a nice grid: integers for integer data, else lo / hi rounded outwards to the step."""
    if integer or (step is not None and float(step).is_integer()):
        lo, hi, step = int(np.floor(lo)), int(np.ceil(hi)), int(step or 1)
    else:
        lo, hi = float(lo), float(hi)
        step = float(step or _nice_step(lo, hi))
        lo, hi = float(round(np.floor(lo / step) * step, 10)), float(round(np.ceil(hi / step) * step, 10))
    return lo, (hi if hi > lo else lo + step), step


def _carry_slider(container, label, lo, hi, default, step, key, single=False, help_=None):
    """Slider whose value survives a change of its limits (a new widget for Streamlit): a handle at an old limit
    goes to the new one, other values are clamped. default: value or (lo, hi) pair."""
    cast = type(lo)
    cur, old = st.session_state.get(key), st.session_state.get(f"{key}__lim")
    if cur is not None and old is not None and tuple(old) != (lo, hi):
        vals = [cur] if single else list(cur)
        vals = [lo if v <= old[0] else hi if v >= old[1] else v for v in vals]
        default = vals[0] if single else tuple(vals)
    clamp = (lambda v: cast(max(lo, min(hi, v))))
    default = clamp(default) if single else (clamp(default[0]), clamp(default[1]))
    v = container.slider(label, min_value=lo, max_value=hi, value=default, step=step, key=key, help=help_)
    st.session_state[f"{key}__lim"] = (lo, hi)
    return v


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
