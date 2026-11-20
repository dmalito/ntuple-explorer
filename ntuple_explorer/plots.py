"""Weighted histograms (plotly) and yield tables on masked numpy arrays."""
from collections import OrderedDict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# categorical colours in a fixed order, never cycled; beyond 8 -> "other" (grey)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#9a9892"
REFERENCE = "#8a8984"        # "before selection" outline
SELECTED = SERIES[0]
SEQUENTIAL = [[0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#1c5cab"], [1, "#0d366b"]]


def _sum(w, m=None):
    if m is None:
        return float(w.sum()), float((w.astype(np.float64) ** 2).sum())
    ww = w[m].astype(np.float64)
    return float(ww.sum()), float((ww ** 2).sum())


def fmt_label(v, labels):
    if labels is not None and v in labels:
        return str(labels[v])
    v = float(v)
    return str(int(v)) if v.is_integer() else f"{v:g}"


# ----------------------------------------------------------------- binning
def binning(x, spec, st_):
    """-> ("cat", values, names) or ("num", edges)."""
    labels = spec.get("labels")
    if isinstance(labels, (list, tuple)):
        labels = dict(enumerate(labels))
    if spec.get("bins") is not None:
        b = spec["bins"]
        if isinstance(b, tuple) and len(b) == 3:
            return ("num", np.linspace(b[0], b[1], int(b[2]) + 1))
        return ("num", np.asarray(b, dtype=float))
    if st_["kind"] == "cat" or labels:
        values = spec.get("values") or st_.get("values") or sorted(labels)
        return ("cat", values, [fmt_label(v, labels) for v in values])
    lo, hi = st_["lo"], st_["hi"]
    if st_.get("integer") and hi - lo <= 60:
        return ("num", np.arange(np.floor(lo) - 0.5, np.ceil(hi) + 1.5))
    if hi <= lo:
        hi = lo + 1
    return ("num", np.linspace(lo, hi, int(spec.get("nbins", 50)) + 1))


def bin_index(x, binspec, overflow=True):
    """-> (bin index per entry, in-range mask, n bins). Lookup table for integer categories, arithmetic for uniform
    edges, searchsorted only for variable edges / non-integer categories."""
    if binspec[0] == "cat":
        values = np.asarray(binspec[1], dtype=float)
        n = len(values)
        if n and np.all(values == np.round(values)) and values.max() - values.min() <= 100000:
            v0 = int(values.min())
            lut = np.full(int(values.max()) - v0 + 1, -1, dtype=np.int64)
            lut[values.astype(np.int64) - v0] = np.arange(n)
            xi = x.astype(np.int64) if x.dtype.kind in "iub" else np.where(np.isfinite(x), x, -2 ** 62).astype(np.int64)
            off = xi - v0
            inr = (off >= 0) & (off < len(lut))
            if x.dtype.kind == "f":
                inr &= (x == np.round(x))
            idx = np.where(inr, lut[np.where(inr, off, 0)], -1)
            return idx, idx >= 0, n
        idx = np.clip(np.searchsorted(values, x), 0, max(n - 1, 0))
        return idx, values[idx] == x, n
    e = np.asarray(binspec[1], dtype=float)
    n = len(e) - 1
    xf = x.astype(np.float64, copy=False)
    if overflow:
        xf = np.clip(xf, e[0], np.nextafter(e[-1], -np.inf))
    d = np.diff(e)
    if np.allclose(d, d[0], rtol=1e-9, atol=0):
        idx = np.floor((xf - e[0]) / d[0]).astype(np.int64)
        idx = np.minimum(idx, n - 1) if overflow else idx
    else:
        idx = np.searchsorted(e, xf, side="right") - 1
    ok = (idx >= 0) & (idx < n)
    if not overflow:
        ok &= np.isfinite(xf) & (xf <= e[-1])
    return idx, ok, n


def _fill(x, w, binspec, overflow=True):
    """sum w and sum w^2 per bin."""
    idx, ok, n = bin_index(x, binspec, overflow)
    w = w.astype(np.float64)
    s = np.bincount(idx[ok], weights=w[ok], minlength=n)[:n]
    s2 = np.bincount(idx[ok], weights=w[ok] ** 2, minlength=n)[:n]
    return s, s2  # used by tests / external callers


def _order_for_cat(binspec):
    """Categorical values must be sorted for searchsorted; keep display order separately."""
    vals = np.asarray(binspec[1], dtype=float)
    order = np.argsort(vals, kind="stable")
    return ("cat", vals[order].tolist(), [binspec[2][i] for i in order])


def _ink(colour):
    """Text colour on top of a fill: white on dark fills, near-black on light ones."""
    h = colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.62 else "#1f1f1f"


def segment_colours(keys, names, totals, colors=None):
    """-> ({key: colour}, [keys folded into "other"]). Colour follows the class, never the current cuts: explicit
    colors[label] first, then the fixed SERIES order (display order; by Σw over all loaded entries when there are
    more classes than colours, the smallest then fold into "other")."""
    colors = colors or {}
    out = {k: colors[str(names[k])] for k in keys if str(names[k]) in colors}
    rest = [k for k in keys if k not in out and totals.get(k, 0) != 0]
    free = [c for c in SERIES if c not in out.values()]
    if len(rest) > len(free):
        rest = sorted(rest, key=lambda k: -totals.get(k, 0))
        n = len(free) - 1
    else:
        n = len(rest)
    out.update(zip(rest[:n], free))
    return out, rest[n:]


# ----------------------------------------------------------------- 1D
_BASE_CACHE = OrderedDict()


def weights(w):
    """(float64 weights, squared weights): compute once per rerun and pass to hist1d."""
    w64 = np.asarray(w, dtype=np.float64)
    return w64, w64 * w64


def hist1d(x, w, base_mask, sel_mask, binspec, title, xlabel=None, normalise=False, logy=False, ratio=True,
           stack=None, overflow=True, show_base=True, base_key=None):
    """Before (base_mask, grey step) vs after (sel_mask, filled) the selection; optional stack by category of the
    selected events (stack = (codes array, {code: name})) and a selected / before ratio panel.

    w: weight array or the (w64, w2) tuple of weights(). base_key: hashable description of (data, column, weight,
    binning) under which the "before" histogram is cached across reruns (it does not depend on the cuts)."""
    if binspec[0] == "cat":
        binspec = _order_for_cat(binspec)
    rows = 2 if ratio else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26] if ratio else [1],
                        vertical_spacing=0.04)
    idx, ok, n = bin_index(x, binspec, overflow)
    w64, w2 = w if isinstance(w, tuple) else weights(w)

    def fill(m):
        k = ok if m is None else (ok & m)
        if k.mean() > 0.3:      # dense: weighted bincount over all entries (no boolean gathers)
            i = np.where(k, idx, n)
            return (np.bincount(i, weights=w64 * k, minlength=n + 1)[:n],
                    np.bincount(i, weights=w2 * k, minlength=n + 1)[:n])
        i = idx[k]
        return (np.bincount(i, weights=w64[k], minlength=n)[:n], np.bincount(i, weights=w2[k], minlength=n)[:n])

    bk = (base_key, repr(binspec), overflow) if base_key is not None and base_mask is None else None
    if bk is not None and bk in _BASE_CACHE:
        hb, hb2 = _BASE_CACHE[bk]
        _BASE_CACHE.move_to_end(bk)
    else:
        hb, hb2 = fill(base_mask)
        if bk is not None:
            _BASE_CACHE[bk] = (hb, hb2)
            while len(_BASE_CACHE) > 256:
                _BASE_CACHE.popitem(last=False)
    hs, hs2 = fill(sel_mask)
    nb_tot, ns_tot = hb.sum(), hs.sum()
    scale_b = 1 / nb_tot if normalise and nb_tot else 1
    scale_s = 1 / ns_tot if normalise and ns_tot else 1

    if binspec[0] == "cat":
        xc = binspec[2]
        width = None
    else:
        e = binspec[1]
        xc = 0.5 * (e[1:] + e[:-1])
        width = np.diff(e)

    hover = "%{x}<br>%{customdata[0]}: %{y:.4g} ± %{customdata[1]:.2g}<extra></extra>"
    if stack is not None:
        codes, names = stack[0], stack[1]
        keys = list(names)
        ci, cok, nc = bin_index(codes, ("cat", sorted(keys), None))
        order = sorted(keys)
        # colour follows the class: preset colours by label, else fixed order by Σw in the whole loaded sample
        tot_all = np.bincount(ci[cok], weights=w64[cok], minlength=nc)[:nc]
        totals = dict(zip(order, tot_all))
        colour, _ = segment_colours(order, names, totals, stack[2] if len(stack) > 2 else None)
        ranked = [order[j] for j in np.argsort(-tot_all, kind="stable") if tot_all[j] != 0]
        k = ok & cok & sel_mask
        flat = ci[k] * n + idx[k]
        H = np.bincount(flat, weights=w64[k], minlength=nc * n)[:nc * n].reshape(nc, n)
        H2 = np.bincount(flat, weights=w2[k], minlength=nc * n)[:nc * n].reshape(nc, n)
        pos = {c: j for j, c in enumerate(order)}
        rest = [c for c in ranked if c not in colour]
        for c in [c for c in ranked if c in colour]:
            h, h2 = H[pos[c]], H2[pos[c]]
            if not h.any():
                continue
            fig.add_trace(go.Bar(x=xc, y=h * scale_s, width=width, name=str(names[c]), marker_color=colour[c],
                                 marker_line=dict(width=1, color="rgba(255,255,255,0.9)"),
                                 customdata=np.c_[np.full(len(h), str(names[c])), np.sqrt(h2) * scale_s],
                                 hovertemplate=hover), row=1, col=1)
        if rest:
            h = sum(H[pos[c]] for c in rest)
            h2 = sum(H2[pos[c]] for c in rest)
            if h.any():
                fig.add_trace(go.Bar(x=xc, y=h * scale_s, width=width, name=f"other ({len(rest)} classes)",
                                     marker_color=OTHER, marker_line=dict(width=1, color="rgba(255,255,255,0.9)"),
                                     customdata=np.c_[np.full(len(h), "other"), np.sqrt(h2) * scale_s],
                                     hovertemplate=hover), row=1, col=1)
        fig.update_layout(barmode="stack")
    else:
        fig.add_trace(go.Bar(x=xc, y=hs * scale_s, width=width, name="selected", marker_color=SELECTED, opacity=0.85,
                             marker_line=dict(width=0),
                             error_y=dict(type="data", array=np.sqrt(hs2) * scale_s, visible=True, thickness=1,
                                          width=0, color="#184f95"),
                             customdata=np.c_[np.full(len(hs), "selected"), np.sqrt(hs2) * scale_s],
                             hovertemplate=hover), row=1, col=1)
    if show_base:
        line = dict(color=REFERENCE, width=2, shape="hvh" if binspec[0] == "num" else "hvh")
        fig.add_trace(go.Scatter(x=xc, y=hb * scale_b, mode="lines", line=line, name="before selection",
                                 customdata=np.c_[np.full(len(hb), "before"), np.sqrt(hb2) * scale_b],
                                 hovertemplate=hover), row=1, col=1)

    if ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(hb != 0, hs / hb, np.nan)
            # binomial-ish error for a subset (weighted): sqrt(var_s (1-2r) + r^2 var_b) / n_b, clipped >= 0
            var = np.clip(hs2 * (1 - 2 * r) + r ** 2 * hb2, 0, None)
            er = np.where(hb != 0, np.sqrt(var) / np.abs(hb), np.nan)
        fig.add_trace(go.Scatter(x=xc, y=r, mode="markers", marker=dict(color=SELECTED, size=7),
                                 error_y=dict(type="data", array=er, thickness=1, width=0, color=SELECTED),
                                 name="selected / before", showlegend=False,
                                 hovertemplate="%{x}<br>fraction kept: %{y:.4f}<extra></extra>"), row=2, col=1)
        fig.update_yaxes(title_text="kept", row=2, col=1, rangemode="tozero", gridcolor="rgba(128,128,128,0.15)")

    fig.update_yaxes(title_text="normalised" if normalise else "Σw", type="log" if logy else "linear", row=1, col=1,
                     gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_xaxes(title_text=xlabel or title, row=rows, col=1, showgrid=False)
    if binspec[0] == "cat":
        fig.update_xaxes(type="category", categoryorder="array", categoryarray=binspec[2])
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), height=380 if ratio else 300,
                      margin=dict(l=10, r=10, t=40, b=10), bargap=0.03 if binspec[0] == "num" else 0.25,
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
                      hovermode="closest")
    return fig


# ----------------------------------------------------------------- composition
NORM_TITLES = {"row": "% of the row (weighted)", "total": "% of the sample (weighted)", "yields": "Σw"}


def composition(rows, row_names, segs, seg_names, w, mask, norm="row", horizontal=True, facet=None, title="",
                colors=None, all_row=None, label_min=6.0, row_title=None, legend_title=None):
    """Stacked bars of one category (segments) inside each class of another (rows), e.g. decay chain x matching
    status. rows / segs: code arrays; row_names / seg_names: {code: label} in display order.
    norm: "row" (each bar adds up to 100 %), "total" (% of the panel), "yields" (Σw).
    facet: None or (codes, {code: label}) -> one panel for all entries + one per facet class.
    -> (plotly figure, table DataFrame)."""
    w64 = w[0] if isinstance(w, tuple) else np.asarray(w, dtype=np.float64)
    all_row = (norm == "row") if all_row is None else all_row
    rk, sk = list(row_names), list(seg_names)
    rsort, ssort = sorted(rk), sorted(sk)
    ri, rok, nr = bin_index(rows, ("cat", rsort, None))
    si, sok, ns = bin_index(segs, ("cat", ssort, None))
    ok = rok & sok
    ri, si = np.where(ok, ri, 0), np.where(ok, si, 0)
    tot_all = np.bincount(si[sok], weights=w64[sok], minlength=ns)[:ns]
    colour, folded = segment_colours(ssort, seg_names, dict(zip(ssort, tot_all)), colors)
    R = [rsort.index(k) for k in rk]
    seg_cols = [k for k in sk if k in colour]
    S = [ssort.index(k) for k in seg_cols]
    seg_labels = [str(seg_names[k]) for k in seg_cols]
    seg_colours = [colour[k] for k in seg_cols]
    if folded:
        seg_labels.append(f"other ({len(folded)} classes)")
        seg_colours.append(OTHER)

    panels = [("all", mask)]
    if facet is not None:
        fc, fn = facet
        panels += [(str(lab), mask & (fc == v)) for v, lab in fn.items()]

    def fill(m):
        k = ok & m
        flat = ri[k] * ns + si[k]
        Wt = np.bincount(flat, weights=w64[k], minlength=nr * ns)[:nr * ns].reshape(nr, ns)
        Nt = np.bincount(flat, minlength=nr * ns)[:nr * ns].reshape(nr, ns)
        Wd, Nd = Wt[R][:, S], Nt[R][:, S]
        if folded:
            F = [ssort.index(k_) for k_ in folded]
            Wd = np.c_[Wd, Wt[R][:, F].sum(1)]
            Nd = np.c_[Nd, Nt[R][:, F].sum(1)]
        return Wd, Nd

    filled = [(name, *fill(m)) for name, m in panels]
    keep = np.any([Wd.sum(1) != 0 for _, Wd, _ in filled], axis=0)
    rlabels = [str(row_names[k]) for k, kp in zip(rk, keep) if kp]
    filled = [(name, Wd[keep], Nd[keep]) for name, Wd, Nd in filled]
    if all_row:
        rlabels = rlabels + ["all"]
        filled = [(name, np.r_[Wd, Wd.sum(0, keepdims=True)], np.r_[Nd, Nd.sum(0, keepdims=True)])
                  for name, Wd, Nd in filled]
    live = np.any([Wd.sum(0) != 0 for _, Wd, _ in filled], axis=0) if filled and len(rlabels) else []

    npan = len(filled)
    fig = make_subplots(rows=1, cols=npan, shared_yaxes=horizontal or norm != "yields",
                        subplot_titles=[n for n, *_ in filled] if npan > 1 else None,
                        horizontal_spacing=0.025 if horizontal else 0.05)
    tables = []
    for p, (pname, Wd, Nd) in enumerate(filled):
        rt = Wd.sum(1)
        ptot = Wd[:-1].sum() if all_row else Wd.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_row = 100 * Wd / rt[:, None]
            share = 100 * rt / ptot if ptot else np.full(len(rt), np.nan)
            V = {"row": pct_row, "total": 100 * Wd / ptot if ptot else Wd * np.nan, "yields": Wd}[norm]
        V = np.nan_to_num(V)
        ylab = rlabels
        if npan == 1 and norm != "yields":
            ylab = [f"{r}  ({s_:.1f}%)" if r != "all" else "all" for r, s_ in zip(rlabels, share)]
        for j, (lab, col) in enumerate(zip(seg_labels, seg_colours)):
            if not live[j]:
                continue
            v = V[:, j]
            txt = [f"{x:.0f}" if (norm != "yields" and x >= label_min) else "" for x in v]
            cd = np.c_[np.full(len(v), lab), pct_row[:, j], Wd[:, j], Nd[:, j]]
            unit = "%" if norm != "yields" else ""
            bar = dict(name=lab, marker_color=col, marker_line=dict(width=1.5, color="rgba(255,255,255,1)"),
                       legendgroup=lab, showlegend=(p == 0), text=txt, textposition="inside",
                       insidetextanchor="middle", textfont=dict(color=_ink(col), size=11), customdata=cd,
                       hovertemplate=("%{" + ("y" if horizontal else "x") + "}<br><b>%{customdata[0]}</b>: "
                                      + ("%{" + ("x" if horizontal else "y") + ":.3g}" + unit + " · " if norm != "row" else "")
                                      + "%{customdata[1]:.2f}% of the row<br>Σw %{customdata[2]:.4g} · "
                                      "N %{customdata[3]:,.0f}<extra></extra>"))
            if horizontal:
                fig.add_trace(go.Bar(x=v, y=ylab, orientation="h", **bar), row=1, col=p + 1)
            else:
                fig.add_trace(go.Bar(x=ylab, y=v, **bar), row=1, col=p + 1)
        if norm == "yields":
            tot = V.sum(1)
            t = dict(mode="text", text=[f"{x:.4g}" for x in tot], showlegend=False, hoverinfo="skip",
                     textfont=dict(size=10, color="#555"))
            if horizontal:
                fig.add_trace(go.Scatter(x=tot, y=ylab, textposition="middle right", **t), row=1, col=p + 1)
            else:
                fig.add_trace(go.Scatter(x=ylab, y=tot, textposition="top center", **t), row=1, col=p + 1)
        tab = pd.DataFrame(V, columns=seg_labels)
        tab.insert(0, row_title or "row", rlabels)
        if npan > 1:
            tab.insert(0, "panel", pname)
        tab["share (%)"] = share
        tab["N"] = Nd.sum(1)
        tab["Σw"] = rt
        tables.append(tab)

    vax = dict(title_text=NORM_TITLES[norm], gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    if norm == "row":
        vax["range"] = [0, 100.5]
    cax = dict(type="category", categoryorder="array", categoryarray=(ylab if npan == 1 else rlabels), showgrid=False)
    for p in range(npan):
        if horizontal:
            fig.update_xaxes(row=1, col=p + 1, **vax)
            fig.update_yaxes(row=1, col=p + 1, autorange="reversed", **{k: v for k, v in cax.items()})
        else:
            fig.update_yaxes(row=1, col=p + 1, **vax)
            fig.update_xaxes(row=1, col=p + 1, **cax)
        if p:
            if horizontal:
                fig.update_xaxes(row=1, col=p + 1, title_text="")
            elif norm != "yields":
                fig.update_yaxes(row=1, col=p + 1, title_text="")
    nrow = len(rlabels)
    height = max(260, 30 * nrow + 150) if horizontal else 440
    fig.update_layout(barmode="stack", bargap=0.28, height=height, title=dict(text=title, x=0, font=dict(size=14)),
                      margin=dict(l=10, r=10, t=95, b=10), uniformtext=dict(minsize=9, mode="hide"),
                      legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0, font=dict(size=11),
                                  title_text=legend_title or ""),
                      hovermode="closest")
    if npan > 1:
        fig.update_annotations(font_size=12)
    table = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    return fig, table


# ----------------------------------------------------------------- 2D
def hist2d(x, y, w, mask, bx, by, xlabel, ylabel, logz=False):
    if bx[0] == "cat":
        bx = _order_for_cat(bx)
    if by[0] == "cat":
        by = _order_for_cat(by)
    xs, ys, ws = x[mask], y[mask], w[mask].astype(np.float64)

    def idx(v, b):
        i, ok, n = bin_index(v, b)
        return i, ok, n, (b[2] if b[0] == "cat" else 0.5 * (np.asarray(b[1])[1:] + np.asarray(b[1])[:-1]))

    ix, okx, nx, cx = idx(xs, bx)
    iy, oky, ny, cy = idx(ys, by)
    ok = okx & oky
    h = np.bincount(iy[ok] * nx + ix[ok], weights=ws[ok], minlength=nx * ny)[:nx * ny].reshape(ny, nx)
    z = np.log10(np.where(h > 0, h, np.nan)) if logz else h
    fig = go.Figure(go.Heatmap(x=cx, y=cy, z=z, colorscale=SEQUENTIAL, customdata=h,
                               colorbar=dict(title="log10 Σw" if logz else "Σw"),
                               hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<br>Σw: %{{customdata:.4g}}"
                                             "<extra></extra>"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), xaxis_title=xlabel, yaxis_title=ylabel)
    if bx[0] == "cat":
        fig.update_xaxes(type="category")
    if by[0] == "cat":
        fig.update_yaxes(type="category")
    return fig


# ----------------------------------------------------------------- tables
def cutflow(results, w, base_mask=None):
    """Cumulative cutflow over the controls that cut, in sidebar order."""
    n0 = len(w) if base_mask is None else int(base_mask.sum())
    m = base_mask
    sw0 = _sum(w, m)[0]
    rows = [dict(step="loaded", cut="", N=n0, sumw=sw0, err=np.sqrt(_sum(w, m)[1]), eff_step=1.0, eff_total=1.0)]
    prev = sw0
    for r in results:
        if r["mask"] is None:
            continue
        m = r["mask"] if m is None else (m & r["mask"])
        sw, sw2 = _sum(w, m)
        rows.append(dict(step=r["label"], cut=r["desc"] or "", N=int(m.sum()), sumw=sw, err=np.sqrt(sw2),
                         eff_step=sw / prev if prev else np.nan, eff_total=sw / sw0 if sw0 else np.nan))
        prev = sw
    return pd.DataFrame(rows)


def yields_by(codes, names, w, mask):
    rows = []
    tot = _sum(w, mask)[0]
    for c, n in names.items():
        m = codes == c if mask is None else (mask & (codes == c))
        sw, sw2 = _sum(w, m)
        rows.append(dict(category=str(n), N=int(m.sum()), sumw=sw, err=np.sqrt(sw2), fraction=sw / tot if tot else np.nan))
    return pd.DataFrame(rows)
