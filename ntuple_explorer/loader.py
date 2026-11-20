"""
Read a ROOT tree into a pandas DataFrame of per-event columns with RDataFrame.

Every column is a ROOT (C++) expression evaluated per entry: a scalar branch name, or a reduction of vector branches
(nx_at(v, 0), v.size(), Sum(v > 0), ...). Raw vectors are never pulled into Python. Results are cached on disk
(.ntuple_cache/) keyed by the file (path, size, mtime), tree, expressions and entry limit.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# data folder: relative file paths and the file list are taken from here (default: the current directory)
ROOT_DIR = Path(os.environ.get("NTUPLE_EXPLORER_DATA") or Path.cwd()).expanduser().resolve()
CACHE_DIR = Path(os.environ.get("NTUPLE_EXPLORER_CACHE") or ROOT_DIR / ".ntuple_cache").expanduser()

# nx_at(v, i): element i of a vector, -999 if the vector is shorter (no out-of-range read)
DEFAULT_HELPERS = """
template <typename V> double nx_at(const V& v, std::size_t i) { return i < v.size() ? double(v[i]) : -999.; }
"""

_declared = set()
_CHAR_TYPES = {"Char_t", "UChar_t", "char", "unsigned char", "signed char", "int8_t", "uint8_t"}
_ROOT = None


def _root():
    global _ROOT
    if _ROOT is None:
        import ROOT
        ROOT.PyConfig.IgnoreCommandLineOptions = True
        ROOT.gROOT.SetBatch(True)
        ROOT.EnableImplicitMT()
        _ROOT = ROOT
    return _ROOT


def declare(code):
    """Declare C++ helper code once per process."""
    if not code or code in _declared:
        return
    if not _root().gInterpreter.Declare(code):
        raise RuntimeError(f"C++ helper code does not compile:\n{code}")
    _declared.add(code)


def resolve(path):
    p = Path(path).expanduser()
    return p if p.is_absolute() else (ROOT_DIR / p)


# ----------------------------------------------------------------- metadata (uproot: no event loop)
def trees(path):
    import uproot
    with uproot.open(resolve(path)) as f:
        return [k.split(";")[0] for k, cls in f.classnames().items() if cls in ("TTree", "TNtuple")]


def branch_info(path, tree):
    """DataFrame: branch, type, is_vector (std::vector / RVec / C array), entries."""
    import uproot
    with uproot.open(resolve(path)) as f:
        t = f[tree]
        rows = []
        for name in t.keys():
            b = t[name]
            if len(b.branches):          # split TLorentzVector parents etc.: keep the leaves
                continue
            tn = b.typename
            rows.append(dict(branch=name, type=tn, is_vector=("vector" in tn or "RVec" in tn or "[" in tn)))
        n = t.num_entries
    df = pd.DataFrame(rows)
    df.attrs["entries"] = n
    return df


# ----------------------------------------------------------------- expression checks
_NOT_BRANCHES = {"true", "false", "std", "ROOT", "size", "double", "int", "float", "bool", "long", "unsigned",
                 "static_cast", "abs", "Sum", "Max", "Min", "Mean", "Any", "All", "nx_at"}


def expr_branches(e):
    """Identifiers in a ROOT expression that are not functions, members, namespaces or keywords."""
    return {b for b in re.findall(r"(?<![\w.:>])([A-Za-z_]\w*)\b(?!\s*(?:\(|::|<))", e) if b not in _NOT_BRANCHES}


def tree_branches(path, tree):
    import uproot
    with uproot.open(resolve(path)) as f:
        return set(f[tree].keys())


def check_exprs(path, tree, exprs, extra_names=()):
    """Raise KeyError for unknown branches before RDataFrame JIT-compiles anything (an unknown identifier in a
    jitted expression aborts the whole Python process)."""
    have = tree_branches(path, tree)
    need = {b for e in exprs for b in expr_branches(e)} - set(extra_names)
    missing = sorted(need - have)
    if missing:
        raise KeyError(f"{tree}: unknown branch(es) {', '.join(missing)}")


def split_missing(path, tree, columns):
    """-> (columns whose branches all exist, {dropped name: missing branches})."""
    have = tree_branches(path, tree)
    ok, dropped = {}, {}
    for name, e in columns.items():
        miss = sorted(expr_branches(e) - have)
        if miss:
            dropped[name] = miss
        else:
            ok[name] = e
    return ok, dropped


# ----------------------------------------------------------------- loading
def _downcast(a):
    if a.dtype == bool:
        return a
    if np.issubdtype(a.dtype, np.floating):
        return a.astype(np.float32) if a.dtype == np.float64 else a
    if np.issubdtype(a.dtype, np.integer) and a.dtype.itemsize > 1 and len(a):
        if a.dtype in (np.uint64, np.int64) and (a.max() > np.iinfo(np.int32).max or a.min() < np.iinfo(np.int32).min):
            return a
        lo, hi = a.min(), a.max()
        for t in (np.int8, np.int16, np.int32):
            if np.iinfo(t).min <= lo and hi <= np.iinfo(t).max:
                return a.astype(t)
    return a


def cache_key(path, tree, columns, entries=None, prefilter=None, helpers=None):
    p = resolve(path)
    st = p.stat()
    blob = json.dumps(dict(path=str(p), size=st.st_size, mtime=st.st_mtime, tree=tree, columns=sorted(columns.items()),
                           entries=entries, prefilter=prefilter, helpers=helpers or ""), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def load(path, tree, columns, entries=None, prefilter=None, helpers=DEFAULT_HELPERS, use_cache=True, progress=None,
         drop_missing=False):
    """{name: ROOT expression} -> DataFrame with one column per name (one event loop).

    entries:      read only the first N entries (None = all; disables multithreading for that read).
    prefilter:    ROOT boolean expression applied before reading (entries failing it are never loaded).
    drop_missing: skip columns that use branches absent from this file (listed in df.attrs["dropped"]) instead of
                  raising; lets one preset serve several ntuple versions.
    """
    p = resolve(path)
    dropped = {}
    if drop_missing:
        columns, dropped = split_missing(p, tree, columns)
    key = cache_key(p, tree, columns, entries, prefilter, helpers)
    cfile = CACHE_DIR / f"{key}.npz"
    if use_cache and cfile.exists():
        with np.load(cfile, allow_pickle=False) as z:
            df = pd.DataFrame({k: z[k] for k in columns})
        df.attrs["dropped"] = dropped
        return df

    check_exprs(p, tree, list(columns.values()) + ([prefilter] if prefilter else []))
    ROOT = _root()
    declare(DEFAULT_HELPERS)
    declare(helpers)
    if entries:
        ROOT.DisableImplicitMT()          # Range() is single-threaded only
    try:
        rdf = ROOT.RDataFrame(tree, str(p))
        if entries:
            rdf = rdf.Range(int(entries))
        if prefilter:
            rdf = rdf.Filter(prefilter)
        have = set(str(c) for c in rdf.GetColumnNames())
        out_names = {}
        for name, e in columns.items():
            if e == name and name in have and rdf.GetColumnType(name) not in _CHAR_TYPES:
                out_names[name] = name
            else:
                if e in have and rdf.GetColumnType(e) in _CHAR_TYPES:
                    e = f"int({e})"          # (u)int8_t is read as char -> object array in AsNumpy
                col = f"__c_{name}"
                rdf = rdf.Define(col, e)
                out_names[name] = col
        if progress:
            progress(f"event loop over {tree} ({len(columns)} columns)…")
        arr = rdf.AsNumpy(list(out_names.values()))
    finally:
        if entries:
            ROOT.EnableImplicitMT()
    data = {}
    for name, col in out_names.items():
        a = np.asarray(arr[col])
        if a.dtype == object:
            raise TypeError(f"column {name!r} = {columns[name]!r} is not a scalar per entry "
                            "(reduce the vector, e.g. nx_at(v, 0), v.size(), Sum(v > 0))")
        data[name] = _downcast(a)
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = cfile.with_suffix(".tmp.npz")
        np.savez(tmp, **data)
        os.replace(tmp, cfile)
    df = pd.DataFrame(data)
    df.attrs["dropped"] = dropped
    return df


# ----------------------------------------------------------------- friend trees (join on event keys)
def match_index(left, right):
    """Row of `right` with the same key tuple as each row of `left` (-1 if none). left / right: lists of integer
    arrays (e.g. [runNumber, eventNumber]). Duplicated keys in `right`: the first row is used."""
    L = [np.asarray(a) for a in left]
    R = [np.asarray(a) for a in right]
    n = len(L[0])
    if not len(R[0]):
        return np.full(n, -1, np.int64)
    if all(a.dtype.kind in "iub" for a in L + R):
        lo = [int(min(a.min(), b.min())) for a, b in zip(L, R)]
        span = [int(max(a.max(), b.max())) - m + 1 for a, b, m in zip(L, R, lo)]
        if float(np.prod([float(s) for s in span])) < 2.0 ** 62:
            def pack(cols):
                k = np.zeros(len(cols[0]), np.int64)
                for c, m, s in zip(cols, lo, span):
                    k = k * s + (c.astype(np.int64) - m)
                return k
            lk, rk = pack(L), pack(R)
            order = np.argsort(rk, kind="stable")
            rs = rk[order]
            pos = np.clip(np.searchsorted(rs, lk), 0, len(rs) - 1)
            found = rs[pos] == lk
            return np.where(found, order[pos], -1).astype(np.int64)
    ri = pd.MultiIndex.from_arrays(R)
    keep = ~ri.duplicated()
    idx = pd.MultiIndex.from_arrays([a[keep] for a in R]).get_indexer(pd.MultiIndex.from_arrays(L))
    return np.where(idx >= 0, np.flatnonzero(keep)[np.maximum(idx, 0)], -1).astype(np.int64)


def join_friend(main, path, spec, columns, helpers=DEFAULT_HELPERS, drop_missing=True):
    """Columns of another tree of the same file, aligned with the rows of `main` by the event keys.

    spec: dict(tree, keys=["runNumber", "eventNumber"], prefilter=None, found="<name>", fill=-1 or {column: fill})
    columns: {name: ROOT expression} on the friend tree. Rows of `main` without a friend entry get `fill` (NaN for
    floating columns); `found` (int8 0/1) says whether the event is in the friend tree (after its prefilter)."""
    keys = list(spec.get("keys", ["runNumber", "eventNumber"]))
    miss = [k for k in keys if k not in main.columns]
    if miss:
        raise KeyError(f"friend {spec['tree']}: key column(s) {miss} not loaded in the main tree")
    cols = {k: k for k in keys}
    cols.update(columns)
    f = load(path, spec["tree"], cols, prefilter=spec.get("prefilter"), helpers=helpers, drop_missing=drop_missing)
    miss = [k for k in keys if k not in f.columns]
    if miss:
        raise KeyError(f"friend {spec['tree']}: key branch(es) {miss} missing")
    idx = match_index([main[k].to_numpy() for k in keys], [f[k].to_numpy() for k in keys])
    found = idx >= 0
    safe = np.maximum(idx, 0)
    fills = spec.get("fill", -1)
    out = {spec.get("found", f"{spec['tree']}_found"): found.astype(np.int8)}
    for name in columns:
        if name not in f.columns:
            continue
        a = f[name].to_numpy()
        if a.dtype.kind == "f":
            fill, dt = np.nan, a.dtype
        else:
            fill = fills.get(name, -1) if isinstance(fills, dict) else fills
            dt = a.dtype if a.dtype.kind != "b" else np.dtype(np.int8)
            if not (np.iinfo(dt).min <= fill <= np.iinfo(dt).max if dt.kind in "iu" else True):
                dt = np.dtype(np.int32)
        r = np.full(len(main), fill, dtype=dt)
        r[found] = a[safe[found]]
        out[name] = r
    res = pd.DataFrame(out, index=main.index)
    res.attrs["dropped"] = f.attrs.get("dropped", {})
    res.attrs["n_friend"] = len(f)
    return res


def clear_cache():
    n = 0
    if CACHE_DIR.exists():
        for f in CACHE_DIR.glob("*.npz"):
            f.unlink()
            n += 1
    return n


# ----------------------------------------------------------------- vector reducers (generic mode)
REDUCERS = {
    "element 0 (leading)": "nx_at({b}, 0)",
    "element 1": "nx_at({b}, 1)",
    "element 2": "nx_at({b}, 2)",
    "size": "int({b}.size())",
    "count > 0": "int(Sum({b} > 0))",
    "count == 1": "int(Sum({b} == 1))",
    "sum": "double(Sum({b}))",
    "max": "({b}.size() ? double(Max({b})) : -999.)",
    "min": "({b}.size() ? double(Min({b})) : -999.)",
}


def reducer_expr(branch, reducer):
    return REDUCERS[reducer].format(b=branch)


def reducer_name(branch, reducer):
    tag = {"element 0 (leading)": "0", "element 1": "1", "element 2": "2", "size": "n", "count > 0": "npos",
           "count == 1": "n1", "sum": "sum", "max": "max", "min": "min"}[reducer]
    return f"{branch}__{tag}"
