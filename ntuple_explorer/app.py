"""
Ntuple Explorer - interactive selections and plots on ROOT ntuples.

    python -m ntuple_explorer --data <folder with .root files> [--presets <folder with preset .py files>]
"""
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np            # noqa: E402
import pandas as pd           # noqa: E402
import streamlit as st        # noqa: E402

import controls as ctl        # noqa: E402
import loader                 # noqa: E402
import plots                  # noqa: E402
import preset as pre          # noqa: E402
from view import View         # noqa: E402

st.set_page_config(page_title="Ntuple Explorer", page_icon="📊", layout="wide")
GENERIC = "Generic (any tree / branch)"


# ----------------------------------------------------------------- cached work
@st.cache_resource(show_spinner=False, max_entries=6)
def cached_load(path, tree, columns_items, entries, prefilter, helpers, drop_missing, friends=(), match=""):
    """Main tree + friend columns (left join) + optionally a second level (outer join, `match`: JSON spec)."""
    df = loader.load(path, tree, dict(columns_items), entries=entries or None, prefilter=prefilter or None,
                     helpers=helpers, drop_missing=drop_missing)
    dropped, errors, parts = dict(df.attrs.get("dropped", {})), {}, [df]
    for name, spec_json in friends:
        spec = json.loads(spec_json)
        try:
            f = loader.join_friend(df, path, spec, spec["columns"], helpers=helpers, drop_missing=drop_missing)
        except Exception as e:      # noqa: BLE001 - the main tree stays usable
            errors[name] = f"{type(e).__name__}: {e}"
            continue
        dropped.update({f"{name}:{k}": v for k, v in f.attrs.get("dropped", {}).items()})
        parts.append(f)
    if len(parts) > 1:
        df = pd.concat(parts, axis=1, copy=False)
    df.attrs.update(dropped=dropped, friend_errors=errors)
    if match:
        spec = json.loads(match)
        df = loader.join_level(df, path, spec, dict(spec["columns"]), helpers=helpers, drop_missing=drop_missing,
                               main_name=spec.get("main", "reco"))
        df.attrs["friend_errors"] = errors
    return df


@st.cache_data(show_spinner=False)
def cached_branch_info(path, tree, mtime, version=2):
    """version: bump when branch_info's output changes, so a running server does not serve the old table."""
    return loader.branch_info(path, tree)


@st.cache_data(show_spinner=False)
def cached_trees(path, mtime):
    return loader.trees(path)


@st.cache_data(show_spinner=False)
def cached_presets():
    return {k: (str(v) if v else None) for k, v in pre.discover().items()}


def root_files():
    return sorted(p.name for p in loader.ROOT_DIR.glob("*.root"))


def mtime(path):
    try:
        return loader.resolve(path).stat().st_mtime
    except OSError:
        return None


TABLE_CSS = """<style>
.nx-table {border-collapse: collapse; font-size: 0.85rem; width: 100%; margin-bottom: 0.8rem;}
.nx-table th {text-align: right; padding: 4px 10px; border-bottom: 1px solid rgba(128,128,128,0.45); font-weight: 600;}
.nx-table td {text-align: right; padding: 3px 10px; border-bottom: 1px solid rgba(128,128,128,0.15);
              font-variant-numeric: tabular-nums;}
.nx-table th:first-child, .nx-table td:first-child {text-align: left;}
.nx-wrap {overflow-x: auto;}
</style>"""


def show_table(df, fmt=None):
    """HTML table (st.dataframe needs pyarrow, which is broken in this environment)."""
    fmt = dict(fmt or {})
    formatters = {}
    for c in df.columns:
        f = fmt.get(c)
        if f is None and df[c].dtype.kind == "f":
            f = "{:.5g}"
        if f is not None:
            formatters[c] = (lambda f_: (lambda x: "" if pd.isna(x) else f_.format(x)))(f)
    html = df.to_html(index=False, classes="nx-table", border=0, formatters=formatters, escape=True)
    st.markdown(TABLE_CSS + f'<div class="nx-wrap">{html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------- sidebar: data source
st.sidebar.title("Ntuple Explorer")
presets = cached_presets()
mode = st.sidebar.selectbox("Mode", [GENERIC] + list(presets), key="mode")
P = None
if mode != GENERIC:
    if presets[mode] is None:
        st.error(f"Preset failed to load: {mode}")
        st.stop()
    P = pre.load_file(presets[mode])
    if P.get("description"):
        st.sidebar.caption(P["description"])

VTAG = {"number": "", "lorentz": "  (4-vector)", "object": "  (objects: size only)"}


def atl_generic(branch, kind, level=None):
    """Generic-mode at_least spec for a vector branch (pt of 4-vectors), optionally on the second level."""
    s = dict(kind="at_least", label=f"{level}: {branch}" if level else branch,
             **({"vector": branch} if kind == "lorentz" else {"values": branch}))
    return dict(s, level=level) if level else s


def generic_columns(scalars, vectors, reds, comps, extra, atl):
    """{name: ROOT expression} of one tree in generic mode: scalar branches, reduced vectors ((branch, kind) pairs),
    extra `name = expression` lines and the k-th largest columns of the at_least specs."""
    cols = {b: b for b in scalars}
    for b, k in vectors:
        cols.update(loader.vector_columns(b, k, reds, comps))
    for s_ in atl:
        cols.update(loader.nth_columns(s_))
    for line in extra.splitlines():
        if "=" in line and line.split("=", 1)[0].strip().isidentifier():
            n, e = line.split("=", 1)
            cols[n.strip()] = e.strip()
    return cols


atl_specs = []          # generic mode: "at least N above X" cuts on vector branches (presets: at_least controls)
match_gen, match_req = None, None       # second level (generic settings / load request)
with st.sidebar.expander("Data", expanded="active" not in st.session_state):
    files = dict(P["files"]) if P and P.get("files") else {f: f for f in root_files()}
    files = {k: v for k, v in files.items() if loader.resolve(v).exists()} or files
    choice = st.selectbox("File", list(files) + ["other path…"], key=f"file_{mode}")
    path = st.text_input("Path", key=f"path_{mode}") if choice == "other path…" else files[choice]
    if not path or not loader.resolve(path).exists():
        st.warning("File not found")
        st.stop()
    entries = st.number_input("Max entries (-1 = all)", min_value=-1, value=-1, step=100000, key=f"entries_{mode}",
                              help="Reads only the first N entries (single-threaded) - quick looks at huge files")
    entries = max(int(entries), 0)      # loader / cache key convention: 0 = all

    if P:
        tree = P["tree"]
        st.caption(f"tree **{tree}** · {len(P['columns'])} columns + {len(P.get('categories', {}))} categories")
        columns = pre.loader_columns(P)
        for fname, f in (P.get("friends") or {}).items():
            st.caption(f"+ joined from **{f['tree']}** ({fname}): {len(pre.friend_columns(f))} columns, "
                       f"matched on {', '.join(f.get('keys', ['runNumber', 'eventNumber']))}")
        prefilter = P.get("prefilter")
        helpers = P.get("helpers") or loader.DEFAULT_HELPERS
        if P.get("match"):
            match_req = pre.match_request(P)
            st.caption(f"+ matched with **{match_req['tree']}** (level *{match_req['name']}*, this tree: "
                       f"*{match_req['main']}*) on {', '.join(match_req['keys'])}: {len(match_req['columns'])} "
                       "columns")
    else:
        tlist = cached_trees(path, mtime(path))
        tree = st.selectbox("Tree", tlist, key="g_tree")
        info = cached_branch_info(path, tree, mtime(path))
        st.caption(f"{info.attrs.get('entries', 0):,} entries · {len(info)} branches")
        kinds = dict(zip(info.branch, info.element))
        num = info.element == "number"
        scal = info.loc[~info.is_vector & num, "branch"].tolist()
        vec = info.loc[info.is_vector, "branch"].tolist()
        sel_s = st.multiselect("Scalar branches", scal, key="g_scal")
        sel_v = st.multiselect("Vector branches", vec, key="g_vec", format_func=lambda b: b + VTAG[kinds[b]])
        reds = st.multiselect("…reduced as", list(loader.REDUCERS), default=["element 0 (leading)"], key="g_red",
                              help="each vector branch becomes one column per reducer; for 4-vectors, element i "
                                   "gives the components chosen below, sum / max / min act on pt, counts do not "
                                   "apply. Also used for the second tree.")
        atl = st.multiselect("'At least N above X' cuts on", [b for b in vec if kinds[b] in ("number", "lorentz")],
                             key="g_atl", format_func=lambda b: b + VTAG[kinds[b]],
                             help=f"loads the 1st … {loader.NTH_MAX}th largest value per entry (pt for 4-vectors); "
                                  "the cut then picks N and the threshold X interactively")
        atl_specs = [atl_generic(b, kinds[b]) for b in atl]
        other = info.loc[~info.is_vector & ~num, "branch"].tolist()
        if other:
            st.caption("Not listed (single objects): " + ", ".join(f"`{b}`" for b in other)
                       + ". Read them with an extra column, e.g. `x_pt = x.Pt()`.")
        extra = st.text_area("Extra columns (`name = ROOT expression`, one per line)", key="g_extra",
                             placeholder="nbtag = Sum(jet_btag == 1)\nlead_pt = nx_pt(jet_p4, 0)")
        prefilter = st.text_input("Pre-filter (ROOT expression, applied while reading)", key="g_pref")
        helpers = loader.DEFAULT_HELPERS

        # ---- second level: another tree of the file, events matched on keys (outer join)
        vec_any = [(b, kinds[b]) for b in sel_v]
        if st.toggle("Match events with a second tree", key="g_match",
                     help="e.g. reco ↔ truth: events of both trees joined on keys such as runNumber, eventNumber; "
                          "each level keeps its own selection and weight; enables the Unfolding tab"):
            others = [x for x in tlist if x != tree]
            if not others:
                st.caption("No other tree in this file.")
            else:
                tree2 = st.selectbox("Second tree", others, key="g_m_tree")
                info2 = cached_branch_info(path, tree2, mtime(path))
                kinds2 = dict(zip(info2.branch, info2.element))
                num2 = info2.element == "number"
                scal2 = set(info2.loc[~info2.is_vector & num2, "branch"])
                common = [b for b in scal if b in scal2]
                mkeys = st.multiselect("Match on", common, key="g_m_keys",
                                       default=[k for k in ("runNumber", "eventNumber") if k in common])
                n1, n2 = st.columns(2)
                lv1 = n1.text_input("Level name, this tree", "reco", key="g_m_main").strip()
                lv2 = n2.text_input("Level name, second tree", "truth", key="g_m_name").strip()
                vec2 = info2.loc[info2.is_vector, "branch"].tolist()
                sel_s2 = st.multiselect(f"{lv2}: scalar branches", sorted(scal2), key="g_m_scal")
                sel_v2 = st.multiselect(f"{lv2}: vector branches", vec2, key="g_m_vec",
                                        format_func=lambda b: b + VTAG[kinds2[b]])
                atl2 = st.multiselect(f"{lv2}: 'at least N above X' cuts on",
                                      [b for b in vec2 if kinds2[b] in ("number", "lorentz")], key="g_m_atl",
                                      format_func=lambda b: b + VTAG[kinds2[b]])
                extra2 = st.text_area(f"{lv2}: extra columns", key="g_m_extra")
                pref2 = st.text_input(f"{lv2}: pre-filter", key="g_m_pref")
                vec_any += [(b, kinds2[b]) for b in sel_v2]
                if not (lv1.isidentifier() and lv2.isidentifier() and lv1 != lv2):
                    st.error("Level names must be two different identifiers (letters, digits, _).")
                elif not mkeys:
                    st.error("Choose the branches to match on.")
                else:
                    atl_specs += [atl_generic(b, kinds2[b], lv2) for b in atl2]
                    match_gen = dict(tree=tree2, name=lv2, main=lv1, keys=mkeys, prefilter=pref2 or None,
                                     scal=sel_s2, vec=[(b, kinds2[b]) for b in sel_v2], extra=extra2,
                                     atl=[atl_generic(b, kinds2[b], lv2) for b in atl2])
        comps = ["pt"]
        if any(k == "lorentz" for _, k in vec_any):
            comps = st.multiselect("…4-vector components", loader.COMPONENTS, default=["pt"], key="g_comp",
                                   help="of each chosen element (units as stored, e.g. MeV)")
        columns = generic_columns(sel_s, [(b, kinds[b]) for b in sel_v], reds, comps, extra,
                                  [s_ for s_ in atl_specs if not s_.get("level")])
        if match_gen:
            columns.update({k: k for k in match_gen["keys"]})
            cols2 = generic_columns(match_gen["scal"], match_gen["vec"], reds, comps, match_gen["extra"],
                                    match_gen["atl"])
            # a second-level column named like a main one (same branch in both trees) gets the level name in front
            cols2 = {(f"{match_gen['name']}_{n}" if n in columns else n): e for n, e in cols2.items()}
            match_req = dict(tree=match_gen["tree"], name=match_gen["name"], main=match_gen["main"],
                             keys=match_gen["keys"], prefilter=match_gen["prefilter"],
                             columns=sorted(cols2.items()))

    c1, c2 = st.columns(2)
    go_load = c1.button("Load", type="primary", use_container_width=True, disabled=not columns)
    if c2.button("Clear cache", use_container_width=True, help="delete .ntuple_cache/*.npz and in-memory data"):
        n = loader.clear_cache()
        cached_load.clear()
        st.session_state.pop("active", None)
        st.toast(f"removed {n} cached files")

request = dict(mode=mode, path=str(path), tree=tree, columns=sorted(columns.items()), entries=int(entries),
               prefilter=prefilter or "", helpers=helpers, drop_missing=bool(P),
               friends=pre.friend_request(P) if P else (),
               match=json.dumps(dict(match_req, entries=int(entries)), sort_keys=True) if match_req else "")
if go_load:
    st.session_state["active"] = request
    st.session_state.pop("load_error", None)

active = st.session_state.get("active")
if not active or active["mode"] != mode:
    st.markdown("## Ntuple Explorer")
    st.info("Choose a file (and branches in generic mode) in the sidebar, then press **Load**. "
            "The first read runs one RDataFrame event loop; later loads of the same columns come from "
            "`.ntuple_cache/` in about a second.")
    st.stop()
if active != request:
    st.sidebar.warning("Data settings changed - press **Load** to apply.")

t0 = time.time()
try:
    with st.spinner(f"Reading {Path(active['path']).name}:{active['tree']} …"):
        df = cached_load(active["path"], active["tree"], tuple(active["columns"]), active["entries"],
                         active["prefilter"], active["helpers"], active["drop_missing"],
                         tuple(tuple(x) for x in active.get("friends", ())), active.get("match", ""))
except Exception as e:      # noqa: BLE001
    st.error(f"Loading failed: {type(e).__name__}: {e}")
    st.stop()
t_load = time.time() - t0
data_key = hashlib.sha1(json.dumps(active, sort_keys=True, default=str).encode()).hexdigest()[:12]
dropped = df.attrs.get("dropped", {})

# ----------------------------------------------------------------- view + controls
labels = pre.labels(P) if P else {}
view = View(df, derived=(P.get("derived") if P else None))
stats_cache = st.session_state.setdefault("stats", {}).setdefault(data_key, {})
pid = Path(presets[mode]).stem if P else "generic"
LEVELS = df.attrs.get("levels")          # [main, second] when two trees are matched (outer join), else None
MAIN, SECOND = LEVELS if LEVELS else (None, None)
LEVEL_OF = df.attrs.get("level_of", {})
LEVEL_KEYS = set(df.attrs.get("keys", [])) | ({f"in_{L}" for L in LEVELS} if LEVELS else set())


def level_of(name):
    """Level of a column: loaded columns from the join, new columns from the columns they use (main if mixed);
    preset derived columns and the in_* flags count as main level."""
    if not LEVELS:
        return None
    if name in LEVEL_OF:
        return LEVEL_OF[name]
    if name in user_defs:
        used = {LEVEL_OF.get(x) for x in re.findall(r"[A-Za-z_]\w*", user_defs[name])} - {None}
        return used.pop() if len(used) == 1 else MAIN
    return MAIN


def stats_for(name):
    return ctl.stats(view, name, SC)


class _StatsProxy:
    def __contains__(self, name):
        return f"{name}|{_src(name)}" in stats_cache

    def __getitem__(self, name):
        return stats_cache[f"{name}|{_src(name)}"]

    def __setitem__(self, name, value):
        stats_cache[f"{name}|{_src(name)}"] = value

    @staticmethod
    def rows(name):
        """Matched dataset: statistics of a column only over the events of its level (not the fill values)."""
        return view[f"in_{level_of(name)}"].astype(bool) if LEVELS and name not in LEVEL_KEYS else None


def _src(name):
    if name in user_defs:
        return f"{user_defs[name]}|{alias_choices}"
    s = view.aliases.get(name)
    return s if isinstance(s, str) else (id(s) if s is not None else "")


SC = _StatsProxy()
unusable = {}           # derived columns that cannot be computed for this file (e.g. an input branch is missing)


def usable(name):
    """False (and the reason in `unusable`) for a column that raises when computed; loaded columns are always fine."""
    if name in view.df.columns and name not in view.aliases:
        return True
    try:
        stats_for(name)
        return True
    except Exception as e:      # noqa: BLE001
        unusable[name] = f"{type(e).__name__}: {e}"
        return False

# ---- state save / load (before widgets are created)
state_prefix = f"ctl_{pid}_"
with st.sidebar.expander("Save / load selection"):
    up = st.file_uploader("Load state (.json)", type="json", key=f"up_{pid}")
    if up is not None:
        raw = up.getvalue()
        h = hashlib.sha1(raw).hexdigest()
        if st.session_state.get(f"applied_{pid}") != h:
            try:
                for k, v in json.loads(raw).items():
                    if k.startswith(state_prefix):
                        st.session_state[k] = v
                st.session_state[f"applied_{pid}"] = h
                st.rerun()
            except Exception as e:      # noqa: BLE001
                st.error(f"bad state file: {e}")
    state_slot = st.empty()

# ---- main page: header (filled once the masks are known), page-cut box, tabs
header_box = st.container()
page_open = bool(st.session_state.get(f"{state_prefix}page_cols")) or any(
    s.get("place") == "main" for s in (P.get("controls", []) if P else []))
page_box = st.expander("Page cuts · applied to every plot and table, not cutflow steps", expanded=page_open)
page_top, page_grid = page_box.container(), page_box.container()
PAGE_NCOL = 3
_page_cells = []


def page_cell():
    """Next cell of the page-cut grid (PAGE_NCOL controls per row)."""
    if not _page_cells:
        _page_cells.extend(page_grid.columns(PAGE_NCOL))
    return _page_cells.pop(0)


# app settings: read before the controls are drawn, widgets at the bottom of the sidebar
RANGE_LABELS = {"snapped": "0.1–99.9 % quantiles, snapped to min / max when close",
                "quantiles": "0.1–99.9 % quantiles", "full": "full range (min – max)"}
ctl.RANGE_MODE = st.session_state.get("opt_slider_range", "snapped")

st.sidebar.markdown("### Selection")
results = []
user_defs = {}          # new columns typed in the app: name -> expression
alias_choices = ()


def _alias_ok(spec):
    """Drop alias options that point to columns missing from this file."""
    opts = {}
    for lab, targets in spec["options"].items():
        if all(callable(s) or s in view for s in targets.values()):
            opts[lab] = targets
    if not opts:
        return None
    s = dict(spec, options=opts)
    if s.get("default") not in opts:
        s["default"] = next(iter(opts))
    return s


def _refs(spec):
    """Columns a control needs (to skip controls on columns dropped for this file)."""
    if spec["kind"] == "at_least":
        return list(loader.nth_columns(spec))
    if spec.get("column"):
        return [spec["column"]]
    return []


def at_least_spec(spec):
    """An at_least spec ready for ctl.render: the names of its k-th largest columns, and what the threshold is on."""
    vals = str(spec.get("values", ""))
    what = spec.get("what") or (spec.get("of", "pt") if spec.get("vector")
                                else vals if vals.isidentifier() else "value")
    return dict(spec, columns=list(loader.nth_columns(spec)), what=what)


specs = list(P.get("controls", [])) if P else []
for i, spec in enumerate(specs):
    if spec["kind"] == "alias":
        spec = _alias_ok(spec)
        if spec is None:
            continue
    if any(c not in view or not usable(c) for c in _refs(spec)):
        continue
    if spec["kind"] in ("multiselect", "range") and spec.get("labels") is None and spec["column"] in labels:
        spec = dict(spec, labels=labels[spec["column"]])
    if spec["kind"] == "at_least":
        spec = at_least_spec(spec)
    on_page = spec.get("place") == "main"
    box = page_cell() if on_page else st.sidebar
    try:
        r = ctl.render(spec, view, f"{state_prefix}{i}", SC, container=box)
    except Exception as e:      # noqa: BLE001
        box.error(f"{spec.get('label', spec['kind'])}: {type(e).__name__}: {e}")
        continue
    if r["aliases"]:
        view.aliases.update(r["aliases"])
        view._memo.clear()
    r["page"] = on_page
    r["level"] = spec.get("level") or MAIN
    results.append(r)
alias_choices = tuple((r["label"], r.get("choice")) for r in results if r["kind"] == "alias")

# ---- new columns: expressions over loaded columns, computed in memory (no reload)
defs_key = f"{state_prefix}newcols"
COMBINE = {
    "OR (any ≠ 0)": lambda cs: " | ".join(f"({c} != 0)" for c in cs),
    "AND (all ≠ 0)": lambda cs: " & ".join(f"({c} != 0)" for c in cs),
    "max": lambda cs: f"max({', '.join(cs)})",
    "min": lambda cs: f"min({', '.join(cs)})",
    "sum": lambda cs: " + ".join(cs),
    "count ≠ 0": lambda cs: f"count({', '.join(f'{c} != 0' for c in cs)})",
}


def _add_combined(cols, op, name):
    """Button callback: append `name = <combination>` to the new-column definitions."""
    if not cols:
        return
    name = name.strip() or "_".join([op.split()[0].lower()] + list(cols))
    cur = st.session_state.get(defs_key, "")
    st.session_state[defs_key] = (cur.rstrip() + "\n" if cur.strip() else "") + f"{name} = {COMBINE[op](cols)}"


def _as_column(a):
    a = np.asarray(a)
    return a.astype(np.int8) if a.dtype.kind == "b" else a


with st.sidebar.expander("New columns", expanded=bool(st.session_state.get(defs_key))):
    defs_txt = st.text_area(
        "One per line: `name = expression`", key=defs_key, height=100,
        placeholder="lep_ok = (pass_e == 1) | (pass_mu == 1)\nnsm_max = max(nsm_a, nsm_b)",
        help="Over loaded columns, computed in memory (no reload). Operators: + - * / ** == != < <= > >=, "
             "& | ~ (or and / or / not, && / || / !), `a if cond else b`. Functions: max, min, where(cond, a, b), "
             "count(cond, ...), isin(x, v1, v2, ...), abs, sqrt, exp, log, log10, sin, cos, tan, arctan2, hypot, "
             "floor, ceil. Conditions give 0/1 columns. New columns can get sliders / selectors and be plotted.")
    for ln, line in enumerate(defs_txt.splitlines(), 1):
        s_ = line.split("#", 1)[0].strip()
        if not s_:
            continue
        n_, _, e_ = s_.partition("=")
        n_, e_ = n_.strip(), e_.strip()
        if not n_.isidentifier() or not e_ or e_.startswith("="):
            st.error(f"line {ln}: expected `name = expression`")
            continue
        if n_ in view:
            st.error(f"line {ln}: `{n_}` already exists")
            continue
        view.derived[n_] = (lambda e: lambda v: _as_column(v.eval(e)))(e_)
        try:
            _ = view[n_]        # (a bare expression would be st.write-n by Streamlit "magic")
            user_defs[n_] = e_
        except Exception as e:      # noqa: BLE001
            del view.derived[n_]
            view._memo.pop(n_, None)
            st.error(f"line {ln} `{n_}`: {type(e).__name__}: {e}")
    st.markdown("**Combine columns**")
    cc = st.multiselect("Columns", [c for c in view.names() if not c.startswith("__") and usable(c)],
                        key=f"{state_prefix}comb_cols")
    k1, k2 = st.columns(2)
    cop = k1.selectbox("as", list(COMBINE), key=f"{state_prefix}comb_op",
                       help="OR / AND / count: of `column ≠ 0`; "
                            "max: a slider on it at ≥ x is an OR of `column ≥ x`")
    cname = k2.text_input("name", key=f"{state_prefix}comb_name", placeholder="automatic")
    st.button("Add column", on_click=_add_combined, args=(tuple(cc), cop, cname), disabled=not cc,
              use_container_width=True)

# ---- extra / generic cuts on any column
all_cols = [c for c in view.names() if not c.startswith("__") and usable(c)]
default_extra = all_cols if (not P and len(all_cols) <= 8) else []
with st.sidebar.expander("Cuts on any column", expanded=not P):
    extra_cols = st.multiselect("Add a slider / selector for", all_cols, default=default_extra,
                                key=f"{state_prefix}extra_cols")
    int_as = st.radio("Integer columns with ≤ 20 values as", ["slider", "selector"], horizontal=True,
                      key=f"{state_prefix}extra_int_as",
                      help="slider: a range (counts such as n jets); selector: pick individual values. "
                           "Columns with labels and 0/1 flags always get a selector. Also used for page cuts.")


def auto_control(c, prefix, box, page):
    """Slider / selector chosen from the column's values; its result (tagged page or not) goes to results."""
    try:
        spec = ctl.auto_spec(view, c, SC, labels=labels.get(c), int_slider=int_as == "slider")
        if page:
            spec["place"] = "main"
        if LEVELS and level_of(c) != MAIN:
            spec["label"] = f"{c} · {level_of(c)}"
        # one widget key per kind: a slider cannot take over the state of a multiselect
        key = f"{prefix}{c}" + ("__range" if spec["kind"] == "range" and stats_for(c)["kind"] == "cat" else "")
        r = ctl.render(spec, view, key, SC, container=box)
        r["page"] = page
        r["level"] = level_of(c)
        results.append(r)
    except Exception as e:      # noqa: BLE001
        box.error(f"{c}: {type(e).__name__}: {e}")


for c in extra_cols:
    auto_control(c, f"{state_prefix}auto_", st.sidebar, False)
for s_ in atl_specs:          # generic mode; skipped until loaded (its columns appear after Load)
    if all(c in view for c in _refs(s_)):
        try:
            r = ctl.render(at_least_spec(s_), view, f"{state_prefix}atl_{s_['label']}", SC, container=st.sidebar)
            results.append(dict(r, page=False, level=s_.get("level") or MAIN))
        except Exception as e:      # noqa: BLE001
            st.sidebar.error(f"{s_['label']}: {type(e).__name__}: {e}")
if not P:
    results.append(dict(ctl.render(dict(kind="expr", label="Expression cut"), view, f"{state_prefix}expr", SC,
                                   container=st.sidebar), page=False, level=MAIN))

# ---- page cuts on any column (main page, above the tabs)
page_cols = page_top.multiselect("Add a page cut (slider / selector) on", all_cols, key=f"{state_prefix}page_cols",
                                 help="Applied to every plot, table and the header numbers, on top of the sidebar "
                                      "selection; in the cutflow they make one last row, 'after page cuts'.")
for c in page_cols:
    auto_control(c, f"{state_prefix}page_", page_cell(), True)

# ---- weights (one per level)
def weight_of(level):
    """(float32 weights, description) of a level (None: the dataset): the preset weight column (the match spec's
    `weight` for the second level), else a selectbox over that level's float columns."""
    wcol = (P.get("weight") if level in (None, MAIN) else P["match"].get("weight")) if P else None
    if wcol and wcol in view:
        src = view.aliases.get(wcol)
        return view[wcol].astype(np.float32), wcol + (f" → {src}" if isinstance(src, str) else "")
    cols_ = [c for c in all_cols if view[c].dtype.kind == "f" and level_of(c) == level]
    key = f"{state_prefix}weight" + ("" if level in (None, MAIN) else f"_{level}")
    wsel = st.sidebar.selectbox("Weight" + (f" ({level})" if level else ""), ["(none)"] + cols_, key=key)
    return (np.ones(len(view), np.float32) if wsel == "(none)" else view[wsel].astype(np.float32)), wsel


sel_results = [r for r in results if not r.get("page")]
page_results = [r for r in results if r.get("page")]
page_idx = [i for i, r in enumerate(results) if r.get("page")]

with header_box:
    st.markdown(f"#### {mode if P else 'Generic'} · `{Path(active['path']).name}` : `{active['tree']}`"
                + (f" ⟷ `{json.loads(active['match'])['tree']}`" if LEVELS else ""))
if LEVELS:
    w_lv = {L: weight_of(L) for L in LEVELS}
    flag = {L: view[f"in_{L}"].astype(bool) for L in LEVELS}
    match_w = st.session_state.get(f"{state_prefix}match_w", MAIN)      # widget in the Unfolding tab
    POPS = [f"{MAIN}-selected", f"{SECOND}-selected", "matched", f"{MAIN} only (fakes)", f"{SECOND} only (misses)"]
    with header_box:
        pop_name = st.radio("Events", POPS, horizontal=True, key=f"{state_prefix}population",
                            help=f"{MAIN} / {SECOND}-selected: in that tree and passing that level's cuts; matched: "
                                 f"both; fakes: {MAIN}-selected only; misses: {SECOND}-selected only. Each control "
                                 f"belongs to the level of its column (main level for derived columns).")
    POP = POPS.index(pop_name)
    wname = [w_lv[MAIN][1], w_lv[SECOND][1], w_lv[match_w][1], w_lv[MAIN][1], w_lv[SECOND][1]][POP]
    w = np.nan_to_num([w_lv[MAIN][0], w_lv[SECOND][0], w_lv[match_w][0], w_lv[MAIN][0], w_lv[SECOND][0]][POP])
    pop_base = [flag[MAIN], flag[SECOND], flag[MAIN] & flag[SECOND], flag[MAIN], flag[SECOND]][POP]
else:
    w, wname = weight_of(None)
    POP, pop_name, pop_base = 0, "selected", None


with st.sidebar.expander("Settings"):
    st.radio("Slider range (without preset limits)", list(ctl.RANGE_MODES), key="opt_slider_range",
             format_func=RANGE_LABELS.get,
             help=f"snapped: an end moves to the true min / max when that is within {ctl.SNAP:.0%} of the range, "
                  "so hard thresholds (e.g. pT > 20 GeV) stay reachable while far tails do not stretch the slider. "
                  "A handle at a slider end never cuts.")


def level_masks(exclude=()):
    """{level: in that tree & its controls' cuts} (results with index in `exclude` left out)."""
    out = {}
    for L in LEVELS:
        m = ctl.combine([r for i, r in enumerate(results) if r["level"] == L and i not in exclude])
        out[L] = flag[L] if m is None else flag[L] & m
    return out


def pop_mask(exclude=()):
    """Mask of the chosen population with all cuts but those of the results in `exclude` (N-1 plots, page cuts)."""
    if not LEVELS:
        m = ctl.combine([r for i, r in enumerate(results) if i not in exclude])
        return np.ones(len(view), bool) if m is None else m
    lm = level_masks(exclude)
    a, b = lm[MAIN], lm[SECOND]
    return [a, b, a & b, a & ~b, b & ~a][POP]


mask = pop_mask()
side_mask = pop_mask(exclude=page_idx)
W = plots.weights(w)            # float64 weights and squares, once per rerun
alias_sig = (tuple((r["label"], r.get("choice")) for r in results if r["kind"] == "alias") + tuple(user_defs.items())
             + (pop_name,))

# state download (after widgets exist)
state = {k: (list(v) if isinstance(v, tuple) else v) for k, v in st.session_state.items()
         if isinstance(k, str) and k.startswith(state_prefix) and isinstance(v, (bool, int, float, str, list, tuple))}
state_slot.download_button("Save state (.json)", json.dumps(state, indent=1), file_name=f"selection_{pid}.json",
                           mime="application/json", use_container_width=True)

# ----------------------------------------------------------------- header
base_all = np.ones(len(view), bool) if pop_base is None else pop_base
n_all, n_sel = int(base_all.sum()), int(mask.sum())
sw_all, sw_sel = float(W[0][base_all].sum()), float(W[0][mask].sum())
page_cutting = any(r["mask"] is not None for r in page_results)
with header_box:
    m1, m2, m3, m4 = st.columns(4)
    where_ = ["in " + MAIN, "in " + SECOND, "in both", "in " + MAIN, "in " + SECOND][POP] if LEVELS else ""
    m1.metric("Entries loaded" if not LEVELS else f"Entries ({where_})", f"{n_all:,}")
    m2.metric("Selected", f"{n_sel:,}", f"{n_sel / n_all:.2%}" if n_all else None, delta_color="off")
    m3.metric("Σw selected", f"{sw_sel:.5g}", f"{sw_sel / sw_all:.2%} of Σw" if sw_all else None, delta_color="off")
    m4.metric("Load", f"{t_load:.1f} s", "cached" if t_load < 2 else "event loop", delta_color="off")
    if LEVELS:
        lm_ = level_masks()
        a_, b_ = lm_[MAIN], lm_[SECOND]
        na, nb, nab = int(a_.sum()), int(b_.sum()), int((a_ & b_).sum())
        st.caption(f"{MAIN}-selected {na:,} · {SECOND}-selected {nb:,} · matched {nab:,}"
                   + (f" ({nab / na:.1%} of {MAIN}" if na else " (")
                   + (f", {nab / nb:.1%} of {SECOND})" if nb else ")")
                   + f" · fakes {na - nab:,} · misses {nb - nab:,}. Loaded: {int(flag[MAIN].sum()):,} in {MAIN}, "
                   f"{int(flag[SECOND].sum()):,} in {SECOND}, {int((flag[MAIN] & flag[SECOND]).sum()):,} in both.")
    if page_cutting:
        n_side = int(side_mask.sum())
        st.caption(f"Sidebar selection: {n_side:,} entries; the page cuts keep {n_sel:,} of them "
                   f"({n_sel / n_side:.2%})." if n_side else "Sidebar selection: no entries.")
    if dropped:
        st.caption("Columns not in this file (skipped): " + ", ".join(f"`{k}`" for k in dropped))
    if unusable:
        st.caption("Derived columns not available for this file (skipped): "
                   + ", ".join(f"`{k}`" for k in unusable), help="\n\n".join(f"{k}: {v}" for k, v in unusable.items()))
    for fname, err in df.attrs.get("friend_errors", {}).items():
        st.warning(f"Friend tree {fname!r} not joined: {err}")
    def _expr(rs):
        desc = [r["desc"] for r in rs if r["desc"] and not r.get("page")]
        pdesc = [r["desc"] for r in rs if r["desc"] and r.get("page")]
        return (" &&\n".join(desc) or "(no cuts)") + ("\n// page cuts (not cutflow steps)\n&& " + " &&\n".join(pdesc)
                                                     if pdesc else "")
    with st.expander("Selection as expression"):
        if LEVELS:
            st.code("\n\n".join(f"// {L} level (in_{L})\n" + _expr([r for r in results if r["level"] == L])
                                for L in LEVELS), language="cpp")
        else:
            st.code(_expr(results), language="cpp")
        extra = "; new columns: " + ", ".join(f"`{n} = {e}`" for n, e in user_defs.items()) if user_defs else ""
        st.caption(f"weight: {wname}. `[…]` entries are column choices or Python-defined cuts{extra}.")

# ----------------------------------------------------------------- tabs
cat_cols = [c for c in all_cols if c in labels or stats_for(c)["kind"] == "cat"]
stack_opts = [c for c in (P.get("stack_options", []) if P else cat_cols) if c in all_cols]
custom = (P.get("custom") or {}) if P else {}
colors = (P.get("colors") or {}) if P else {}


def names_for(c):
    """{value: label} of a categorical column in display order (labels first, then other values present)."""
    vals = stats_for(c).get("values") or []
    lab = labels.get(c) or {}
    out = {k: str(v) for k, v in lab.items()}
    for v in vals:
        if v not in out:
            out[v] = plots.fmt_label(v, None)
    return out


# ---- variables to plot: a collapsible box above the tabs
plot_specs = [s for s in (P.get("plots", []) if P else []) if s["column"] in all_cols]
default_plot_cols = ([s["column"] for s in plot_specs]
                     or [c for c in all_cols if c not in ("runNumber", "eventNumber")][:6])
plots_key = f"{state_prefix}plots"
if plots_key in st.session_state:       # drop columns that no longer exist (e.g. a removed new column)
    st.session_state[plots_key] = [c for c in st.session_state[plots_key] if c in all_cols]
n_plot = len(st.session_state.get(plots_key, default_plot_cols))
with st.expander(f"Variables to plot ({n_plot})", expanded=False):
    plot_cols = st.multiselect("Variables (Plots tab)", all_cols, default=default_plot_cols, key=plots_key,
                               label_visibility="collapsed")

tab_names = (["Plots", "2D"] + (["Composition"] if P and P.get("compositions") else []) + ["Cutflow & yields"]
             + (["Unfolding"] if LEVELS else []))
all_tabs = st.tabs(tab_names + list(custom))
tabs = dict(zip(tab_names, all_tabs))

with tabs["Plots"]:
    o2, o3, o4, o5, o6 = st.columns([1, 1, 1, 1, 2])
    norm = o2.toggle("Normalise", key="opt_norm")
    logy = o3.toggle("Log y", key="opt_logy")
    ratio = o4.toggle("Ratio", value=True, key="opt_ratio", help="selected / before selection per bin")
    nm1 = o5.toggle("N-1", value=True, key="opt_nm1",
                    help="each plot ignores the cuts on its own variable (classic N-1 plots)")
    stack = o6.selectbox("Stack selected by", ["(none)"] + stack_opts, key=f"{state_prefix}stack")
    ncol = st.select_slider("Plots per row", [1, 2, 3], value=2, key="opt_ncol")

    stack_arg = None
    if stack != "(none)":
        codes = view[stack]
        names = labels.get(stack) or {v: plots.fmt_label(v, None) for v in np.unique(codes).tolist()}
        stack_arg = (codes, names, colors.get(stack))

    by_col = {s["column"]: s for s in plot_specs}
    grid = st.columns(ncol)
    for k, c in enumerate(plot_cols):
        spec = dict(by_col.get(c, {"column": c}))
        if c in labels:
            spec.setdefault("labels", labels[c])
        try:
            x = view[c]
            s_ = stats_for(c)
            if c in labels and s_["kind"] == "cat":
                spec["values"] = sorted(set(s_["values"]) | set(labels[c]))
            bs = plots.binning(x, spec, s_)
            if nm1:
                own = [i for i, r in enumerate(results) if r.get("column") == c]
                m_plot = mask if not own else pop_mask(exclude=own)
                tag = " (N-1)" if own else ""
            else:
                m_plot, tag = mask, ""
            fig = plots.hist1d(x, W, pop_base, m_plot, bs, spec.get("title", c) + tag, xlabel=c, normalise=norm,
                               logy=logy, ratio=ratio, stack=stack_arg,
                               base_key=(data_key, c, wname, alias_sig))
            grid[k % ncol].plotly_chart(fig, use_container_width=True, key=f"plot_{c}")
        except Exception as e:      # noqa: BLE001
            grid[k % ncol].error(f"{c}: {type(e).__name__}: {e}")

with tabs["2D"]:
    a1, a2, a3, a4, a5 = st.columns([2, 2, 2, 2, 1])
    xs = a1.selectbox("x", all_cols, index=0, key=f"{state_prefix}x2d")
    ys = a2.selectbox("y", all_cols, index=min(1, len(all_cols) - 1), key=f"{state_prefix}y2d")
    norms2d = {"Σw": "none", "% of total": "total", "% of row": "row", "% of column": "column"}
    n2d = a3.selectbox("Normalise", list(norms2d), key=f"{state_prefix}norm2d",
                       help="row: each y bin adds up to 100 %; column: each x bin adds up to 100 %")
    texts2d = {"none": None, "value": "value", "entries (N)": "entries"}
    t2d = a4.selectbox("Numbers in bins", list(texts2d), key=f"{state_prefix}text2d",
                       help="value: the coloured quantity; entries: unweighted number of entries")
    logz = a5.toggle("Log z", key="opt_logz")
    try:
        bx = plots.binning(view[xs], {"labels": labels.get(xs)}, stats_for(xs))
        by = plots.binning(view[ys], {"labels": labels.get(ys)}, stats_for(ys))
        st.plotly_chart(plots.hist2d(view[xs], view[ys], w, mask, bx, by, xs, ys, logz, norm=norms2d[n2d],
                                     text=texts2d[t2d]), use_container_width=True)
    except Exception as e:      # noqa: BLE001
        st.error(f"{type(e).__name__}: {e}")

def show_composition(cs, key):
    """One composition panel from a spec dict(rows, segments, norm, facet, horizontal, where, title, label_min)."""
    m = mask if cs.get("selection", True) else np.ones(len(view), bool)
    if cs.get("where"):
        m = m & view.eval(cs["where"]).astype(bool)
    fac = cs.get("facet")
    fig, tab = plots.composition(
        view[cs["rows"]], names_for(cs["rows"]), view[cs["segments"]], names_for(cs["segments"]), W, m,
        norm=cs.get("norm", "row"), horizontal=cs.get("horizontal", True),
        facet=(view[fac], names_for(fac)) if fac else None, title=cs.get("title", ""),
        colors=colors.get(cs["segments"]), label_min=cs.get("label_min", 6.0), row_title=cs["rows"])
    st.plotly_chart(fig, use_container_width=True, key=f"comp_{key}")
    with st.expander("Table" + (f" · where `{cs['where']}`" if cs.get("where") else "")):
        num = {c: "{:.2f}" for c in tab.columns if c not in ("panel", cs["rows"], "N", "Σw")}
        show_table(tab, {**num, "N": "{:,.0f}", "Σw": "{:.5g}"})
        st.download_button("CSV", tab.to_csv(index=False), file_name=f"composition_{cs['rows']}_{cs['segments']}.csv",
                           mime="text/csv", key=f"compcsv_{key}")


if "Composition" in tabs:
    with tabs["Composition"]:
        comp_specs = [c for c in (P.get("compositions") or [] if P else [])
                      if all(c[k] in all_cols for k in ("rows", "segments") if c.get(k))
                      and (not c.get("facet") or c["facet"] in all_cols)]
        if cat_cols:
            d0 = comp_specs[0] if comp_specs else {}
            norms = {"% of row": "row", "% of total": "total", "Σw (yields)": "yields"}

            def _ix(opts, v, fallback=0):
                return opts.index(v) if v in opts else fallback

            b1, b2, b3, b4, b5 = st.columns([2, 2, 2, 2, 1])
            rows_c = b1.selectbox("Rows (bars)", cat_cols, index=_ix(cat_cols, d0.get("rows")),
                                  key=f"{state_prefix}comp_rows")
            seg_c = b2.selectbox("Segments (stacked)", cat_cols,
                                 index=_ix(cat_cols, d0.get("segments"), min(1, len(cat_cols) - 1)),
                                 key=f"{state_prefix}comp_seg")
            nlab = b3.radio("Normalise", list(norms), horizontal=True, key=f"{state_prefix}comp_norm",
                            index=_ix(list(norms.values()), d0.get("norm", "row")))
            fac_opts = ["(none)"] + [c for c in cat_cols if c not in (rows_c, seg_c)]
            fac_c = b4.selectbox("Split into panels by", fac_opts, key=f"{state_prefix}comp_facet",
                                 help="one panel for all entries plus one per class (small multiples)")
            horiz = b5.toggle("Horizontal", value=d0.get("horizontal", True), key=f"{state_prefix}comp_h")
            c1, c2 = st.columns([4, 1])
            where = c1.text_input("Only entries with (expression, on top of the sidebar selection)",
                                  key=f"{state_prefix}comp_where",
                                  placeholder="e.g. match_status == 0")
            use_sel = c2.toggle("Sidebar selection", value=True, key=f"{state_prefix}comp_sel")
            try:
                show_composition(dict(rows=rows_c, segments=seg_c, norm=norms[nlab], horizontal=horiz,
                                      facet=None if fac_c == "(none)" else fac_c, where=where, selection=use_sel,
                                      title=f"{seg_c} per {rows_c}"), "builder")
            except Exception as e:      # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")
        else:
            st.info("No categorical columns loaded (integer-like with ≤ 20 values, or with labels).")
        if comp_specs:
            st.markdown("---\n**Preset panels** (sidebar selection applied)")
            for i, cs in enumerate(comp_specs):
                try:
                    show_composition(cs, f"preset{i}")
                except Exception as e:      # noqa: BLE001
                    st.error(f"{cs.get('title', i)}: {type(e).__name__}: {e}")

with tabs["Cutflow & yields"]:
    h1, h2 = st.columns([5, 1])
    h1.markdown("**Cutflow** (controls applied cumulatively, in sidebar order"
                + (", one per level: each starts from the events of that tree)" if LEVELS else ")"))
    cf_log = h2.toggle("Log x", key="opt_cf_log")
    for L in (LEVELS or [None]):
        if L is None:
            cf = plots.cutflow(sel_results, w, page=page_results)
        else:
            st.markdown(f"**{L} level** · weight: {w_lv[L][1]}")
            cf = plots.cutflow([r for r in sel_results if r["level"] == L], np.nan_to_num(w_lv[L][0]),
                               base_mask=flag[L], page=[r for r in page_results if r["level"] == L])
        st.plotly_chart(plots.cutflow_fig(cf, logx=cf_log), use_container_width=True, key=f"cutflow_fig_{L}")
        show_table(cf, {"sumw": "{:.5g}", "err": "{:.3g}", "eff_step": "{:.4f}", "eff_total": "{:.4f}"})
        st.download_button("Cutflow CSV", cf.to_csv(index=False), file_name=f"cutflow_{pid}{'_' + L if L else ''}.csv",
                           mime="text/csv", key=f"cutflow_csv_{L}")
    if LEVELS:
        lm_ = level_masks()
        a_, b_ = lm_[MAIN], lm_[SECOND]
        wm, ws, wx = (np.nan_to_num(w_lv[x][0]).astype(np.float64) for x in (MAIN, SECOND, match_w))
        rows = [(f"{MAIN}-selected", a_, wm), (f"{SECOND}-selected", b_, ws), ("matched", a_ & b_, wx),
                (f"{MAIN} only (fakes)", a_ & ~b_, wm), (f"{SECOND} only (misses)", b_ & ~a_, ws)]
        summ = pd.DataFrame([dict(events=n_, N=int(m_.sum()), sumw=float(w_[m_].sum())) for n_, m_, w_ in rows])
        with np.errstate(divide="ignore", invalid="ignore"):
            summ[f"N / {MAIN}-sel."] = summ.N / summ.N[0]
            summ[f"N / {SECOND}-sel."] = summ.N / summ.N[1]
        st.markdown(f"**Matching** ({MAIN} ⟷ {SECOND} on {', '.join(df.attrs.get('keys', []))}; matched Σw with "
                    f"the {match_w} weight)")
        show_table(summ, {"N": "{:,.0f}", "sumw": "{:.5g}", f"N / {MAIN}-sel.": "{:.4f}",
                          f"N / {SECOND}-sel.": "{:.4f}"})
    if cat_cols:
        yc = st.selectbox("Yields by", cat_cols, index=cat_cols.index("origin") if "origin" in cat_cols else 0,
                          key=f"{state_prefix}yields_by")
        codes = view[yc]
        names = labels.get(yc) or {v: plots.fmt_label(v, None) for v in np.unique(codes).tolist()}
        yt = plots.yields_by(codes, names, w, mask)
        show_table(yt, {"sumw": "{:.5g}", "err": "{:.3g}", "fraction": "{:.4f}"})
        st.download_button("Yields CSV", yt.to_csv(index=False), file_name=f"yields_{yc}_{pid}.csv", mime="text/csv")



def partner(c, candidates):
    """Second-level column most likely paired with `c`: the one sharing the longest name ending (pt0 ↔ pl_pt0)."""
    rev = c[::-1]
    return max(candidates, key=lambda x: (len(os.path.commonprefix([x[::-1], rev])), -len(x))) if candidates else None


def unfold_bins(txt, xr_c, xt_c, preset_bins=None):
    """Binning shared by both levels: preset bins, typed `lo:hi:n` or edges `e0, e1, …`, else automatic (the reco
    column's classes / 10 bins over its slider range)."""
    txt = (txt or "").strip()
    if txt:
        if ":" in txt:
            lo_, hi_, n_ = (float(v) for v in txt.split(":"))
            return ("num", np.linspace(lo_, hi_, int(n_) + 1))
        e_ = sorted(float(v) for v in re.split(r"[,\s]+", txt) if v)
        if len(e_) < 2:
            raise ValueError("give at least two bin edges, or lo:hi:n")
        return ("num", np.asarray(e_))
    if preset_bins is not None:
        return plots.binning(view[xr_c], {"bins": preset_bins}, stats_for(xr_c))
    sr, st_ = stats_for(xr_c), stats_for(xt_c)
    if sr["kind"] == "cat" or xr_c in labels:
        vals = sorted(set(sr.get("values") or []) | set(st_.get("values") or []) | set(labels.get(xr_c) or []))
        return ("cat", vals, [plots.fmt_label(v, labels.get(xr_c)) for v in vals])
    lo_, hi_ = ctl.slider_limits(sr)
    return ("num", np.linspace(lo_, hi_ if hi_ > lo_ else lo_ + 1, 11))


class NoPair(Exception):
    """No variable pair chosen in the Unfolding tab (nothing to show)."""


if "Unfolding" in tabs:
    with tabs["Unfolding"]:
        ucols = {L: [c for c in all_cols if level_of(c) == L and c not in df.attrs.get("keys", [])
                     and not c.startswith("in_")] for L in LEVELS}
        pairs = {(u.get("title") or f"{u[MAIN]} ⟷ {u[SECOND]}"): u for u in ((P.get("unfold") or []) if P else [])
                 if u[MAIN] in all_cols and u[SECOND] in all_cols}
        u1, u2, u3 = st.columns(3)
        pick = u1.selectbox("Variable", list(pairs) + ["any pair"], key=f"{state_prefix}unf_pair")
        xr_c = xt_c = pbins = None
        if pick == "any pair" and (not ucols[MAIN] or not ucols[SECOND]):
            st.info(f"Load at least one column on each level ({MAIN}, {SECOND}).")
        elif pick == "any pair":
            xr_c = u2.selectbox(f"{MAIN} variable", ucols[MAIN], key=f"{state_prefix}unf_x_{MAIN}")
            guess = partner(xr_c, ucols[SECOND])
            xt_c = u3.selectbox(f"{SECOND} variable", ucols[SECOND], index=ucols[SECOND].index(guess),
                                key=f"{state_prefix}unf_x_{SECOND}_{xr_c}")
            pbins = None
        else:
            xr_c, xt_c, pbins = pairs[pick][MAIN], pairs[pick][SECOND], pairs[pick].get("bins")
            u2.markdown(f"{MAIN}: `{xr_c}`  \n{SECOND}: `{xt_c}`")
        b1, b2, b3, b4 = st.columns([2, 2, 2, 2])
        bins_txt = b1.text_input("Bins", key=f"{state_prefix}unf_bins",
                                 placeholder="auto · 0, 25, 50, 100 · 0:200:20",
                                 help="empty: automatic (or the preset's); edges separated by commas; lo:hi:n for n "
                                      "equal bins. Same binning on both levels.")
        unorm = {"% of truth row": "row", "Σw": "none", "% of reco column": "column", "% of total": "total"}
        un = b2.selectbox("Migration matrix", list(unorm), key=f"{state_prefix}unf_norm",
                          help=f"rows: {SECOND} bins (y), columns: {MAIN} bins (x); matched events only")
        utext = {"value": "value", "entries (N)": "entries", "none": None}
        ut = b3.selectbox("Numbers in bins", list(utext), key=f"{state_prefix}unf_text")
        b4.radio("Matched events weighted with", LEVELS, horizontal=True, key=f"{state_prefix}match_w",
                 help="weight that fills the migration matrix (and the 'matched' events); acceptance and efficiency "
                      "come from the same matrix")
        uover = st.toggle("Values outside the binning in the edge bins", key=f"{state_prefix}unf_over",
                          help="off: entries outside the bins are left out (they migrate out of the range)")
        try:
            if xr_c is None:
                raise NoPair
            bspec = unfold_bins(bins_txt, xr_c, xt_c, pbins)
            lm_ = level_masks()
            sel_r, sel_t = lm_[MAIN], lm_[SECOND]
            wr, wt, wx = (np.nan_to_num(w_lv[x][0]) for x in (MAIN, SECOND, match_w))
            M, C, utab = plots.unfolding(view[xr_c], view[xt_c], wx, wr, wt, sel_r, sel_t, bspec, overflow=uover)
            g1, g2 = st.columns(2)
            g1.plotly_chart(plots.hist2d(view[xr_c], view[xt_c], wx, sel_r & sel_t, bspec, bspec,
                                         f"{xr_c} ({MAIN})", f"{xt_c} ({SECOND})", norm=unorm[un],
                                         text=utext[ut], overflow=uover), use_container_width=True, key="unf_mig")
            g2.plotly_chart(plots.unfolding_fig(utab, f"{xr_c} ⟷ {xt_c}"), use_container_width=True, key="unf_fig")
            show_table(utab, {c: "{:.4f}" for c in ("acceptance", "efficiency", "purity", "stability")}
                       | {c: "{:.5g}" for c in ("sumw_reco", "sumw_truth", "matched_reco", "matched_truth")}
                       | {c: "{:,.0f}" for c in ("N_reco", "N_truth", "N_matched_reco", "N_matched_truth")})
            d1, d2 = st.columns(2)
            d1.download_button("Table CSV", utab.to_csv(index=False), file_name=f"unfolding_{xr_c}.csv",
                               mime="text/csv", key="unf_csv")
            labs = utab["bin"].tolist()
            d2.download_button("Migration matrix CSV (rows: truth bins)",
                               pd.DataFrame(M, index=labs, columns=labs).to_csv(), file_name=f"migration_{xr_c}.csv",
                               mime="text/csv", key="unf_mig_csv")
            st.caption(f"Columns *_reco / *_truth: per {MAIN} / {SECOND} bin. acceptance = matched / {MAIN}-selected "
                       f"(per {MAIN} bin), efficiency = matched / {SECOND}-selected (per {SECOND} bin), purity / "
                       f"stability = diagonal / column / row sum. Weights: {MAIN} {w_lv[MAIN][1]}, {SECOND} "
                       f"{w_lv[SECOND][1]}, matched {w_lv[match_w][1]}.")
        except NoPair:
            pass
        except Exception as e:      # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")

for tab, (title, fn) in zip(all_tabs[len(tab_names):], custom.items()):
    with tab:
        try:
            items = fn(dict(view=view, mask=mask, weight=w, results=results, labels=labels, levels=LEVELS,
                            level_masks=level_masks() if LEVELS else None, population=pop_name))
        except Exception as e:      # noqa: BLE001
            st.error(f"{type(e).__name__}: {e}")
            continue
        for it in items:
            if isinstance(it, tuple) and it[0] == "metrics":
                cols = st.columns(len(it[1]))
                for col, (lab, val) in zip(cols, it[1]):
                    col.metric(lab, val)
            elif isinstance(it, tuple) and it[0] == "table":
                show_table(it[1])
            elif isinstance(it, pd.DataFrame):
                show_table(it)
            elif isinstance(it, str):
                st.markdown(it)
            else:
                st.plotly_chart(it, use_container_width=True)
