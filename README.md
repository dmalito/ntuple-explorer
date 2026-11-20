# Ntuple Explorer

A local web app to explore ROOT ntuples interactively: pick a file and a tree, put sliders and selectors on any
column, and see histograms, stacks, compositions, cutflows and yields update as you change the selection.

- **Generic**: works on any flat or vector-branch TTree, with no code to write.
- **Presets**: a short Python file can describe an analysis: derived columns, categories, ready-made
  controls, default plots and extra tabs. You then get the same study at one click instead of one notebook per
  question.
- **Fast on millions of events**: the file is read once with ROOT `RDataFrame` (multithreaded). The columns are
  cached on disk, and every interaction runs on numpy arrays in memory (~1–3 s on 5–10 M events).

## Install

ROOT with PyROOT (6.28 or newer) has to be importable from Python. It is not on PyPI; use conda-forge
(`conda install -c conda-forge root`), an LCG view, or your own ROOT build. Then:

```bash
git clone <this repository> && cd ntuple-explorer
pip install -r requirements.txt
```

## Run

```bash
python -m ntuple_explorer --data /path/to/ntuples                         # generic mode
python -m ntuple_explorer --data /path/to/ntuples --presets /path/to/my_presets
```

Then open http://localhost:8501.

| option | meaning |
| --- | --- |
| `--data DIR` | folder whose `.root` files are listed in the app (any other path can be typed in); relative paths in presets are resolved against it |
| `--presets DIR` | folder with preset files; can be given several times |
| `--cache DIR` | where the column cache goes (default `<data>/.ntuple_cache`) |
| `--port N` | port (default 8501) |

The server listens on **localhost only**. The app compiles the ROOT (C++) expressions typed into it, so it must not
be reachable by other people. `--address` changes this; use it only on a machine and network you trust.

## How it works

1. **Load**: every column is a ROOT expression evaluated per entry in one RDataFrame event loop:
   - a scalar branch;
   - a reduction of vector branches: `nx_at(v, 0)` (element 0, −999 if the vector is shorter), `v.size()`,
     `Sum(v > 0)`, …;
   - any other expression.

   Raw vectors are never pulled into Python. The result is cached in `.ntuple_cache/`, keyed on file (path, size,
   modification time), tree and expressions. A second load of the same columns takes well under a second.
2. **Select**: the sidebar controls build boolean masks with numpy; nothing is re-read from disk.
3. **Plot**: weighted histograms with Plotly.

Unknown branch names are reported as a normal error before anything is compiled: an unknown identifier in a jitted
RDataFrame expression would otherwise abort the whole Python process.

### Tabs

| tab | content |
| --- | --- |
| Plots | 1D histograms: all loaded entries (grey line) vs selected (filled), optional ratio panel (fraction kept per bin), normalisation, log y, stacking by a category. In N-1 mode (default) each plot ignores the cuts on its own variable |
| 2D | weighted 2D histogram of any two columns |
| Composition | one category stacked inside the classes of another (e.g. truth origin × reconstruction status): bars at 100 % per row, % of the total or Σw; optional panels split by a third category; extra `where` cut; table + CSV |
| Cutflow & yields | the controls applied cumulatively in sidebar order (N, Σw, efficiencies) and yields per class of a category; CSV |
| custom | any tabs a preset defines |

The sidebar also has **Save / load selection** (all control values as JSON) and **Selection as expression**
(the active cuts as text).

## Generic mode

Pick a file, a tree, scalar branches, vector branches with reducers (element 0/1/2, size, count > 0, sum, max, …),
extra `name = ROOT expression` columns and an optional pre-filter (applied while reading). Then add a control for
any loaded column:
- integer-like columns with ≤ 20 values get a multiselect;
- other columns get a range slider (0.1–99.9 % quantiles; a handle left at its limit means "no cut on that side").

The expression cut box takes pandas.eval syntax over loaded columns (`&`, `|`, `~`, `and`, `or`, `==`, `abs()`).

## Presets

A preset is a Python file defining `PRESET = dict(...)`. Files whose name starts with `_` are shared helper modules,
not presets. A preset file is re-read on every interaction, so edits apply on the next click. Edits to `_` helper
modules, and new preset files, need a restart.

```python
# my_presets/ttbar_ljets.py
import numpy as np

PRESET = dict(
    name="tt̄ l+jets",
    files={"nominal": "ttbar.root"},                       # relative to --data
    tree="reco",
    columns=dict(                                           # ROOT expressions, read once
        njet="int(jet_pt.size())",
        nbjet="int(Sum(jet_isbtagged == 1))",
        lep_pt="nx_at(el_pt, 0) > 0 ? nx_at(el_pt, 0) / 1000. : nx_at(mu_pt, 0) / 1000.",
        met="met_met / 1000.",
        is_e="int(el_pt.size() == 1)",
        w="weight_mc * weight_pileup",
    ),
    categories={"channel": {"e+jets": "el_pt.size() == 1", "μ+jets": "mu_pt.size() == 1", "other": None}},
    derived=dict(ht=lambda v: v["lep_pt"] + v["met"]),       # numpy columns computed after loading
    weight="w",
    controls=[
        dict(kind="multiselect", column="channel", label="Channel"),
        dict(kind="range", column="njet", limits=(0, 10), default=(4, 10), label="n jets"),
        dict(kind="range", column="nbjet", limits=(0, 4), default=(1, 4), label="n b-jets"),
        dict(kind="range", column="met", label="MET [GeV]"),
        dict(kind="alias", label="Weight", options={"MC weight": {"weight": "w"},
                                                    "unweighted": {"weight": lambda v: np.ones(len(v))}}),
        dict(kind="expr", label="Extra cut"),
    ],
    plots=[dict(column="lep_pt", bins=(0, 300, 30)), dict(column="met", bins=(0, 300, 30)), dict(column="njet")],
    stack_options=["channel"],
    compositions=[dict(rows="njet", segments="channel", norm="row", title="channel mix per jet multiplicity")],
)
```

| key | meaning |
| --- | --- |
| `name` | shown in the Mode dropdown (required) |
| `tree` | tree name (required) |
| `columns` | `{name: ROOT expression}` loaded once (required). Columns using branches missing from a file are skipped, so one preset can serve several ntuple versions |
| `files` | `{label: path}`; relative paths are relative to `--data` |
| `helpers` | C++ code declared before reading (functions to use in the expressions) |
| `prefilter` | ROOT boolean expression; entries failing it are never loaded |
| `categories` | `{name: {class label: ROOT bool expression or None}}` → one int column: index of the first true class; `None` = everything else |
| `labels` | `{column: {value: label}}` (or a list) for axes and selectors |
| `derived` | `{name: f(view) -> array}`: numpy columns computed after loading |
| `weight` | column name of the event weight |
| `friends` | `{name: dict(tree, columns, categories, keys=["runNumber", "eventNumber"], prefilter, found, fill)}`: columns of another tree of the same file, joined to each entry on the event keys (e.g. truth-level info for reco events). `found` names an int8 column, 1 if the event is in that tree after its prefilter; missing entries get `fill` (default −1, NaN for floats) |
| `controls` | list of control specs (below), rendered in order; later controls see the column choices made by earlier ones |
| `plots` | `[dict(column, title, bins=(lo, hi, n) or edges, nbins)]` shown by default |
| `stack_options` | categorical columns offered for stacking |
| `compositions` | `[dict(rows, segments, norm="row"/"total"/"yields", facet=None, horizontal=True, where=None, title, label_min=6)]`: panels of the Composition tab |
| `colors` | `{column: {class label: colour}}` for stacks and compositions (default: a fixed palette order) |
| `custom` | `{tab title: f(ctx) -> list}`; `ctx = dict(view, mask, weight, results, labels)`; returned items: plotly figures, DataFrames, markdown strings, `("table", df)`, `("metrics", [(label, value)])` |

Control specs (`kind`):

| kind | fields | effect |
| --- | --- | --- |
| `alias` | `options={label: {name: column or f(view)}}`, `default` | switches which column a name points to (e.g. the weight, or one of several algorithms) |
| `select` | `options={label: expression or None}`, `default` | one of several cuts |
| `multiselect` | `column`, `labels`, `default=[values]` | keep the chosen values |
| `checkboxes` | `options={label: expression}`, `default=[labels]`, `combine="or"/"and"` | e.g. lepton channels |
| `range` | `column`, `limits=(lo, hi)`, `default=(lo, hi)`, `step` | slider; a handle at a limit = open-ended |
| `toggle` | `expr`, `default`, `expr_off` | cut on/off |
| `expr` | `default` | free cut expression |

Common optional fields: `label`, `help`, `dropdown=True` (selectbox instead of radio).
Expressions in controls are pandas.eval strings over view columns, or `f(view) -> bool array`.

## Notes

- Memory: columns are down-cast (float32, smallest integer type). About 30 columns on 5 M events take ~400 MB.
- `Clear cache` in the sidebar deletes the cache folder. Cached columns are also invalidated automatically when
  the ROOT file changes.
- Tables are rendered as HTML, so they work even where pyarrow is broken.
- `Max entries` in the sidebar reads only the first N entries, for quick looks at huge files.
