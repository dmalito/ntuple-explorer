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
| `--presets DIR` | folder with preset files; can be given several times. On a preset name clash the first folder wins |
| `--cache DIR` | where the column cache goes (default `<data>/.ntuple_cache`) |
| `--port N` | port (default 8501) |

Without `--data` / `--presets`, the launcher uses the environment variables `NTUPLE_EXPLORER_DATA` and `NTUPLE_EXPLORER_PRESETS` (several folders separated by `:`), e.g. set once in your shell profile.

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
2. **Select**: the controls build boolean masks with numpy; nothing is re-read from disk (see *Selections* below).
3. **Plot**: weighted histograms with Plotly.

Unknown branch names are reported as a normal error before anything is compiled: an unknown identifier in a jitted
RDataFrame expression would otherwise abort the whole Python process.

### Tabs

| tab | content |
| --- | --- |
| Plots | 1D histograms of the columns picked in *Variables to plot* (a collapsible box above the tabs): all loaded entries (grey line) vs selected (filled), optional ratio panel (fraction kept per bin), normalisation, log y, stacking by a category. In N-1 mode (default) each plot ignores the cuts on its own variable |
| 2D | weighted 2D histogram of any two columns: Σw or normalised to the total, per row or per column (e.g. migration-style matrices); optional numbers in each bin (the shown value or the unweighted entries) |
| Composition | one category stacked inside the classes of another (e.g. truth origin × reconstruction status): bars at 100 % per row, % of the total or Σw; optional panels split by a third category; extra `where` cut; table + CSV. Shown only when the preset defines `compositions` |
| Cutflow & yields | the controls applied cumulatively in sidebar order: bar chart and table (N, Σw, efficiencies) and yields per class of a category; CSV. With two matched trees: one cutflow per level and a matching summary |
| Unfolding | only with two matched trees (see *Matching two trees*): migration matrix, acceptance, efficiency, purity and stability of one variable; tables and CSV |
| custom | any tabs a preset defines |

The sidebar also has **Save / load selection** (all control values as JSON, page cuts and new columns included) and
**Selection as expression** (the active cuts as text).

## Selections

There are two kinds of cuts, and every plot, table and header number uses both:

- **Sidebar selection**: the preset controls and *Cuts on any column*. Each control that cuts is one cutflow step, in
  sidebar order.
- **Page cuts**: sliders and selectors in the *Page cuts* box above the tabs, for quick looks on top of the
  selection. Add one with *Add a page cut on*, or mark a preset control `place="main"`. They are not cutflow steps:
  the cutflow ends with a single *after page cuts* row, and the header says how many entries of the sidebar selection
  they keep.

**New columns** (sidebar) are computed in memory from loaded columns, without reloading: one `name = expression` per
line, for example

```text
lep_ok  = pass_e == 1 || pass_mu == 1     # 0/1 column -> a selector
nsm_max = max(nsm_a, nsm_b)               # slider at >= 1 = "nsm_a >= 1 or nsm_b >= 1"
nsm_tot = nsm_a + nsm_b
```

*Combine columns* writes such lines for you: pick columns and OR / AND (of `column ≠ 0`), max, min, sum or count.
A new column behaves like a loaded one: sliders and selectors (sidebar or page), plots, stacks, 2D, compositions.

Expressions (new columns, expression cuts, preset `select` / `checkboxes` / `toggle`) accept `+ - * / ** %`,
comparisons (also chained, `1 < x <= 3`), `& | ~`, `and or not`, `&& || !` (`|` and `&` bind looser than comparisons,
so `a > 0 | b > 0` is an OR of two conditions), `a if cond else b`, and the functions `max`, `min`,
`where(cond, a, b)`, `count(cond, ...)` (how many are true), `isin(x, v1, v2, ...)`, `abs`, `sqrt`, `exp`, `log`,
`log10`, `sin`, `cos`, `tan`, `arctan2`, `hypot`, `floor`, `ceil`. Plain expressions run through numexpr, the
others through numpy.

## Matching two trees (reco ↔ truth, unfolding)

Two trees of the same file can be matched event by event, on keys such as `runNumber` and `eventNumber`: generic mode
has *Match events with a second tree* in the data section (second tree, keys, level names, its branches), presets
have `match` (below). Both trees are read and *outer-joined*: one row per event found in either tree, with the flags
`in_<level>` (e.g. `in_reco`, `in_truth`). A level's columns are NaN (floats) or −1 (integers) for events missing from
it; slider ranges and categories come from that level's events only.

Each level has its own weight and its own cuts: every control belongs to the level of its column (presets:
`level="truth"`; derived columns and the `in_*` flags count as the main level). A switch above the tabs picks the
events every plot and table shows:

| events | definition |
| --- | --- |
| *reco*-selected | in the reco tree and passing the reco cuts |
| *truth*-selected | in the truth tree and passing the truth cuts |
| matched | both |
| fakes | reco-selected, not truth-selected |
| misses | truth-selected, not reco-selected |

The Cutflow tab gets one cutflow per level and a matching summary (N, Σw, fractions). The **Unfolding** tab takes a
reco variable and its truth partner (guessed from the name, or a preset `unfold` pair) with one binning for both
(automatic, edges `0, 25, 50, 100` or `lo:hi:n`) and shows, with M[i, j] the matched Σw in truth bin i and reco bin j:

- the migration matrix (truth bins as rows; normalised per truth row, per reco column, to the total, or Σw);
- acceptance per reco bin = Σ_i M[i, j] / Σw(reco-selected in bin j);
- efficiency per truth bin = Σ_j M[i, j] / Σw(truth-selected in bin i);
- purity = M[j, j] / Σ_i M[i, j], stability = M[i, i] / Σ_j M[i, j].

The matrix is filled with the reco or the truth weight (a choice in the tab, also used for the *matched* events);
acceptance and efficiency come from the same matrix. Values outside the binning are left out (migrations out of the
range) unless *Values outside the binning in the edge bins* is on.

## Generic mode

Pick a file, a tree, scalar branches, vector branches with reducers (element 0/1/2, size, count > 0, sum, max, …),
extra `name = ROOT expression` columns and an optional pre-filter (applied while reading). Then add a control for
any loaded column:

- columns with exactly two values (0/1, −1/1, …) get a 3-way switch *all / a / b*; other columns with labels get a
  multiselect (clickable chips for page cuts);
- other integer-like columns with ≤ 20 values (counts such as n jets, n soft muons) get an integer range slider, or a
  multiselect if you switch "Integer columns with ≤ 20 values as" to *selector* in the same expander;
- other columns get a range slider (a handle left at its limit means "no cut on that side"). Its range is set in
  the sidebar *Settings*: the 0.1–99.9 % quantiles *snapped* to the true min / max when those are within 5 % of
  the range (default: hard thresholds such as pT > 20 GeV stay reachable, far tails do not stretch the slider),
  plain quantiles, or the full min–max range. Preset `limits` always win.

The expression cut box takes the expressions described in *Selections*.

Vectors of 4-vectors (`ROOT::Math::LorentzVector`, `TLorentzVector`, split or not) are listed once, marked
*(4-vector)*. Element *i* gives the components chosen in *…4-vector components* (pt, eta, phi, E, m, in the stored
units), e.g. `jet_p4__pt0`; *size* is the multiplicity; *sum* / *max* / *min* act on pt. Vectors of other objects
offer *size* only; single objects are not listed, but extra columns can read them (`x_pt = x.Pt()`). The helpers
`nx_pt(v, i)`, `nx_eta`, `nx_phi`, `nx_e`, `nx_m` (−999 if the element is missing) and `nx_sum_pt(v)`,
`nx_max_pt(v)`, `nx_min_pt(v)` can be used in any ROOT expression, as can `nx_pts(v)` / `nx_etas` / `nx_phis` /
`nx_es` / `nx_ms` (the component of every element, as a vector) and `nx_kth(values, k)` (k-th largest, NaN if
fewer).

*'At least N above X' cuts on* (number vectors and 4-vectors, pt) adds an *at least N objects with value ≥ X*
control to the selection after Load (the `at_least` control kind below, up to N = 4).

## Presets

A preset is a Python file defining `PRESET = dict(...)`, in a `--presets` folder or in `presets/` at the repository
root (git-ignored, for your own analyses; searched after the `--presets` folders). Files whose name starts with `_`
are shared helper modules,
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
| `match` | `dict(tree, name="truth", main="reco", keys=["runNumber", "eventNumber"], columns, categories, prefilter, weight, fill=-1)`: a second level matched event by event (see *Matching two trees*); `columns` names must differ from the main ones, `weight` is a column of this level |
| `unfold` | `[dict(<main level>=column, <second level>=column, bins=(lo, hi, n) or edges, title)]`, e.g. `dict(reco="lep_pt", truth="pl_lep_pt", bins=[25, 50, 100, 200])`: variable pairs offered in the Unfolding tab |
| `friends` | `{name: dict(tree, columns, categories, keys=["runNumber", "eventNumber"], prefilter, found, fill)}`: columns of another tree of the same file, joined to each entry on the event keys (e.g. truth-level info for reco events). `found` names an int8 column, 1 if the event is in that tree after its prefilter; missing entries get `fill` (default −1, NaN for floats) |
| `controls` | list of control specs (below), rendered in order; later controls see the column choices made by earlier ones |
| `plots` | `[dict(column, title, bins=(lo, hi, n) or edges, nbins)]` shown by default |
| `stack_options` | categorical columns offered for stacking |
| `compositions` | `[dict(rows, segments, norm="row"/"total"/"yields", facet=None, horizontal=True, where=None, title, label_min=6)]`: panels of the Composition tab (the tab only appears when this is set) |
| `colors` | `{column: {class label: colour}}` for stacks and compositions (default: a fixed palette order) |
| `custom` | `{tab title: f(ctx) -> list}`; `ctx = dict(view, mask, weight, results, labels)`; returned items: plotly figures, DataFrames, markdown strings, `("table", df)`, `("metrics", [(label, value)])` |

Control specs (`kind`):

| kind | fields | effect |
| --- | --- | --- |
| `alias` | `options={label: {name: column or f(view)}}`, `default` | switches which column a name points to (e.g. the weight, or one of several algorithms) |
| `select` | `options={label: expression or None}`, `default` | one of several cuts |
| `multiselect` | `column`, `labels`, `default=[values]` | keep the chosen values: a multiselect box in the sidebar, clickable chips on the page (`place="main"`); a column with exactly two values (0/1, −1/1, two classes) gets a 3-way switch *all / a / b* instead, in both places |
| `checkboxes` | `options={label: expression}`, `default=[labels]`, `combine="or"/"and"` | e.g. lepton channels |
| `range` | `column`, `limits=(lo, hi)`, `default=(lo, hi)`, `step` | slider; a handle at a limit = open-ended |
| `toggle` | `expr`, `default`, `expr_off` | cut on/off |
| `expr` | `default` | free cut expression |
| `at_least` | `vector` (4-vector branch) + `of="pt"` (or eta, phi, E, m), or `values` (any vector expression, e.g. `jet_pt / 1000.`); `where` (object requirement, e.g. `abs(jet_eta) < 2.5` or `abs(nx_etas(jet_p4)) < 2.5`), `max_n=4`, `default=(N, X)`, `what` | *at least N objects with value ≥ X*: N (any, 1 … max_n) and a threshold slider; at the slider's lower end only the count is required. Works because "at least N above X" = "the N-th largest above X": the loader stores the 1st … max_n-th largest value per entry (sorted in C++); `where` is fixed at load time |

Common optional fields: `label`, `help`, `dropdown=True` (selectbox instead of radio), `level` (with `match`: the
level the cut belongs to, default the main one), `place="main"` (a page cut
above the tabs instead of a sidebar cut; see *Selections*).
Expressions in controls are strings over view columns (syntax in *Selections*), or `f(view) -> bool array`.

## Notes

- Memory: columns are down-cast (float32, smallest integer type). About 30 columns on 5 M events take ~400 MB.
- `Clear cache` in the sidebar deletes the cache folder. Cached columns are also invalidated automatically when
  the ROOT file changes.
- Tables are rendered as HTML, so they work even where pyarrow is broken.
- `Max entries` in the sidebar reads only the first N entries, for quick looks at huge files (-1, the default,
  reads all).
