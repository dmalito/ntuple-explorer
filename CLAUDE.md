# Ntuple Explorer: notes for working on this repo

Local Streamlit app to explore ROOT ntuples interactively: sliders and selectors on per-event columns, histograms,
compositions, cutflows. User docs and the full preset format are in `README.md`; this file covers how the code is
built and how to work on it.

## Principles

- **Generic first.** Nothing analysis-specific goes in this repo: no branch names, physics definitions or file paths
  from one analysis. Analyses live in preset files in external folders (`--presets DIR`) or in the git-ignored
  `presets/` folder at the repo root (never commit it). If a feature is needed by a
  preset, add a generic mechanism (as `friends`, `compositions`, `colors` were), not a special case.
- **Read once, select in memory.** ROOT is touched only at load time: one `RDataFrame` event loop produces flat
  per-event numpy columns (vectors are reduced in C++). Every widget interaction is numpy on those arrays and must
  stay fast on ~10 M rows (aim: < 3 s per interaction with a full preset page).
- **Localhost only.** The app JIT-compiles C++ typed into its text boxes. The launcher and `.streamlit/config.toml`
  bind to localhost; keep it that way by default.

## Layout (`ntuple_explorer/`)

| file | role |
| --- | --- |
| `__main__.py` | launcher: `python -m ntuple_explorer --data DIR --presets DIR [--cache DIR] [--port N]`; sets `NTUPLE_EXPLORER_DATA` / `_PRESETS` / `_CACHE` and runs `streamlit run app.py` bound to localhost |
| `app.py` | the Streamlit page: sidebar (data source, controls, *New columns*, save/load state), header metrics, *Page cuts* box (controls with `place="main"` + page cuts on any column; results tagged `page=True`: in every mask, but only one final cutflow row), with two matched trees (`LEVELS`): a level per control (`level_of()`), `level_masks()`, the population switch (`pop_mask()`, `pop_base`, weight per population), per-level cutflows, Unfolding tab; tabs (Plots, 2D, Composition only if the preset has `compositions`, Cutflow & yields, Unfolding, preset custom tabs). `cached_load` (st.cache_resource) = main tree + joined friend trees |
| `loader.py` | ROOT side: `load()` (Define + one `AsNumpy`, down-cast, npz cache), `branch_info()` / `trees()` via uproot (metadata only), `check_exprs()` / `split_missing()`, `join_friend()` + `match_index()` (left join of another tree's columns on event keys), `join_level()` (outer join: a second level, flags `in_<level>`, attrs `levels` / `level_of`), generic-mode `REDUCERS` + `vector_columns()` (per element kind: number / 4-vector / object, from `element_kind()`), default helper `nx_at(v, i)` and `VECTOR_HELPERS` (`nx_pt` … `nx_max_pt`) |
| `view.py` | `View`: name → array through aliases (set by alias controls), derived callables (preset, and the app's *New columns*), loaded columns; memoised per rerun, cycle detection; `eval()` = pandas.eval + numexpr, falling back to `np_eval()` (whitelisted AST on numpy: `FUNCS` helpers, `&& \|\| !`, `a if c else b`) |
| `controls.py` | widget specs → `dict(label, mask, desc, column, aliases, kind, choice)`; `stats()` decides cat vs num; `slider_limits()` (`RANGE_MODE` setting), `_slider_grid()`, `_carry_slider()` (value survives a limit change); `at_least` = N radio + threshold slider on the k-th largest columns (`loader.nth_columns()`, hidden `__nth_*`); `combine()` ANDs masks |
| `plots.py` | weighted histograms with Plotly: `bin_index()` (lookup table / arithmetic / searchsorted), `hist1d()` (before vs after, ratio, stacks, cached "before" histogram), `hist2d()`, `composition()`, `cutflow()`, `yields_by()`, `unfolding()` + `unfolding_fig()`, palette + `segment_colours()` |
| `preset.py` | preset discovery in the `NTUPLE_EXPLORER_PRESETS` folders, `validate()` (allowed `KEYS`, control `KINDS`, composition `NORMS`), categories → nested-ternary int expression, friend columns |

## Things that bite

- A derived column (preset `derived`, alias target, *New column*) whose inputs are missing from the file raises when
  computed. `usable()` in `app.py` probes it once per rerun (through the stats cache) and drops it from `all_cols`;
  check membership in `all_cols`, not `name in view`, before offering a column anywhere.

- **An unknown identifier in a jitted RDataFrame expression aborts the whole Python process** (and the Streamlit
  server with it). Every expression is checked against the tree's branches first (`check_exprs`, regex
  `expr_branches`, `_NOT_BRANCHES` for keywords). Keep this check in front of anything new that reaches `Define` / `Filter`.
- `(u)int8_t` / `char` branches come back from `AsNumpy` as object arrays: `load()` wraps them in `int(...)`.
- A column that is still a vector per entry raises a `TypeError` asking for a reduction; never pull raw vectors.
- A JIT compile error *inside the event loop* (e.g. a numeric reducer such as `nx_at` on a vector of 4-vectors)
  is an uncaught C++ exception: `abort`, whole process gone. `branch_info()` lists top-level branches only (split
  objects appear once; uproot's `parent/parent.fX` sub-branch names are not valid column names) and tags the
  element kind, so generic mode only offers reducers that compile for it.
- Full loads run multithreaded: entries come back in a different order than in the file. Columns of one load are
  consistent with each other; compare with uproot / other code by event keys (`runNumber`, `eventNumber`), not
  by position.
- Streamlit "magic" writes any bare expression statement in `app.py` with `st.write`, which imports pyarrow and
  fails here: assign results (`_ = view[name]`), never leave an expression on its own line.
- Matched datasets fill a level's columns with NaN / −1 on events missing from it. Statistics (categories, slider
  ranges) must use that level's rows only: `ctl.stats()` asks the cache for `rows(name)` (`_StatsProxy.rows`).
  `hist1d()` caches the "before" histogram under `base_key`, which must identify the base mask (the population).
- A widget's ID depends on its parameters (e.g. a slider's min / max): when they change, Streamlit makes a new
  widget from `value` and writes to `st.session_state[key]` are ignored. To carry a value over, compute it and pass
  it as `value` (see the range control: open handles follow the limits, kept in `<key>__lim`).
- AppTest cannot drive single-choice button groups (`st.segmented_control`, `st.pills` single): any of them on the
  page breaks every `run()`. Use `st.radio(horizontal=True)` for single choices; multi-choice `st.pills` is fine.
- `st.dataframe` needs pyarrow, which is broken in some conda envs: tables are rendered as HTML (`show_table`).
  Do not reintroduce `st.dataframe`.
- Streamlit renders all tabs on every rerun, so every chart on the page costs time on each interaction.
  Measure before adding default panels.
- Colours follow the class, never its rank under the current cuts: explicit preset `colors` first, then the fixed
  `SERIES` order (by Σw over all loaded entries only when there are more than 8 classes, the rest fold into "other").
- Preset files are re-imported on every rerun (edits apply on the next click); `_*.py` helper modules and new preset
  files need a server restart.
- Cache key = (absolute path, size, mtime, tree, column expressions, entries, prefilter, helpers): changing helper
  code invalidates the cache of every preset that uses it.

## Testing

No test suite yet. What has been used so far:

1. A synthetic ROOT file made with `RDataFrame(N).Define(...).Snapshot(...)` (random vectors etc.), so tests need no
   real data. Keep such files out of git (`*.root` is ignored).
2. Headless runs with `streamlit.testing.v1.AppTest`: set the `NTUPLE_EXPLORER_*` env vars, then
   `AppTest.from_file("ntuple_explorer/app.py", default_timeout=300).run()`. Drive widgets by key
   (`mode`, `file_<mode>`, `g_tree`, `g_scal`, `g_vec`, `g_red`, `ctl_<preset stem>_<control index>`,
   `ctl_<pid>_comp_rows`, `ctl_<pid>_newcols`, `ctl_<pid>_page_cols`, …; a two-valued multiselect is the radio
   `<key>__sw`, page chips are the button group `<key>__pills`, an integer slider on a few-valued column is
   `<key>__range`; an `at_least` control is `<key>__n` (radio) + `<key>__x` (slider), generic ones `ctl_generic_atl_<branch>`; matching: `g_match`, `g_m_tree`, `g_m_keys`, `g_m_scal`, `g_m_vec`, …, `ctl_<pid>_population`, `ctl_<pid>_weight_<level>`, `ctl_<pid>_unf_*`, `ctl_<pid>_match_w`; app settings are `opt_*`, e.g. `opt_slider_range`) and check that `at.exception` and `at.error` are empty and the `Selected` metric changes.
3. Numbers checked against an independent computation (numpy / the analysis' own code) for any preset feature that
   produces physics results.

kaleido is not installed, so figures cannot be rendered to images in tests: check the figure structure
(`fig.to_dict()`) and ask the user to look at the page.

## Conventions

- Python, 120-column lines, numpy-vectorised code, short docstrings saying what a function returns.
- Commit only when the user asks.

## Next steps (after v0.1)

- Choose and add a licence (the user's decision).
- Example data: a small ATLAS Open Data sample fetched by a download script in `examples/` (do not commit ROOT
  files), plus an example preset for its branches.
- A real `tests/` folder built from the testing recipe above; possibly `pyproject.toml` with a console script.
- Not supported yet: one entry per object (per-muon / per-jet histograms), multi-sample stacks with cross-section
  normalisation, reco ↔ truth migration matrices as a built-in tab.
- Event matching of two trees (outer join, populations, Unfolding tab) is in. Still open: object-level matching
  (e.g. reco ↔ truth muons by ΔR or barcode as a generic mechanism), statistical uncertainties on acceptance /
  efficiency, more than two levels.
