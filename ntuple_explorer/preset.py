"""
Presets: Python files defining PRESET = dict(...), found in the folders of NTUPLE_EXPLORER_PRESETS (the launcher's
--presets option). See README.md for the schema.
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

import loader


def preset_dirs():
    """Folders searched for presets, in order: NTUPLE_EXPLORER_PRESETS (os.pathsep-separated), presets/ at the
    repository root (git-ignored: local presets), presets/ inside the package."""
    dirs = [Path(d).expanduser() for d in os.environ.get("NTUPLE_EXPLORER_PRESETS", "").split(os.pathsep) if d]
    here = Path(__file__).resolve().parent
    dirs += [here.parent / "presets", here / "presets"]
    return [d for d in dirs if d.is_dir()]


KEYS = {"name", "description", "files", "tree", "helpers", "prefilter", "columns", "categories", "labels", "derived",
        "weight", "controls", "plots", "stack_options", "custom", "postprocess", "plots_2d", "friends", "compositions",
        "colors", "match", "unfold"}
NORMS = {"row", "total", "yields"}
PLACES = {"sidebar", "main"}       # sidebar: selection + cutflow steps; main: page cuts above the tabs
KINDS = {"alias", "select", "multiselect", "checkboxes", "range", "toggle", "expr", "at_least"}


def discover():
    """{preset name: file path} of *.py in the preset folders (files starting with _ are helpers, skipped). On a name
    clash the first folder wins: the user's folders before the bundled one."""
    out = {}
    for f in [f for d in preset_dirs() for f in sorted(d.glob("*.py"))]:
        if f.name.startswith("_"):
            continue
        try:
            p = load_file(f)
            out.setdefault(p["name"], f)
        except Exception as e:      # noqa: BLE001 - listed with the error, app keeps running
            out[f"{f.stem} (error: {type(e).__name__}: {e})"] = None
    return out


def load_file(path):
    path = Path(path)
    name = f"ntuple_explorer_preset_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "PRESET"):
        raise AttributeError(f"{path.name} does not define PRESET")
    p = dict(mod.PRESET)
    validate(p, path.name)
    return p


def validate(p, where="preset"):
    unknown = set(p) - KEYS
    if unknown:
        raise KeyError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(KEYS)})")
    for k in ("name", "tree", "columns"):
        if k not in p:
            raise KeyError(f"{where}: missing {k!r}")
    lv = levels(p) or [None]
    m = p.get("match")
    if m is not None:
        if "tree" not in m or "columns" not in m:
            raise KeyError(f"{where}: match needs 'tree' and 'columns'")
        if lv[0] == lv[1] or not all(str(x).isidentifier() for x in lv):
            raise ValueError(f"{where}: match level names {lv} must be two different identifiers")
        clash = set(friend_columns(m)) & set(loader_columns(p))
        if clash:
            raise KeyError(f"{where}: match columns {sorted(clash)} clash with main columns")
    for i, u in enumerate(p.get("unfold") or []):
        if not m or not all(u.get(x) for x in lv):
            raise KeyError(f"{where}: unfold {i} needs a match and the columns {lv} (one per level)")
    for i, c in enumerate(p.get("controls", [])):
        if c.get("level") is not None and c["level"] not in lv:
            raise ValueError(f"{where}: control {i} level {c['level']!r}, expected one of {lv}")
        if c.get("kind") not in KINDS:
            raise ValueError(f"{where}: control {i} has kind {c.get('kind')!r}, expected one of {sorted(KINDS)}")
        if c["kind"] in ("multiselect", "range") and "column" not in c:
            raise KeyError(f"{where}: control {i} ({c['kind']}) needs 'column'")
        if c["kind"] in ("alias", "select", "checkboxes") and not c.get("options"):
            raise KeyError(f"{where}: control {i} ({c['kind']}) needs 'options'")
        if c["kind"] == "at_least" and not (c.get("vector") or c.get("values")):
            raise KeyError(f"{where}: control {i} (at_least) needs 'vector' (4-vector branch) or 'values' (expression)")
        if c["kind"] == "at_least" and str(c.get("of", "pt")).lower() not in {x.lower() for x in loader.COMPONENTS}:
            raise ValueError(f"{where}: control {i} of={c.get('of')!r}, expected one of {loader.COMPONENTS}")
        if c.get("place", "sidebar") not in PLACES:
            raise ValueError(f"{where}: control {i} place {c.get('place')!r}, expected one of {sorted(PLACES)}")
    for name in p.get("categories", {}):
        if name in p["columns"]:
            raise KeyError(f"{where}: category {name!r} also defined in columns")
    for fname, f in (p.get("friends") or {}).items():
        if "tree" not in f or "columns" not in f:
            raise KeyError(f"{where}: friend {fname!r} needs 'tree' and 'columns'")
        clash = set(friend_columns(f)) & set(loader_columns(p))
        if clash:
            raise KeyError(f"{where}: friend {fname!r} columns {sorted(clash)} clash with main columns")
    for i, c in enumerate(p.get("compositions") or []):
        if "rows" not in c or "segments" not in c:
            raise KeyError(f"{where}: composition {i} needs 'rows' and 'segments'")
        if c.get("norm", "row") not in NORMS:
            raise ValueError(f"{where}: composition {i} norm {c.get('norm')!r}, expected one of {sorted(NORMS)}")


def category_expr(classes):
    """{label: ROOT bool expr | None} -> int ROOT expression: index of the first true class; the class with None
    gets every entry in no other class (-1 if there is no such class)."""
    rest = [i for i, e in enumerate(classes.values()) if e is None]
    tail = str(rest[0]) if rest else "-1"
    e = tail
    for i, ex in reversed(list(enumerate(classes.values()))):
        if ex is not None:
            e = f"(({ex}) ? {i} : {e})"
    return f"int{e}" if e.startswith("(") else f"int({e})"


def overlap_expr(classes):
    return " + ".join(f"int(bool({ex}))" for ex in classes.values() if ex is not None) or "0"


def loader_columns(p):
    """All ROOT expressions to load: preset columns + one int column per category (+ an overlap counter) + the k-th
    largest values behind at_least controls."""
    cols = dict(p["columns"])
    for name, classes in p.get("categories", {}).items():
        cols[name] = category_expr(classes)
        cols[f"__overlap_{name}"] = overlap_expr(classes)
    lv = levels(p)
    for c in p.get("controls", []):
        if c.get("kind") == "at_least" and (not lv or c.get("level", lv[0]) == lv[0]):
            cols.update(loader.nth_columns(c))
    if lv:
        cols.update({k: k for k in p["match"].get("keys", ["runNumber", "eventNumber"])})
    return cols


def levels(p):
    """[main level, second level] names of a preset with `match`, else None."""
    m = p.get("match")
    return [m.get("main", "reco"), m.get("name", "truth")] if m else None


def match_request(p):
    """Load request (dict) of the second level: its columns (+ categories, + at_least columns of its controls)."""
    m = p["match"]
    name = levels(p)[1]
    cols = friend_columns(m)
    for c in p.get("controls", []):
        if c.get("kind") == "at_least" and c.get("level") == name:
            cols.update(loader.nth_columns(c))
    return dict(tree=m["tree"], name=name, main=levels(p)[0], keys=list(m.get("keys", ["runNumber", "eventNumber"])),
                prefilter=m.get("prefilter"), fill=m.get("fill", -1), columns=sorted(cols.items()))


def friend_columns(f):
    """ROOT expressions to load from a friend tree: its columns + one int column per category."""
    cols = dict(f["columns"])
    for name, classes in (f.get("categories") or {}).items():
        cols[name] = category_expr(classes)
    return cols


def friend_request(p):
    """Hashable description of the friends (part of the load request): ((name, json spec), ...)."""
    out = []
    for name, f in (p.get("friends") or {}).items():
        spec = {k: v for k, v in f.items() if k not in ("columns", "categories")}
        spec["columns"] = friend_columns(f)
        out.append((name, json.dumps(spec, sort_keys=True)))
    return tuple(out)


def labels(p):
    """{column: {value: label}} incl. categories."""
    out = {}
    for name, classes in p.get("categories", {}).items():
        out[name] = dict(enumerate(classes))
    for f in (p.get("friends") or {}).values():
        for name, classes in (f.get("categories") or {}).items():
            out[name] = dict(enumerate(classes))
    for k, v in (p.get("labels") or {}).items():
        out[k] = dict(enumerate(v)) if isinstance(v, (list, tuple)) else dict(v)
    return out
