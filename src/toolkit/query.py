"""
query.py — searches the PDB for candidate assemblies and fetches their metadata.

This wraps rcsb.api rather than reimplementing it: rcsb.api already handles
attribute-based search and data retrieval against the PDB reliably. What this
module adds is a cage-design-oriented interface on top of it — short,
memorable field names instead of raw dotted attribute paths, arbitrary
AND/OR combination of any attributes, and range support for numeric/date
fields — all as plain functions, so any interface (script, CLI, or a
NiceGUI form later) can drive the same logic.

IMPORTANT CONTEXT — two backends, two schemas:
rcsb.org actually runs two separate services behind the `rcsbapi` package:
  - the Search API   (what AttributeQuery talks to)
  - the Data API     (a GraphQL service, what DataQuery talks to)
These are maintained somewhat independently, and — this is the key
constraint the rest of this module is built around — they don't support
the same set of attributes. A field that conceptually means the same thing
(e.g. "resolution") can live at a DIFFERENT dotted path in each one, because
the Data API mirrors the actual object graph (assembly -> entry ->
resolution) while the Search API flattens everything into one flat
namespace. Worse, some fields (e.g. per-model confidence/"model_quality")
are Data-API-only: they describe metadata you can FETCH about a structure
you already have an ID for, but you cannot ask the Search API to filter
candidates by them directly.

That split is why this module is organized as three independent, composable
stages instead of one do-everything call:

  1. build_query(attribute, value, operator) -- builds ONE Search API query
                                  object. Build as many of these as you need
                                  as separate, named, individually-inspectable
                                  variables, then combine them yourself with
                                  plain `&`/`|`, or hand a list of them to
                                  search_ids() and let it AND/OR them for
                                  you. A list of criterion dicts is also
                                  still accepted, for callers building
                                  criteria programmatically (e.g. a NiceGUI
                                  form) rather than by hand.

  1b. search_ids(criteria)    -- Search API only. Turns build_query()
                                  objects (or dicts, or a mix) into a list of
                                  matching IDs. Only attributes the Search
                                  API actually understands can be used here —
                                  passing a Data-API-only attribute raises
                                  immediately rather than failing confusingly
                                  downstream.

  2. fetch_metadata(ids, ...) -- Data API only. Takes IDs (typically the
                                  output of search_ids) and returns one
                                  DataFrame row per ID, with whatever fields
                                  you ask for — including fields that were
                                  never searchable in step 1.

  3. filter_metadata(df, ...) -- Pure pandas, no API calls. Filters an
                                  already-fetched metadata DataFrame on ANY
                                  column, exact-match or range or contains,
                                  AND/OR combined. This is where
                                  Data-API-only attributes (like
                                  model_quality) get to act as filters, even
                                  though they could never be a `search_ids`
                                  criterion.

query_candidates() composes all three for the common case — search, then
fetch, then filter — so most callers never need to touch the individual
stages directly, but each stage stays usable (and testable) on its own.

A note on debugging zero-result searches: because build_query() produces
independent, reusable objects, the fastest way to find out WHICH criterion
is responsible for an unexpectedly empty result is to run search_ids() on
each one alone (e.g. `search_ids(build_query("experimental_method",
"X-RAY DIFFRACTION"))`) rather than only ever running the full AND'd
combination. A criterion that's individually valid can still legitimately
AND down to zero real matches — e.g. large designed cages are disproportionately
solved by cryo-EM rather than X-ray, so combining a symmetry filter with
`experimental_method="X-RAY DIFFRACTION"` can genuinely have few or no hits
even though nothing is broken.
"""

from functools import reduce
import pandas as pd
from rcsbapi.search import AttributeQuery
from rcsbapi.data import DataQuery


# ============================================================================
# SEARCH_ATTRIBUTES — short name -> Search API dotted path.
# Anything registered here can be used as a search_ids() criterion.
# ============================================================================

SEARCH_ATTRIBUTES = {
    # HIERARCHY & SYMMETRY
    "symmetry": "rcsb_struct_symmetry.symbol",
    "oligomeric_count": "rcsb_assembly_info.polymer_entity_instance_count",
    "stoichiometry": "rcsb_struct_symmetry.stoichiometry",
    "oligomeric_state": "rcsb_struct_symmetry.oligomeric_state",

    # SCALE & MASS
    "polymer_entity_count": "rcsb_assembly_info.polymer_entity_count",
    "composition": "rcsb_assembly_info.polymer_composition",
    "weight": "rcsb_assembly_info.molecular_weight",

    # DESIGN & ORIGIN
    "source_organism": "rcsb_entity_source_organism.scientific_name",
    "polymer_entity_type": "rcsb_entry_info.selected_polymer_entity_types",

    # EXPERIMENTAL & DEPOSITION
    "assembly_method": "pdbx_struct_assembly.method_details",
    "resolution": "rcsb_entry_info.resolution_combined",
    "experimental_method": "rcsb_entry_info.experimental_method",

    # TEXT QUERIES
    "keywords": "struct_keywords.pdbx_keywords",
    "title": "struct.title",
    "date": "rcsb_accession_info.deposit_date",
    "pfam_id": "rcsb_polymer_entity_annotation.annotation_id",
}


# ============================================================================
# DATA_ATTRIBUTES — short name -> Data API dotted path.
# Superset of SEARCH_ATTRIBUTES: everything searchable is also fetchable
# (at a possibly different path — see the CONFIRMED overrides below), plus
# some fields that are fetch-only (e.g. model_quality has no Search API
# equivalent at all).
#
# Strategy: start as an exact copy of SEARCH_ATTRIBUTES (default assumption
# is "same path unless proven otherwise"), then explicitly override any
# field confirmed to differ, and add any fetch-only field on top. This
# keeps one single source of truth (SEARCH_ATTRIBUTES) instead of two dicts
# drifting out of sync, and makes every deviation visible and explained
# here rather than buried.
# ============================================================================
DATA_ATTRIBUTES = dict(SEARCH_ATTRIBUTES)

# --- CONFIRMED mismatches (verified by hand-testing) -----------------------

DATA_ATTRIBUTES["resolution"] = "entry.rcsb_entry_info.resolution_combined"
DATA_ATTRIBUTES["weight"] = "entry.rcsb_entry_info.molecular_weight"
DATA_ATTRIBUTES["source_organism"] = "entry.polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name"
DATA_ATTRIBUTES["polymer_entity_type"] = "entry.rcsb_entry_info.selected_polymer_entity_types"
DATA_ATTRIBUTES["experimental_method"] = "entry.rcsb_entry_info.experimental_method"
DATA_ATTRIBUTES["keywords"] = "entry.struct_keywords.pdbx_keywords"
DATA_ATTRIBUTES["title"] = "entry.struct.title"
DATA_ATTRIBUTES["date"] = "entry.rcsb_accession_info.deposit_date"
DATA_ATTRIBUTES["pfam_id"] = "entry.polymer_entities.rcsb_polymer_entity_annotation.annotation_id"

# --- Fetch-only fields: NO Search API equivalent exists at all -------------
# These can be fetched via fetch_metadata()/query_candidates(), and filtered
# locally via filter_metadata() afterward, but can NEVER be passed to
# search_ids() — there's no Search API path to build an AttributeQuery from.
DATA_ATTRIBUTES["model_quality"] = "entry.rcsb_ma_qa_metric_global.ma_qa_metric_global.value"

# --- SUSPECTED mismatches, NOT yet confirmed --------------------------------
# Left as-is (unprefixed) until tested — flip these on once confirmed,
# following the resolution example above. Everything else (symmetry,
# stoichiometry, weight, interface_area, etc.) is genuinely assembly/
# interface-level data, so it's *expected* to already match between the two
# APIs without needing a prefix — but that's still an assumption, not a
# confirmed fact, until each one is actually exercised through
# fetch_metadata().

# Short names that exist in DATA_ATTRIBUTES but have no Search API path —
# i.e. legal for fetch_metadata()/filter_metadata(), illegal for search_ids().
# Computed once, from the two registries above, so it can never drift out
# of sync with them by hand-editing a third list.
FETCH_ONLY_ATTRIBUTES = set(DATA_ATTRIBUTES) - set(SEARCH_ATTRIBUTES)

# Fields for which a range ("from X to Y") search generally makes more sense
# than an exact match. Informational only (e.g. for a future NiceGUI form
# deciding whether to render a range slider or a text box) — not enforced.
RANGE_HINTED_ATTRIBUTES = {
    "weight", "interface_area", "resolution", "interface_count",
    "oligomeric_count", "polymer_entity_count", "model_quality", "date",
}


# ============================================================================
# STAGE 1 — Search API: criteria -> assembly IDs
# ============================================================================

def build_query(attribute, value, operator="exact_match"):
    """
    Build ONE Search API query object (an AttributeQuery, or a combination
    of two for "range") — the basic unit meant to be built and named
    individually, then combined however you like:

        cage_symmetry  = build_query("symmetry", "T")
        good_res       = build_query("resolution", (0, 3.0), operator="range")
        xray_only      = build_query("experimental_method", "X-RAY DIFFRACTION")

        # combine with plain Python operators — these are real query
        # objects, `&`/`|` work directly, exactly like rcsb.api's own:
        ids = search_ids(cage_symmetry & good_res & xray_only)

        # ...or hand search_ids() the individual pieces as a list and let
        # it AND/OR them for you:
        ids = search_ids([cage_symmetry, good_res, xray_only])

    Building each criterion as its own named variable (rather than only
    ever assembling one big list-of-dicts) makes each one independently
    runnable and printable — handy when an AND'd search unexpectedly comes
    back empty and you need to check which single criterion is responsible
    (run search_ids() on each one alone; see the module docstring).

    attribute : short name from SEARCH_ATTRIBUTES, or a raw dotted path if
        you need something not yet registered there.
    value     : a single value for exact/text operators, or a (low, high)
        tuple when operator="range".
    operator  : "exact_match" (default), "range", or any other rcsb.api
        comparison operator (e.g. "contains_phrase", "greater").

    Raises ValueError if `attribute` is a short name that's registered in
    DATA_ATTRIBUTES but has no Search API equivalent (see
    FETCH_ONLY_ATTRIBUTES) — that's the one case this function can catch
    early and explain, rather than silently building a query against a
    made-up path and letting the Search API fail confusingly later.
    """
    if attribute in FETCH_ONLY_ATTRIBUTES:
        raise ValueError(
            f"'{attribute}' is metadata-only — the Search API has no "
            f"equivalent field, so it cannot be used to filter for IDs. "
            f"Fetch it with fetch_metadata()/query_candidates(fetch_fields=...) "
            f"and filter on it with filter_metadata() instead."
        )

    path = SEARCH_ATTRIBUTES.get(attribute, attribute)

    if operator == "range":
        low, high = value
        q_low = AttributeQuery(attribute=path, operator="greater_or_equal", value=low)
        q_high = AttributeQuery(attribute=path, operator="less_or_equal", value=high)
        return q_low & q_high

    # If a list/tuple/set was passed, upgrade a plain exact_match to "in".
    if isinstance(value, (list, tuple, set)):
        if operator == "exact_match":
            operator = "in"
        value = list(value)
    elif operator == "in":
        value = [value]

    return AttributeQuery(attribute=path, operator=operator, value=value)


def combine_queries(queries, mode="and"):
    """
    Combine several query objects (from build_query(), or criterion dicts)
    into one.

    mode="and" -> matches all of the queries (rcsb.api's `&` operator)
    mode="or"  -> matches any of the queries (rcsb.api's `|` operator)
    """
    if mode not in ("and", "or"):
        raise ValueError("mode must be 'and' or 'or'")
    queries = [build_query(q["attribute"], q["value"], q.get("operator", "exact_match")) if isinstance(q, dict) else q for q in queries]
    op = (lambda a, b: a & b) if mode == "and" else (lambda a, b: a | b)
    return reduce(op, queries)


def _collect_query_attributes(query_obj):
    """
    Recursively walks a (possibly AND/OR-combined) query object and
    collects every attribute PATH it touches, by reading Terminal.params
    ("attribute") and Group.nodes — the two public fields rcsb.api's own
    query objects expose for exactly this kind of introspection.

    Used internally by query_candidates() so that, even when search
    criteria are handed in as pre-built build_query() objects rather than
    attribute-name dicts, the fields they reference still get auto-included
    in the fetched DataFrame — see search_ids()'s docstring for why both
    forms are supported.
    """
    nodes = getattr(query_obj, "nodes", None)
    if nodes is not None:
        attrs = set()
        for node in nodes:
            attrs |= _collect_query_attributes(node)
        return attrs
    attr = getattr(query_obj, "params", {}).get("attribute")
    return {attr} if attr else set()


def search_ids(criteria, mode="and", return_type="assembly"):
    """
    The single entry point for STAGE 1: turns filter criteria into a list
    of matching IDs, using the Search API only.

    criteria : any of
        - a single already-built query object — e.g. build_query(...), or
          an expression combining several with & / | (e.g. `q1 & q2`).
          Used as-is, unmodified.
        - a list of build_query()-style query objects — AND/OR'd together
          via `mode`. This is the recommended way to define several
          criteria individually and then combine them, rather than
          assembling one list-of-dicts by hand:
              search_ids([build_query("symmetry", "T"),
                          build_query("resolution", (0, 3.0), operator="range")])
        - a list of criterion dicts (still fully supported — useful when
          criteria are generated programmatically, e.g. from a NiceGUI
          form), each shaped like one of:
              {"attribute": "symmetry", "value": "T"}
              {"attribute": "resolution", "value": (0, 2.5), "operator": "range"}
              {"attribute": "keywords", "value": "nanocage", "operator": "contains_phrase"}
          Dicts and pre-built query objects can be freely mixed in the
          same list.
    mode     : "and" or "or" — applies across ALL criteria in the list.
        (No mixed AND/OR groups yet — e.g. (A OR B) AND C would need a
        nested-group version; flag it if you need that later. You CAN get
        this today by pre-combining with plain `&`/`|` and passing the
        result as a single already-built query object instead of a list.)
    return_type : the kind of ID to return — "assembly" (default), "entry",
        "polymer_entity", etc. — whatever rcsb.api's AttributeQuery accepts.

    Returns a plain list of ID strings (e.g. ["6XYZ-1", "6XYZ-2", ...]).
    Raises ValueError if `criteria` is empty, or if any criterion names a
    fetch-only attribute (see build_query()).
    """
    if not criteria:
        raise ValueError("must supply at least one criterion")

    if callable(criteria):
        # A single already-built/combined query object — every rcsb.api
        # query object (AttributeQuery, Group, ...) is directly callable
        # (that's how it executes itself against the Search API), so this
        # is a reliable way to detect "already a query" regardless of which
        # concrete rcsb.api class it happens to be. Checking against a
        # specific type name (e.g. isinstance(..., (list, tuple))) is what
        # you want to AVOID here — rcsb.api's own classes for a combined
        # query aren't list/tuple subclasses, but relying on that
        # indirectly (checking what criteria is NOT) is more fragile than
        # checking directly for what a query object always IS: callable.
        combined = criteria
    elif isinstance(criteria, dict):
        combined = build_query(criteria["attribute"], criteria["value"], criteria.get("operator", "exact_match"))
    elif isinstance(criteria, (list, tuple)):
        queries = [build_query(c["attribute"], c["value"], c.get("operator", "exact_match")) if isinstance(c, dict) else c for c in criteria]
        combined = combine_queries(queries, mode=mode) if len(queries) > 1 else queries[0]
    else:
        raise TypeError(
            f"criteria must be a single query object (from build_query() or a "
            f"combined expression), a criterion dict, or a list of either — got {type(criteria)!r}"
        )

    return list(combined(return_type=return_type))


# ============================================================================
# STAGE 2 — Data API: IDs -> metadata DataFrame
# ============================================================================

def extract_leaf_values(val):
    """
    Recursively extracts all primitive/leaf values from any nested dict/list
    structure. Returns a single scalar if 1 item, a joined string or list if
    multiple, or None if empty.
    """
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None

    # If it's already a clean primitive value, keep it
    if isinstance(val, (str, int, float, bool)):
        return val

    leaves = []

    def _walk(node):
        if isinstance(node, dict):
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)
        elif node is not None:
            leaves.append(node)

    _walk(val)

    # Deduplicate while preserving order
    unique_leaves = list(dict.fromkeys(leaves))

    if not unique_leaves:
        return None
    if len(unique_leaves) == 1:
        return unique_leaves[0]  # Return as a clean single scalar ("Homo sapiens", 1024.5)

    # Return string-joined if all are strings, else return as a clean primitive list
    if all(isinstance(x, str) for x in unique_leaves):
        return ", ".join(unique_leaves)

    return unique_leaves


def _plural_key(input_type):
    """RCSB's GraphQL response nests records under a pluralized key derived
    from input_type ('assembly' -> 'assemblies', 'entry' -> 'entries')."""
    return f"{input_type[:-1]}ies" if input_type.endswith("y") else f"{input_type}s"


def _resolve_path(node, path_parts):
    """
    Resolves a dotted path (already split into parts) against one nested
    JSON record from the Data API's GraphQL response, FANNING OUT over any
    list encountered partway through the path rather than stopping there.

    This matters because GraphQL always returns has-many relationships as
    a list — e.g. an entry's "polymer_entities" — even when a particular
    entry happens to only have one. pandas' pd.json_normalize (the
    previous approach here) stops flattening the instant it hits a list,
    so any DATA_ATTRIBUTES path that passes THROUGH a has-many
    relationship (not just ends at one) — e.g. "source_organism" =
    "entry.polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name",
    or "pfam_id", both of which go through "polymer_entities" — never
    reached its target field at all. The column that came out was named
    after wherever json_normalize gave up (something like
    "entry.polymer_entities"), not the short name registered in
    DATA_ATTRIBUTES, which is exactly the mismatch that made
    filter_metadata() report the column as "not present" even though the
    field WAS fetched, just under an unexpected name.

    Resolving each requested path by hand — descending through dicts one
    key at a time, and fanning out to resolve the SAME remaining path
    against every element when a list is hit — sidesteps json_normalize's
    column-naming behavior entirely, and handles any number of nested
    has-many relationships along a path (e.g. a list inside a list)
    automatically, with no special-casing per field.

    Returns whatever raw value matched — a scalar, dict, list, or (after
    fanning out) a list of collected sub-results — for extract_leaf_values
    to flatten into a clean cell value. Returns None if any key along the
    path is missing or a non-dict/list node is encountered before the path
    is exhausted.
    """
    if not path_parts:
        return node
    if isinstance(node, list):
        results = [_resolve_path(item, path_parts) for item in node]
        return [r for r in results if r is not None]
    if not isinstance(node, dict) or path_parts[0] not in node:
        return None
    return _resolve_path(node[path_parts[0]], path_parts[1:])


def fetch_metadata(ids, fields, input_type="assembly"):
    """
    The single entry point for STAGE 2: fetches metadata for a known list
    of IDs (typically straight out of search_ids), using the Data API only.

    ids    : list of ID strings, e.g. what search_ids() returned.
    fields : short names (from DATA_ATTRIBUTES) or raw dotted paths, for
        whatever metadata columns you want back. Unlike search_ids, this
        stage has no attribute restriction — any field DATA_ATTRIBUTES (or
        the Data API directly) knows about is fair game, including
        fetch-only fields like "model_quality".
    input_type : the kind of ID being passed in — "assembly" (default),
        "entry", etc. Must match what the IDs actually are.

    Returns a DataFrame with one row per ID and one column per requested
    field, ALWAYS named by its short name (falling back to the raw path
    for anything not registered in DATA_ATTRIBUTES) — never by whatever
    name a generic flattening pass happened to produce. Values are
    flattened via extract_leaf_values so each cell is a clean scalar
    rather than a nested dict/list, with multi-valued fields (e.g. two
    different source organisms across an assembly's polymer entities)
    collected and comma-joined rather than silently keeping only one.

    A leading "<input_type>_id" column (e.g. "assembly_id" for the
    default input_type="assembly", matching distance.py's
    assembly_id_column convention; "entry_id" for input_type="entry",
    matching download.py's id_column convention) is always included too,
    holding each row's own rcsb_id — e.g. "6XYZ-1". The Data API's
    DataQuery adds this automatically to every query regardless of what's
    in `fields` (its add_rcsb_id default is True), so it's always present
    on the raw record; it just wasn't being surfaced as an actual column
    before. This is what lets every row still be traced back to which
    candidate it is after filter_metadata() drops rows and resets the
    index — without it, that link only ever existed positionally (row N
    corresponds to ids[N]), which silently breaks the moment any rows get
    dropped or reordered.
    """
    if not ids:
        return pd.DataFrame()

    id_column = f"{input_type}_id"

    # short name -> dotted path, in the order requested (dict preserves
    # insertion order, so column order matches `fields`).
    field_paths = {f: DATA_ATTRIBUTES.get(f, f) for f in fields if f != id_column}

    data_query = DataQuery(input_type=input_type, input_ids=ids, return_data_list=list(field_paths.values()))
    raw_results = data_query.exec()

    records = raw_results.get("data", {}).get(_plural_key(input_type), [])

    rows = []
    for record in records:
        row = {id_column: _resolve_path(record, ["rcsb_id"])}
        for short_name, path in field_paths.items():
            row[short_name] = extract_leaf_values(_resolve_path(record, path.split(".")))
        rows.append(row)

    return pd.DataFrame(rows, columns=[id_column] + list(field_paths.keys()))


# ============================================================================
# STAGE 3 — pure pandas: filter an existing metadata DataFrame
# ============================================================================

def _cell_matches(cell, value, operator):
    """
    Evaluates one filter criterion against one DataFrame cell. `cell` may
    be a plain scalar, None/NaN, or a list (extract_leaf_values keeps
    genuinely multi-valued, non-string cells as a list — e.g. several
    distinct organism names).
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return False

    if operator == "range":
        low, high = value
        if isinstance(cell, list):
            return any(low <= v <= high for v in cell)
        return low <= cell <= high

    if operator == "contains_phrase":
        text = ", ".join(cell) if isinstance(cell, list) else str(cell)
        return str(value).lower() in text.lower()

    if operator in ("greater", "greater_or_equal", "less", "less_or_equal"):
        values = cell if isinstance(cell, list) else [cell]
        if operator == "greater":
            return any(v > value for v in values)
        if operator == "greater_or_equal":
            return any(v >= value for v in values)
        if operator == "less":
            return any(v < value for v in values)
        return any(v <= value for v in values)

    # exact_match / in — anything else falls back to plain membership.
    wanted = value if isinstance(value, (list, tuple, set)) else [value]
    if isinstance(cell, list):
        return any(v in wanted for v in cell)
    return cell in wanted


def filter_metadata(df, criteria, mode="and"):
    """
    The single entry point for STAGE 3: filters an already-fetched metadata
    DataFrame (from fetch_metadata) against any number of criteria, purely
    in pandas — no API calls, so this works identically whether the
    attribute was ever searchable via the Search API or not.

    criteria : same shape as search_ids() criteria — list of dicts with
        "attribute" (a DataFrame column name, i.e. a short name or whatever
        the column ended up being called), "value", and optional
        "operator" (default "exact_match"; also supports "range",
        "contains_phrase", "in", "greater"/"greater_or_equal"/"less"/
        "less_or_equal").
    mode     : "and" or "or" across ALL criteria in the list.

    Raises ValueError if `criteria` is empty, or if a criterion names a
    column that isn't actually in `df` (i.e. wasn't fetched) — this is
    almost always a call-site mistake (forgetting to include the field in
    fetch_fields) worth catching immediately.

    Returns a new DataFrame (row order preserved, index reset) — the input
    df is never modified in place.
    """
    if not criteria:
        raise ValueError("must supply at least one criterion")
    if mode not in ("and", "or"):
        raise ValueError("mode must be 'and' or 'or'")
    if df.empty:
        return df

    missing = [c["attribute"] for c in criteria if c["attribute"] not in df.columns]
    if missing:
        raise ValueError(
            f"cannot filter on {missing} — not present in the DataFrame. "
            f"Make sure these fields were included in fetch_fields."
        )

    masks = [
        df[c["attribute"]].apply(lambda cell: _cell_matches(cell, c["value"], c.get("operator", "exact_match")))
        for c in criteria
    ]
    combined_mask = reduce((lambda a, b: a & b) if mode == "and" else (lambda a, b: a | b), masks)

    return df[combined_mask].reset_index(drop=True)


# ============================================================================
# Convenience orchestrator — search, then fetch, then (optionally) filter.
# ============================================================================

def _attribute_names_from_search_criteria(search_criteria):
    """
    Extracts the short/attribute names referenced by a search_ids()-style
    `criteria` argument, regardless of whether it's a dict list, a
    build_query() object list, a single pre-combined query expression, or
    any mix — so query_candidates() can auto-include them in fetch_fields
    either way.
    """
    items = search_criteria if isinstance(search_criteria, (list, tuple)) else [search_criteria]
    path_to_short = {v: k for k, v in SEARCH_ATTRIBUTES.items()}
    names = []
    for item in items:
        if isinstance(item, dict):
            names.append(item["attribute"])
        else:
            for path in sorted(_collect_query_attributes(item)):
                names.append(path_to_short.get(path, path))
    return names


def query_candidates(search_criteria, fetch_fields=None, filter_criteria=None, mode="and", return_type="assembly"):
    """
    Composes the three stages above for the common case: run search_ids()
    against search_criteria, fetch_metadata() for the resulting IDs, and
    (if given) filter_metadata() the result against filter_criteria.

    search_criteria : anything search_ids() accepts — a list of dicts, a
        list of build_query() objects, a single pre-combined query
        expression, or a mix. Every attribute referenced here must be
        Search-API-capable (see FETCH_ONLY_ATTRIBUTES).
    fetch_fields    : list of short names/paths to fetch metadata for. Any
        attribute referenced in search_criteria or filter_criteria is
        included automatically even if you don't list it explicitly, so
        the returned DataFrame always has a column to show why a row
        matched. Extra fields (not used for filtering) are fine too — pass
        whatever you want to see in the result.
    filter_criteria : optional list of dicts for filter_metadata(), applied
        AFTER fetching — this is where fetch-only attributes (like
        model_quality) get to act as filters, since they could never be a
        search_criteria entry.
    mode            : "and"/"or", applied independently within
        search_criteria and within filter_criteria (not across the two).
    return_type     : the kind of ID being searched/fetched — "assembly"
        (default), "entry", etc.

    Returns a metadata DataFrame — empty if the search matched nothing.
    """
    fields = list(fetch_fields) if fetch_fields else []
    for name in _attribute_names_from_search_criteria(search_criteria):
        if name not in fields:
            fields.append(name)
    for c in (filter_criteria or []):
        if c["attribute"] not in fields:
            fields.append(c["attribute"])

    ids = search_ids(search_criteria, mode=mode, return_type=return_type)
    if not ids:
        return pd.DataFrame()

    df = fetch_metadata(ids, fields, input_type=return_type)

    if filter_criteria and not df.empty:
        df = filter_metadata(df, filter_criteria, mode=mode)

    return df