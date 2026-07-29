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
  - the Search API   (what AttributeQuery / our `query()` function talks to)
  - the Data API     (a GraphQL service, what DataQuery talks to)
These are maintained somewhat independently. A field that conceptually
means the same thing (e.g. "resolution") can live at a DIFFERENT dotted
path in each one, because the Data API mirrors the actual object graph
(assembly -> entry -> resolution) while the Search API flattens everything
into one flat namespace for convenience. This is why we now keep TWO
registries below instead of one shared dict.
"""

from functools import reduce
import pandas as pd
from rcsbapi.search import AttributeQuery
from rcsbapi.data import DataQuery


# ============================================================================
# SEARCH_ATTRIBUTES — short name -> Search API dotted path.
# This is what AttributeQuery (used inside query()) needs, and it's what
# search_ids() / search_by_criteria() ultimately query against.
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
# This is what DataQuery (used inside fetch_metadata) needs.
#
# Strategy: start as an exact copy of SEARCH_ATTRIBUTES (our default
# assumption is "same path unless proven otherwise"), then explicitly
# override any field we've confirmed differs. This keeps one single source
# of truth (SEARCH_ATTRIBUTES) instead of two dicts drifting out of sync,
# and makes every deviation visible and explained below rather than buried.
# ============================================================================
DATA_ATTRIBUTES = dict(SEARCH_ATTRIBUTES)

# --- CONFIRMED mismatches (verified by hand-testing) -----------------------

DATA_ATTRIBUTES["resolution"] = "entry.rcsb_entry_info.resolution_combined"
DATA_ATTRIBUTES["weight"] = "entry.rcsb_entry_info.molecular_weight"
DATA_ATTRIBUTES["source_organism"] =  "entry.polymer_entities.rcsb_entity_source_organism.ncbi_scientific_name"
DATA_ATTRIBUTES["polymer_entity_type"] = "entry.rcsb_entry_info.selected_polymer_entity_types"
DATA_ATTRIBUTES["experimental_method"] = "entry.rcsb_entry_info.experimental_method"
DATA_ATTRIBUTES["model_quality"] = "entry.rcsb_ma_qa_metric_global.ma_qa_metric_global.value"
DATA_ATTRIBUTES["keywords"] = "entry.struct_keywords.pdbx_keywords"
DATA_ATTRIBUTES["title"] = "entry.struct.title"
DATA_ATTRIBUTES["date"] = "entry.rcsb_accession_info.deposit_date"
DATA_ATTRIBUTES["pfam_id"] = "entry.polymer_entities.rcsb_polymer_entity_annotation.annotation_id"

# --- SUSPECTED mismatches, NOT yet confirmed --------------------------------
# Left as-is (unprefixed) until tested — flip
# these on once confirmed, following the resolution example above:
#
# DATA_ATTRIBUTES["experimental_method"] = "entry." + SEARCH_ATTRIBUTES["experimental_method"]
# DATA_ATTRIBUTES["date"] = "entry." + SEARCH_ATTRIBUTES["date"]
#
# Everything else (symmetry, stoichiometry, weight, interface_area, etc.)
# is genuinely assembly/interface-level data, so we'd *expect* those to
# already match between the two APIs without needing a prefix — but that's
# still an assumption, not a confirmed fact, until each one is actually
# exercised through fetch_metadata().


# Fields for which a range ("from X to Y") search generally makes more sense than an exact match
RANGE_HINTED_ATTRIBUTES = {
    "weight", "interface_area", "resolution", "interface_count",
    "oligomeric_count", "polymer_entity_count", "model_quality", "date",
}


def query(attribute, value, operator):
    """
    Build a single Search API query object from one filter criterion.

    attribute : short name from SEARCH_ATTRIBUTES, or a raw dotted path if you need something not yet registered
    value     : a single value for exact/text operators, or a (low, high) tuple when operator="range"
    operator  : "exact_match" (default), "range", or any other rcsb.api comparison operator (e.g. "contains_phrase", "greater")

    For operator="range", this builds two combined sub-queries
    (>= low AND <= high) using rcsb.api's overloaded `&` operator on query
    objects, rather than relying on a single native "range" operator.
    """
    path = SEARCH_ATTRIBUTES.get(attribute, attribute)

    if operator == "range":
        low, high = value
        q_low = AttributeQuery(attribute=path, operator="greater_or_equal", value=low)
        q_high = AttributeQuery(attribute=path, operator="less_or_equal", value=high)
        return q_low & q_high

    # 2. General list/array operator handling
    # If a list/tuple/set was passed, upgrade exact_match to "in"
    if isinstance(value, (list, tuple, set)):
        if operator == "exact_match":
            operator = "in"
        value = list(value)

    # If "in" was explicitly specified, ensure value is a list
    elif operator == "in":
        value = [value]

    return AttributeQuery(attribute=path, operator=operator, value=value)


def combine_queries(queries, mode="and"):
    """
    Combine several query() objects into one.

    mode="and" -> matches all of the queries (rcsb.api's `&` operator)
    mode="or"  -> matches any of the queries (rcsb.api's `|` operator)

    """
    if mode not in ("and", "or"):
        raise ValueError("mode must be 'and' or 'or'")
    op = (lambda a, b: a & b) if mode == "and" else (lambda a, b: a | b)
    return reduce(op, queries)


def search_ids(query_obj, return_type="assembly"):
    """Execute a query (single or combined) and return the list of matching IDs."""
    return list(query_obj(return_type=return_type))


def search_by_criteria(criteria, mode="and", return_type="assembly"):
    """
    General-purpose, DATA-DRIVEN search: any number of filters, any mix of
    exact-match and range criteria, combined with AND or OR — all described
    as plain data (a list of dicts) rather than Python code.
    criteria : list of dicts, each shaped like one of:
        {"attribute": "symmetry", "value": "T"}
        {"attribute": "resolution", "value": (0, 2.5), "operator": "range"}
        {"attribute": "keywords", "value": "nanocage", "operator": "contains_phrase"}
    mode     : "and" or "or" — applies across ALL criteria in the list.
               (No mixed AND/OR groups yet — e.g. (A OR B) AND C would need
               a nested-group version; flag it if you need that later.)
    """
    if not criteria:
        raise ValueError("must supply at least one criterion")

    queries = [query(**c) for c in criteria]
    combined = combine_queries(queries, mode=mode) if len(queries) > 1 else queries[0]
    return search_ids(combined, return_type=return_type)


def extract_leaf_values(val):
    """
    Recursively extracts all primitive/leaf values from any nested dict/list structure.
    Returns a single scalar if 1 item, a joined string or list if multiple, or None/NaN if empty.
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

def fetch_metadata(ids, fields, input_type="assembly"):
    
    paths = [DATA_ATTRIBUTES.get(f, f) for f in fields]

    data_query = DataQuery(input_type=input_type, input_ids=ids, return_data_list=paths)
    raw_results = data_query.exec()

    # Determine plural key returned by rcsbapi ('assemblies', 'entries', etc.)
    plural_key = f"{input_type}s" if not input_type.endswith('y') else f"{input_type[:-1]}ies"

    # Extract the actual list of records from the GraphQL response structure
    records = raw_results.get("data", {}).get(plural_key, [])

    # Flatten the records — now 1 row per record/assembly!
    df = pd.json_normalize(records)

    # Rename flat dotted columns back to friendly short names
    path_to_short = {v: k for k, v in DATA_ATTRIBUTES.items()}
    df = df.rename(columns=lambda c: path_to_short.get(c, c))

    clean_df = df.map(extract_leaf_values)

    return clean_df
