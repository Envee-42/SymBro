"""
query.py — searches the PDB for candidate assemblies and fetches their metadata.

This wraps rcsb.api rather than reimplementing it: rcsb.api already handles
attribute-based search and data retrieval against the PDB reliably. What this
module adds is a cage-design-oriented interface on top of it — short,
memorable field names instead of raw dotted attribute paths, arbitrary
AND/OR combination of any attributes, and range support for numeric/date
fields — all as plain functions, so any interface (script, CLI, or a
NiceGUI form later) can drive the same logic.
"""

from functools import reduce
import pandas as pd
from rcsbapi.search import AttributeQuery
from rcsbapi.data import DataQuery


# --------------------------------------------------------------------------
# Attribute registry: short name -> full rcsb.api dotted path.
#
# A user picks filters by short name here, or passes a raw dotted path
# directly for anything not yet registered. Extend freely.
#
# NOTE on "keywords": rcsb.api's actual field name is very likely
# `struct_keywords.pdbx_keywords` (underscore) rather than the colon
# version below — the colon looks like it may be a typo. Worth confirming
# against the schema before relying on that one field; left as given for
# now so nothing silently changes underneath you.
# --------------------------------------------------------------------------
ATTRIBUTE_REGISTRY = {
    # HIERARCHY & SYMMETRY
    "symmetry": "rcsb_struct_symmetry.symbol",
    "oligomeric_count": "pdbx_struct_assembly.oligomeric_count",
    "stoichiometry": "rcsb_struct_symmetry.stoichiometry",

    # SCALE & MASS
    "polymer_entity_count": "rcsb_assembly_info.polymer_entity_count",
    "composition": "rcsb_assembly_info.polymer_composition",
    "weight": "rcsb_assembly_info.molecular_weight",
    "interface_area": "rcsb_interface_info.interface_area",

    # PROTEIN DESIGN & ORIGIN
    "source_organism": "rcsb_entity_source_organism.scientific_name",
    "source_type": "rcsb_entity_source_organism.type",
    "polymer_entity_type": "rcsb_entry_info.selected_polymer_entity_types",
    "interface_count": "rcsb_interface_info.interface_count",

    # EXPERIMENTAL & DEPOSITION
    "assembly_method": "pdbx_struct_assembly.method_details",
    "resolution": "entry.rcsb_entry_info.resolution_combined",
    "experimental_method": "entry.rcsb_entry_info.experimental_method",
    "model_quality": "ma_qa_metric_global.value",

    # TEXT QUERIES
    "keywords": "struct_keywords.pdbx:keywords",
    "title": "struct.title",
    "date": "rcsb_accession_info.deposit_date",
    "pfam_id": "rcsb_polymer_entity_annotation.annotation_id",
}

# Attributes for which range ("from X to Y") search generally makes more
# sense than exact match. Not enforced anywhere — purely a hint you can use
# when building a UI, so range-appropriate fields default to a range widget.
RANGE_HINTED_ATTRIBUTES = {
    "weight", "interface_area", "resolution", "interface_count",
    "oligomeric_count", "polymer_entity_count", "model_quality", "date",
}


def build_criterion_query(attribute, value, operator="exact_match"):
    """
    Build a single query from one filter criterion.

    attribute : short name from ATTRIBUTE_REGISTRY, or a raw dotted path
    value     : a single value for exact/text operators, or a (low, high)
                tuple when operator="range"
    operator  : "exact_match" (default), "range", or any other rcsb.api
                comparison operator (e.g. "contains_phrase", "greater")

    For operator="range", this builds two combined sub-queries
    (>= low AND <= high) rather than relying on a single "range" operator
    this is the safest approach until we've confirmed rcsb.api's exact range
    syntax against a live call (see the TODO at the bottom of this file).
    """
    path = ATTRIBUTE_REGISTRY.get(attribute, attribute)
    attr = Attr(path)

    if operator == "range":
        low, high = value
        return (attr >= low) & (attr <= high)

    return AttributeQuery(attribute=path, operator=operator, value=value)


def combine_queries(queries, mode="and"):
    """
    Combine several query objects into one.

    mode="and" -> matches all of the queries
    mode="or"  -> matches any of the queries
    """
    if mode not in ("and", "or"):
        raise ValueError("mode must be 'and' or 'or'")
    op = (lambda a, b: a & b) if mode == "and" else (lambda a, b: a | b)
    return reduce(op, queries)


def search_ids(query, return_type="assembly"):
    """Execute a query (single or combined) and return the list of matching IDs."""
    return list(query(return_type=return_type))


def search_by_criteria(criteria, mode="and", return_type="assembly"):
    """
    General-purpose search: any number of filters, any mix of exact-match
    and range criteria, combined with AND or OR.

    criteria : list of dicts, each shaped like one of:
        {"attribute": "symmetry", "value": "T"}
        {"attribute": "resolution", "value": (0, 2.5), "operator": "range"}
        {"attribute": "keywords", "value": "nanocage", "operator": "contains_phrase"}
    mode     : "and" or "or" — applies across ALL criteria in the list.
               (No mixed AND/OR groups yet — a nested-group version can be
               added later if you need e.g. (A OR B) AND C; flag it if so.)

    Example — symmetry T or O, resolution under 2.5A, natural source only:
        search_by_criteria([
            {"attribute": "symmetry", "value": "T"},
            {"attribute": "symmetry", "value": "O"},
        ], mode="or")
    """
    if not criteria:
        raise ValueError("must supply at least one criterion")

    queries = [build_criterion_query(**c) for c in criteria]
    combined = combine_queries(queries, mode=mode) if len(queries) > 1 else queries[0]
    return search_ids(combined, return_type=return_type)


def search_by_symmetry(symbols, return_type="assembly"):
    """
    Convenience wrapper over search_by_criteria for the common case:
    search by one or more symmetry symbols, combined with OR.

    symbols : a single symbol ("T") or a list of symbols (["T", "O"])
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    criteria = [{"attribute": "symmetry", "value": s} for s in symbols]
    return search_by_criteria(criteria, mode="or", return_type=return_type)


def fetch_metadata(ids, fields, input_type="assembly"):
    """
    Fetch metadata attributes for a list of IDs and return them as a DataFrame.

    fields : list of short names (from ATTRIBUTE_REGISTRY) or raw dotted paths.
             DataFrame columns are named using the short name where available,
             otherwise the raw path as given.

    NOTE: the exact shape of DataQuery's returned JSON needs to be confirmed
    against a real response before this flattening logic can be trusted —
    see the TODO at the bottom of this file. Still pending your test run.
    """
    paths = [ATTRIBUTE_REGISTRY.get(f, f) for f in fields]

    query = DataQuery(input_type=input_type, input_ids=ids, return_data_list=paths)
    raw_results = query.exec()

    df = pd.json_normalize(raw_results)
    path_to_short = {v: k for k, v in ATTRIBUTE_REGISTRY.items()}
    df = df.rename(columns=lambda c: path_to_short.get(c, c))
    return df


# --------------------------------------------------------------------------
# TODO (still needs your test run to confirm — carried over from last time):
# No network access here, so two things are unverified against a real call:
#   1. fetch_metadata's json_normalize flattening
#   2. The >= / <= combination used for operator="range" in
#      build_criterion_query — rcsb.api may have a single native "range"
#      operator that's cleaner than two combined queries; worth checking
#      their docs or testing once you can.
#
# When you get a chance, run a small real query (2-3 IDs, a couple fields,
# and one range-based criterion) and paste back what comes out — we'll true
# up both functions against it.
# --------------------------------------------------------------------------