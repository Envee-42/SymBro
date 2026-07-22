"""
query.py — searches the PDB for candidate assemblies and fetches their metadata.

This wraps rcsb.api rather than reimplementing it: rcsb.api already handles
attribute-based search and data retrieval against the PDB reliably. What this
module adds is a cage-design-oriented interface on top of it — short,
memorable field names instead of raw dotted attribute paths, and easy
combination of multiple search criteria (e.g. "symmetry is T OR O").
"""

from functools import reduce
import pandas as pd
from rcsbapi.search import AttributeQuery
from rcsbapi.data import DataQuery


# --------------------------------------------------------------------------
# Attribute registry: short name -> full rcsb.api dotted path.
#
# This is what makes the tool modular. A user building their own candidate
# list doesn't need to know rcsb.api's schema — they pick from short names
# here, or pass a raw dotted path directly if they need something not yet
# in the registry (see fetch_metadata below).
#
# Extend this dict as you identify more attributes relevant to cage design.
# --------------------------------------------------------------------------
ATTRIBUTE_REGISTRY = {
    # HIERARCHY & SYMMETRY
    "symmetry": "rcsb_struct_symmetry.symbol",
    "oligomeric_count": "pdbx_struct_assembly.oligomeric_count",
    "stoichiometry": "rcsb_struct_symmetry.stoichiometry",

    # SCALE & MASS
    "polymer_entity_count": "rcsb_assembly_info.polymer_entity_count",
    "composition": "rcsb_assembly_info.polymer_composition", # Like homo or hetero
    "weight": "rcsb_assembly_info.molecular_weight", # kDA or MDa, range or interval
    "interface_area": "rcsb_interface_info.interface_area", # Value or range of interface surface (Angstroms squared)

    # PROTEIN DESIGN & ORIGIN
    "source_organism": "rcsb_entity_source_organism.scientific_name", # Species of origin or de novo
    "source_type": "rcsb_entity_source_organism.type", # Natural, synthetic, genetically engineered
    "polymer_entity_type": "rcsb_entry_info.selected_polymer_entity_types", # Protein only or more
    "interface_count": "rcsb_interface_info.interface_count", # Number of distinct contact points in the assembly

    # EXPERIMENTAL & DEPOSITION
    "assembly_method": "pdbx_struct_assembly.method_details",
    "resolution": "entry.rcsb_entry_info.resolution_combined",
    "experimental_method": "entry.rcsb_entry_info.experimental_method",
    "model_quality": "ma_qa_metric_global.value", # plDDT score for computational or predicted models (AF2 / AF3)

    # TEXT QUERIES
    "keywords": "struct_keywords.pdbx:keywords", 
    "title": "struct.title", # Free-text search targets
    "date": "rcsb_accession_info.deposit_date", # Date ranges to filter for more or less recent structs
    "pfam_id": "rcsb_polymer_entity_annotation.annotation_id" # Protein Family accession IDs    
}


def build_attribute_query(attribute, value, operator="exact_match"):
    """
    Build a single rcsb.api AttributeQuery.

    attribute : short name from ATTRIBUTE_REGISTRY, or a raw dotted path
    value     : the value to match (e.g. "T")
    operator  : rcsb.api comparison operator, default "exact_match"
    """
    path = ATTRIBUTE_REGISTRY.get(attribute, attribute)
    return AttributeQuery(attribute=path, operator=operator, value=value)


def combine_queries(queries, mode="or"):
    """
    Combine several AttributeQuery objects into one.

    mode="or"  -> matches any of the queries (e.g. symmetry T OR O)
    mode="and" -> matches all of the queries (e.g. symmetry T AND resolution < 2.0)

    rcsb.api overloads Python's | and & operators on Query objects for this
    exact purpose, so we just fold the list with the right operator.
    """
    if mode not in ("or", "and"):
        raise ValueError("mode must be 'or' or 'and'")
    op = (lambda a, b: a | b) if mode == "or" else (lambda a, b: a & b)
    return reduce(op, queries)


def search_ids(query, return_type="assembly"):
    """Execute a query (single or combined) and return the list of matching IDs."""
    return list(query(return_type=return_type))


def search_by_symmetry(symbols, return_type="assembly"):
    """
    Convenience wrapper: search by one or more symmetry symbols, combined with OR.

    symbols : a single symbol ("T") or a list of symbols (["T", "O"])
    """
    if isinstance(symbols, str):
        symbols = [symbols]
    queries = [build_attribute_query("symmetry", s) for s in symbols]
    combined = combine_queries(queries, mode="or") if len(queries) > 1 else queries[0]
    return search_ids(combined, return_type=return_type)


def fetch_metadata(ids, fields, input_type="assembly"):
    """
    Fetch metadata attributes for a list of IDs and return them as a DataFrame.

    fields : list of short names (from ATTRIBUTE_REGISTRY) or raw dotted paths.
             DataFrame columns are named using the short name where available,
             otherwise the raw path as given.

    NOTE: the exact shape of DataQuery's returned JSON (how deeply it nests
    entry-level vs. assembly-level fields) needs to be confirmed against a
    real response before this flattening logic can be trusted end to end —
    see the note at the bottom of this file.
    """
    paths = [ATTRIBUTE_REGISTRY.get(f, f) for f in fields]

    query = DataQuery(input_type=input_type, input_ids=ids, return_data_list=paths)
    raw_results = query.exec()

    # Flatten the nested response into one row per ID. json_normalize handles
    # dotted-path nesting automatically; we then rename columns back to the
    # short names the user asked for, where we have one.
    df = pd.json_normalize(raw_results)
    path_to_short = {v: k for k, v in ATTRIBUTE_REGISTRY.items()}
    df = df.rename(columns=lambda c: path_to_short.get(c, c))
    return df


# --------------------------------------------------------------------------
# TODO (needs your test run to confirm):
# I don't have network access in this environment, so I can't hit rcsb.api
# to see the real shape of DataQuery().exec()'s output. The json_normalize
# call above is a reasonable first guess, but if the response nests IDs and
# fields differently than expected, the column names won't line up cleanly.
#
# Run this and paste back:
#   raw = DataQuery(input_type="assembly", input_ids=t_id_list[:2],
#                    return_data_list=[...]).exec()
#   print(raw)
# just for 1-2 IDs, and we'll adjust fetch_metadata's flattening to match
# the actual structure.
# --------------------------------------------------------------------------