
from toolkit.query import build_query, search_ids, fetch_metadata, filter_metadata, query_candidates, DATA_ATTRIBUTES, combine_queries, _resolve_path
from toolkit.download import download_candidates, clear_temp_dir
from toolkit.geometry import rings, orientation
from toolkit.geometry import structure
from toolkit import isolate

clear_temp_dir()
isolate.clear_temp_subunits_dir()

q1 = build_query("symmetry", "T", "exact_match")
q2 = build_query("resolution", (0, 5), "range")
q3 = build_query("symmetry", "O", "exact_match")

merged = combine_queries([q1, q3], "or")

filter_criteria = [
    {"attribute": "resolution", "value": (0, 10), "operator": "range"},
    {"attribute": "keywords", "value": "FERRITIN", "operator": "in"} ]

one_shot_df = query_candidates(
    search_criteria=[merged],  # a list of built objects
    fetch_fields=["weight", "keywords"], 
    filter_criteria=filter_criteria,
    return_type="assembly",
)
print(one_shot_df)

downloaded_df = download_candidates(one_shot_df)

dist_df = rings.from_structure(downloaded_df)

print(dist_df)

orientation_df = orientation.from_rings(dist_df, structures=downloaded_df, symmetry_type="C3")

row = orientation_df.iloc[0]
# Filters for the matching row and extracts the filepath
filepath = downloaded_df.loc[downloaded_df["assembly_id"] == row["assembly_id"], "filepath"].values[0]


termini_ss_df = structure.from_rings(orientation_df, downloaded_df, "C3")

orientation_df = orientation_df.merge(
    termini_ss_df[["assembly_id", "symmetry_type", "termini_ss"]],
    on=["assembly_id", "symmetry_type"],
    how="left",
)

print(orientation_df)

extracted_df = isolate.from_rings(orientation_df, downloaded_df)

print(extracted_df[["assembly_id", "symmetry_type", "chain_groups", "filepath"]])

orientation.plot_chain_group_vectors(filepath, row["chain_groups"])

