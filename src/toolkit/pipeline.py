
from toolkit.query import build_query, search_ids, fetch_metadata, filter_metadata, query_candidates, DATA_ATTRIBUTES, combine_queries, _resolve_path
from toolkit.download import download_candidates, clear_temp_dir
from toolkit.geometry.distance import run_ring_analysis

q1 = build_query("symmetry", "T", "exact_match")
q2 = build_query("symmetry", "O", "exact_match")

#test a nested query by the way

merged = combine_queries([q1, q2], "or")

filter_criteria = [
    {"attribute": "source_organism", "value": "Escherichia coli", "operator": "exact_match"},
    {"attribute": "resolution", "value": (2, 2.2), "operator": "range"}
   ]


one_shot_df = query_candidates(
    search_criteria=[merged],  # a list of built objects
    fetch_fields=["source_organism", "resolution", "weight"], 
    filter_criteria=filter_criteria,
    return_type="assembly",
)
print(one_shot_df)

downloads = download_candidates(one_shot_df)

dist_df = run_ring_analysis(downloads)

print(dist_df)

