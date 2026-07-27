# -------- TESTING -----------
from query import query, combine_queries, search_ids, fetch_metadata
from download import download_candidates, clear_temp_dir
from geometry.distance import process_candidates, run_ring_analysis

q1 = query("symmetry", "T", "exact_match")
q2 = query("resolution", (2, 2), "range")
qs = combine_queries([q1, q2], "and")

cands = search_ids(qs, "assembly")
cand_df = fetch_metadata(cands, ["symmetry", "weight", "oligomeric_count"], "assembly")
cand_df["assembly_id"] = cands

down_df = download_candidates(cands)

combined_df = down_df.merge(cand_df, on="assembly_id")

ranked_df = run_ring_analysis(combined_df, ring_size=3, tolerance=1.5)

print(ranked_df)
clear_temp_dir()