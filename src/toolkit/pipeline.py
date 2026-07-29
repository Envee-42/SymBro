# -------- TESTING -----------
from query import query, combine_queries, search_ids, fetch_metadata
from query import unified_search_and_fetch
from download import download_candidates, clear_temp_dir
from geometry.termini import get_chain_ca_geometry
from toolkit.geometry.distance import run_ring_analysis
from geometry.orientation import annotate_ring_orientation, plot_ring_orientation
import gemmi

my_criteria = [
    # 1. Standard API search
    {"attribute": "symmetry", "value": "T"},
    
    # 2. Range API search
    {"attribute": "weight", "value": (500000, 1000000), "operator": "range"},
    
    # 3. Text search
    {"attribute": "keywords", "value": "nanocage", "operator": "contains_phrase"},
    
    # 4. Local-only filter (the orchestrator handles this in Pandas automatically)
    #    (Assuming model_quality isn't in SEARCH_ATTRIBUTES)
    {"attribute": "model_quality", "value": (0.8, 1.0), "operator": "range"},
    
    # 5. Bring these columns along for the ride, but don't filter by them
    {"attribute": "source_organism", "operator": "fetch_only"},
    {"attribute": "date", "operator": "fetch_only"}
]

dfc = unified_search_and_fetch(criteria=my_criteria)

print(dfc)