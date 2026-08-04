from toolkit.query import build_query, combine_queries, query_candidates
from toolkit.download import download_candidates, clear_temp_dir
from toolkit.geometry import rings, orientation
from toolkit.geometry import structure
from toolkit import isolate, rfdiffusion, colab



q1 = build_query("symmetry", "T", "exact_match")
q2 = build_query("symmetry", "O", "exact_match")
q3 = build_query("entry_id", "4v6b", "exact_match")


merged = combine_queries([q1, q2], "or")

filter_criteria = [
    {"attribute": "resolution", "value": (0, 5), "operator": "range"},
    {"attribute": "description", "value": "FERRITIN", "operator": "contains_words"} ]

one_shot_df = query_candidates(
    search_criteria=[q3],  # a list of built objects
    fetch_fields=["weight", "title"],
    filter_criteria=None,
    return_type="assembly",
)
print(one_shot_df)

downloaded_df = download_candidates(one_shot_df)

dist_df = rings.from_structure(downloaded_df)
print(dist_df)

orientation_df = orientation.from_rings(dist_df, structures=downloaded_df, symmetry_type="C3")

termini_ss_df = structure.from_rings(orientation_df, downloaded_df, "C3")

orientation_df = orientation_df.merge(
    termini_ss_df[["assembly_id", "symmetry_type", "termini_ss"]],
    on=["assembly_id", "symmetry_type"],
    how="left",
)
print(orientation_df)

# Isolate EVERY ring in orientation_df, each from ITS OWN assembly's
# structure file (looked up per-row) rather than reusing one filepath for
# every row regardless of which assembly it actually belongs to.
pdb_paths = {}
rename_maps = {}
for _, ring_row in orientation_df.iterrows():
    ring_assembly_id = ring_row["assembly_id"]
    ring_filepath = downloaded_df.loc[
        downloaded_df["assembly_id"] == ring_assembly_id, "filepath"
    ].values[0]

    path, rename_map = isolate.extract_ring_structure(
        ring_filepath, ring_row["chain_groups"],
        f"temporary_subunits/{ring_assembly_id}_ring.pdb",
        file_format="pdb",
    )
    pdb_paths[ring_assembly_id] = path
    rename_maps[ring_assembly_id] = rename_map

final_df = rfdiffusion.process_ring_dataframe(orientation_df, pdb_paths)
print(final_df)

# Build an RFdiffusion job for one specific ring — row 0, as an example.
row = final_df.loc[4]
assembly_id = row["assembly_id"]
chain_group = row["chain_groups"]
 
short_order = rfdiffusion.remap_chain_order(chain_group, rename_maps[assembly_id])
 
job = rfdiffusion.prepare_fusion_job(
    pdb_paths[assembly_id], short_order,
    linker_length=(10, 15),
)
 
# colab was imported but never used in the original script — this is the
# natural next step if your cluster is still down: package the job for
# the companion notebook instead of calling rfdiffusion.submit(job).
bundle_zip = colab.prepare_colab_bundle(job)
print(bundle_zip)