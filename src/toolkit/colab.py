"""
colab.py — an emergency/interim path to run RFdiffusion on Google Colab's
free-tier GPU, for when neither a local GPU nor a configured Singularity/
SLURM cluster is reachable (e.g. your institution's cluster is down).

Why this is two plain functions and not a third submit() backend
------------------------------------------------------------------
"local" and "singularity" both mean "Symseeker's own Python process spawns
a subprocess on a machine it already controls" — submit()/poll_status()
work because that subprocess shares a filesystem with Symseeker, so
polling for output files to appear is meaningful.

Colab is a different execution model entirely: it's a separate Jupyter
notebook running on a Google-controlled machine, with its own filesystem
and its own GPU. Symseeker cannot spawn it, cannot poll it, and cannot see
its output files appear on disk — there is a human in the loop, physically
uploading a bundle in and downloading results back out. So instead of
faking a backend that doesn't fit that shape, Colab support is two plain
functions that bookend the human step:

    prepare_colab_bundle(job)   BEFORE Colab — packages an RFdiffusionJob
                                 (+ its input PDB) into a small, self-
                                 contained .zip the user uploads to the
                                 companion notebook.

    import_colab_results(job)   AFTER Colab — takes whatever the user
                                 downloaded back from the notebook and
                                 drops it at job.output_prefix, then
                                 returns a dict shaped exactly like
                                 rfdiffusion.poll_status()'s "completed"
                                 result — so a NiceGUI results panel, or a
                                 script chaining straight into
                                 ProteinMPNN, never needs to special-case
                                 "this job happened to run on Colab."

Both functions call rfdiffusion._build_overrides() rather than re-deriving
the Hydra key=value syntax — there is exactly one place that knows how an
RFdiffusionJob turns into RFdiffusion's CLI overrides, and Colab reuses it
so the run that happens inside the notebook is guaranteed to use the same
contigs/hotspots/num_designs you already validated locally, never a
hand-retyped copy that could drift.

The companion notebook (symseeker_rfdiffusion_colab.ipynb, shipped
alongside this file) is deliberately dumb: install RFdiffusion + weights
once per Colab session, read job_manifest.json, run inference with its
overrides verbatim, zip the outputs for download. All the "is this contig
valid, is this hotspot actually on the structure" checking already
happened locally, before the bundle was ever built.
"""

import glob
import json
import os
import re
import shutil
import zipfile
from dataclasses import replace
from typing import Optional

from toolkit.rfdiffusion import RFdiffusionJob, _build_overrides

MANIFEST_NAME = "job_manifest.json"
BUNDLE_INPUT_NAME = "input.pdb"
BUNDLE_OUTPUT_PREFIX = "output/design"


def prepare_colab_bundle(job: RFdiffusionJob, bundle_dir: Optional[str] = None, zip_bundle: bool = True) -> str:
    """
    Packages `job` into a self-contained folder (zipped by default) meant
    to be uploaded, as-is, to the companion notebook.

    Why a bundle instead of just handing Colab the job object: Colab's
    filesystem is not Symseeker's filesystem, so job.input_pdb's real path
    (e.g. "temporary_subunits/ring_C3.pdb") means nothing there. The
    bundle copies the input PDB in under a fixed name and rewrites the
    Hydra overrides to point at that fixed name and a fixed, bundle-
    relative output_prefix — so the notebook never has to know, or guess,
    any of Symseeker's own paths.

    Returns the path actually produced: a .zip file if zip_bundle=True
    (the default — a single file is what Colab's upload widget wants), or
    the bundle folder itself if zip_bundle=False.
    """
    if bundle_dir is None:
        base = os.path.splitext(os.path.basename(job.output_prefix))[0] or "job"
        bundle_dir = f"{base}_colab_bundle"

    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir)
    os.makedirs(os.path.join(bundle_dir, "output"), exist_ok=True)

    if not os.path.exists(job.input_pdb):
        raise FileNotFoundError(
            f"job.input_pdb={job.input_pdb!r} does not exist — build/submit "
            f"the job against a real file (e.g. isolate.py's output) before "
            f"packaging it for Colab."
        )
    shutil.copy(job.input_pdb, os.path.join(bundle_dir, BUNDLE_INPUT_NAME))

    # Same overrides a local/singularity run would build — just pointed at
    # the bundle's fixed, self-contained paths instead of job's real ones.
    bundle_job = replace(job, input_pdb=BUNDLE_INPUT_NAME, output_prefix=BUNDLE_OUTPUT_PREFIX)
    overrides = _build_overrides(bundle_job)

    manifest = {
        "overrides": overrides,
        "num_designs": job.num_designs,
        "contigs": job.contigs,
        "hotspot_res": job.hotspot_res,
        # Not read by the notebook — carried through so
        # import_colab_results() can report which job these results
        # actually belong to if something looks off.
        "original_output_prefix": job.output_prefix,
    }
    with open(os.path.join(bundle_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)

    if not zip_bundle:
        return bundle_dir

    zip_path = f"{bundle_dir}.zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _dirs, files in os.walk(bundle_dir):
            for name in files:
                full = os.path.join(root, name)
                zf.write(full, os.path.relpath(full, bundle_dir))
    shutil.rmtree(bundle_dir)
    return zip_path


def import_colab_results(job: RFdiffusionJob, downloaded_path: str, cleanup: bool = True) -> dict:
    """
    Takes whatever the user downloaded back from the Colab notebook (a
    .zip, or an already-unzipped folder, containing an "output/"
    subfolder of "design_<N>.pdb" files — the bundle's fixed
    BUNDLE_OUTPUT_PREFIX names them that way) and copies the designs to
    job.output_prefix — the SAME location a local/singularity run would
    have written them — renamed to match ("{output_prefix}_<N>.pdb").

    Returns a dict shaped exactly like rfdiffusion.poll_status()'s return
    value, so calling code (a NiceGUI results panel, a script chaining
    into ProteinMPNN next) can treat a finished Colab job identically to a
    finished local/singularity one — no "if it came from Colab" branch
    needed anywhere downstream. The two differences: returncode is always
    None (there's no local subprocess exit code for a job that ran on
    someone else's machine) and log_path is always None (the notebook's
    own output cells are that job's log, not a file Symseeker can reach).
    """
    extract_dir = downloaded_path
    temp_extract = None
    if os.path.isfile(downloaded_path) and downloaded_path.endswith(".zip"):
        temp_extract = f"{downloaded_path}__extracted"
        if os.path.exists(temp_extract):
            shutil.rmtree(temp_extract)
        with zipfile.ZipFile(downloaded_path) as zf:
            zf.extractall(temp_extract)
        extract_dir = temp_extract

    found = sorted(glob.glob(os.path.join(extract_dir, "**", "design_*.pdb"), recursive=True))
    if not found:
        raise FileNotFoundError(
            f"No 'design_*.pdb' files found under {downloaded_path!r} — make "
            f"sure you downloaded the notebook's results.zip and pointed this "
            f"at it (or its unzipped folder) as-is, without renaming anything "
            f"inside it."
        )

    output_dir = os.path.dirname(job.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    design_paths = []
    index_pattern = re.compile(r"design_(\d+)\.pdb$")
    for i, src in enumerate(found):
        m = index_pattern.search(src)
        index = m.group(1) if m else str(i)
        dest = f"{job.output_prefix}_{index}.pdb"
        shutil.copy(src, dest)
        design_paths.append(dest)
    design_paths.sort()

    if temp_extract and cleanup:
        shutil.rmtree(temp_extract)

    state = "completed" if len(design_paths) >= job.num_designs else "completed_partial"

    return {
        "state": state,
        "returncode": None,
        "designs_written": len(design_paths),
        "designs_expected": job.num_designs,
        "design_paths": design_paths,
        "log_path": None,
    }