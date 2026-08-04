"""
rfdiffusion.py — generates RFdiffusion inputs (contigs, hotspots) from
already-computed geometry, and dispatches inference as a non-blocking
background job.

RFdiffusion itself stays external (see external_tools_architecture.md):
its install is heavy (CUDA-pinned conda env, SE(3)-Transformer, DGL) and
deliberately NOT a dependency of this package. What symseeker owns is
everything AROUND the actual diffusion run: building its fiddly
Hydra-config-override command correctly, generating its contig/hotspot
syntax from data symseeker already has (rather than the user hand-typing
"A1-150/20-30/B1-100"), and launching it in a way that never blocks the
calling process — critical for a NiceGUI app, whose whole UI runs on one
asyncio event loop.

Command syntax below (contig grammar, hotspot grammar, output file
naming) was verified directly against RosettaCommons/RFdiffusion's
README, examples/design_ppi.sh, and scripts/run_inference.py — not
recalled from memory — since a wrong bracket or missing space here
produces a silently-broken design job.

Design, in short:

- CHAIN NAMING WARNING, read this first: RFdiffusion's contig/hotspot
  grammar assumes short chain ids immediately followed by digits (e.g.
  "A150" unambiguously means chain A, residue 150). isolate.py's DEFAULT
  mmCIF output preserves RCSB's original assembly-expanded chain names
  (e.g. "A-13") to avoid lossy renaming — but those same names make
  RFdiffusion's contig grammar unparseable ("A-13150-180" cannot be told
  apart from chain "A", "A-1", or "A-13"). So: always extract the ring
  fed to this module with isolate.py's `file_format="pdb"` option (which
  reassigns short A/B/C/... ids and returns an explicit rename_map), then
  translate chain_groups through that rename_map via remap_chain_order()
  before calling anything here. Every function below assumes it's
  operating on an already-short-chain-id structure; none of them attempt
  to detect or fix a too-long chain name themselves.

- Contig/hotspot strings are built from the ACTUAL resolved residue
  numbering read out of the target structure (gemmi res.seqid.num), and
  validated against it before ever being handed to RFdiffusion — a typo'd
  hotspot or an out-of-range fixed segment fails here, immediately, with
  a clear message, rather than deep inside a multi-minute GPU job.

- build_linker_fusion_contig() is the direct answer to the project's own
  stated goal ("so the user can easily design new fusions"): given a
  chain_groups-style tuple (already in real physical ring order — see
  rings.py's own docstring on why chain_groups IS ring order) it
  generates the fixed/diffused-linker/fixed/... contig automatically, no
  manual residue-range typing at all.

- submit() launches RFdiffusion via subprocess.Popen (never
  subprocess.run, which blocks the caller until the process exits) and
  returns an RFdiffusionRun handle immediately. RFdiffusion writes no
  completion/summary file (confirmed against run_inference.py), so
  poll_status() detects progress the same way the tool itself resumes an
  interrupted run: by counting "{output_prefix}_<N>.pdb" files on disk.
  Both are cheap enough to call every second or two from a NiceGUI
  ui.timer callback without any special threading.

- Two backends are implemented: "local" (bare subprocess.Popen — assumes
  RFdiffusion is directly installed in a reachable Python env) and
  "singularity" (runs a .sif container image — the realistic path for
  most real users, since RFdiffusion's install is heavy and most target
  users are on shared HPC/SLURM clusters where Docker is usually blocked
  but Singularity/Apptainer isn't; confirmed against prosculpt's own
  production installation.yaml, a directly comparable published tool).
  The singularity backend bind-mounts each host folder the job touches
  to the SAME path inside the container (rather than remapping to a
  fixed in-container path and rewriting every Hydra override to match,
  which is what prosculpt itself does) — simpler, and there's no path to
  get subtly wrong between host and container. A "slurm" backend is the
  documented next step (see external_tools_architecture.md) —
  submit()'s backend= parameter and RFdiffusionRun's shape are already
  set up so adding it never changes how poll_status()/cancel() are
  called.

- Per-user tool paths (which backend, the .sif image path, GPU/bind
  settings) are never hardcoded here — they're read from config.py's
  installation config (a gitignored YAML file, following prosculpt's own
  installation.yaml convention) via submit(job, config=...), or can
  still be passed explicitly/individually for scripting and tests.
"""

import glob
import math
import os
import re
import subprocess
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gemmi
import pandas as pd

from toolkit.config import get_tool_config


# ============================================================================
# Chain-naming bridge to isolate.py
# ============================================================================

def remap_chain_order(chain_order: Sequence[str], rename_map: Dict[str, str]) -> Tuple[str, ...]:
    """
    Translates a chain_groups-style tuple of ORIGINAL chain names (as
    rings.py/orientation.py/structure.py report them) into their
    RFdiffusion-safe short ids, using the rename_map isolate.py's
    extract_ring_structure()/from_rings() returns when writing
    file_format="pdb". A name with no entry in rename_map is assumed to
    already be short enough and is passed through unchanged.
    """
    return tuple(rename_map.get(name, name) for name in chain_order)


# ============================================================================
# Contig-map construction
# ============================================================================

def _validate_fixed_segment(model: gemmi.Model, chain: str, start: int, end: int) -> None:
    if start > end:
        raise ValueError(f"fixed segment start ({start}) must be <= end ({end}) for chain {chain!r}")

    gemmi_chain = model.find_chain(chain)
    if gemmi_chain is None:
        raise ValueError(f"chain {chain!r} not found in structure — available chains: {[c.name for c in model]}")

    resnums = {res.seqid.num for res in gemmi_chain.get_polymer()}
    if not resnums:
        raise ValueError(f"chain {chain!r} has no resolved polymer residues")

    missing = [n for n in (start, end) if n not in resnums]
    if missing:
        raise ValueError(
            f"chain {chain!r} fixed segment {start}-{end} is not fully within its resolved "
            f"residues (missing residue number(s) {missing}) — resolved range is "
            f"{min(resnums)}-{max(resnums)}"
        )


def _format_diffuse(min_len: int, max_len: Optional[int] = None) -> str:
    if max_len is None:
        max_len = min_len
    if not (isinstance(min_len, int) and isinstance(max_len, int)) or min_len < 0 or max_len < min_len:
        raise ValueError(f"diffuse segment length must be a valid (min, max) with 0 <= min <= max — got ({min_len}, {max_len})")
    return f"{min_len}-{max_len}"


def build_contig_string(model: gemmi.Model, segments: Sequence[tuple]) -> str:
    """
    Low-level, fully explicit contig builder — a 1:1 mirror of
    RFdiffusion's own contigmap.contigs grammar, no inferred behavior.

    segments : ordered list of tuples, each one of:
        ("fixed", chain, start, end)        — keep this chain's residues
            start-end from the input structure verbatim. Validated
            against `model`'s actual resolved residue numbers.
        ("diffuse", length)                 — generate a new segment of
            exactly `length` residues.
        ("diffuse", min_length, max_length) — generate a new segment
            whose length is sampled in [min_length, max_length].
        ("break",)                          — an explicit chain break
            (RFdiffusion's "0" token): everything before this point and
            everything after it become SEPARATE chains in the output,
            rather than one continuous fused chain. Confirmed syntax is
            "/0 " (slash, zero, SPACE) joining directly to the next
            segment with no further slash — e.g. a target chain kept
            fixed with a freshly-diffused binder chain after it:
            [A1-150/0 70-100]. Segments on either side of a "break" with
            NO break between them (e.g. fixed/diffuse/fixed) instead
            fuse into one continuous new chain — that's the pattern
            build_linker_fusion_contig() below generates automatically
            for ring-subunit fusions.

    Returns the bracketed, RFdiffusion-ready contig string, e.g.
    "[A1-150/20-30/B1-100]" or "[A1-150/0 70-100]" — ready to assign
    straight to RFdiffusionJob(contigs=...).

    Raises ValueError for an invalid segment, an out-of-range fixed
    segment, or an unrecognized segment kind.
    """
    result = ""
    for segment in segments:
        kind = segment[0]

        if kind == "fixed":
            if len(segment) != 4:
                raise ValueError(f"'fixed' segment must be ('fixed', chain, start, end) — got {segment!r}")
            _, chain, start, end = segment
            _validate_fixed_segment(model, chain, start, end)
            token = f"{chain}{start}-{end}"
        elif kind == "diffuse":
            if len(segment) == 2:
                token = _format_diffuse(segment[1])
            elif len(segment) == 3:
                token = _format_diffuse(segment[1], segment[2])
            else:
                raise ValueError(f"'diffuse' segment must be ('diffuse', length) or ('diffuse', min, max) — got {segment!r}")
        elif kind == "break":
            # RFdiffusion's chain-break token is "0", joined to the
            # FOLLOWING segment by a space rather than the usual "/" —
            # confirmed: 'contigmap.contigs=[A1-150/0 70-100]'.
            result += "/0 "
            continue
        else:
            raise ValueError(f"unrecognized segment kind {kind!r} — expected 'fixed', 'diffuse', or 'break'")

        if result and not result.endswith(" "):
            result += "/"
        result += token

    return f"[{result}]"


def build_linker_fusion_contig(
    model: gemmi.Model, chain_order: Sequence[str], linker_length: Union[int, Tuple[int, int]],
) -> str:
    """
    Direct answer to "design new fusions": builds a fixed/diffused-linker/
    fixed/.../fixed contig across chain_order, with a diffused linker of
    `linker_length` residues inserted between every consecutive pair — no
    chain-break tokens, since the entire point of a fusion is that every
    original chain plus its linkers becomes ONE continuous new chain.

    chain_order : the chains to fuse head-to-tail, in fusion order —
        typically a chain_groups tuple straight from
        rings.py/orientation.py (already the real physical ring order,
        each chain's C-terminus adjacent to the next chain's N-terminus —
        see rings.py's own docstring on why chain_groups IS ring order),
        remapped to short ids first via remap_chain_order() (see module
        docstring's chain-naming warning).
    linker_length : an int for an exact-length linker, or a (min, max)
        tuple/list for a sampled-length range.

    Each chain's own fixed residue range is read directly from `model`'s
    actual resolved auth_seq_id numbering — nothing is typed by hand.

    Returns the bracketed contig string, ready for
    RFdiffusionJob(contigs=...). Use build_contig_string directly (with
    an explicit ("break",) segment) instead if you want chains kept as
    SEPARATE output chains rather than fused into one.
    """
    if len(chain_order) < 2:
        raise ValueError(f"chain_order needs at least 2 chains to build a fusion linker between them — got {tuple(chain_order)!r}")

    if isinstance(linker_length, (tuple, list)):
        min_len, max_len = linker_length
    else:
        min_len = max_len = linker_length

    segments = []
    for i, chain_name in enumerate(chain_order):
        start, end = _chain_residue_range(model, chain_name)
        segments.append(("fixed", chain_name, start, end))
        if i < len(chain_order) - 1:
            segments.append(("diffuse", min_len, max_len))

    return build_contig_string(model, segments)


def _chain_residue_range(model: gemmi.Model, chain_name: str) -> Tuple[int, int]:
    chain = model.find_chain(chain_name)
    if chain is None:
        raise KeyError(f"chain {chain_name!r} not found in structure — available chains: {[c.name for c in model]}")
    resnums = [res.seqid.num for res in chain.get_polymer() if res.find_atom("CA", "*")]
    if not resnums:
        raise ValueError(f"chain {chain_name!r} has no resolved CA atoms — cannot determine its residue range")
    return min(resnums), max(resnums)


# ============================================================================
# Hotspot construction
# ============================================================================

def build_hotspot_string(model: gemmi.Model, hotspots: Sequence[Tuple[str, int]]) -> str:
    """
    hotspots : list of (chain, resnum) tuples — chain should already be a
        short, RFdiffusion-safe id (see module docstring / remap_chain_order
        if starting from rings.py/orientation.py's original chain names).
        Deliberately NOT accepting a combined "A59"-style string here:
        this project's chain names routinely contain digits/hyphens
        themselves (e.g. "A-13"), which makes splitting a combined string
        into "chain" + "resnum" genuinely ambiguous — an explicit tuple
        has no such ambiguity.

    Every hotspot is validated against `model` (chain exists, residue
    number is an actually-resolved residue) before being accepted, so a
    typo surfaces here immediately rather than as an opaque failure deep
    inside RFdiffusion. Raises ValueError naming every invalid entry at
    once (not just the first) if any are found.

    Returns the bracketed, ppi.hotspot_res-ready string, e.g.
    "[A59,A83,A91]".
    """
    tokens = []
    bad = []
    for chain, resnum in hotspots:
        gemmi_chain = model.find_chain(chain)
        resnums = {res.seqid.num for res in gemmi_chain.get_polymer()} if gemmi_chain is not None else set()
        if gemmi_chain is None or resnum not in resnums:
            bad.append((chain, resnum))
        else:
            tokens.append(f"{chain}{resnum}")

    if bad:
        raise ValueError(f"hotspot residue(s) not found in structure: {bad}")

    return f"[{','.join(tokens)}]"


# ============================================================================
# Job spec
# ============================================================================

@dataclass
class RFdiffusionJob:
    """A fully-specified, ready-to-submit RFdiffusion inference call."""

    input_pdb: str
    output_prefix: str
    contigs: str
    hotspot_res: Optional[str] = None
    num_designs: int = 10
    diffuser_T: int = 50
    # Escape hatch: any other Hydra "key=value" override RFdiffusion
    # accepts (e.g. "denoiser.noise_scale_ca") that this module doesn't
    # model explicitly — passed straight through so a new RFdiffusion
    # flag never needs a code change here to be usable.
    extra_overrides: Dict[str, str] = field(default_factory=dict)


def _build_overrides(job: RFdiffusionJob, input_pdb: Optional[str] = None, output_prefix: Optional[str] = None) -> List[str]:
    """
    Builds just the Hydra "key=value" override tokens shared by every
    backend — NOT prefixed with how to invoke RFdiffusion itself (that
    part differs: "python scripts/run_inference.py ..." for a bare local
    install, vs. just these overrides appended after the image name for
    a container, since a container's entrypoint already knows to run the
    script — confirmed against prosculpt's own production RFdiffusion
    invocation, which uses `singularity run --nv ... rfdiff.sif
    inference.schedule_directory_path=...` with no python/script prefix
    at all).

    input_pdb/output_prefix let a caller override those two specific
    paths (e.g. to the absolute, container-visible path) without
    mutating `job` itself — used by the singularity backend below.
    """
    input_pdb = job.input_pdb if input_pdb is None else input_pdb
    output_prefix = job.output_prefix if output_prefix is None else output_prefix

    overrides = [
        f"inference.input_pdb={input_pdb}",
        f"inference.output_prefix={output_prefix}",
        f"inference.num_designs={job.num_designs}",
        f"diffuser.T={job.diffuser_T}",
        f"contigmap.contigs={job.contigs}",
    ]
    if job.hotspot_res:
        overrides.append(f"ppi.hotspot_res={job.hotspot_res}")
    for key, value in job.extra_overrides.items():
        overrides.append(f"{key}={value}")
    return overrides


def build_command(job: RFdiffusionJob, script_path: str = "scripts/run_inference.py", python_executable: str = "python") -> List[str]:
    """
    Builds the argv LIST (never a shell string) for invoking RFdiffusion
    directly via "python scripts/run_inference.py ..." — the "local"
    backend's command. See _submit_singularity for the container
    equivalent.

    Returned as a list on purpose: subprocess.Popen given a list bypasses
    the shell entirely, so RFdiffusion's fiddly contig syntax (square
    brackets, and the literal space in a chain-break token — see
    build_contig_string) never needs manual shell-quoting or escaping —
    each element reaches the process verbatim as one argv entry.
    """
    return [python_executable, script_path] + _build_overrides(job)


def prepare_fusion_job(
    ring_pdb_path: str, chain_order: Sequence[str], linker_length: Union[int, Tuple[int, int]],
    output_dir: Optional[str] = None, num_designs: int = 10,
    hotspots: Optional[Sequence[Tuple[str, int]]] = None, diffuser_T: int = 50,
    extra_overrides: Optional[Dict[str, str]] = None,
) -> RFdiffusionJob:
    """
    One call tying isolate.py's short-chain-id PDB output directly to a
    submit()-ready job:

        pdb_path, rename_map = isolate.extract_ring_structure(
            filepath, chain_groups, "temporary_subunits/ring.pdb", file_format="pdb")
        short_order = rfdiffusion.remap_chain_order(chain_groups, rename_map)
        job = rfdiffusion.prepare_fusion_job(pdb_path, short_order, linker_length=(15, 25))
        run = rfdiffusion.submit(job)

    Reads `ring_pdb_path` once with gemmi, then builds the fixed/diffused
    fusion contig across chain_order (build_linker_fusion_contig) and, if
    given, the hotspot string (build_hotspot_string) against that same
    parsed structure, so both are validated together in one pass.

    output_dir defaults to an "rfdiffusion_designs" folder next to
    ring_pdb_path itself — same workspace-visible-scratch-folder
    philosophy as download.py/isolate.py, rather than an OS temp
    directory; output_prefix is derived from the ring file's own
    basename.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(ring_pdb_path)) or ".", "rfdiffusion_designs")
    os.makedirs(output_dir, exist_ok=True)

    structure = gemmi.read_structure(ring_pdb_path)
    model = structure[0]

    contigs = build_linker_fusion_contig(model, chain_order, linker_length)
    hotspot_res = build_hotspot_string(model, hotspots) if hotspots else None

    stem = os.path.splitext(os.path.basename(ring_pdb_path))[0]
    output_prefix = os.path.join(output_dir, stem)

    return RFdiffusionJob(
        input_pdb=ring_pdb_path, output_prefix=output_prefix, contigs=contigs,
        hotspot_res=hotspot_res, num_designs=num_designs, diffuser_T=diffuser_T,
        extra_overrides=dict(extra_overrides or {}),
    )


# ============================================================================
# Batch ring-dataframe enrichment
# ============================================================================

def process_ring_dataframe(df: pd.DataFrame, pdb_paths: Dict[str, str]) -> pd.DataFrame:
    """
    Enriches a ring-grain dataframe (e.g. orientation_df) with everything
    prepare_fusion_job() needs to build a job straight from a row:

    1. Drops "orientation_junctions" if present — internal bookkeeping
       column not needed past this point.
    2. Derives a recommended (min, max) linker length in RESIDUES from
       "mean_distance" (the N/C-terminal CA-CA gap in Angstroms, from
       termini.py/rings.py), at roughly 3.2 A/residue for a near-extended
       chain plus a couple of residues' slack either side, rounded UP
       (math.ceil) so the recommendation never suggests a linker too
       short to physically close the gap:

           recommended_min = ceil(mean_distance / 3.2 + 2)
           recommended_max = ceil(mean_distance / 3.2 + 7)

       These feed straight into prepare_fusion_job's own
       linker_length=(recommended_min, recommended_max) argument.
    3. Reads each row's already-isolated ring PDB and appends each
       chain's residue count as chain_lengths: {chain_id: length}. Uses
       gemmi (matching every other structure-reading step in this
       project — isolate.py, structure.py, termini.py, rings.py, and
       orientation.py all read structures the same way) rather than a
       second parsing library. chain_lengths is keyed by whatever chain
       ids are actually IN the file — the short A/B/C ids isolate.py's
       file_format="pdb" assigns, not the original long RCSB names — so
       translate through the same rename_map if you need to match it
       back up against chain_groups.

    pdb_paths : {assembly_id: pdb_path} — exactly where each row's
        isolated ring file actually lives. Deliberately explicit rather
        than guessed from assembly_id + a folder: isolate.py's own
        naming (f"{assembly_id}_{symmetry_type}_{chains}.pdb") and any
        custom output_path you pick when calling extract_ring_structure()
        directly rarely line up with a hand-rolled filename guess, and a
        missed guess used to fail silently (chain_lengths just came back
        empty, no error). Build this dict from whatever you already have:

            # from single-row extract_ring_structure() calls:
            pdb_paths = {}
            for _, row in orientation_df.iterrows():
                path, _ = isolate.extract_ring_structure(
                    filepath, row["chain_groups"],
                    f"temporary_subunits/{row['assembly_id']}_ring.pdb",
                    file_format="pdb",
                )
                pdb_paths[row["assembly_id"]] = path

            # or straight from isolate.py's own batch output:
            isolated_df = isolate.from_rings(orientation_df, downloaded_df)
            pdb_paths = dict(zip(isolated_df["assembly_id"], isolated_df["filepath"]))

    A row whose assembly_id isn't in pdb_paths, or whose file no longer
    exists on disk, is reported and given chain_lengths={} rather than
    raising — same fail-soft convention isolate_assembly_rings() already
    uses for an unresolvable chain_group.
    """
    df = df.copy()

    if "orientation_junctions" in df.columns:
        df = df.drop(columns=["orientation_junctions"])

    mean_distance = df["mean_distance"]
    df["recommended_min"] = ((mean_distance / 3.2) + 2).apply(math.ceil)
    df["recommended_max"] = ((mean_distance / 3.2) + 7).apply(math.ceil)

    def fetch_lengths(assembly_id: str) -> Dict[str, int]:
        file_path = pdb_paths.get(assembly_id)
        if not file_path or not os.path.exists(file_path):
            print(f"Warning: no isolated PDB found for {assembly_id!r} (pdb_paths has {file_path!r})")
            return {}

        structure = gemmi.read_structure(file_path)
        model = structure[0]
        return {
            chain.name: len(chain.get_polymer())
            for chain in model
            if len(chain.get_polymer())
        }

    df["chain_lengths"] = df["assembly_id"].apply(fetch_lengths)

    return df


# ============================================================================
# Non-blocking dispatch
# ============================================================================

@dataclass
class RFdiffusionRun:
    """Handle to a dispatched (possibly still-running) RFdiffusion job."""

    job: RFdiffusionJob
    process: subprocess.Popen
    command: List[str]
    log_path: str


def submit(job: RFdiffusionJob, backend: Optional[str] = None, config: Optional[dict] = None, **backend_kwargs) -> RFdiffusionRun:
    """
    Dispatches `job` WITHOUT blocking the calling process or event loop —
    the whole point being that a NiceGUI request handler can call this,
    get an RFdiffusionRun handle back immediately (RFdiffusion itself
    typically takes minutes, not seconds), and poll it from a ui.timer
    callback instead of the UI freezing on a blocking call.

    backend : "local" (bare subprocess.Popen) or "singularity" (runs a
        .sif container image — see _submit_singularity; the recommended
        path for most real users, see module docstring). If not given
        explicitly, resolved from config["rfdiffusion"]["backend"] —
        defaults to "local" if neither is set, so existing calls that
        pass neither backend= nor config= keep working unchanged.
    config  : the dict load_installation_config() returns (or an
        equivalent hand-built dict, e.g. for tests) — supplies
        per-backend settings (singularity_image, python_executable,
        etc.) that aren't given explicitly via backend_kwargs.
        backend_kwargs always win over anything from config, and config
        is entirely optional: everything can still be passed by hand.

    "slurm" is the documented next step (external_tools_architecture.md)
    — it would return the same RFdiffusionRun shape, so
    poll_status()/cancel() keep working unchanged once added.
    """
    tool_config = get_tool_config(config or {}, "rfdiffusion")
    if backend is None:
        backend = tool_config.get("backend", "local")

    if backend == "local":
        defaults = {
            "python_executable": tool_config.get("python_executable", "python"),
            "script_path": tool_config.get("script_path", "scripts/run_inference.py"),
        }
        return _submit_local(job, **{**defaults, **backend_kwargs})

    if backend == "singularity":
        defaults = {
            "image": tool_config.get("singularity_image"),
            "executable": tool_config.get("singularity_executable", "singularity"),
            "model_directory": tool_config.get("model_directory"),
            "bind_paths": tool_config.get("bind_paths", []),
            "use_gpu": tool_config.get("use_gpu", True),
        }
        merged = {**defaults, **backend_kwargs}
        if not merged.get("image"):
            raise ValueError(
                "backend='singularity' needs an 'image' (the .sif path) — pass image=... directly, "
                "or set rfdiffusion.singularity_image in your installation config."
            )
        return _submit_singularity(job, **merged)

    raise NotImplementedError(
        f"backend {backend!r} is not implemented yet — 'local' and 'singularity' are available today. "
        f"See external_tools_architecture.md for the planned slurm backend."
    )


def _submit_local(
    job: RFdiffusionJob, script_path: str = "scripts/run_inference.py",
    python_executable: str = "python", cwd: Optional[str] = None,
) -> RFdiffusionRun:
    argv = build_command(job, script_path=script_path, python_executable=python_executable)

    output_dir = os.path.dirname(job.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # subprocess.Popen (never subprocess.run/check_call, which block
    # until the child exits) so this returns immediately -- the child
    # inherits its own duplicated copy of the log file descriptor, so
    # closing our handle at the end of the `with` block doesn't affect it.
    log_path = f"{job.output_prefix}.log"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, cwd=cwd)

    return RFdiffusionRun(job=job, process=process, command=argv, log_path=log_path)


def _submit_singularity(
    job: RFdiffusionJob, image: str, executable: str = "singularity", run_mode: str = "run",
    model_directory: Optional[str] = None, bind_paths: Sequence[str] = (), use_gpu: bool = True,
    cwd: Optional[str] = None,
) -> RFdiffusionRun:
    """
    Runs RFdiffusion from a Singularity/Apptainer .sif image. Confirmed
    against prosculpt's own production installation.yaml: their RFdiffusion
    image is invoked with `singularity run --nv`, not `exec` — meaning the
    image's own entrypoint already runs `python scripts/run_inference.py`,
    so only the Hydra overrides get appended after the image path, no
    python/script prefix (unlike the local backend's build_command).

    Every host directory the job touches (the input PDB's folder, the
    output folder, and the model weights folder if given) is bind-mounted
    to the SAME absolute path inside the container — so job.input_pdb and
    job.output_prefix never need rewriting between host and container;
    what RFdiffusion sees inside the container is identical to what's on
    disk outside it. This trades one thing prosculpt's own approach has
    (a fixed, predictable in-container path) for one this project cares
    about more: there's no separate host/container path to keep in sync,
    so there's no way for that translation to silently go wrong.

    executable : "singularity" (default) or "apptainer" — some clusters
        install the Apptainer fork under that name instead; both accept
        the same CLI.
    """
    input_pdb_abs = os.path.abspath(job.input_pdb)
    output_prefix_abs = os.path.abspath(job.output_prefix)

    output_dir = os.path.dirname(output_prefix_abs)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    mount_dirs = {os.path.dirname(input_pdb_abs), output_dir}
    model_directory_abs = os.path.abspath(model_directory) if model_directory else None
    if model_directory_abs:
        mount_dirs.add(model_directory_abs)
    mount_dirs = sorted(d for d in mount_dirs if d)

    argv = [executable, run_mode]
    if use_gpu:
        argv.append("--nv")
    for d in mount_dirs:
        argv += ["-B", f"{d}:{d}"]
    for extra in bind_paths:
        argv += ["-B", extra]
    argv.append(image)
    argv += _build_overrides(job, input_pdb=input_pdb_abs, output_prefix=output_prefix_abs)
    if model_directory_abs:
        argv.append(f"inference.model_directory_path={model_directory_abs}")

    log_path = f"{output_prefix_abs}.log"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(argv, stdout=log_file, stderr=subprocess.STDOUT, cwd=cwd)

    # RFdiffusionRun.job must carry the SAME output_prefix actually passed
    # to the container (the absolute path), or poll_status()'s glob would
    # look in the wrong place if job.output_prefix had been relative.
    resolved_job = replace(job, input_pdb=input_pdb_abs, output_prefix=output_prefix_abs)
    return RFdiffusionRun(job=resolved_job, process=process, command=argv, log_path=log_path)


def poll_status(run: RFdiffusionRun) -> dict:
    """
    Cheap, non-blocking status check — safe to call from a NiceGUI
    ui.timer every second or two. RFdiffusion writes no completion/
    summary file (confirmed against scripts/run_inference.py), so
    progress is read the same way the tool itself resumes an interrupted
    run: by counting "{output_prefix}_<N>.pdb" files that have appeared
    on disk.

    Returns a dict:
        state            : "running" | "completed" | "completed_partial" | "failed"
        returncode       : None while still running, else the process's exit code
        designs_written  : how many "{output_prefix}_<N>.pdb" files exist so far
        designs_expected : job.num_designs
        design_paths     : sorted list of the design PDB paths found so far
        log_path         : where RFdiffusion's own stdout/stderr is being captured
    """
    returncode = run.process.poll()

    pattern = re.compile(re.escape(run.job.output_prefix) + r"_\d+\.pdb$")
    design_paths = sorted(p for p in glob.glob(f"{run.job.output_prefix}_*.pdb") if pattern.search(p))

    if returncode is None:
        state = "running"
    elif returncode != 0:
        state = "failed"
    elif len(design_paths) >= run.job.num_designs:
        state = "completed"
    else:
        state = "completed_partial"

    return {
        "state": state,
        "returncode": returncode,
        "designs_written": len(design_paths),
        "designs_expected": run.job.num_designs,
        "design_paths": design_paths,
        "log_path": run.log_path,
    }


def cancel(run: RFdiffusionRun) -> None:
    """Terminates a still-running job (e.g. behind a UI 'cancel' button)
    — a no-op if it already finished."""
    if run.process.poll() is None:
        run.process.terminate()