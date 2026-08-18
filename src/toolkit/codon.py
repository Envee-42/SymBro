"""
codon.py — reverse-translates validated protein sequences into
synthesis-ready DNA, host-codon-optimized via DNAChisel + python_codon_tables.

Needs the "codon" extra (`pip install symbro[codon]`) -- DNAChisel and
python_codon_tables aren't base dependencies, since only the last stage
of the pipeline needs them (same reasoning RFdiffusion/ProteinMPNN/the
structure-prediction backends stay external/optional -- see
pyproject.toml's own comment on why torch, unlike these, IS a base dep).

QUICKSTART
==========

    from toolkit import codon

    result = codon.optimize_sequence("MKTAYIAKQ...", host="e_coli")
    result["dna_sequence"]   # the optimized DNA, ready to order
    result["gc_content"]     # overall GC fraction of that DNA

Or, batched over a DataFrame with assembly_id/component_id/candidate_id/
sequence columns (the shape pipeline.join_predict_with_pmpnn() produces):

    df = codon.optimize_candidates(joined_df, host="e_coli")
    codon.write_fasta(df, "designs.fasta")

Normally reached via `symbro codon` / pipeline.run_codon(), not called
directly -- see cli.py's own `codon` command.

SCOPE, DELIBERATELY
====================

This produces a good STARTING sequence per candidate: codon-optimized for
the chosen host, and free of the handful of things that routinely get a
gene synthesis order flagged or rejected outright by a vendor --

  - GC content out of a safe range, both overall and in local windows
    (a uniform 70%-GC sequence can pass an overall-GC check while still
    having synthesis-breaking local stretches, which is why this checks
    both -- see EnforceGCContent's windowed use below).
  - long homopolymer runs (AAAAAA, ...) -- hard to synthesize accurately.
  - long repeated subsequences / hairpins -- same reason.
  - the two most common Type IIS Golden Gate cloning enzyme sites
    (BsaI, BsmBI) landing inside the coding sequence by accident.

It does NOT attempt a full expression-vector assembly (no Golden Gate/
vector logic, no ordering-format-specific submission) the way
Baker-lab-style Domesticator3 does (github.com/rdkibler/domesticator_3 --
what prosculpt's own `codon_optimization/domesticator_gg_assembly.ipynb`
shells out to; see that project's `.gb` vector file, `--nstruct` flag,
`.params` output). That's a deliberate scope choice, not a missing
feature: vector/cloning strategy is genuinely project-specific (which
enzyme, which vector, which overhangs), and DNAChisel's own building
blocks below can already avoid whatever enzyme you tell them to.

Every sequence this produces is still meant to be reviewed by hand before
ordering, same as output from any codon-optimization tool -- this module
gets you most of the way there and flags anything it couldn't fully
satisfy (see the "warnings" key optimize_sequence() returns), it doesn't
replace judgment about the specific construct you're building.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    import dnachisel as _dc
    _IMPORT_ERROR: Optional[Exception] = None
except ImportError as _exc:  # pragma: no cover -- exercised only without the "codon" extra
    _dc = None
    _IMPORT_ERROR = _exc

try:
    import python_codon_tables as _pct
except ImportError as _exc:  # pragma: no cover -- same "codon" extra as dnachisel
    _pct = None
    if _IMPORT_ERROR is None:
        _IMPORT_ERROR = _exc


# python_codon_tables' own bundled organism names (get_codons_table() takes
# these short forms directly -- confirmed against its actual API, not the
# longer "<name>_<taxid>" strings available_codon_tables_names lists).
SUPPORTED_HOSTS: Tuple[str, ...] = (
    "e_coli", "s_cerevisiae", "h_sapiens", "c_elegans", "b_subtilis",
    "d_melanogaster", "g_gallus", "m_musculus",
)
DEFAULT_HOST = "e_coli"

# The two Type IIS enzymes prosculpt's own Domesticator3-based codon
# optimization step avoids for its Golden Gate assembly (github.com/
# rdkibler/domesticator_3) -- a reasonable default here too, since both
# are common enough in general cloning that avoiding them costs nothing
# for anyone not using them, and helps anyone who is.
DEFAULT_ENZYMES_AVOIDED: Tuple[str, ...] = ("BsaI", "BsmBI")

_STANDARD_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")

_CODON_COLUMNS: Tuple[str, ...] = (
    "assembly_id", "component_id", "candidate_id", "host", "protein_sequence",
    "dna_sequence", "gc_content", "warnings",
)


def _require_dnachisel() -> None:
    if _dc is None or _pct is None:
        raise ImportError(
            "Codon optimization needs the 'codon' extra: pip install symbro[codon] "
            f"(underlying import error: {_IMPORT_ERROR})"
        )


def _validate_host(host: str) -> None:
    if host not in SUPPORTED_HOSTS:
        raise ValueError(
            f"host={host!r} not supported -- must be one of {SUPPORTED_HOSTS} "
            f"(python_codon_tables' own bundled organisms)."
        )


def _best_stop_codon(host: str) -> str:
    """The host's own most-frequently-used stop codon (e.g. TAA for E.
    coli, ~64% usage per Kazusa) -- reverse_translate() itself doesn't
    add a stop codon at all, so callers that want one get the host's own
    preferred choice rather than a hardcoded "TAA" that may not actually
    be the best choice for every host."""
    table = _pct.get_codons_table(host)
    return max(table["*"], key=table["*"].get)


def optimize_sequence(
    protein_sequence: str,
    host: str = DEFAULT_HOST,
    method: str = "use_best_codon",
    gc_min: float = 0.3,
    gc_max: float = 0.65,
    gc_window: int = 100,
    homopolymer_max: int = 5,
    avoid_hairpins: bool = True,
    avoid_repeats_kmer: Optional[int] = 15,
    avoid_enzymes: Sequence[str] = DEFAULT_ENZYMES_AVOIDED,
    add_stop_codon: bool = True,
) -> Dict[str, object]:
    """
    Reverse-translates one protein sequence into host-codon-optimized
    DNA, subject to the synthesis-safety constraints described in this
    module's own docstring.

    method : DNAChisel's own CodonOptimize() methods -- "use_best_codon"
        (default; every codon becomes the single most-frequent synonymous
        codon for the host -- equivalent to CAI optimization), "match_codon_usage"
        (final codon usage matches the host's overall usage profile,
        rather than always picking the single best codon -- the
        literature's more common choice when avoiding an unnaturally
        repetitive/low-diversity sequence matters more than maximizing
        CAI), or "harmonize_rca" (matches each codon's RELATIVE usage
        rank to the original sequence's own host -- rarely applicable
        here, since RFdiffusion/ProteinMPNN-designed sequences have no
        real "original host").
    gc_min/gc_max/gc_window : GC content bounds, checked both over the
        whole sequence and in every gc_window-sized sliding window (the
        windowed check is what actually catches locally-GC-extreme
        stretches a whole-sequence-only check would miss).
    homopolymer_max : longest allowed run of the same nucleotide (e.g.
        the default 5 disallows any run of 6+, matching Domesticator3's
        own AAAAAA/CCCCCC/GGGGGG/TTTTTT avoidance). 0/None disables this
        check.
    avoid_hairpins : DNAChisel's own AvoidHairpins() with its default
        stem_size/hairpin_window -- catches self-complementary regions
        that can misfold during synthesis or PCR.
    avoid_repeats_kmer : disallow any repeated subsequence of this length
        anywhere in the sequence (DNAChisel's UniquifyAllKmers) -- None/0
        disables this check. 15 is DNAChisel's own commonly-used default
        for this kind of check.
    avoid_enzymes : restriction site name(s) (DNAChisel's own
        EnzymeSitePattern names, e.g. "BsaI", "BsmBI", "EcoRI", ...) to
        keep out of the final sequence. Empty sequence/None disables this
        check entirely.
    add_stop_codon : appends the host's own preferred stop codon (see
        _best_stop_codon()) after optimization. Set False if this
        sequence is meant to be fused into a larger ORF (e.g. behind an
        N-terminal tag) rather than expressed on its own.

    Returns {"dna_sequence", "gc_content" (overall, including any
    appended stop codon), "host", "constraints_report" (DNAChisel's own
    human-readable pass/fail summary -- worth reading before ordering),
    "warnings" (list[str], populated only if a constraint couldn't be
    fully satisfied -- an empty list is the common case, not a
    guarantee every input can always satisfy every constraint, e.g. an
    extremely short or low-complexity protein sequence might not have
    enough synonymous-codon freedom to hit a tight GC window)}.

    Raises ValueError for an unsupported host or a protein_sequence
    containing non-standard amino acid characters. Raises RuntimeError
    in the (should-be-impossible, given EnforceTranslation below) case
    that the final DNA doesn't translate back to the input protein --
    this is the one thing that must be exactly right for the result to
    be usable at all, so it's never returned silently wrong.
    """
    _require_dnachisel()
    _validate_host(host)

    cleaned = protein_sequence.strip().upper()
    if not cleaned:
        raise ValueError("protein_sequence is empty -- nothing to optimize.")
    bad_chars = set(cleaned) - _STANDARD_AMINO_ACIDS
    if bad_chars:
        raise ValueError(
            f"protein_sequence contains non-standard amino acid character(s) {sorted(bad_chars)}: "
            f"{protein_sequence!r}"
        )

    dna = _dc.reverse_translate(cleaned)

    constraints: List[object] = [
        _dc.EnforceTranslation(),
        _dc.EnforceGCContent(mini=gc_min, maxi=gc_max, window=gc_window),
    ]
    if homopolymer_max:
        constraints += [
            _dc.AvoidPattern(_dc.HomopolymerPattern(nt, homopolymer_max + 1)) for nt in "ATGC"
        ]
    if avoid_hairpins:
        constraints.append(_dc.AvoidHairpins())
    if avoid_repeats_kmer:
        constraints.append(_dc.UniquifyAllKmers(k=avoid_repeats_kmer))
    for enzyme in avoid_enzymes or ():
        constraints.append(_dc.AvoidPattern(_dc.EnzymeSitePattern(enzyme)))

    problem = _dc.DnaOptimizationProblem(
        sequence=dna, constraints=constraints,
        objectives=[_dc.CodonOptimize(species=host, method=method)],
        logger=None,
    )

    warnings: List[str] = []
    try:
        problem.resolve_constraints()
    except _dc.NoSolutionError as exc:
        warnings.append(f"Could not fully satisfy every constraint: {exc}")
    problem.optimize()

    final_dna = problem.sequence
    if add_stop_codon:
        final_dna += _best_stop_codon(host)

    check_region = final_dna[: len(final_dna) - 3] if add_stop_codon else final_dna
    if _dc.translate(check_region) != cleaned:
        raise RuntimeError(
            "Internal error: the optimized DNA does not translate back to the input protein "
            "sequence despite EnforceTranslation -- please report this as a bug rather than "
            "using this sequence."
        )

    gc_content = (final_dna.count("G") + final_dna.count("C")) / len(final_dna)

    return {
        "dna_sequence": final_dna,
        "gc_content": gc_content,
        "host": host,
        "constraints_report": problem.constraints_text_summary(),
        "warnings": warnings,
    }


def optimize_candidates(df: pd.DataFrame, host: str = DEFAULT_HOST, **kwargs) -> pd.DataFrame:
    """
    Batched optimize_sequence() over every row of df -- df must have
    assembly_id/component_id/candidate_id/sequence columns, the shape
    pipeline.join_predict_with_pmpnn() produces. kwargs are passed
    straight through to optimize_sequence() (host/method/gc_min/gc_max/
    gc_window/homopolymer_max/avoid_hairpins/avoid_repeats_kmer/
    avoid_enzymes/add_stop_codon).

    A row is skipped (reported, not fatal to the rest of the batch --
    same fail-soft convention as pipeline.run_pmpnn()/run_predict()) if
    its sequence is missing (a join_predict_with_pmpnn() mismatch) or if
    optimize_sequence() itself raises for that one row (e.g. an
    unsupported host, or a sequence DNAChisel couldn't satisfy).

    Checks the "codon" extra is actually importable, and that host= is
    one of SUPPORTED_HOSTS, ONCE up front -- rather than letting either
    problem surface as the SAME error repeated as a "Skipped" line per
    row (both would otherwise only be caught inside optimize_sequence(),
    called fresh per row). That would bury a clear, batch-wide, fixable
    problem (missing dependency, or a typo'd --host) inside what looks
    like N separate per-candidate failures -- and for the ImportError
    case specifically, would never even reach cli.py's own dedicated
    `except ImportError` handler, since the per-row try/except below
    would swallow it first.

    Returns a DataFrame in _CODON_COLUMNS' shape -- empty (but
    correctly-columned) if every row was skipped, same convention as
    _PMPNN_SEQUENCE_COLUMNS/_PREDICT_COLUMNS.
    """
    _require_dnachisel()
    _validate_host(host)
    rows = []
    for _, row in df.iterrows():
        label = f"{row['assembly_id']}/{row.get('candidate_id')}"
        sequence = row.get("sequence")
        if not isinstance(sequence, str) or not sequence:
            print(f"Skipped {label}: no sequence available (predict.pkl/pmpnn.pkl checkpoint mismatch?).")
            continue
        try:
            result = optimize_sequence(sequence, host=host, **kwargs)
        except Exception as exc:
            print(f"Skipped {label}: {exc}")
            continue
        rows.append({
            "assembly_id": row["assembly_id"],
            "component_id": row.get("component_id"),
            "candidate_id": row.get("candidate_id"),
            "host": result["host"],
            "protein_sequence": sequence,
            "dna_sequence": result["dna_sequence"],
            "gc_content": result["gc_content"],
            "warnings": "; ".join(result["warnings"]) if result["warnings"] else None,
        })

    return pd.DataFrame(rows, columns=list(_CODON_COLUMNS)) if rows else pd.DataFrame(columns=list(_CODON_COLUMNS))


def write_fasta(codon_df: pd.DataFrame, path: str) -> str:
    """Writes one FASTA record per row (header: assembly_id_candidate_id,
    sequence: dna_sequence) -- ready to hand to a gene synthesis vendor,
    after your own manual review (see this module's own docstring).
    Returns path."""
    with open(path, "w") as f:
        for _, row in codon_df.iterrows():
            header = f"{row['assembly_id']}_{row['candidate_id']}"
            f.write(f">{header}\n{row['dna_sequence']}\n")
    return path
