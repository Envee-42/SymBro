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

LICENSING — confirmed directly against each project's own PyPI/GitHub
license metadata, not recalled from memory: DNAChisel
(Edinburgh-Genome-Foundry/DnaChisel) is MIT (Copyright 2017 Edinburgh
Genome Foundry, University of Edinburgh). python_codon_tables
(Edinburgh-Genome-Foundry/python_codon_tables, same group) is CC0-1.0 --
a public-domain dedication, even more permissive than MIT. Neither
imposes any copyleft or redistribution restriction, so there's no
conflict with this project's own Apache-2.0 license -- the cleanest
case of any optional backend (compare af3.py's module docstring, whose
weights carry real restrictions these don't). Both are pulled in only
via the "codon" extra (`pip install symbro[codon]`), never vendored.

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
    max_attempts: int = 1,
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
    max_attempts : DNAChisel's constraint solver has genuine call-to-call
        randomness (confirmed empirically -- no seed is exposed anywhere
        in its API, and re-running the exact same input can succeed where
        a prior attempt didn't). On a hard case -- most notably a protein
        with near-identical repeated domains, this project's own
        fused-ring designs' own shape, which fights UniquifyAllKmers --
        a single attempt can fail a meaningful fraction of the time
        (confirmed: roughly 1-in-3 on a real repeated-domain design)
        even though the constraints ARE jointly satisfiable most tries.
        Default 1 keeps a single attempt, exactly today's behavior --
        this is deliberately NOT retried automatically, so one call's
        cost stays predictable unless you ask for more. Pass a higher
        value (e.g. 3) to retry within this one call instead of having
        to notice a warning and re-invoke `symbro codon`/this function by
        hand: keeps the first attempt whose `warnings` comes back empty,
        or -- only if every attempt still has one -- returns the LAST
        attempt's best-effort result, with `warnings` noting how many
        attempts were tried. See `symbro codon --max-attempts` for the
        CLI equivalent.

    Returns {"dna_sequence", "gc_content" (overall, including any
    appended stop codon), "host", "constraints_report" (DNAChisel's own
    human-readable pass/fail summary -- worth reading before ordering),
    "warnings" (list[str], populated only if a constraint couldn't be
    fully satisfied -- an empty list is the common case, not a
    guarantee every input can always satisfy every constraint, e.g. an
    extremely short or low-complexity protein sequence might not have
    enough synonymous-codon freedom to hit a tight GC window)}.

    Raises ValueError for an unsupported host, an out-of-range
    max_attempts, or a protein_sequence containing non-standard amino
    acid characters. Raises RuntimeError in the (should-be-impossible,
    given EnforceTranslation below) case that some attempt's DNA doesn't
    translate back to the input protein -- this is the one thing that
    must be exactly right for a result to be usable at all, so it's
    never returned silently wrong, and never retried past either: a
    translation mismatch is a real bug to report, not a solver hiccup
    worth re-rolling.
    """
    _require_dnachisel()
    _validate_host(host)
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts!r}.")

    cleaned = protein_sequence.strip().upper()
    if not cleaned:
        raise ValueError("protein_sequence is empty -- nothing to optimize.")
    bad_chars = set(cleaned) - _STANDARD_AMINO_ACIDS
    if bad_chars:
        raise ValueError(
            f"protein_sequence contains non-standard amino acid character(s) {sorted(bad_chars)}: "
            f"{protein_sequence!r}"
        )

    result: Dict[str, object] = {}
    for attempt in range(1, max_attempts + 1):
        result = _optimize_once(
            cleaned, host=host, method=method, gc_min=gc_min, gc_max=gc_max, gc_window=gc_window,
            homopolymer_max=homopolymer_max, avoid_hairpins=avoid_hairpins,
            avoid_repeats_kmer=avoid_repeats_kmer, avoid_enzymes=avoid_enzymes,
            add_stop_codon=add_stop_codon,
        )
        if not result["warnings"]:
            return result
        if attempt < max_attempts:
            print(f"    codon-optimization attempt {attempt}/{max_attempts} still had an unresolved "
                  f"constraint, retrying (DNAChisel's solver has genuine call-to-call randomness)...")

    if max_attempts > 1:
        result["warnings"] = [
            f"After {max_attempts} attempts, none fully satisfied every constraint -- "
            f"returning the last attempt's best-effort result."
        ] + list(result["warnings"])
    return result


def _optimize_once(
    cleaned: str,
    host: str,
    method: str,
    gc_min: float,
    gc_max: float,
    gc_window: int,
    homopolymer_max: int,
    avoid_hairpins: bool,
    avoid_repeats_kmer: Optional[int],
    avoid_enzymes: Sequence[str],
    add_stop_codon: bool,
) -> Dict[str, object]:
    """One single reverse-translate-and-optimize attempt against an
    already-validated, already-uppercased protein sequence -- the body
    optimize_sequence() used to be, factored out so its max_attempts loop
    can call it fresh (a genuinely new DNAChisel solver run, not a retry
    of the same one) more than once. Not meant to be called directly --
    see optimize_sequence()'s own docstring for the full parameter
    reference and return shape, which this mirrors exactly."""
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
    else:
        # DNAChisel's own precondition, confirmed directly against its
        # source (ObjectivesMaximizerMixin.optimize_by_exhaustive_search):
        # "Optimization can only be done when all constraints are
        # verified" -- calling optimize() after resolve_constraints()
        # already failed doesn't just skip improving the objective, it
        # raises ITS OWN NoSolutionError (a different one than the try/
        # except above catches), previously uncaught here. That silently
        # turned "one constraint couldn't be fully satisfied" (meant to
        # surface as a warning on an otherwise-still-returned, best-effort
        # sequence -- see this function's own docstring) into an
        # unhandled exception -- which optimize_candidates()'s per-row
        # try/except then swallowed as a plain "Skipped", silently
        # dropping the candidate from the batch instead of returning it
        # flagged. Confirmed reproducible: a protein with near-identical
        # repeated domains (this project's own fused-ring designs are
        # exactly this shape) hitting UniquifyAllKmers often fails
        # resolve_constraints() and previously crashed here every time,
        # not just warned.
        constraint_satisfying_sequence = problem.sequence
        problem.optimize()
        if not problem.all_constraints_pass():
            # Confirmed empirically (not assumed from docs, which read as
            # though optimize() should only ever accept objective-improving
            # moves that keep every constraint satisfied): on a protein
            # with near-identical repeated domains, CodonOptimize's
            # objective-improvement step can push codon choices back
            # toward a k-mer-repeating pattern that resolve_constraints()
            # had *just* resolved -- with NO exception raised, and
            # `warnings` would otherwise stay empty. Trusting an empty
            # `warnings` list here would silently ship a sequence that
            # violates its own EnforceGCContent/UniquifyAllKmers/etc, which
            # is worse than just not codon-optimizing it -- so revert to
            # the last confirmed constraint-satisfying sequence (from
            # immediately after resolve_constraints(), before optimize()
            # touched it) instead of keeping optimize()'s output.
            # problem.sequence is a plain instance attribute here (not a
            # property with side effects), and every DNAChisel method used
            # below/by callers (all_constraints_pass(),
            # constraints_text_summary(), .sequence itself) re-evaluates
            # against whatever it currently holds -- confirmed directly by
            # reproducing this exact failure and checking all three read
            # correctly post-revert, not assumed.
            problem.sequence = constraint_satisfying_sequence
            warnings.append(
                "CodonOptimize's objective-improvement step produced a sequence that broke a "
                "constraint DNAChisel had already resolved (caught via all_constraints_pass(), "
                "since DNAChisel itself raises no exception for this) -- reverted to the last "
                "constraint-satisfying sequence, which is less codon-optimized as a result."
            )

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
    avoid_enzymes/add_stop_codon/max_attempts -- see that function's own
    docstring for max_attempts specifically: pass e.g. max_attempts=3 to
    retry a candidate that hits DNAChisel's genuine solver randomness
    within this one call, instead of re-invoking `symbro codon` by hand).

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
