"""
cli.py — the `symbro` command. A thin layer over pipeline.py: parses
arguments, calls the matching pipeline stage function, and prints a
clear, human-readable summary or error. No pipeline logic lives here --
see pipeline.py's module docstring for why, and for the ".symbro/" state
folder convention every command below relies on.

Typical workflow, run in order:

    symbro query         search RCSB PDB for candidate assemblies
    symbro download      download the matching structure files
    symbro local         (ALTERNATIVE to query+download) register your own
                         local PDB/CIF file(s) instead of searching RCSB
    symbro geometry      detect symmetry rings (and orientation/termini for one type)
    symbro isolate       extract each ring's PDB, ready for RFdiffusion
    symbro view          render a structure as an interactive 3D HTML file
                         (a direct path, or --stage downloaded/rings + --assembly-id)
    symbro rfdiffusion   submit RFdiffusion, one job per assembly (blocks until done
                         unless --detach, which needs backend='slurm')
    symbro status        check on --detach'd RFdiffusion jobs
    symbro pmpnn         run ProteinMPNN against each assembly's best design(s)
    symbro predict       fold candidates back (boltz/af2/af3) and screen by self-consistency
    symbro codon         reverse-translate validated designs into host-codon-optimized DNA
                         (needs the "codon" extra: pip install symbro\[codon])
    symbro clean         clear scratch files + checkpoints between runs

Each command picks up automatically where the previous one left off (via
a small ".symbro/" folder in your current directory) -- run `symbro
--help` or `symbro <command> --help` any time for details.
"""

from __future__ import annotations

import os
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from toolkit import pipeline

app = typer.Typer(
    name="symbro",
    help=__doc__,
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True, style="red")


def _version_callback(value: bool):
    if value:
        try:
            from importlib.metadata import version
            console.print(f"symbro {version('symbro')}")
        except Exception:
            console.print("symbro (version unknown -- not installed via pip)")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    state_dir: str = typer.Option(
        pipeline.DEFAULT_STATE_DIR, "--state-dir",
        help="Where stage checkpoints are read from and written to.",
    ),
    version: Optional[bool] = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show the installed symbro version and exit.",
    ),
):
    ctx.obj = {"state_dir": state_dir}
    # Every default path in this pipeline (state_dir, and download's/
    # isolate's own data_dir/output_dir defaults) resolves relative to
    # os.getcwd() -- i.e. the whole CLI already assumes you run `symbro`
    # from your project root. Echoing the resolved absolute paths here
    # makes that assumption visible and checkable (e.g. in a cluster batch
    # job's log) instead of silent -- see paths.py for why checkpoints
    # rely on it holding at command-invocation time only, not beyond.
    if ctx.invoked_subcommand is not None:
        console.print(
            f"[dim]Working from: {os.getcwd()}  (state dir: {os.path.abspath(state_dir)})[/dim]"
        )


def _fail(message: str) -> "typer.NoReturn":
    # rich_markup_mode="rich" (see the Typer() constructor above) means
    # console.print() interprets "[...]" in ITS OWN format string as
    # markup tags -- an exception message that happens to contain a
    # literal "[...]" (e.g. codon.py's own "pip install symbro[codon]")
    # would otherwise get silently swallowed as an unrecognized tag
    # rather than printed. rich.markup.escape() keeps `message` itself
    # untouched (str(exc) elsewhere, e.g. in tests, still sees the real
    # text) and only escapes it at this print boundary.
    err_console.print(f"[bold red]✗[/bold red] {escape(message)}")
    raise typer.Exit(code=1)


def _format_cell(value) -> str:
    """Compact, readable rendering for a table cell -- including structured
    values (chain_groups, orientation_junctions, termini_ss, ...) that a
    plain str() would print as an ugly Python repr. Never drops data: this
    is only used for the terminal preview, the .pkl/.csv checkpoints keep
    the real objects regardless of how this renders them.
    """
    if value is None:
        return "—"  # em dash
    if isinstance(value, float):
        return f"{value:.2f}"
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "—"
        if all(isinstance(v, tuple) for v in value):
            # a list of tuples, e.g. termini_ss's (chain, n_ss, c_ss) or
            # orientation_junctions' (from_chain, to_chain, angle)
            return ", ".join("/".join(_format_cell(x) for x in item) for item in value)
        return ", ".join(_format_cell(v) for v in value)  # a flat tuple, e.g. chain_groups
    if isinstance(value, dict):
        return ", ".join(f"{k}={_format_cell(v)}" for k, v in value.items())
    return str(value)


def _summarize(df, empty_hint: str, max_cols: int = 12) -> None:
    if df is None or df.empty:
        console.print(f"[yellow]No rows.[/yellow] {empty_hint}")
        return
    table = Table(show_header=True, header_style="bold cyan")
    preview_cols = list(df.columns)[:max_cols]
    for col in preview_cols:
        table.add_column(col)
    for _, row in df.head(10).iterrows():
        table.add_row(*(_format_cell(row[c]) for c in preview_cols))
    console.print(table)
    if len(df) > 10:
        console.print(f"  ... and {len(df) - 10} more row(s).")
    if len(df.columns) > max_cols:
        console.print(f"  ({len(df.columns) - max_cols} more column(s) not shown here -- see the .csv)")


def _parse_criterion(raw: str, flag_name: str = "--criterion") -> dict:
    """Parses "attribute=value:operator" (operator optional, defaults to exact_match).

    A comma-separated value becomes a list (e.g. "in" matching); with
    operator "range", exactly two comma-separated numbers are required.

    flag_name : which CLI flag to name in error messages -- shared by both
        --criterion (search-time) and --filter (post-fetch), which use this
        identical raw-string syntax; passing the actual flag the user typed
        keeps a bad-input error pointing at the right option instead of
        always blaming --criterion.
    """
    if "=" not in raw:
        raise ValueError(f'{flag_name} "{raw}" is missing "=" -- expected format: attribute=value[:operator]')
    attribute, rest = raw.split("=", 1)
    if ":" in rest:
        value_part, operator = rest.rsplit(":", 1)
    else:
        value_part, operator = rest, "exact_match"

    if operator == "range":
        parts = [p.strip() for p in value_part.split(",")]
        if len(parts) != 2:
            raise ValueError(f'{flag_name} "{raw}": operator "range" needs exactly two comma-separated numbers, e.g. resolution=0,3:range')
        try:
            value = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise ValueError(f'{flag_name} "{raw}": range values must be numbers') from None
    elif "," in value_part:
        value = [p.strip() for p in value_part.split(",")]
    else:
        value = value_part.strip()

    return {"attribute": attribute.strip(), "value": value, "operator": operator}


@app.command()
def query(
    ctx: typer.Context,
    symmetry: Optional[List[str]] = typer.Option(
        None, "--symmetry", help="Symmetry type to match, e.g. C3 (repeatable -- OR'd together)."
    ),
    entry_id: Optional[List[str]] = typer.Option(
        None, "--entry-id", help="Specific PDB entry ID, e.g. 4V6B (repeatable)."
    ),
    resolution_min: Optional[float] = typer.Option(None, help="Minimum resolution (Angstrom)."),
    resolution_max: Optional[float] = typer.Option(None, help="Maximum resolution (Angstrom)."),
    description: Optional[str] = typer.Option(
        None, help='Words the structure title/description must contain, e.g. "ferritin".'
    ),
    criterion: Optional[List[str]] = typer.Option(
        None, "--criterion",
        help='Advanced: raw "attribute=value[:operator]" search criterion (repeatable). '
             'e.g. --criterion "experimental_method=X-RAY DIFFRACTION"',
    ),
    match: str = typer.Option(
        "and", help="How to combine all the above: 'and' or 'or'. Applied separately to "
                    "--filter (below), not jointly across the two."
    ),
    fetch_field: Optional[List[str]] = typer.Option(
        None, "--fetch-field", help="Extra metadata field to include in the result (repeatable)."
    ),
    metadata_filter: Optional[List[str]] = typer.Option(
        None, "--filter",
        help='Advanced: raw "attribute=value[:operator]" filter, same syntax as --criterion, '
             'applied AFTER results are fetched rather than sent to RCSB\'s search (repeatable). '
             'Use this for fetch-only attributes -- e.g. model_quality -- that have no Search '
             'API equivalent and can never appear in --criterion. '
             'e.g. --filter "model_quality=70:greater_or_equal"',
    ),
    return_type: str = typer.Option("assembly", help="'assembly' or 'entry'."),
):
    """Search RCSB PDB for candidate structures matching your criteria."""
    if resolution_min is not None and resolution_max is None:
        resolution_max = 100.0
    if resolution_max is not None and resolution_min is None:
        resolution_min = 0.0
    resolution_range = (resolution_min, resolution_max) if resolution_min is not None else None

    try:
        extra_criteria = [_parse_criterion(c) for c in (criterion or [])]
        filter_criteria = [_parse_criterion(c, flag_name="--filter") for c in (metadata_filter or [])]
        df = pipeline.run_query(
            symmetry=symmetry, entry_id=entry_id, resolution_range=resolution_range,
            description=description, extra_criteria=extra_criteria, match=match,
            fetch_fields=fetch_field, filter_criteria=filter_criteria, return_type=return_type,
            state_dir=ctx.obj["state_dir"],
        )
    except ValueError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Search failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Found {len(df)} candidate(s).")
    _summarize(df, empty_hint="Try loosening your search criteria or --filter.")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/candidates.pkl[/cyan] (preview: candidates.csv)")
        console.print("  Next: [bold]symbro download[/bold]")


@app.command()
def download(
    ctx: typer.Context,
    data_dir: Optional[str] = typer.Option(
        None, help="Where to save downloaded structures (default: temporary_files/)."
    ),
    overwrite: bool = typer.Option(False, help="Re-download files that already exist locally."),
):
    """Download structure files for every candidate found by `symbro query`."""
    try:
        df = pipeline.run_download(data_dir=data_dir, overwrite=overwrite, state_dir=ctx.obj["state_dir"])
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Download failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Downloaded {len(df)} structure(s).")
    _summarize(df, empty_hint="")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/downloaded.pkl[/cyan] (preview: downloaded.csv)")
        console.print("  Next: [bold]symbro geometry --symmetry-type <e.g. C3>[/bold]")


@app.command()
def local(
    ctx: typer.Context,
    paths: List[str] = typer.Argument(
        ..., help="Local PDB/CIF file path(s) to register as candidates (repeatable)."
    ),
    assembly_id: Optional[List[str]] = typer.Option(
        None, "--assembly-id",
        help="Label for each path, same order and count as PATHS (repeatable). "
             "Default: derived from each file's own name.",
    ),
    data_dir: Optional[str] = typer.Option(
        None, help="Where to copy registered files (default: temporary_files/local/)."
    ),
    overwrite: bool = typer.Option(False, help="Re-copy files that already exist at the destination."),
):
    """Register your own local PDB/CIF file(s) as candidates -- an alternative to
    `symbro query` + `symbro download` for a structure you already have."""
    try:
        df = pipeline.run_local(
            paths, assembly_ids=assembly_id or None, data_dir=data_dir, overwrite=overwrite,
            state_dir=ctx.obj["state_dir"],
        )
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Registering local structure(s) failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Registered {len(df)} local structure(s).")
    _summarize(df, empty_hint="")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/downloaded.pkl[/cyan] (preview: downloaded.csv)")
        console.print("  Next: [bold]symbro geometry --symmetry-type <e.g. C3>[/bold]")


@app.command()
def geometry(
    ctx: typer.Context,
    symmetry_type: Optional[str] = typer.Option(
        None,
        help="Narrow to one symmetry order (e.g. C3) and also compute orientation + "
             "termini secondary structure. Omit on a first pass to just see what's "
             "present in your candidates before choosing one.",
    ),
    validate_symmetry: bool = typer.Option(
        True, "--validate-symmetry/--no-validate-symmetry",
        help="Drop assemblies whose detected rings don't include any of RCSB's own "
             "annotated symmetry (a real, if uncommon, sign of a PDB annotation issue "
             "-- wrong assembly marked biological, etc.) rather than passing them "
             "through as candidates. You'll be warned either way. Pass "
             "--no-validate-symmetry to keep every detected ring regardless of what "
             "RCSB annotated -- e.g. if you're deliberately investigating a mismatch.",
    ),
    warn_incomplete_axes: bool = typer.Option(
        True, "--warn-incomplete-axes/--no-warn-incomplete-axes",
        help="Warn (never drop) about a surviving row whose axis_count falls short of "
             "the most disjoint rings its own chain count could support -- e.g. only 6 "
             "of 8 possible C3 triplets found. Independent of --validate-symmetry: the "
             "axis type itself may be entirely correctly detected, this just flags that "
             "not every eligible chain formed a ring, often real per-structure disorder "
             "rather than an annotation problem. Pass --no-warn-incomplete-axes to skip.",
    ),
):
    """Detect symmetry rings in every downloaded structure."""
    try:
        df = pipeline.run_geometry(
            symmetry_type=symmetry_type, state_dir=ctx.obj["state_dir"],
            validate_annotated_symmetry=validate_symmetry,
            warn_incomplete_axes=warn_incomplete_axes,
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Geometry analysis failed: {exc}")

    if symmetry_type is None:
        summary = pipeline.detected_symmetry_types(df)
        if summary.empty:
            console.print("[yellow]No symmetry rings detected in any downloaded structure.[/yellow]")
        else:
            console.print("[bold green]✓[/bold green] Symmetry types detected:")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("symmetry_type")
            table.add_column("assemblies", justify="right")
            table.add_column("components", justify="right")
            table.add_column("total_axis_count", justify="right")
            for _, row in summary.iterrows():
                table.add_row(
                    row["symmetry_type"], str(row["assemblies"]), str(row["components"]), str(row["total_axis_count"]),
                )
            console.print(table)
            console.print("  [dim]\"components\" > \"assemblies\" for a symmetry_type means at least one "
                           "assembly has more than one structurally distinct component at that order -- "
                           "e.g. a two-protein cage -- each of which will get its own row/file downstream.[/dim]")
            console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/geometry.pkl[/cyan] (preview: geometry.csv)")
            console.print("  Next: re-run with [bold]--symmetry-type[/bold] on one of the above, e.g. "
                           f"[bold]symbro geometry --symmetry-type {summary.iloc[0]['symmetry_type']}[/bold]")
    else:
        console.print(f"[bold green]✓[/bold green] {len(df)} assembly/assemblies with {symmetry_type} symmetry.")
        _summarize(df, empty_hint=f"No assemblies had {symmetry_type} symmetry -- try a different --symmetry-type.")
        if not df.empty:
            console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/geometry.pkl[/cyan] (preview: geometry.csv)")
            console.print("  Next: [bold]symbro isolate[/bold]")


@app.command()
def isolate(
    ctx: typer.Context,
    symmetry_type: Optional[str] = typer.Option(
        None, help="Narrow to one symmetry order first. Only needed if `symbro geometry` "
                    "was run without --symmetry-type."
    ),
    component_id: Optional[int] = typer.Option(
        None, help="Only isolate this one structurally distinct component (see `symbro geometry`'s "
                    "'components' count) instead of every component found -- e.g. to extract just one "
                    "protein of a multi-component cage. Default: isolate every component."
    ),
    file_format: str = typer.Option("pdb", help="'pdb' or 'cif'. RFdiffusion expects 'pdb'."),
    output_dir: Optional[str] = typer.Option(
        None, help="Where to write extracted ring structures (default: temporary_subunits/)."
    ),
):
    """Extract each candidate's ring structure(s) -- the files RFdiffusion needs next.

    A multi-component assembly (e.g. a two-protein cage) extracts one file per component by
    default -- narrow to a single one with --component-id.
    """
    try:
        df = pipeline.run_isolate(
            symmetry_type=symmetry_type, component_id=component_id, file_format=file_format,
            output_dir=output_dir, state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except (ValueError,) as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Isolating rings failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Extracted {len(df)} ring structure(s).")
    _summarize(df, empty_hint="Try a different --symmetry-type/--component-id, or check `symbro geometry`'s output.")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/rings.pkl[/cyan] (preview: rings.csv)")
        console.print("  Next: [bold]symbro rfdiffusion[/bold]")


@app.command()
def view(
    ctx: typer.Context,
    path: Optional[str] = typer.Argument(
        None, help="A .pdb/.ent or .cif/.mmcif file to render directly. Omit to look one up "
                    "from a checkpoint instead, via --stage/--assembly-id.",
    ),
    stage: Optional[str] = typer.Option(
        None, help="Look up the structure from a checkpoint instead of a direct PATH: "
                    "'downloaded' (the full assembly) or 'rings' (one isolated ring/component, "
                    "from `symbro isolate`). Requires --assembly-id.",
    ),
    assembly_id: Optional[str] = typer.Option(None, help="Required together with --stage."),
    component_id: Optional[int] = typer.Option(
        None, help="For --stage rings on a multi-component assembly (e.g. a two-protein cage) "
                    "-- narrows to one component's row. Omit for a single-component assembly.",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Where to write the HTML view. Default: next to the "
                                      "structure file, same basename with a .html extension.",
    ),
):
    """
    Render a structure as a self-contained, interactive 3D HTML view.

    Open the result in any browser -- no server, no network, no external
    tool or GPU needed, either to generate it or to view it (the 3Dmol.js
    viewer is bundled inside symbro itself and embedded directly in the
    output file). Colors by chain; rotate/zoom/pan with the mouse.
    """
    from toolkit import viz

    if (path is None) == (stage is None):
        _fail("Pass exactly one of PATH or --stage.")
    if stage is not None and assembly_id is None:
        _fail("--stage requires --assembly-id.")

    if stage is not None:
        try:
            path = pipeline.resolve_structure_path(
                stage, assembly_id, component_id=component_id, state_dir=ctx.obj["state_dir"],
            )
        except pipeline.StageNotFoundError as exc:
            _fail(str(exc))
        except ValueError as exc:
            _fail(str(exc))

    if not os.path.exists(path):
        _fail(f"No such file: {path!r}")

    resolved_output = output or os.path.splitext(path)[0] + ".html"
    try:
        written = viz.render_to_file(path, resolved_output, title=assembly_id or os.path.basename(path))
    except ValueError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Rendering failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Wrote [cyan]{written}[/cyan]")
    console.print("  Open it in any browser to view -- no server needed.")


@app.command()
def rfdiffusion(
    ctx: typer.Context,
    assembly_id: Optional[str] = typer.Option(
        None, help="Only submit this assembly's ring (default: every row in rings.pkl)."
    ),
    linker_min: Optional[int] = typer.Option(
        None, help="Diffused linker min length (residues). Default: each ring's own "
                    "geometry-informed recommendation from `symbro geometry` (see its "
                    "recommended_linker_length column) -- pass both --linker-min and "
                    "--linker-max together to override with a fixed range for every job instead."
    ),
    linker_max: Optional[int] = typer.Option(
        None, help="Diffused linker max length (residues). See --linker-min."
    ),
    num_designs: int = typer.Option(10, help="How many designs each assembly's job should produce."),
    diffuser_t: int = typer.Option(50, "--diffuser-t", help="RFdiffusion's diffuser.T (noise steps)."),
    backend: Optional[str] = typer.Option(
        None, help="'local', 'singularity', or 'slurm' -- default: installation.yaml's "
                    "rfdiffusion.backend, or 'local' if that isn't set either."
    ),
    detach: bool = typer.Option(
        False, "--detach", help="Submit and return immediately instead of blocking until done -- "
                                 "only works with backend='slurm'. Check progress later with "
                                 "`symbro status`."
    ),
    poll_interval: int = typer.Option(20, help="Seconds between status checks while blocking."),
    timeout: Optional[int] = typer.Option(
        None, help="Give up waiting after this many seconds (default: no timeout, wait until done). "
                    "The job(s) keep running regardless; SLURM jobs stay checkable via `symbro "
                    "status` afterward, local/singularity jobs may not (see the warning if it happens)."
    ),
):
    """Submit RFdiffusion -- one job per (assembly, component) row in `symbro isolate`'s output.

    Diffused linker length defaults to each ring's own geometry-informed recommendation
    unless both --linker-min and --linker-max are given, which then apply uniformly to
    every job instead.
    """
    if (linker_min is None) != (linker_max is None):
        _fail("--linker-min and --linker-max must be given together (or neither, to use each "
              "ring's own geometry-informed recommendation).")
    linker_length = (linker_min, linker_max) if linker_min is not None else None

    try:
        df = pipeline.run_rfdiffusion(
            assembly_id=assembly_id, linker_length=linker_length,
            num_designs=num_designs, diffuser_T=diffuser_t, backend=backend,
            detach=detach, poll_interval=poll_interval, timeout=timeout,
            state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except (ValueError, RuntimeError) as exc:
        _fail(f"RFdiffusion submission failed: {exc}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"RFdiffusion failed: {exc}")

    console.print(f"[bold green]✓[/bold green] {len(df)} job(s) submitted.")
    if not df.empty:
        # Skip the raw "run" column in the preview -- it's an
        # RFdiffusionRun dataclass, and _format_cell()'s generic str()
        # fallback would dump its full, noisy repr into every cell.
        preview = df.copy()
        preview["slurm_job_id"] = preview["run"].apply(lambda r: r.slurm_job_id or "—")
        preview["designs_written"] = preview["design_paths"].apply(len)
        preview = preview[["assembly_id", "symmetry_type", "component_id", "state", "slurm_job_id", "designs_written"]]
        _summarize(preview, empty_hint="")
    else:
        _summarize(df, empty_hint="")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/rfdiffusion.pkl[/cyan] (preview: rfdiffusion.csv)")
        if detach:
            console.print("  Next: [bold]symbro status[/bold] to check progress, then "
                           "[bold]symbro pmpnn[/bold] once done.")
        else:
            console.print("  Next: [bold]symbro pmpnn[/bold]")


@app.command()
def status(ctx: typer.Context):
    """Check on --detach'd RFdiffusion jobs and refresh their status."""
    try:
        df = pipeline.run_status(state_dir=ctx.obj["state_dir"])
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Status check failed: {exc}")

    still_running = df[~df["state"].isin(["completed", "completed_partial", "failed"])]
    console.print(
        f"[bold green]✓[/bold green] {len(df)} RFdiffusion job(s) tracked, "
        f"{len(still_running)} still running."
    )
    preview = df[["assembly_id", "symmetry_type", "component_id", "state", "design_paths"]].copy()
    preview["design_paths"] = preview["design_paths"].apply(len)
    preview = preview.rename(columns={"design_paths": "designs_written"})
    _summarize(preview, empty_hint="")
    if still_running.empty and not df.empty:
        console.print("  All jobs done. Next: [bold]symbro pmpnn[/bold]")


@app.command()
def pmpnn(
    ctx: typer.Context,
    assembly_id: Optional[str] = typer.Option(
        None, help="Only run this assembly (default: every completed row in rfdiffusion.pkl). A "
                    "multi-component assembly may still match more than one row -- narrow further "
                    "with --component-id."
    ),
    component_id: Optional[int] = typer.Option(
        None, help="Only run this one structurally distinct component of --assembly-id (see "
                    "`symbro geometry`'s 'components' count)."
    ),
    top_n: int = typer.Option(1, help="Submit only the top N RFdiffusion designs per row, by pLDDT."),
    min_plddt: Optional[float] = typer.Option(None, help="Also require at least this mean pLDDT."),
    select: Optional[List[str]] = typer.Option(
        None, "--select", help="Explicit design PDB path (repeatable) -- bypasses --top-n/"
                                "--min-plddt entirely. Requires --assembly-id (and --component-id "
                                "for a multi-component assembly)."
    ),
    num_seq_per_target: int = typer.Option(8, help="ProteinMPNN sequences per selected design."),
    sampling_temp: float = typer.Option(0.1, help="ProteinMPNN sampling temperature."),
    batch_size: int = typer.Option(8, help="Must evenly divide --num-seq-per-target."),
    poll_interval: float = typer.Option(5.0, help="Seconds between status checks."),
    timeout: int = typer.Option(
        900, help="Give up waiting after this many seconds, per assembly. ProteinMPNN is always a "
                   "local process -- unlike RFdiffusion's SLURM jobs, it does not survive this "
                   "command exiting."
    ),
):
    """Run ProteinMPNN against each assembly's best RFdiffusion design(s)."""
    try:
        df = pipeline.run_pmpnn(
            assembly_id=assembly_id, component_id=component_id, top_n=top_n, min_plddt=min_plddt, select=select,
            num_seq_per_target=num_seq_per_target, sampling_temp=sampling_temp,
            batch_size=batch_size, poll_interval=poll_interval, timeout=timeout,
            state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except (ValueError, RuntimeError) as exc:
        _fail(f"ProteinMPNN run failed: {exc}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"ProteinMPNN failed: {exc}")

    console.print(f"[bold green]✓[/bold green] {len(df)} sequence row(s) written.")
    _summarize(df, empty_hint="No assembly had a completed RFdiffusion job ready -- check `symbro status`.")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/pmpnn.pkl[/cyan] (preview: pmpnn.csv)")
        console.print("  Next: [bold]symbro predict[/bold]")


@app.command()
def predict(
    ctx: typer.Context,
    predictor: Optional[str] = typer.Option(
        None, help="'boltz', 'af2'/'alphafold2', or 'af3'/'alphafold3' -- default: installation.yaml's "
                    "structure_prediction.default."
    ),
    assembly_id: Optional[str] = typer.Option(
        None, help="Only screen this assembly (default: every row in pmpnn.pkl). A multi-component "
                    "assembly may still match more than one row -- narrow further with --component-id."
    ),
    component_id: Optional[int] = typer.Option(
        None, help="Only screen this one structurally distinct component of --assembly-id."
    ),
    top_n: int = typer.Option(3, help="Shortlist size per (assembly, component), by ProteinMPNN's own score."),
    max_rmsd: float = typer.Option(2.0, help="Max CA-RMSD (Angstrom) against the RFdiffusion backbone to pass."),
    min_plddt: float = typer.Option(70.0, help="Min mean pLDDT to pass."),
    backend: Optional[str] = typer.Option(
        None, help="'local', 'singularity', or 'slurm' -- default: installation.yaml's <predictor>.backend."
    ),
    af3_model_dir: Optional[str] = typer.Option(
        None, "--af3-model-dir", help="AF3 only -- default: installation.yaml's af3.model_dir."
    ),
    af3_db_dir: Optional[str] = typer.Option(
        None, "--af3-db-dir", help="AF3 only -- default: installation.yaml's af3.db_dir."
    ),
    af3_run_data_pipeline: Optional[bool] = typer.Option(
        None, "--af3-run-data-pipeline/--af3-no-data-pipeline",
        help="AF3 only -- default: installation.yaml's af3.run_data_pipeline, or af3.py's own "
             "default (True) if that's unset too."
    ),
    af3_terms_acknowledged: bool = typer.Option(
        False, "--af3-terms-acknowledged",
        help="AF3 only -- confirms you've read https://github.com/google-deepmind/alphafold3/blob/"
             "main/WEIGHTS_TERMS_OF_USE.md. Ignored for boltz/af2. Can also be set once via "
             "installation.yaml's af3.terms_acknowledged."
    ),
):
    """Fold ProteinMPNN's best candidate(s) back and screen by self-consistency (RMSD/pLDDT)."""
    try:
        df = pipeline.run_predict(
            predictor=predictor, assembly_id=assembly_id, component_id=component_id,
            top_n=top_n, max_rmsd=max_rmsd, min_plddt=min_plddt, backend=backend,
            af3_model_dir=af3_model_dir, af3_db_dir=af3_db_dir,
            af3_terms_acknowledged=af3_terms_acknowledged, af3_run_data_pipeline=af3_run_data_pipeline,
            state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except (ValueError, RuntimeError) as exc:
        _fail(f"Structure prediction failed: {exc}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Structure prediction failed: {exc}")

    console.print(f"[bold green]✓[/bold green] {len(df)} candidate(s) passed self-consistency screening.")
    _summarize(df, empty_hint="No candidate passed --max-rmsd/--min-plddt -- check `symbro pmpnn`'s own output, "
                               "or loosen those thresholds.")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/predict.pkl[/cyan] (preview: predict.csv)")


@app.command()
def codon(
    ctx: typer.Context,
    host: str = typer.Option(
        "e_coli", help="Codon usage table to optimize for -- one of e_coli, s_cerevisiae, h_sapiens, "
                        "c_elegans, b_subtilis, d_melanogaster, g_gallus, m_musculus "
                        "(python_codon_tables' own bundled organisms)."
    ),
    assembly_id: Optional[str] = typer.Option(None, help="Only optimize this assembly (default: every row in predict.pkl)."),
    method: str = typer.Option(
        "use_best_codon", help="DNAChisel codon-optimization method: use_best_codon (CAI-style, most "
                                "frequent codon every time), match_codon_usage (matches the host's overall "
                                "codon usage profile), or harmonize_rca (rarely applicable here -- see "
                                "codon.py's own module docstring)."
    ),
    gc_min: float = typer.Option(0.3, help="Minimum GC content fraction (0-1), checked overall and per --gc-window."),
    gc_max: float = typer.Option(0.65, help="Maximum GC content fraction (0-1), checked overall and per --gc-window."),
    gc_window: int = typer.Option(100, help="Window size (nt) for the local GC content check."),
    homopolymer_max: int = typer.Option(5, help="Longest allowed run of one nucleotide (0 disables this check)."),
    avoid_hairpins: bool = typer.Option(True, "--avoid-hairpins/--allow-hairpins", help="Avoid self-complementary hairpin regions."),
    avoid_repeats_kmer: int = typer.Option(15, help="Disallow any repeated subsequence of this length (0 disables this check)."),
    avoid_enzyme: List[str] = typer.Option(
        ["BsaI", "BsmBI"], "--avoid-enzyme",
        help="Restriction site(s) to keep out of the sequence (repeatable; DNAChisel enzyme names, e.g. "
             "EcoRI). Defaults to the two Type IIS Golden Gate enzymes prosculpt's own codon step avoids "
             "-- pass this flag with no values to disable."
    ),
    add_stop_codon: bool = typer.Option(
        True, "--add-stop-codon/--no-stop-codon",
        help="Append the host's own preferred stop codon. Use --no-stop-codon if this sequence is meant "
             "to be fused into a larger ORF (e.g. behind an N-terminal tag)."
    ),
    fasta_path: Optional[str] = typer.Option(
        None, "--fasta", help="Where to write the orderable FASTA (default: <state-dir>/codon.fasta)."
    ),
):
    """
    Reverse-translate symbro predict's validated designs into host-codon-optimized DNA (needs the
    "codon" extra: pip install symbro\[codon]), applying standard gene-synthesis safety checks (GC
    content, homopolymer runs, hairpins, repeats, common Golden Gate enzyme sites). Produces a strong
    starting sequence per candidate -- still meant to be reviewed by hand before ordering, not a fully
    automated expression-vector assembly. See codon.py's own module docstring for the full scope.
    """
    try:
        df = pipeline.run_codon(
            assembly_id=assembly_id, host=host, method=method, gc_min=gc_min, gc_max=gc_max,
            gc_window=gc_window, homopolymer_max=homopolymer_max, avoid_hairpins=avoid_hairpins,
            avoid_repeats_kmer=avoid_repeats_kmer or None, avoid_enzymes=avoid_enzyme or None,
            add_stop_codon=add_stop_codon, fasta_path=fasta_path, state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except ImportError as exc:
        _fail(str(exc))
    except (ValueError, RuntimeError) as exc:
        _fail(f"Codon optimization failed: {exc}")
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Codon optimization failed: {exc}")

    console.print(f"[bold green]✓[/bold green] {len(df)} sequence(s) codon-optimized for {host!r}.")
    _summarize(df, empty_hint="No candidate could be optimized -- check the messages printed above "
                               "(likely a predict.pkl/pmpnn.pkl checkpoint mismatch).")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/codon.pkl[/cyan] (preview: codon.csv), "
                       f"FASTA: [cyan]{fasta_path or os.path.join(ctx.obj['state_dir'], 'codon.fasta')}[/cyan]")


@app.command()
def clean(
    ctx: typer.Context,
    keep_state: bool = typer.Option(
        False, "--keep-state", help="Don't clear .symbro/ checkpoints -- only scratch files. See "
                                     "the warning below before using this."
    ),
    keep_downloads: bool = typer.Option(False, "--keep-downloads", help="Don't clear temporary_files/."),
    keep_subunits: bool = typer.Option(
        False, "--keep-subunits", help="Don't clear temporary_subunits/ (isolated rings + RFdiffusion designs)."
    ),
    keep_simulations: bool = typer.Option(
        False, "--keep-simulations", help="Don't clear temporary_simulations/ (ProteinMPNN output)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be cleared without deleting anything."),
):
    """
    Clear scratch files and checkpoints between pipeline runs -- the
    "start fresh" button. Clears everything by default (every
    .symbro/*.pkl checkpoint, plus temporary_files/, temporary_subunits/,
    temporary_simulations/); use --keep-* to narrow it. Clearing scratch
    files without also clearing the matching checkpoint(s) leaves stale
    filepath references behind, which is why --keep-state exists but
    isn't the default -- only reach for it if you specifically know why.
    """
    cleared = pipeline.clean(
        state=not keep_state, downloads=not keep_downloads,
        subunits=not keep_subunits, simulations=not keep_simulations,
        dry_run=dry_run, state_dir=ctx.obj["state_dir"],
    )

    verb = "Would clear" if dry_run else "Cleared"
    nothing = not cleared["state"] and not any(
        cleared[k] for k in ("temporary_files", "temporary_subunits", "temporary_simulations")
    )
    if nothing:
        console.print("[yellow]Nothing to clear.[/yellow]")
        return

    console.print(f"[bold green]✓[/bold green] {verb}:")
    if cleared["state"]:
        console.print(f"  checkpoints: {', '.join(cleared['state'])}")
    for key, label in (
        ("temporary_files", "temporary_files/"),
        ("temporary_subunits", "temporary_subunits/"),
        ("temporary_simulations", "temporary_simulations/"),
    ):
        if cleared[key]:
            console.print(f"  {label}")

    if dry_run:
        console.print("  (dry run -- nothing was actually deleted; drop --dry-run to apply.)")


if __name__ == "__main__":
    app()
