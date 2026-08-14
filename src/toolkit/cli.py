"""
cli.py — the `symbro` command. A thin layer over pipeline.py: parses
arguments, calls the matching pipeline stage function, and prints a
clear, human-readable summary or error. No pipeline logic lives here --
see pipeline.py's module docstring for why, and for the ".symbro/" state
folder convention every command below relies on.

Typical workflow, run in order:

    symbro query         search RCSB PDB for candidate assemblies
    symbro download      download the matching structure files
    symbro geometry      detect symmetry rings (and orientation/termini for one type)
    symbro isolate       extract each ring's PDB, ready for RFdiffusion
    symbro rfdiffusion   submit RFdiffusion, one job per assembly (blocks until done
                         unless --detach, which needs backend='slurm')
    symbro status        check on --detach'd RFdiffusion jobs
    symbro pmpnn         run ProteinMPNN against each assembly's best design(s)
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
    err_console.print(f"[bold red]✗[/bold red] {message}")
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


def _parse_criterion(raw: str) -> dict:
    """Parses "attribute=value:operator" (operator optional, defaults to exact_match).

    A comma-separated value becomes a list (e.g. "in" matching); with
    operator "range", exactly two comma-separated numbers are required.
    """
    if "=" not in raw:
        raise ValueError(f'--criterion "{raw}" is missing "=" -- expected format: attribute=value[:operator]')
    attribute, rest = raw.split("=", 1)
    if ":" in rest:
        value_part, operator = rest.rsplit(":", 1)
    else:
        value_part, operator = rest, "exact_match"

    if operator == "range":
        parts = [p.strip() for p in value_part.split(",")]
        if len(parts) != 2:
            raise ValueError(f'--criterion "{raw}": operator "range" needs exactly two comma-separated numbers, e.g. resolution=0,3:range')
        try:
            value = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise ValueError(f'--criterion "{raw}": range values must be numbers') from None
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
    match: str = typer.Option("and", help="How to combine all the above: 'and' or 'or'."),
    fetch_field: Optional[List[str]] = typer.Option(
        None, "--fetch-field", help="Extra metadata field to include in the result (repeatable)."
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
        df = pipeline.run_query(
            symmetry=symmetry, entry_id=entry_id, resolution_range=resolution_range,
            description=description, extra_criteria=extra_criteria, match=match,
            fetch_fields=fetch_field, return_type=return_type,
            state_dir=ctx.obj["state_dir"],
        )
    except ValueError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Search failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Found {len(df)} candidate(s).")
    _summarize(df, empty_hint="Try loosening your search criteria.")
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
def geometry(
    ctx: typer.Context,
    symmetry_type: Optional[str] = typer.Option(
        None,
        help="Narrow to one symmetry order (e.g. C3) and also compute orientation + "
             "termini secondary structure. Omit on a first pass to just see what's "
             "present in your candidates before choosing one.",
    ),
):
    """Detect symmetry rings in every downloaded structure."""
    try:
        df = pipeline.run_geometry(symmetry_type=symmetry_type, state_dir=ctx.obj["state_dir"])
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Geometry analysis failed: {exc}")

    if symmetry_type is None:
        counts = pipeline.detected_symmetry_types(df)
        if counts.empty:
            console.print("[yellow]No symmetry rings detected in any downloaded structure.[/yellow]")
        else:
            console.print("[bold green]✓[/bold green] Symmetry types detected:")
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("symmetry_type")
            table.add_column("assemblies", justify="right")
            for sym, count in counts.items():
                table.add_row(sym, str(count))
            console.print(table)
            console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/geometry.pkl[/cyan] (preview: geometry.csv)")
            console.print("  Next: re-run with [bold]--symmetry-type[/bold] on one of the above, e.g. "
                           f"[bold]symbro geometry --symmetry-type {counts.index[0]}[/bold]")
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
    file_format: str = typer.Option("pdb", help="'pdb' or 'cif'. RFdiffusion expects 'pdb'."),
    output_dir: Optional[str] = typer.Option(
        None, help="Where to write extracted ring structures (default: temporary_subunits/)."
    ),
):
    """Extract each candidate's ring structure -- the files RFdiffusion needs next."""
    try:
        df = pipeline.run_isolate(
            symmetry_type=symmetry_type, file_format=file_format,
            output_dir=output_dir, state_dir=ctx.obj["state_dir"],
        )
    except pipeline.StageNotFoundError as exc:
        _fail(str(exc))
    except typer.Exit:
        raise
    except Exception as exc:
        _fail(f"Isolating rings failed: {exc}")

    console.print(f"[bold green]✓[/bold green] Extracted {len(df)} ring structure(s).")
    _summarize(df, empty_hint="Try a different --symmetry-type, or check `symbro geometry`'s output.")
    if not df.empty:
        console.print(f"  Saved to [cyan]{ctx.obj['state_dir']}/rings.pkl[/cyan] (preview: rings.csv)")
        console.print("  Next: [bold]symbro rfdiffusion[/bold]")


@app.command()
def rfdiffusion(
    ctx: typer.Context,
    assembly_id: Optional[str] = typer.Option(
        None, help="Only submit this assembly's ring (default: every row in rings.pkl)."
    ),
    linker_min: int = typer.Option(15, help="Diffused linker min length (residues)."),
    linker_max: int = typer.Option(25, help="Diffused linker max length (residues)."),
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
    """Submit RFdiffusion -- one job per assembly in `symbro isolate`'s output."""
    try:
        df = pipeline.run_rfdiffusion(
            assembly_id=assembly_id, linker_length=(linker_min, linker_max),
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
        preview = preview[["assembly_id", "symmetry_type", "state", "slurm_job_id", "designs_written"]]
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
    preview = df[["assembly_id", "symmetry_type", "state", "design_paths"]].copy()
    preview["design_paths"] = preview["design_paths"].apply(len)
    preview = preview.rename(columns={"design_paths": "designs_written"})
    _summarize(preview, empty_hint="")
    if still_running.empty and not df.empty:
        console.print("  All jobs done. Next: [bold]symbro pmpnn[/bold]")


@app.command()
def pmpnn(
    ctx: typer.Context,
    assembly_id: Optional[str] = typer.Option(
        None, help="Only run this assembly (default: every completed row in rfdiffusion.pkl)."
    ),
    top_n: int = typer.Option(1, help="Submit only the top N RFdiffusion designs per assembly, by pLDDT."),
    min_plddt: Optional[float] = typer.Option(None, help="Also require at least this mean pLDDT."),
    select: Optional[List[str]] = typer.Option(
        None, "--select", help="Explicit design PDB path (repeatable) -- bypasses --top-n/"
                                "--min-plddt entirely. Requires --assembly-id."
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
            assembly_id=assembly_id, top_n=top_n, min_plddt=min_plddt, select=select,
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
        console.print("  Structure-prediction screening commands are coming in the next round.")


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
