"""
viz.py -- generates a single, self-contained HTML file that renders a
structure in an interactive 3D viewer, via 3Dmol.js (vendored under
toolkit/static/3dmol/ -- see that directory's own NOTICE.md for exact
version/provenance/license). The entire library is embedded inline, not
CDN-linked, so both GENERATING and OPENING the result need zero network
access -- deliberately the one piece of this pipeline that works for
anyone who just `pip install`s symbro, with no installation.yaml entry,
no external tool, no GPU. See NOTICE.md before ever updating the vendored
library, and README.md's "3D visualization" section for the CLI-facing
docs (`symbro view`).

    from toolkit import viz

    viz.render_to_file("temporary_subunits/1ABC-1_C3_A-B-C.pdb", "ring.html")

Scope, deliberately kept lean: renders exactly ONE structure per call,
colored by chain. No overlay of two structures (e.g. an RFdiffusion
backbone against its predicted candidate) and no drawing of a detected
symmetry axis as a 3D shape -- both real, useful extensions, left for a
later pass rather than folded into this first cut.
"""

from __future__ import annotations

import os
from importlib import resources
from typing import Optional

_STATIC_ANCHOR = "toolkit"  # anchor on the top-level package only, then
# joinpath() into static/3dmol/ below -- "3dmol" (leading digit) isn't a
# valid Python identifier, so it can't be part of a dotted import target,
# but joinpath() is pure filesystem traversal past the anchor and doesn't
# care.
_JS_RELATIVE_PATH = ("static", "3dmol", "3Dmol-min.js")

_FORMAT_BY_EXTENSION = {".pdb": "pdb", ".ent": "pdb", ".cif": "cif", ".mmcif": "cif"}

# HTML's own script-tag termination rule: a literal "</script" ends ANY
# <script> block textually, regardless of its type= attribute -- applied
# to both the (trusted, vendored) JS library and the (arbitrary) structure
# text below, cheaply removing this as a failure mode entirely rather than
# assuming neither will ever contain that substring.
def _script_safe(text: str) -> str:
    return text.replace("</script", "<\\/script")


def structure_format(filepath: str) -> str:
    """Maps a file extension to the format string 3Dmol.js's addModel()
    expects ("pdb" or "cif"). Raises ValueError for anything else -- no
    guessing from file content, since getting this wrong silently renders
    nothing rather than erroring."""
    ext = os.path.splitext(filepath)[1].lower()
    fmt = _FORMAT_BY_EXTENSION.get(ext)
    if fmt is None:
        raise ValueError(
            f"Unrecognized structure file extension {ext!r} for {filepath!r} -- "
            f"expected one of {sorted(_FORMAT_BY_EXTENSION)}."
        )
    return fmt


def _load_3dmol_js() -> str:
    path = resources.files(_STATIC_ANCHOR)
    for part in _JS_RELATIVE_PATH:
        path = path.joinpath(part)
    return path.read_text(encoding="utf-8")


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__ -- SymBro 3D view</title>
<style>
  html, body { margin: 0; padding: 0; height: 100%; background: #14161a; }
  #viewer { width: 100%; height: 100%; position: relative; }
  #label {
    position: absolute; top: 10px; left: 14px; z-index: 10;
    color: #e8e8e8; font: 13px/1.4 -apple-system, "Segoe UI", sans-serif;
    text-shadow: 0 1px 2px rgba(0,0,0,0.8);
  }
</style>
</head>
<body>
<div id="viewer"><div id="label">__TITLE__</div></div>
<script type="text/plain" id="symbro-structure-data">__STRUCTURE_TEXT__</script>
<script>
__JS_LIB__
</script>
<script>
(function () {
  var data = document.getElementById("symbro-structure-data").textContent;
  var viewer = $3Dmol.createViewer("viewer", {backgroundColor: "0x14161a"});
  viewer.addModel(data, "__STRUCTURE_FORMAT__");
  viewer.setStyle({}, {cartoon: {colorscheme: "chain"}});
  viewer.zoomTo();
  viewer.render();
})();
</script>
</body>
</html>
"""


def render_html(filepath: str, title: Optional[str] = None) -> str:
    """
    Reads a structure file and returns a complete, self-contained HTML
    document string that renders it in an interactive, chain-colored
    cartoon view -- open it in any browser, no server/network needed.

    filepath : a .pdb/.ent or .cif/.mmcif file. Its format is inferred
        from the extension (see structure_format()) and its raw text is
        embedded verbatim -- 3Dmol.js parses PDB/mmCIF itself, so no
        gemmi round-trip is needed here.
    title : shown in both the browser tab and an on-page label. Defaults
        to the file's own basename.
    """
    fmt = structure_format(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        structure_text = f.read()

    html = _HTML_TEMPLATE
    html = html.replace("__TITLE__", title or os.path.basename(filepath))
    html = html.replace("__STRUCTURE_FORMAT__", fmt)
    # Order matters: these two are the largest substitutions (the vendored
    # library, then the structure itself) and are done via plain
    # str.replace rather than str.format() specifically because BOTH can
    # freely contain "{"/"}" (minified JS throughout; a PDB file only
    # incidentally, but never assume) -- str.format() would misparse them
    # as format fields.
    html = html.replace("__JS_LIB__", _script_safe(_load_3dmol_js()))
    html = html.replace("__STRUCTURE_TEXT__", _script_safe(structure_text))
    return html


def render_to_file(filepath: str, output_path: str, title: Optional[str] = None) -> str:
    """render_html() plus writing the result to output_path (parent
    directories created as needed). Returns output_path."""
    html = render_html(filepath, title=title)
    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
