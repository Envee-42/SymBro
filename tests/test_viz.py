"""
test_viz.py -- viz.py's own unit tests. No mocking anywhere: unlike every
external backend this project wires up (RFdiffusion/ProteinMPNN/AF2/
Boltz/AF3), viz.py never crosses a real I/O boundary (no network, no
subprocess) -- it's local file read + string templating, so every test
here runs the real code path against a real, gemmi-built structure (the
same conftest.py `ring_pdb`/`_build_ring_structure` fixtures the rest of
this suite already uses).
"""
import os

import pytest

from toolkit import viz
from conftest import _build_ring_structure


# ----------------------------------------------------------------------
# structure_format
# ----------------------------------------------------------------------

@pytest.mark.parametrize("filename, expected", [
    ("ring.pdb", "pdb"),
    ("ring.ent", "pdb"),
    ("ring.PDB", "pdb"),  # case-insensitive
    ("ring.cif", "cif"),
    ("ring.mmcif", "cif"),
])
def test_structure_format_recognized_extensions(filename, expected):
    assert viz.structure_format(filename) == expected


def test_structure_format_unrecognized_extension_raises():
    with pytest.raises(ValueError, match=r"\.xyz"):
        viz.structure_format("ring.xyz")


# ----------------------------------------------------------------------
# render_html
# ----------------------------------------------------------------------

def test_render_html_embeds_structure_text_and_library(ring_pdb):
    html = viz.render_html(ring_pdb)

    with open(ring_pdb) as f:
        raw = f.read()
    assert raw in html  # the real ATOM records are in there verbatim
    assert "$3Dmol.createViewer" in html
    assert "addModel" in html
    assert 'id="symbro-structure-data"' in html
    # the vendored library is genuinely embedded, not CDN-linked --
    # confirmed both by a UMD-build marker string and by there being no
    # http(s):// script src anywhere in the output.
    assert "3dmol v2.5.5" in html or "$3Dmol=" in html
    assert "<script src=" not in html


def test_render_html_uses_correct_format_string(ring_pdb):
    html = viz.render_html(ring_pdb)
    assert 'addModel(data, "pdb")' in html


def test_render_html_title_defaults_to_basename(ring_pdb):
    html = viz.render_html(ring_pdb)
    assert os.path.basename(ring_pdb) in html


def test_render_html_custom_title_is_used(ring_pdb):
    html = viz.render_html(ring_pdb, title="TEST-1 C3 ring")
    assert "TEST-1 C3 ring" in html


def test_render_html_neutralizes_embedded_script_close_tag(project_dir):
    # Not realistic PDB content, but a real defensive guard against
    # whatever ends up in a structure file breaking the surrounding HTML
    # -- a literal "</script" would otherwise terminate the embedding
    # <script> block early regardless of its type= attribute.
    path = os.path.join(str(project_dir), "weird.pdb")
    with open(path, "w") as f:
        f.write("REMARK </script>alert(1)</script>\nEND\n")

    html = viz.render_html(path)
    assert "</script>alert(1)</script>" not in html
    assert "<\\/script>alert(1)<\\/script>" in html


# ----------------------------------------------------------------------
# render_to_file
# ----------------------------------------------------------------------

def test_render_to_file_writes_html(project_dir, ring_pdb):
    out_path = os.path.join(str(project_dir), "view.html")
    result = viz.render_to_file(ring_pdb, out_path)

    assert result == out_path
    assert os.path.exists(out_path)
    with open(out_path) as f:
        content = f.read()
    assert "$3Dmol.createViewer" in content


def test_render_to_file_creates_parent_directories(project_dir, ring_pdb):
    out_path = os.path.join(str(project_dir), "nested", "dir", "view.html")
    viz.render_to_file(ring_pdb, out_path)
    assert os.path.exists(out_path)


def test_render_to_file_unrecognized_extension_raises_before_writing(project_dir):
    src = os.path.join(str(project_dir), "structure.xyz")
    with open(src, "w") as f:
        f.write("not a real structure\n")
    out_path = os.path.join(str(project_dir), "view.html")

    with pytest.raises(ValueError):
        viz.render_to_file(src, out_path)
    assert not os.path.exists(out_path)
