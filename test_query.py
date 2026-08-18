"""
test_query.py — pure-pandas / pure-logic coverage for query.py.

query.py's own `from rcsbapi.search import AttributeQuery` / `from
rcsbapi.data import DataQuery` (module level) trigger a REAL network fetch
of RCSB's live schema at import time (confirmed directly: rcsbapi's own
SearchSchema.__init__ calls httpx.get() against search.rcsb.org as soon as
`rcsbapi.search` is imported) -- there's no way to import toolkit.query at
all without either real internet access or stubbing that import out. This
file stubs sys.modules['rcsbapi.search']/['rcsbapi.data'] with minimal
fakes BEFORE importing toolkit.query, so build_query()/_cell_matches()/
filter_metadata()/extract_leaf_values() -- everything that's pure logic,
no actual API call -- can be exercised for real, offline, in CI. This is
the same "mock only at the true I/O boundary" convention the rest of this
project's test suite already follows; here the boundary happens to be the
import itself rather than a function call.

search_ids()/fetch_metadata() (the two functions that DO make a real call)
are deliberately NOT covered here -- that needs either a live RCSB
connection or a much heavier mock of rcsbapi's own query-execution
internals, out of scope for this pass.
"""
import sys
import types

import pandas as pd
import pytest


def _install_rcsbapi_stub():
    """Fake, import-safe stand-ins for AttributeQuery/DataQuery -- enough
    for build_query() to construct one and return it (never called/
    executed here), without touching the network."""
    if "toolkit.query" in sys.modules:
        return  # already imported for real (or already stubbed) this process

    class _FakeAttributeQuery:
        def __init__(self, attribute=None, operator=None, value=None):
            self.attribute, self.operator, self.value = attribute, operator, value

        def __and__(self, other):
            return self

        def __or__(self, other):
            return self

        def __call__(self, *args, **kwargs):
            raise NotImplementedError("stub -- real execution not exercised by this test file")

    class _FakeDataQuery:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            raise NotImplementedError("stub -- real execution not exercised by this test file")

    search_mod = types.ModuleType("rcsbapi.search")
    search_mod.AttributeQuery = _FakeAttributeQuery
    data_mod = types.ModuleType("rcsbapi.data")
    data_mod.DataQuery = _FakeDataQuery
    rcsbapi_mod = types.ModuleType("rcsbapi")
    rcsbapi_mod.search = search_mod
    rcsbapi_mod.data = data_mod

    sys.modules.setdefault("rcsbapi", rcsbapi_mod)
    sys.modules["rcsbapi.search"] = search_mod
    sys.modules["rcsbapi.data"] = data_mod


_install_rcsbapi_stub()
from toolkit import query  # noqa: E402 -- must come after the stub install above


# ----------------------------------------------------------------------
# build_query() -- range value validation (this session's fix)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", ["not-a-tuple", (1, 2, 3), (1,), 5, [1]])
def test_build_query_range_rejects_malformed_value(bad_value):
    with pytest.raises(ValueError, match="operator='range' needs value"):
        query.build_query("resolution", bad_value, operator="range")


def test_build_query_range_accepts_a_real_pair():
    q = query.build_query("resolution", (0, 3.0), operator="range")
    assert q is not None  # both AttributeQuerys built and &'d without raising


def test_build_query_rejects_fetch_only_attribute():
    with pytest.raises(ValueError, match="metadata-only"):
        query.build_query("model_quality", 90)


# ----------------------------------------------------------------------
# _cell_matches() -- contains_phrase TypeError fix + range validation
# ----------------------------------------------------------------------
def test_cell_matches_contains_phrase_on_numeric_list_cell():
    """THE regression test: a multi-valued, non-string cell (e.g. two
    distinct assembly weights) used to raise TypeError via ", ".join(cell)
    -- str()'d now, so this must return a real bool instead of crashing."""
    cell = [1024.5, 2048.0]
    assert query._cell_matches(cell, "1024", "contains_phrase") is True
    assert query._cell_matches(cell, "9999", "contains_phrase") is False


def test_cell_matches_contains_phrase_on_string_list_cell_still_works():
    cell = ["Homo sapiens", "Escherichia coli"]
    assert query._cell_matches(cell, "sapiens", "contains_phrase") is True


def test_cell_matches_contains_phrase_on_scalar_cell():
    assert query._cell_matches(1024.5, "1024", "contains_phrase") is True


@pytest.mark.parametrize("bad_value", ["not-a-tuple", (1, 2, 3), 5])
def test_cell_matches_range_rejects_malformed_value(bad_value):
    with pytest.raises(ValueError, match="operator='range' needs value"):
        query._cell_matches(1.5, bad_value, "range")


def test_cell_matches_range_on_list_and_scalar_cells():
    assert query._cell_matches(2.0, (1.0, 3.0), "range") is True
    assert query._cell_matches([0.5, 2.0], (1.0, 3.0), "range") is True
    assert query._cell_matches([0.5, 0.6], (1.0, 3.0), "range") is False


def test_cell_matches_none_and_nan_never_match():
    assert query._cell_matches(None, "x", "exact_match") is False
    assert query._cell_matches(float("nan"), "x", "exact_match") is False


# ----------------------------------------------------------------------
# extract_leaf_values() -- flattening behavior filter_metadata()/
# _cell_matches() both depend on
# ----------------------------------------------------------------------
def test_extract_leaf_values_single_scalar():
    assert query.extract_leaf_values({"a": {"b": 1024.5}}) == 1024.5


def test_extract_leaf_values_multiple_strings_joined():
    assert query.extract_leaf_values(["Homo sapiens", "Homo sapiens", "E. coli"]) == "Homo sapiens, E. coli"


def test_extract_leaf_values_multiple_non_strings_kept_as_list():
    assert query.extract_leaf_values([1024.5, 2048.0]) == [1024.5, 2048.0]


def test_extract_leaf_values_empty_is_none():
    assert query.extract_leaf_values(None) is None
    assert query.extract_leaf_values([]) is None
    assert query.extract_leaf_values(float("nan")) is None


# ----------------------------------------------------------------------
# filter_metadata() -- pure-pandas stage 3, real AND/OR/range/contains_phrase
# ----------------------------------------------------------------------
@pytest.fixture
def metadata_df():
    return pd.DataFrame([
        {"assembly_id": "A-1", "symmetry": "C3", "weight": [1024.5, 2048.0], "resolution": 1.8},
        {"assembly_id": "A-2", "symmetry": "C2", "weight": 512.0, "resolution": 3.2},
        {"assembly_id": "A-3", "symmetry": "C3", "weight": 4096.0, "resolution": None},
    ])


def test_filter_metadata_exact_match(metadata_df):
    out = query.filter_metadata(metadata_df, [{"attribute": "symmetry", "value": "C3"}])
    assert set(out["assembly_id"]) == {"A-1", "A-3"}


def test_filter_metadata_range(metadata_df):
    out = query.filter_metadata(metadata_df, [{"attribute": "resolution", "value": (0, 2.0), "operator": "range"}])
    assert set(out["assembly_id"]) == {"A-1"}  # A-3's None resolution never matches a range


def test_filter_metadata_contains_phrase_on_numeric_list_column(metadata_df):
    """The exact real-world shape the contains_phrase fix targets: a
    multi-valued numeric column (weight) filtered with contains_phrase."""
    out = query.filter_metadata(metadata_df, [{"attribute": "weight", "value": "1024", "operator": "contains_phrase"}])
    assert set(out["assembly_id"]) == {"A-1"}


def test_filter_metadata_and_vs_or(metadata_df):
    criteria = [{"attribute": "symmetry", "value": "C3"}, {"attribute": "resolution", "value": (0, 2.0), "operator": "range"}]
    and_out = query.filter_metadata(metadata_df, criteria, mode="and")
    or_out = query.filter_metadata(metadata_df, criteria, mode="or")
    assert set(and_out["assembly_id"]) == {"A-1"}
    assert set(or_out["assembly_id"]) == {"A-1", "A-3"}


def test_filter_metadata_raises_on_missing_column(metadata_df):
    with pytest.raises(ValueError, match="not.*in|not present|not.*column"):
        query.filter_metadata(metadata_df, [{"attribute": "not_a_real_column", "value": "x"}])


def test_filter_metadata_raises_on_empty_criteria(metadata_df):
    with pytest.raises(ValueError):
        query.filter_metadata(metadata_df, [])
