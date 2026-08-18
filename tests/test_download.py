"""test_download.py — the timeout + per-file fail-soft fix applied this session."""
import requests

from toolkit import download


class _FakeResponse:
    def __init__(self, content=b"fake gz bytes", status=200):
        self._content = content
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error")

    @property
    def content(self):
        return self._content


def test_download_structure_passes_a_timeout(project_dir, monkeypatch):
    """The original bug: requests.get(url) had no timeout= at all."""
    captured = {}

    def fake_get(url, timeout=None):
        captured["timeout"] = timeout
        import gzip
        return _FakeResponse(content=gzip.compress(b"structure data"))

    monkeypatch.setattr(requests, "get", fake_get)
    download.download_structure("out.cif", "http://example.test/x.cif.gz")

    assert captured["timeout"] is not None
    assert captured["timeout"] == download.DOWNLOAD_TIMEOUT


def test_download_candidates_skips_one_failure_not_the_whole_batch(project_dir, monkeypatch):
    import gzip

    def fake_get(url, timeout=None):
        # RCSB_DOWNLOAD_URL always lowercases the entry id into the URL
        # (see download.py's own build_download_table()) -- check
        # case-insensitively so this doesn't depend on that detail.
        if "bad" in url.lower():
            return _FakeResponse(status=404)
        return _FakeResponse(content=gzip.compress(b"structure data"))

    monkeypatch.setattr(requests, "get", fake_get)
    df = download.download_candidates(["OK1-1", "BAD1-1", "OK2-1"])

    assert set(df["assembly_id"]) == {"OK1-1", "OK2-1"}  # the bad one is dropped, not fatal
    assert len(df) == 2


def test_download_candidates_all_succeed(project_dir, monkeypatch):
    import gzip

    monkeypatch.setattr(requests, "get", lambda url, timeout=None: _FakeResponse(content=gzip.compress(b"x")))
    df = download.download_candidates(["OK1-1", "OK2-1"])
    assert len(df) == 2
    for p in df["filepath"]:
        from toolkit.paths import resolve_path
        assert __import__("os").path.exists(resolve_path(p))
