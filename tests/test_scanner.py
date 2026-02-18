import pytest
from zvecsearch.scanner import scan_paths


@pytest.fixture
def tmp_files(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.markdown").write_text("# B")
    (tmp_path / "c.txt").write_text("not md")
    (tmp_path / ".hidden.md").write_text("# hidden")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.md").write_text("# D")
    return tmp_path


def test_scan_finds_markdown(tmp_files):
    results = scan_paths([tmp_files])
    names = {r.path.name for r in results}
    assert "a.md" in names
    assert "b.markdown" in names
    assert "d.md" in names
    assert "c.txt" not in names


def test_scan_ignores_hidden(tmp_files):
    results = scan_paths([tmp_files])
    names = {r.path.name for r in results}
    assert ".hidden.md" not in names


def test_scan_single_file(tmp_files):
    results = scan_paths([tmp_files / "a.md"])
    assert len(results) == 1
    assert results[0].path.name == "a.md"


def test_scan_deduplicates(tmp_files):
    results = scan_paths([tmp_files, tmp_files / "a.md"])
    paths = [r.path for r in results]
    assert len(paths) == len(set(paths))


def test_scanned_file_has_metadata(tmp_files):
    results = scan_paths([tmp_files / "a.md"])
    assert results[0].mtime > 0
    assert results[0].size > 0
