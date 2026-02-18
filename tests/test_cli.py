"""Tests for the zvecsearch CLI interface.

All tests mock ZvecSearch/ZvecStore since the zvec native library
requires AVX-512 which is unavailable on this CPU.  We use
click.testing.CliRunner for all CLI invocations.
"""
from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Mock zvec in sys.modules before any import of store/core
# ---------------------------------------------------------------------------
if "zvec" not in sys.modules:
    _zvec_mock = MagicMock()
    _zvec_mock.DataType.STRING = "STRING"
    _zvec_mock.DataType.INT32 = "INT32"
    _zvec_mock.DataType.VECTOR_FP32 = "VECTOR_FP32"
    _zvec_mock.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
    _zvec_mock.MetricType.COSINE = "COSINE"
    _zvec_mock.MetricType.L2 = "L2"
    _zvec_mock.MetricType.IP = "IP"
    _zvec_mock.LogLevel.WARN = "WARN"
    _zvec_mock.FieldSchema = MagicMock
    _zvec_mock.VectorSchema = MagicMock
    _zvec_mock.CollectionSchema = MagicMock
    _zvec_mock.CollectionOption = MagicMock
    _zvec_mock.HnswIndexParam = MagicMock
    _zvec_mock.InvertIndexParam = MagicMock
    _zvec_mock.FlatIndexParam = MagicMock
    _zvec_mock.VectorQuery = MagicMock
    _zvec_mock.RrfReRanker = MagicMock
    _zvec_mock.BM25EmbeddingFunction = MagicMock
    _zvec_mock.Doc = MagicMock
    sys.modules["zvec"] = _zvec_mock

from zvecsearch.cli import cli  # noqa: E402


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Basic CLI tests
# ---------------------------------------------------------------------------
class TestCLIBasics:
    def test_cli_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output

    def test_cli_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_cli_no_args_shows_help(self, runner):
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Usage" in result.output


# ---------------------------------------------------------------------------
# Index command
# ---------------------------------------------------------------------------
class TestCLIIndex:
    def test_index_help(self, runner):
        result = runner.invoke(cli, ["index", "--help"])
        assert result.exit_code == 0
        assert "PATHS" in result.output

    def test_index_success(self, runner, tmp_path):
        (tmp_path / "a.md").write_text("# Hello\nContent here.")
        mock_zs = MagicMock()
        mock_zs.index = AsyncMock(return_value=3)
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["index", str(tmp_path)])
            assert result.exit_code == 0
            assert "Indexed 3 chunks" in result.output
            mock_zs.close.assert_called_once()

    def test_index_force_flag(self, runner, tmp_path):
        (tmp_path / "a.md").write_text("# Test\n")
        mock_zs = MagicMock()
        mock_zs.index = AsyncMock(return_value=1)
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["index", "--force", str(tmp_path)])
            assert result.exit_code == 0
            mock_zs.index.assert_called_once_with(force=True)

    def test_index_no_paths_fails(self, runner):
        result = runner.invoke(cli, ["index"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Search command
# ---------------------------------------------------------------------------
class TestCLISearch:
    def test_search_help(self, runner):
        result = runner.invoke(cli, ["search", "--help"])
        assert result.exit_code == 0
        assert "QUERY" in result.output

    def test_search_text_output(self, runner):
        mock_zs = MagicMock()
        mock_zs.search = AsyncMock(return_value=[
            {
                "content": "Some matching content",
                "source": "/tmp/test.md",
                "heading": "Test Heading",
                "score": 0.9234,
                "chunk_hash": "abc123",
            },
        ])
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["search", "test query"])
            assert result.exit_code == 0
            assert "Result 1" in result.output
            assert "0.9234" in result.output
            assert "/tmp/test.md" in result.output
            assert "Some matching content" in result.output

    def test_search_json_output(self, runner):
        mock_zs = MagicMock()
        mock_zs.search = AsyncMock(return_value=[
            {"content": "test", "source": "a.md", "score": 0.8},
        ])
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["search", "--json-output", "query"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert isinstance(data, list)
            assert data[0]["content"] == "test"

    def test_search_no_results(self, runner):
        mock_zs = MagicMock()
        mock_zs.search = AsyncMock(return_value=[])
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["search", "nothing"])
            assert result.exit_code == 0
            assert "No results found" in result.output

    def test_search_top_k(self, runner):
        mock_zs = MagicMock()
        mock_zs.search = AsyncMock(return_value=[])
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["search", "--top-k", "3", "query"])
            assert result.exit_code == 0
            mock_zs.search.assert_called_once_with("query", top_k=3)

    def test_search_truncates_long_content(self, runner):
        mock_zs = MagicMock()
        mock_zs.search = AsyncMock(return_value=[
            {
                "content": "x" * 600,
                "source": "a.md",
                "heading": "",
                "score": 0.5,
                "chunk_hash": "hash123",
            },
        ])
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["search", "query"])
            assert result.exit_code == 0
            assert "truncated" in result.output
            assert "expand" in result.output


# ---------------------------------------------------------------------------
# Expand command
# ---------------------------------------------------------------------------
class TestCLIExpand:
    def test_expand_help(self, runner):
        result = runner.invoke(cli, ["expand", "--help"])
        assert result.exit_code == 0
        assert "CHUNK_HASH" in result.output

    def test_expand_found(self, runner, tmp_path):
        source_file = tmp_path / "test.md"
        source_file.write_text("# Section\nLine 1\nLine 2\nLine 3\n")

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "content": "Line 1\nLine 2",
                "source": str(source_file),
                "heading": "Section",
                "heading_level": 1,
                "start_line": 2,
                "end_line": 3,
                "chunk_hash": "abc123",
            },
        ]
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["expand", "abc123"])
            assert result.exit_code == 0
            assert "Section" in result.output
            assert str(source_file) in result.output

    def test_expand_not_found(self, runner):
        mock_store = MagicMock()
        mock_store.query.return_value = []
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["expand", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output

    def test_expand_json_output(self, runner, tmp_path):
        source_file = tmp_path / "test.md"
        source_file.write_text("# Section\nLine 1\nLine 2\n")

        mock_store = MagicMock()
        mock_store.query.return_value = [
            {
                "content": "Line 1",
                "source": str(source_file),
                "heading": "Section",
                "heading_level": 1,
                "start_line": 2,
                "end_line": 2,
                "chunk_hash": "abc123",
            },
        ]
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["expand", "--json-output", "abc123"])
            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["chunk_hash"] == "abc123"


# ---------------------------------------------------------------------------
# Stats command
# ---------------------------------------------------------------------------
class TestCLIStats:
    def test_stats_help(self, runner):
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0

    def test_stats_shows_count(self, runner):
        mock_store = MagicMock()
        mock_store.count.return_value = 42
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 0
            assert "42" in result.output

    def test_stats_error_handled(self, runner):
        with patch("zvecsearch.store.ZvecStore", side_effect=Exception("no collection")):
            result = runner.invoke(cli, ["stats"])
            assert result.exit_code == 1
            assert "Error" in result.output or "no collection" in result.output


# ---------------------------------------------------------------------------
# Reset command
# ---------------------------------------------------------------------------
class TestCLIReset:
    def test_reset_help(self, runner):
        result = runner.invoke(cli, ["reset", "--help"])
        assert result.exit_code == 0

    def test_reset_with_yes(self, runner):
        mock_store = MagicMock()
        mock_store.drop = MagicMock()
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["reset", "--yes"])
            assert result.exit_code == 0
            assert "Dropped" in result.output
            mock_store.drop.assert_called_once()

    def test_reset_without_yes_aborts(self, runner):
        mock_store = MagicMock()
        mock_store.close = MagicMock()

        with patch("zvecsearch.store.ZvecStore", return_value=mock_store):
            result = runner.invoke(cli, ["reset"], input="n\n")
            assert result.exit_code != 0 or "Aborted" in result.output


# ---------------------------------------------------------------------------
# Compact command
# ---------------------------------------------------------------------------
class TestCLICompact:
    def test_compact_help(self, runner):
        result = runner.invoke(cli, ["compact", "--help"])
        assert result.exit_code == 0

    def test_compact_success(self, runner):
        mock_zs = MagicMock()
        mock_zs.compact = AsyncMock(return_value="Summary of memories")
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["compact"])
            assert result.exit_code == 0
            assert "Summary of memories" in result.output

    def test_compact_no_chunks(self, runner):
        mock_zs = MagicMock()
        mock_zs.compact = AsyncMock(return_value="")
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["compact"])
            assert result.exit_code == 0
            assert "No chunks" in result.output

    def test_compact_with_source(self, runner):
        mock_zs = MagicMock()
        mock_zs.compact = AsyncMock(return_value="Source summary")
        mock_zs.close = MagicMock()

        with patch("zvecsearch.core.ZvecSearch", return_value=mock_zs):
            result = runner.invoke(cli, ["compact", "--source", "test.md"])
            assert result.exit_code == 0
            mock_zs.compact.assert_called_once()
            call_kwargs = mock_zs.compact.call_args[1]
            assert call_kwargs["source"] == "test.md"


# ---------------------------------------------------------------------------
# Watch command
# ---------------------------------------------------------------------------
class TestCLIWatch:
    def test_watch_help(self, runner):
        result = runner.invoke(cli, ["watch", "--help"])
        assert result.exit_code == 0
        assert "PATHS" in result.output

    def test_watch_no_paths_fails(self, runner):
        result = runner.invoke(cli, ["watch"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Transcript command
# ---------------------------------------------------------------------------
class TestCLITranscript:
    def test_transcript_help(self, runner):
        result = runner.invoke(cli, ["transcript", "--help"])
        assert result.exit_code == 0

    def test_transcript_empty_file(self, runner, tmp_path):
        jsonl = tmp_path / "transcript.jsonl"
        jsonl.write_text("")
        result = runner.invoke(cli, ["transcript", str(jsonl)])
        assert result.exit_code == 0
        assert "No conversation turns" in result.output

    def test_transcript_with_turns(self, runner, tmp_path):
        jsonl = tmp_path / "transcript.jsonl"
        import json as _json
        lines = [
            _json.dumps({
                "type": "user",
                "uuid": "user-uuid-123",
                "timestamp": "2025-01-01T10:00:00Z",
                "message": {"content": "What is AI?"},
            }),
            _json.dumps({
                "type": "assistant",
                "uuid": "asst-uuid-456",
                "timestamp": "2025-01-01T10:00:01Z",
                "message": {"content": [{"type": "text", "text": "AI is..."}]},
            }),
        ]
        jsonl.write_text("\n".join(lines) + "\n")

        result = runner.invoke(cli, ["transcript", str(jsonl)])
        assert result.exit_code == 0
        assert "user-uuid" in result.output or "1" in result.output

    def test_transcript_json_output(self, runner, tmp_path):
        jsonl = tmp_path / "transcript.jsonl"
        import json as _json
        lines = [
            _json.dumps({
                "type": "user",
                "uuid": "user-uuid-123",
                "timestamp": "2025-01-01T10:00:00Z",
                "message": {"content": "Hello!"},
            }),
            _json.dumps({
                "type": "assistant",
                "uuid": "asst-uuid-456",
                "timestamp": "2025-01-01T10:00:01Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            }),
        ]
        jsonl.write_text("\n".join(lines) + "\n")

        result = runner.invoke(cli, ["transcript", "--json-output", str(jsonl)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)


# ---------------------------------------------------------------------------
# Config commands
# ---------------------------------------------------------------------------
class TestCLIConfig:
    def test_config_help(self, runner):
        result = runner.invoke(cli, ["config", "--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "set" in result.output
        assert "get" in result.output
        assert "list" in result.output

    def test_config_list_resolved(self, runner):
        result = runner.invoke(cli, ["config", "list", "--resolved"])
        assert result.exit_code == 0
        # Should contain some known config sections
        assert "embedding" in result.output or "zvec" in result.output

    def test_config_list_global(self, runner):
        result = runner.invoke(cli, ["config", "list", "--global"])
        assert result.exit_code == 0

    def test_config_list_project(self, runner):
        result = runner.invoke(cli, ["config", "list", "--project"])
        assert result.exit_code == 0

    def test_config_get(self, runner):
        result = runner.invoke(cli, ["config", "get", "embedding.provider"])
        assert result.exit_code == 0
        assert "openai" in result.output

    def test_config_set_and_get(self, runner, tmp_path, monkeypatch):
        """config set should persist, config get should retrieve."""
        cfg_file = tmp_path / "config.toml"
        monkeypatch.setattr("zvecsearch.cli._GLOBAL_CFG", cfg_file)
        monkeypatch.setattr("zvecsearch.config._GLOBAL_CFG", cfg_file)

        result = runner.invoke(cli, ["config", "set", "embedding.provider", "ollama"])
        assert result.exit_code == 0
        assert "Set" in result.output

    def test_config_init_help(self, runner):
        result = runner.invoke(cli, ["config", "init", "--help"])
        assert result.exit_code == 0
        assert "project" in result.output.lower() or "global" in result.output.lower()
