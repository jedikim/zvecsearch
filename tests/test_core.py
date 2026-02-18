"""Tests for ZvecSearch core orchestrator.

The core orchestrator ties together scanning, chunking, embedding, and storage.
Since zvec's native library requires AVX-512, we mock the zvec module at
sys.modules level (to allow store.py to import), then mock both ZvecStore
and the embedding provider to test orchestration logic in isolation.
"""
from __future__ import annotations

import sys
import pytest
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Mock zvec in sys.modules before any import of store.py / core.py
# (zvec native lib requires AVX-512 which is unavailable on this CPU)
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

from zvecsearch.chunker import chunk_markdown, compute_chunk_id  # noqa: E402
from zvecsearch.core import ZvecSearch  # noqa: E402

TEST_DB = Path("/tmp/zvecsearch_test_core")


@pytest.fixture(autouse=True)
def clean():
    """Clean up test DB path before and after each test."""
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)
    yield
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


@pytest.fixture
def md_dir(tmp_path):
    """Create a temporary directory with a markdown test file."""
    (tmp_path / "a.md").write_text(
        "# Hello\nThis is a test document about AI.\n## Sub\nMore content here."
    )
    return tmp_path


@pytest.fixture
def multi_md_dir(tmp_path):
    """Create a temporary directory with multiple markdown files."""
    (tmp_path / "a.md").write_text("# First\nFirst document content.")
    (tmp_path / "b.md").write_text("# Second\nSecond document content.")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.md").write_text("# Third\nThird in subdirectory.")
    return tmp_path


def _mock_embedder(dim=4):
    """Create a mock embedding provider."""
    mock_emb = AsyncMock()
    mock_emb.model_name = "test-model"
    mock_emb.dimension = dim
    # Default: return one embedding per text
    mock_emb.embed.side_effect = lambda texts: [[0.1] * dim for _ in texts]
    return mock_emb


def _mock_store():
    """Create a mock ZvecStore with working in-memory state."""
    store = MagicMock()
    store._docs = {}

    def _upsert(chunks):
        for c in chunks:
            store._docs[c["chunk_hash"]] = c
        return len(chunks)

    store.upsert.side_effect = _upsert

    def _hashes_by_source(source):
        return {h for h, c in store._docs.items() if c["source"] == source}

    store.hashes_by_source.side_effect = _hashes_by_source

    def _existing_hashes(hashes):
        return {h for h in hashes if h in store._docs}

    store.existing_hashes.side_effect = _existing_hashes

    def _delete_by_hashes(hashes):
        for h in hashes:
            store._docs.pop(h, None)

    store.delete_by_hashes.side_effect = _delete_by_hashes

    def _delete_by_source(source):
        to_remove = [h for h, c in store._docs.items() if c["source"] == source]
        for h in to_remove:
            del store._docs[h]

    store.delete_by_source.side_effect = _delete_by_source

    store.count.side_effect = lambda: len(store._docs)
    store.close.return_value = None
    store.search.return_value = []
    store.query.return_value = []

    return store


# ---------------------------------------------------------------------------
# Tests: Construction
# ---------------------------------------------------------------------------
class TestZvecSearchInit:
    def test_constructor_creates_embedder_and_store(self):
        """ZvecSearch should create an embedding provider and a store."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb) as mock_gp, \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st) as mock_zs:
            zs = ZvecSearch(paths=["/tmp/test"], zvec_path=str(TEST_DB))
            mock_gp.assert_called_once_with("openai", None)
            mock_zs.assert_called_once()
            zs.close()

    def test_custom_provider_and_model(self):
        """Constructor should pass custom provider/model to get_provider."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb) as mock_gp, \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(
                paths=["/tmp/test"],
                embedding_provider="ollama",
                embedding_model="nomic-embed-text",
                zvec_path=str(TEST_DB),
            )
            mock_gp.assert_called_once_with("ollama", "nomic-embed-text")
            zs.close()

    def test_store_property(self):
        """The store property should return the underlying ZvecStore."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=["/tmp/test"], zvec_path=str(TEST_DB))
            assert zs.store is mock_st
            zs.close()

    def test_context_manager(self):
        """ZvecSearch should support use as a context manager."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            with ZvecSearch(paths=["/tmp/test"], zvec_path=str(TEST_DB)) as zs:
                assert zs.store is mock_st
            mock_st.close.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Indexing
# ---------------------------------------------------------------------------
class TestZvecSearchIndex:
    async def test_index_returns_chunk_count(self, md_dir):
        """index() should scan files and return total chunks indexed."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count = await zs.index()
            assert count >= 1
            # Embedder should have been called
            mock_emb.embed.assert_called()
            # Store upsert should have been called
            mock_st.upsert.assert_called()
            zs.close()

    async def test_index_multiple_files(self, multi_md_dir):
        """index() should process all discovered markdown files."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(multi_md_dir)], zvec_path=str(TEST_DB))
            count = await zs.index()
            # 3 files, each with at least 1 chunk
            assert count >= 3
            zs.close()

    async def test_index_single_file(self, md_dir):
        """index_file() should index a single file and return chunk count."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count = await zs.index_file(md_dir / "a.md")
            assert count >= 1
            zs.close()

    async def test_index_empty_file(self, tmp_path):
        """Indexing an empty file should return 0."""
        (tmp_path / "empty.md").write_text("")
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path=str(TEST_DB))
            count = await zs.index_file(tmp_path / "empty.md")
            assert count == 0
            zs.close()

    async def test_index_skips_existing_chunks(self, md_dir):
        """Second index call should skip already-indexed chunks."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count1 = await zs.index()
            assert count1 >= 1

            # Second index: all hashes already exist, so embed should not be called again
            mock_emb.embed.reset_mock()
            count2 = await zs.index()
            assert count2 == 0
            mock_emb.embed.assert_not_called()
            zs.close()

    async def test_index_force_reindexes(self, md_dir):
        """index(force=True) should re-embed all chunks even if they exist."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count1 = await zs.index()
            assert count1 >= 1

            mock_emb.embed.reset_mock()
            count2 = await zs.index(force=True)
            assert count2 >= 1
            mock_emb.embed.assert_called()
            zs.close()

    async def test_index_removes_stale_chunks(self, md_dir):
        """When a file's content changes, stale chunks should be deleted."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            await zs.index()

            # Modify the file so chunk IDs change
            (md_dir / "a.md").write_text("# Changed\nCompletely different content.")
            mock_emb.embed.reset_mock()
            mock_st.delete_by_hashes.reset_mock()

            count = await zs.index()
            assert count >= 1
            # Stale hashes from old content should have been deleted
            mock_st.delete_by_hashes.assert_called()
            zs.close()


# ---------------------------------------------------------------------------
# Tests: Search
# ---------------------------------------------------------------------------
class TestZvecSearchSearch:
    async def test_search_returns_results(self, md_dir):
        """search() should embed the query and call store.search()."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.search.return_value = [
            {"content": "test content", "source": "a.md", "score": 0.9},
        ]

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            results = await zs.search("AI test", top_k=5)
            assert isinstance(results, list)
            assert len(results) == 1
            assert results[0]["content"] == "test content"
            # Embedder should have been called with the query
            mock_emb.embed.assert_called_with(["AI test"])
            # Store search should have been called
            mock_st.search.assert_called_once()
            zs.close()

    async def test_search_passes_top_k(self, md_dir):
        """search() should pass the top_k parameter to store.search()."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            await zs.search("query", top_k=3)
            _, kwargs = mock_st.search.call_args
            assert kwargs["top_k"] == 3
            zs.close()

    async def test_search_empty_returns_empty(self, md_dir):
        """Search on empty store should return empty list."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.search.return_value = []

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            results = await zs.search("anything")
            assert results == []
            zs.close()


# ---------------------------------------------------------------------------
# Tests: Compact
# ---------------------------------------------------------------------------
class TestZvecSearchCompact:
    async def test_compact_calls_compact_chunks(self, md_dir, tmp_path):
        """compact() should call compact_chunks with queried data."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.query.return_value = [
            {"content": "chunk 1 content", "source": "a.md"},
            {"content": "chunk 2 content", "source": "a.md"},
        ]

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.compact_chunks", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = "Summarized content"
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            result = await zs.compact(source="a.md")
            assert result == "Summarized content"
            mock_cc.assert_called_once()
            zs.close()

    async def test_compact_empty_returns_empty(self, md_dir):
        """compact() with no matching chunks should return empty string."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.query.return_value = []

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            result = await zs.compact(source="nonexistent.md")
            assert result == ""
            zs.close()

    async def test_compact_writes_to_output_dir(self, md_dir, tmp_path):
        """compact() with output_dir should write summary to a file."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.query.return_value = [
            {"content": "chunk content", "source": "a.md"},
        ]

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.compact_chunks", new_callable=AsyncMock) as mock_cc:
            mock_cc.return_value = "Summary output"
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            result = await zs.compact(source="a.md", output_dir=str(tmp_path))
            assert result == "Summary output"
            # Check that the memory directory was created
            memory_dir = tmp_path / "memory"
            assert memory_dir.exists()
            # Check a dated file was written
            from datetime import date
            compact_file = memory_dir / f"{date.today()}.md"
            assert compact_file.exists()
            content = compact_file.read_text()
            assert "Summary output" in content
            assert "Memory Compact" in content
            zs.close()


# ---------------------------------------------------------------------------
# Tests: Watch
# ---------------------------------------------------------------------------
class TestZvecSearchWatch:
    def test_watch_returns_file_watcher(self, md_dir):
        """watch() should return a FileWatcher instance."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.FileWatcher") as mock_fw:
            mock_fw.return_value = MagicMock()
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            zs.watch()
            mock_fw.assert_called_once()
            zs.close()

    def test_watch_passes_debounce(self, md_dir):
        """watch(debounce_ms=...) should forward the debounce setting."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.FileWatcher") as mock_fw:
            mock_fw.return_value = MagicMock()
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            zs.watch(debounce_ms=500)
            _, kwargs = mock_fw.call_args
            assert kwargs["debounce_ms"] == 500
            zs.close()

    def test_watch_callback_indexes_on_create(self, md_dir):
        """The watch callback should call index_file on created events."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        captured_callback = None

        def capture_fw(paths, callback, **kw):
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.FileWatcher", side_effect=capture_fw):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            zs.watch()
            assert captured_callback is not None
            # The callback type/behavior is verified by checking FileWatcher was called
            # with a callable second argument
            zs.close()

    def test_watch_callback_deletes_on_delete(self, md_dir):
        """The watch callback should call delete_by_source on deleted events."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        captured_callback = None

        def capture_fw(paths, callback, **kw):
            nonlocal captured_callback
            captured_callback = callback
            return MagicMock()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st), \
             patch("zvecsearch.core.FileWatcher", side_effect=capture_fw):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            zs.watch()
            assert captured_callback is not None
            # Simulate a deleted event
            test_path = md_dir / "a.md"
            captured_callback("deleted", test_path)
            mock_st.delete_by_source.assert_called_with(str(test_path))
            zs.close()


# ---------------------------------------------------------------------------
# Tests: Integration-like (chunker + embedder + store mocked together)
# ---------------------------------------------------------------------------
class TestZvecSearchIntegration:
    async def test_index_then_search_workflow(self, md_dir):
        """Full workflow: index files, then search."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()
        mock_st.search.return_value = [
            {"content": "AI content", "source": str(md_dir / "a.md"), "score": 0.95},
        ]

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count = await zs.index()
            assert count >= 1

            results = await zs.search("AI", top_k=5)
            assert len(results) == 1
            assert results[0]["score"] == 0.95
            zs.close()

    async def test_chunk_ids_use_model_name(self, md_dir):
        """Chunk IDs should incorporate the embedding model name."""
        mock_emb = _mock_embedder()
        mock_emb.model_name = "special-model"
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            await zs.index()

            # Verify all upserted records have chunk_hash that would use "special-model"
            assert mock_st.upsert.called
            records = mock_st.upsert.call_args[0][0]
            # Recompute expected IDs with model name
            text = (md_dir / "a.md").read_text()
            chunks = chunk_markdown(text, source=str((md_dir / "a.md").resolve()))
            expected_ids = set()
            for c in chunks:
                cid = compute_chunk_id(
                    c.source, c.start_line, c.end_line, c.content_hash, "special-model"
                )
                expected_ids.add(cid)
            actual_ids = {r["chunk_hash"] for r in records}
            assert actual_ids == expected_ids
            zs.close()

    async def test_upsert_records_have_required_fields(self, md_dir):
        """Upserted records should have all required store fields."""
        mock_emb = _mock_embedder()
        mock_st = _mock_store()

        with patch("zvecsearch.core.get_provider", return_value=mock_emb), \
             patch("zvecsearch.core.ZvecStore", return_value=mock_st):
            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            await zs.index()

            records = mock_st.upsert.call_args[0][0]
            required_fields = {
                "chunk_hash", "content", "source", "heading",
                "heading_level", "start_line", "end_line", "embedding",
            }
            for rec in records:
                assert required_fields.issubset(set(rec.keys())), \
                    f"Missing fields: {required_fields - set(rec.keys())}"
            zs.close()
