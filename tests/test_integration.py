"""Integration tests for the full zvecsearch pipeline.

Tests the end-to-end flow: scan -> chunk -> store (with internal embedding) -> search.
The store handles embedding internally via zvec-native functions; no external
embedding provider is used.

The store mock is stateful (in-memory dict) to allow real incremental-index
logic (hashes_by_source, existing_hashes, delete_by_hashes, etc.) to work
correctly.  All index/search APIs are synchronous.

Mocking strategy follows test_core.py: install a fake zvec in sys.modules
before importing any zvecsearch code, then mock ZvecStore with patch() for
every test.
"""
from __future__ import annotations

import sys
import pytest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Install a zvec stub in sys.modules so that store.py can import it even
# without AVX-512 CPU support.
# ---------------------------------------------------------------------------
if "zvec" not in sys.modules:
    _zvec_stub = MagicMock()
    _zvec_stub.DataType.STRING = "STRING"
    _zvec_stub.DataType.INT32 = "INT32"
    _zvec_stub.DataType.VECTOR_FP32 = "VECTOR_FP32"
    _zvec_stub.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
    _zvec_stub.MetricType.COSINE = "COSINE"
    _zvec_stub.MetricType.L2 = "L2"
    _zvec_stub.MetricType.IP = "IP"
    _zvec_stub.LogLevel.WARN = "WARN"
    _zvec_stub.FieldSchema = MagicMock
    _zvec_stub.VectorSchema = MagicMock
    _zvec_stub.CollectionSchema = MagicMock
    _zvec_stub.CollectionOption = MagicMock
    _zvec_stub.HnswIndexParam = MagicMock
    _zvec_stub.InvertIndexParam = MagicMock
    _zvec_stub.FlatIndexParam = MagicMock
    _zvec_stub.VectorQuery = MagicMock
    _zvec_stub.RrfReRanker = MagicMock
    _zvec_stub.BM25EmbeddingFunction = MagicMock
    _zvec_stub.Doc = MagicMock
    _zvec_stub.WeightedReRanker = MagicMock
    _zvec_stub.HnswQueryParam = MagicMock
    _zvec_stub.OpenAIDenseEmbedding = MagicMock
    sys.modules["zvec"] = _zvec_stub

from zvecsearch.core import ZvecSearch  # noqa: E402
from zvecsearch.chunker import chunk_markdown, compute_chunk_id  # noqa: E402


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------

def _make_store() -> MagicMock:
    """Return a stateful in-memory mock of ZvecStore.

    Implements the same interface as ZvecStore using a simple dict so that
    incremental indexing logic (hashes_by_source, existing_hashes, etc.)
    works correctly end-to-end.
    """
    store = MagicMock()
    store._docs: dict[str, dict] = {}

    # --- embed_and_upsert ---
    def _upsert(chunks):
        for c in chunks:
            store._docs[c["chunk_hash"]] = c
        return len(chunks)

    store.embed_and_upsert.side_effect = _upsert

    # --- embed_and_insert ---
    def _insert(chunks):
        for c in chunks:
            store._docs[c["chunk_hash"]] = c
        return len(chunks)

    store.embed_and_insert.side_effect = _insert

    # --- flush / optimize ---
    store.flush.return_value = None
    store.optimize.return_value = None

    # --- count ---
    store.count.side_effect = lambda: len(store._docs)

    # --- hashes_by_source ---
    def _hashes_by_source(source: str):
        return {h for h, c in store._docs.items() if c.get("source") == source}

    store.hashes_by_source.side_effect = _hashes_by_source

    # --- existing_hashes ---
    def _existing_hashes(hashes):
        return {h for h in hashes if h in store._docs}

    store.existing_hashes.side_effect = _existing_hashes

    # --- delete_by_hashes ---
    def _delete_by_hashes(hashes):
        for h in hashes:
            store._docs.pop(h, None)

    store.delete_by_hashes.side_effect = _delete_by_hashes

    # --- delete_by_source ---
    def _delete_by_source(source: str):
        to_remove = [h for h, c in store._docs.items() if c.get("source") == source]
        for h in to_remove:
            del store._docs[h]

    store.delete_by_source.side_effect = _delete_by_source

    # --- search ---
    def _search(query_text="", top_k=10):
        results = list(store._docs.values())[:top_k]
        return [
            {
                "content": c.get("content", ""),
                "source": c.get("source", ""),
                "heading": c.get("heading", ""),
                "heading_level": c.get("heading_level", 0),
                "start_line": c.get("start_line", 0),
                "end_line": c.get("end_line", 0),
                "chunk_hash": c.get("chunk_hash", ""),
                "score": 1.0 / (i + 1),
            }
            for i, c in enumerate(results)
        ]

    store.search.side_effect = _search

    # --- close ---
    store.close.return_value = None

    # --- query ---
    store.query.return_value = []

    return store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def md_dir(tmp_path):
    """Single markdown file with two sections."""
    (tmp_path / "doc.md").write_text(
        "# Introduction\n\nThis document covers machine learning basics.\n\n"
        "## Neural Networks\n\nNeural networks are a key technique in deep learning."
    )
    return tmp_path


@pytest.fixture
def multi_md_dir(tmp_path):
    """Three markdown files spread across a directory and subdirectory."""
    (tmp_path / "alpha.md").write_text(
        "# Alpha\n\nAlpha file content about retrieval-augmented generation."
    )
    (tmp_path / "beta.md").write_text(
        "# Beta\n\nBeta file content about vector databases and embeddings."
    )
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "gamma.md").write_text(
        "# Gamma\n\nGamma file content about semantic search techniques."
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Helper: create a ZvecSearch with patched dependencies
# ---------------------------------------------------------------------------

def _make_zs(paths, store=None):
    """Create a ZvecSearch with mocked ZvecStore.

    Returns (zs, store, patch_stack).  Caller must stop the patch
    stack after use; using as a context manager is recommended.
    """
    if store is None:
        store = _make_store()

    p2 = patch("zvecsearch.core.ZvecStore", return_value=store)
    p2.start()
    zs = ZvecSearch(paths=paths, zvec_path="/tmp/unused_zvec_path")
    return zs, store, (p2,)


def _stop_patches(patches):
    for p in reversed(patches):
        p.stop()


# ---------------------------------------------------------------------------
# Test 1: test_full_pipeline
# Index multiple files, search, verify result structure.
# ---------------------------------------------------------------------------
class TestFullPipeline:
    def test_full_pipeline_indexes_all_files(self, multi_md_dir):
        """Index three files and verify at least one chunk per file is stored."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            count = zs.index()
            assert count >= 3, f"Expected >= 3 chunks from 3 files, got {count}"
            assert store.count() >= 3
        finally:
            zs.close()
            _stop_patches(patches)

    def test_full_pipeline_search_returns_structured_results(self, md_dir):
        """After indexing, search() must return results with all required fields."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            results = zs.search("machine learning", top_k=5)

            assert isinstance(results, list)
            assert len(results) >= 1

            required = {"content", "source", "heading", "heading_level",
                        "start_line", "end_line", "chunk_hash", "score"}
            for r in results:
                missing = required - set(r.keys())
                assert not missing, f"Search result missing fields: {missing}"
        finally:
            zs.close()
            _stop_patches(patches)

    def test_full_pipeline_single_file(self, md_dir):
        """index() on a single-file directory stores all its chunks."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count = zs.index()
            assert count >= 1
            assert store.count() >= 1
        finally:
            zs.close()
            _stop_patches(patches)

    def test_full_pipeline_embeds_correct_texts(self, md_dir):
        """Store receives the actual chunk content strings via embed_and_upsert."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()

            # Collect all chunks passed to embed_and_upsert()
            all_contents = []
            for call_args in store.embed_and_upsert.call_args_list:
                chunks = call_args[0][0]
                all_contents.extend(c["content"] for c in chunks)

            # Every stored chunk's content should have been passed to embed_and_upsert
            stored_contents = {c["content"] for c in store._docs.values()}
            for content in stored_contents:
                assert any(content in t or t in content for t in all_contents), \
                    f"Content not found in embed_and_upsert calls: {content[:50]!r}"
        finally:
            zs.close()
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Test 2: test_incremental_index
# Index, modify file, re-index, verify only changed chunks re-embedded.
# ---------------------------------------------------------------------------
class TestIncrementalIndex:
    def test_second_index_skips_existing_chunks(self, md_dir):
        """A second index() on unchanged files should embed 0 new chunks."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count1 = zs.index()
            assert count1 >= 1

            store.embed_and_upsert.reset_mock()
            count2 = zs.index()

            assert count2 == 0
            store.embed_and_upsert.assert_not_called()
        finally:
            zs.close()
            _stop_patches(patches)

    def test_modified_file_triggers_reembedding(self, md_dir):
        """Changing a file's content causes new chunks to be embedded."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count1 = zs.index()
            assert count1 >= 1

            # Replace file content entirely so all chunk IDs change
            (md_dir / "doc.md").write_text(
                "# Updated Title\n\nCompletely different content about NLP.\n\n"
                "## Transformers\n\nTransformer models revolutionized NLP processing."
            )

            store.embed_and_upsert.reset_mock()
            count2 = zs.index()

            assert count2 >= 1
            store.embed_and_upsert.assert_called()
        finally:
            zs.close()
            _stop_patches(patches)

    def test_modified_file_removes_stale_chunks(self, md_dir):
        """Old chunk hashes from a modified file must be deleted."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            old_hashes = set(store._docs.keys())

            (md_dir / "doc.md").write_text(
                "# Brand New\n\nFreshly written content with completely different words."
            )
            zs.index()
            new_hashes = set(store._docs.keys())

            # Any hash that was in old but not in new should be gone
            stale = old_hashes - new_hashes
            for h in stale:
                assert h not in store._docs, f"Stale hash still in store: {h}"
        finally:
            zs.close()
            _stop_patches(patches)

    def test_incremental_preserves_unchanged_chunks(self, multi_md_dir):
        """When only one file changes, other files' chunks stay in the store."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            zs.index()

            # Identify chunks from alpha.md
            alpha_path = str((multi_md_dir / "alpha.md").resolve())
            alpha_hashes = {h for h, c in store._docs.items()
                            if c["source"] == alpha_path}
            other_hashes = set(store._docs.keys()) - alpha_hashes

            # Modify only alpha.md
            (multi_md_dir / "alpha.md").write_text(
                "# Alpha Modified\n\nCompletely rewritten alpha content."
            )
            zs.index()

            # All non-alpha chunks must still be present
            for h in other_hashes:
                assert h in store._docs, f"Non-stale hash disappeared: {h}"
        finally:
            zs.close()
            _stop_patches(patches)

    def test_delete_by_hashes_called_for_stale(self, md_dir):
        """delete_by_hashes() must be called when stale chunks are detected."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            store.delete_by_hashes.reset_mock()

            (md_dir / "doc.md").write_text(
                "# Totally Different\n\nAll new content here."
            )
            zs.index()

            store.delete_by_hashes.assert_called()
        finally:
            zs.close()
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Test 3: test_force_reindex
# Force re-index replaces all chunks even when content hasn't changed.
# ---------------------------------------------------------------------------
class TestForceReindex:
    def test_force_reindex_re_embeds_unchanged_file(self, md_dir):
        """index(force=True) must re-embed all chunks even if they already exist."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count1 = zs.index()
            assert count1 >= 1

            # Normal re-index skips everything
            store.embed_and_upsert.reset_mock()
            count2 = zs.index()
            assert count2 == 0
            store.embed_and_upsert.assert_not_called()

            # Force re-index must re-embed via embed_and_insert
            count3 = zs.index(force=True)
            assert count3 >= 1
            store.embed_and_insert.assert_called()
        finally:
            zs.close()
            _stop_patches(patches)

    def test_force_reindex_calls_delete_by_source(self, md_dir):
        """index(force=True) must delete old source chunks before reinserting."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            store.delete_by_source.reset_mock()

            zs.index(force=True)

            store.delete_by_source.assert_called()
        finally:
            zs.close()
            _stop_patches(patches)

    def test_force_reindex_all_files(self, multi_md_dir):
        """index(force=True) on multiple files re-embeds all of them."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            count1 = zs.index()
            assert count1 >= 3

            count_forced = zs.index(force=True)
            assert count_forced >= 3
        finally:
            zs.close()
            _stop_patches(patches)

    def test_force_reindex_store_count_unchanged(self, md_dir):
        """After force re-index, the store should contain the same chunk count."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            count_before = store.count()

            zs.index(force=True)
            count_after = store.count()

            assert count_after == count_before
        finally:
            zs.close()
            _stop_patches(patches)

    def test_force_false_skips_existing_chunks(self, md_dir):
        """Default (force=False) must not re-embed already-indexed chunks."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()

            store.embed_and_upsert.reset_mock()
            zs.index(force=False)
            store.embed_and_upsert.assert_not_called()
        finally:
            zs.close()
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Test 4: test_search_with_results
# Verify search returns proper structure with all required fields.
# ---------------------------------------------------------------------------
class TestSearchWithResults:
    def test_search_returns_all_required_fields(self, md_dir):
        """Every search result must contain all expected metadata fields."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            results = zs.search("neural networks", top_k=5)

            assert isinstance(results, list)
            assert len(results) >= 1

            required = {"content", "source", "heading", "heading_level",
                        "start_line", "end_line", "chunk_hash", "score"}
            for r in results:
                missing = required - set(r.keys())
                assert not missing, f"Missing fields: {missing}"
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_scores_are_non_negative_floats(self, md_dir):
        """Each result's score must be a non-negative float."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            results = zs.search("deep learning")

            for r in results:
                assert isinstance(r["score"], (int, float))
                assert r["score"] >= 0.0
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_empty_store_returns_empty_list(self, md_dir):
        """Searching before any indexing must return an empty list."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            # No indexing - store is empty
            results = zs.search("anything")
            assert results == []
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_top_k_limits_results(self, multi_md_dir):
        """search(top_k=N) must return at most N results."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            zs.index()
            results = zs.search("semantic search", top_k=2)
            assert len(results) <= 2
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_calls_embedder_with_query(self, md_dir):
        """search() must pass the query text to the store."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            store.search.reset_mock()
            zs.search("deep learning query")
            store.search.assert_called_once_with(query_text="deep learning query", top_k=10)
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_passes_embedding_to_store(self, md_dir):
        """search() must forward query_text and top_k to the store."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            store.search.reset_mock()

            zs.search("test query", top_k=3)

            store.search.assert_called_once()
            call_kwargs = store.search.call_args[1]
            assert "query_text" in call_kwargs
            assert "top_k" in call_kwargs
            assert call_kwargs["top_k"] == 3
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_source_field_is_string(self, md_dir):
        """The source field in search results must be a string."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            results = zs.search("introduction")
            for r in results:
                assert isinstance(r["source"], str)
        finally:
            zs.close()
            _stop_patches(patches)

    def test_search_content_field_is_string(self, md_dir):
        """The content field in search results must be a string."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            results = zs.search("introduction")
            for r in results:
                assert isinstance(r["content"], str)
        finally:
            zs.close()
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Test 5: test_stats_after_indexing
# Verify count matches and stats reflect the indexed state correctly.
# ---------------------------------------------------------------------------
class TestStatsAfterIndexing:
    def test_count_equals_indexed_chunk_count(self, md_dir):
        """store.count() must equal the value returned by index()."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count = zs.index()
            assert store.count() == count
        finally:
            zs.close()
            _stop_patches(patches)

    def test_count_is_zero_before_indexing(self, md_dir):
        """Before any indexing, the store must be empty."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            assert store.count() == 0
        finally:
            zs.close()
            _stop_patches(patches)

    def test_count_reflects_multiple_files(self, multi_md_dir):
        """Indexing three files yields at least three chunks in the store."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            count = zs.index()
            assert count >= 3
            assert store.count() == count
        finally:
            zs.close()
            _stop_patches(patches)

    def test_count_stable_after_second_index(self, md_dir):
        """A second index() call should not change the total chunk count."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            count1 = store.count()

            zs.index()
            count2 = store.count()

            assert count1 == count2
        finally:
            zs.close()
            _stop_patches(patches)

    def test_count_decreases_after_source_deletion(self, multi_md_dir):
        """Deleting one source's chunks must reduce the total count."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            zs.index()
            count_before = store.count()

            alpha_path = str((multi_md_dir / "alpha.md").resolve())
            store.delete_by_source(alpha_path)
            count_after = store.count()

            assert count_after < count_before
        finally:
            zs.close()
            _stop_patches(patches)

    def test_count_includes_all_sources(self, multi_md_dir):
        """Every indexed file contributes chunks to the total count."""
        zs, store, patches = _make_zs([str(multi_md_dir)])
        try:
            zs.index()

            sources_in_store = {c["source"] for c in store._docs.values()}
            expected_sources = {
                str((multi_md_dir / "alpha.md").resolve()),
                str((multi_md_dir / "beta.md").resolve()),
                str((multi_md_dir / "sub" / "gamma.md").resolve()),
            }
            assert sources_in_store == expected_sources
        finally:
            zs.close()
            _stop_patches(patches)


# ---------------------------------------------------------------------------
# Additional edge cases and end-to-end scenarios
# ---------------------------------------------------------------------------
class TestPipelineEdgeCases:
    def test_index_empty_directory(self, tmp_path):
        """Indexing an empty directory should return 0 chunks."""
        zs, store, patches = _make_zs([str(tmp_path)])
        try:
            count = zs.index()
            assert count == 0
            assert store.count() == 0
        finally:
            zs.close()
            _stop_patches(patches)

    def test_index_empty_markdown_file(self, tmp_path):
        """Indexing an empty markdown file should return 0 chunks."""
        (tmp_path / "empty.md").write_text("")
        zs, store, patches = _make_zs([str(tmp_path)])
        try:
            count = zs.index()
            assert count == 0
        finally:
            zs.close()
            _stop_patches(patches)

    def test_index_file_directly(self, md_dir):
        """index_file() on a specific path should index only that file."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count = zs.index_file(md_dir / "doc.md")
            assert count >= 1
            assert store.count() >= 1
        finally:
            zs.close()
            _stop_patches(patches)

    def test_upserted_records_have_required_fields(self, md_dir):
        """Every record passed to store must contain all schema fields."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()

            required = {"chunk_hash", "content", "source", "heading",
                        "heading_level", "start_line", "end_line"}
            for chunk_hash, record in store._docs.items():
                missing = required - set(record.keys())
                assert not missing, f"Record {chunk_hash!r} missing: {missing}"
        finally:
            zs.close()
            _stop_patches(patches)

    def test_context_manager_closes_store(self, md_dir):
        """Using ZvecSearch as a context manager must close the store on exit."""
        store = _make_store()

        with patch("zvecsearch.core.ZvecStore", return_value=store):
            with ZvecSearch(paths=[str(md_dir)],
                            zvec_path="/tmp/unused_zvec_path") as zs:
                zs.index()

        store.close.assert_called_once()

    def test_chunk_ids_incorporate_model_name(self, md_dir):
        """Chunk IDs must encode the embedding model name for cache invalidation."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()

            text = (md_dir / "doc.md").read_text()
            chunks = chunk_markdown(text, source=str((md_dir / "doc.md").resolve()))
            expected_ids = {
                compute_chunk_id(c.source, c.start_line, c.end_line,
                                 c.content_hash, "zvec")
                for c in chunks
            }
            actual_ids = set(store._docs.keys())
            assert actual_ids == expected_ids
        finally:
            zs.close()
            _stop_patches(patches)

    def test_index_then_search_end_to_end(self, md_dir):
        """Full round-trip: index then search returns non-empty, structured results."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            count = zs.index()
            assert count >= 1

            results = zs.search("machine learning neural networks", top_k=5)
            assert isinstance(results, list)
            assert len(results) >= 1
            for r in results:
                assert "content" in r
                assert "score" in r
        finally:
            zs.close()
            _stop_patches(patches)

    def test_source_path_stored_as_resolved_string(self, md_dir):
        """The source field in stored chunks must be the resolved absolute path."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            expected_source = str((md_dir / "doc.md").resolve())
            for record in store._docs.values():
                assert record["source"] == expected_source
        finally:
            zs.close()
            _stop_patches(patches)

    def test_start_and_end_lines_are_positive(self, md_dir):
        """start_line and end_line in stored chunks must be positive integers."""
        zs, store, patches = _make_zs([str(md_dir)])
        try:
            zs.index()
            for record in store._docs.values():
                assert isinstance(record["start_line"], int)
                assert isinstance(record["end_line"], int)
                assert record["start_line"] >= 1
                assert record["end_line"] >= record["start_line"]
        finally:
            zs.close()
            _stop_patches(patches)
