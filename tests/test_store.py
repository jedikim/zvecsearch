"""Tests for ZvecStore - zvec-native storage with embedded embedding.

Since zvec's native C++ backend requires specific CPU features (AVX-512) that may
not be available in all environments, these tests mock the zvec dependency and verify
that ZvecStore correctly orchestrates zvec API calls.
"""
import pytest
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeDoc:
    """Minimal Doc stand-in for testing."""
    def __init__(self, id, fields=None, vectors=None, score=None):
        self.id = id
        self.fields = fields or {}
        self.vectors = vectors or {}
        self.score = score or 0.0

    def field(self, name):
        return self.fields.get(name)


class FakeStatus:
    """Simulates a zvec status return from upsert/insert."""
    def __init__(self, ok=True, code=0, message=""):
        self._ok = ok
        self._code = code
        self._message = message

    def ok(self):
        return self._ok

    def code(self):
        return self._code

    def message(self):
        return self._message


# ---------------------------------------------------------------------------
# Build a comprehensive mock of the zvec module so that store.py can import
# it even when the real native library is unavailable.
# ---------------------------------------------------------------------------

_zvec_stub = MagicMock()

# Enums
_zvec_stub.DataType.STRING = "STRING"
_zvec_stub.DataType.INT32 = "INT32"
_zvec_stub.DataType.VECTOR_FP32 = "VECTOR_FP32"
_zvec_stub.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
_zvec_stub.MetricType.COSINE = "COSINE"
_zvec_stub.MetricType.L2 = "L2"
_zvec_stub.MetricType.IP = "IP"
_zvec_stub.LogLevel.WARN = "WARN"

# Classes that store.py instantiates
_zvec_stub.FieldSchema = MagicMock
_zvec_stub.VectorSchema = MagicMock
_zvec_stub.CollectionSchema = MagicMock
_zvec_stub.CollectionOption = MagicMock
_zvec_stub.HnswIndexParam = MagicMock
_zvec_stub.InvertIndexParam = MagicMock
_zvec_stub.VectorQuery = MagicMock
_zvec_stub.RrfReRanker = MagicMock
_zvec_stub.WeightedReRanker = MagicMock
_zvec_stub.HnswQueryParam = MagicMock
_zvec_stub.OpenAIDenseEmbedding = MagicMock
_zvec_stub.Doc = FakeDoc

# Patch zvec in sys.modules before importing store
sys.modules["zvec"] = _zvec_stub

from zvecsearch.store import ZvecStore, GeminiDenseEmbedding  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TEST_DB = Path("/tmp/zvecsearch_test_db")


def _make_mock_collection():
    """Create a mock collection that simulates zvec Collection behavior."""
    collection = MagicMock()
    collection._docs = {}
    collection._destroyed = False

    # stats property returns an object with doc_count
    def _stats():
        s = SimpleNamespace()
        s.doc_count = len(collection._docs)
        return s
    type(collection).stats = property(lambda self: _stats())

    # upsert: store docs by id and return FakeStatus list
    def _upsert(docs):
        if not isinstance(docs, list):
            docs = [docs]
        statuses = []
        for doc in docs:
            collection._docs[doc.id] = doc
            statuses.append(FakeStatus(ok=True))
        return statuses
    collection.upsert.side_effect = _upsert

    # insert: store docs by id and return FakeStatus list
    def _insert(docs):
        if not isinstance(docs, list):
            docs = [docs]
        statuses = []
        for doc in docs:
            collection._docs[doc.id] = doc
            statuses.append(FakeStatus(ok=True))
        return statuses
    collection.insert.side_effect = _insert

    # flush: no-op
    collection.flush.return_value = None

    # optimize: no-op
    collection.optimize.return_value = None

    # delete: remove by ids
    def _delete(ids):
        if isinstance(ids, str):
            ids = [ids]
        statuses = []
        for id_ in ids:
            status = FakeStatus(ok=(id_ in collection._docs))
            if id_ in collection._docs:
                del collection._docs[id_]
            statuses.append(status)
        return statuses
    collection.delete.side_effect = _delete

    # delete_by_filter: parse simple 'source == "X"' pattern
    def _delete_by_filter(filter_expr):
        import re
        m = re.search(r'source\s*==\s*"([^"]+)"', filter_expr)
        if m:
            source_val = m.group(1)
            to_remove = [
                did for did, doc in collection._docs.items()
                if doc.fields.get("source") == source_val
            ]
            for did in to_remove:
                del collection._docs[did]
    collection.delete_by_filter.side_effect = _delete_by_filter

    # fetch: return docs by ids
    def _fetch(ids):
        if isinstance(ids, str):
            ids = [ids]
        return {id_: collection._docs[id_] for id_ in ids if id_ in collection._docs}
    collection.fetch.side_effect = _fetch

    # query: simulate search - returns docs matching filter or all docs
    def _query(vectors=None, topk=10, filter=None, output_fields=None,
               reranker=None, query_param=None):
        import re
        results = list(collection._docs.values())
        if filter:
            m = re.search(r'source\s*==\s*"([^"]+)"', filter)
            if m:
                source_val = m.group(1)
                results = [d for d in results if d.fields.get("source") == source_val]
        scored = []
        for i, doc in enumerate(results[:topk]):
            doc.score = 1.0 / (i + 1)
            scored.append(doc)
        return scored
    collection.query.side_effect = _query

    # destroy
    def _destroy():
        collection._docs.clear()
        collection._destroyed = True
    collection.destroy.side_effect = _destroy

    return collection


@pytest.fixture(autouse=True)
def clean_db():
    """Clean up test DB path before and after each test."""
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)
    yield
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


@pytest.fixture
def store():
    """Set up mocked zvec environment and return (ZvecStore, mock_collection)."""
    import zvecsearch.store as store_module
    store_module._zvec_initialized = False

    mock_collection = _make_mock_collection()

    with patch("zvecsearch.store.zvec") as zvec_mod:
        # Wire up the module mock
        zvec_mod.DataType = _zvec_stub.DataType
        zvec_mod.MetricType = _zvec_stub.MetricType
        zvec_mod.LogLevel = _zvec_stub.LogLevel
        zvec_mod.FieldSchema = MagicMock
        zvec_mod.VectorSchema = MagicMock
        zvec_mod.CollectionSchema = MagicMock
        zvec_mod.CollectionOption = MagicMock
        zvec_mod.HnswIndexParam = MagicMock
        zvec_mod.InvertIndexParam = MagicMock
        zvec_mod.VectorQuery = MagicMock
        zvec_mod.RrfReRanker = MagicMock()
        zvec_mod.WeightedReRanker = MagicMock()
        zvec_mod.HnswQueryParam = MagicMock()
        zvec_mod.Doc = FakeDoc

        # Dense embedding mock (default = DefaultLocalDenseEmbedding)
        dense_emb_instance = MagicMock()
        dense_emb_instance.embed.side_effect = lambda text: [0.1, 0.2, 0.3, 0.4]
        dense_emb_instance.dim = 4
        dense_emb_instance.dimension = 4
        zvec_mod.DefaultLocalDenseEmbedding = MagicMock(return_value=dense_emb_instance)
        zvec_mod.OpenAIDenseEmbedding = MagicMock(return_value=dense_emb_instance)

        # Sparse embedding mock (default = DefaultLocalSparseEmbedding)
        sparse_emb_instance = MagicMock()
        sparse_emb_instance.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
        zvec_mod.DefaultLocalSparseEmbedding = MagicMock(return_value=sparse_emb_instance)

        # BM25 mock (for non-default providers)
        bm25_doc_instance = MagicMock()
        bm25_doc_instance.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
        bm25_query_instance = MagicMock()
        bm25_query_instance.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
        _bm25_instances = iter([bm25_doc_instance, bm25_query_instance])
        zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(_bm25_instances))

        # Reranker mock (default = DefaultLocalReRanker)
        zvec_mod.DefaultLocalReRanker = MagicMock()

        # create_and_open / open return our mock collection
        zvec_mod.create_and_open.return_value = mock_collection
        zvec_mod.open.return_value = mock_collection
        zvec_mod.init.return_value = None

        s = ZvecStore(
            path=str(TEST_DB),
            collection="test_chunks",
        )
        yield s, mock_collection, zvec_mod


def _sample_chunks(n=3):
    """Generate sample chunk dicts for testing (no embedding field)."""
    chunks = []
    for i in range(n):
        chunks.append({
            "chunk_hash": f"hash_{i}",
            "content": f"This is test content number {i} about topic {i}",
            "source": "test.md",
            "heading": f"Section {i}",
            "heading_level": 1,
            "start_line": i * 10 + 1,
            "end_line": i * 10 + 10,
        })
    return chunks


# ---------------------------------------------------------------------------
# Tests: Creation
# ---------------------------------------------------------------------------
class TestZvecStoreCreation:
    def test_constructor_sets_path(self, store):
        s, _, _ = store
        assert "test_chunks" in s._path

    def test_dense_emb_initialized(self, store):
        s, _, zvec_mod = store
        zvec_mod.DefaultLocalDenseEmbedding.assert_called_once()
        assert s._dense_emb is not None

    def test_bm25_doc_initialized(self, store):
        s, _, _ = store
        assert s._bm25_doc is not None

    def test_bm25_query_initialized(self, store):
        s, _, _ = store
        assert s._bm25_query is not None

    def test_zvec_init_called(self, store):
        _, _, zvec_mod = store
        zvec_mod.init.assert_called()

    def test_create_and_open_called(self, store):
        _, _, zvec_mod = store
        zvec_mod.create_and_open.assert_called_once()

    def test_default_reranker_is_default(self, store):
        s, _, _ = store
        assert s._reranker_type == "default"

    def test_default_quantize_type(self, store):
        s, _, _ = store
        assert s._quantize_type == "int8"

    def test_initial_count_is_zero(self, store):
        s, _, _ = store
        assert s.count() == 0


# ---------------------------------------------------------------------------
# Tests: embed_and_upsert
# ---------------------------------------------------------------------------
class TestEmbedAndUpsert:
    def test_returns_count(self, store):
        s, _, _ = store
        count = s.embed_and_upsert(_sample_chunks(3))
        assert count == 3

    def test_calls_dense_emb_embed(self, store):
        s, _, _ = store
        chunks = _sample_chunks(2)
        s.embed_and_upsert(chunks)
        assert s._dense_emb.embed.call_count == 2

    def test_calls_bm25_doc_embed(self, store):
        s, _, _ = store
        chunks = _sample_chunks(2)
        s.embed_and_upsert(chunks)
        assert s._bm25_doc.embed.call_count == 2

    def test_does_not_flush(self, store):
        s, mock_col, _ = store
        mock_col.flush.reset_mock()
        s.embed_and_upsert(_sample_chunks(2))
        mock_col.flush.assert_not_called()

    def test_empty_chunks_returns_zero(self, store):
        s, _, _ = store
        count = s.embed_and_upsert([])
        assert count == 0

    def test_stores_docs_in_collection(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        assert s.count() == 3

    def test_upsert_is_idempotent(self, store):
        s, _, _ = store
        chunks = _sample_chunks(2)
        s.embed_and_upsert(chunks)
        s.embed_and_upsert(chunks)  # same hashes
        assert s.count() == 2

    def test_doc_has_correct_fields(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(1))
        doc = mock_col._docs["hash_0"]
        assert doc.id == "hash_0"
        assert doc.fields["content"] == "This is test content number 0 about topic 0"
        assert doc.fields["source"] == "test.md"
        assert doc.fields["heading"] == "Section 0"
        assert doc.fields["heading_level"] == 1
        assert doc.fields["start_line"] == 1
        assert doc.fields["end_line"] == 10

    def test_doc_has_dense_and_sparse_vectors(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(1))
        doc = mock_col._docs["hash_0"]
        assert "embedding" in doc.vectors
        assert "sparse_embedding" in doc.vectors

    def test_calls_collection_upsert(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(1))
        mock_col.upsert.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: embed_and_insert
# ---------------------------------------------------------------------------
class TestEmbedAndInsert:
    def test_returns_count(self, store):
        s, _, _ = store
        count = s.embed_and_insert(_sample_chunks(3))
        assert count == 3

    def test_uses_collection_insert(self, store):
        s, mock_col, _ = store
        s.embed_and_insert(_sample_chunks(2))
        mock_col.insert.assert_called_once()
        mock_col.upsert.assert_not_called()

    def test_does_not_flush(self, store):
        s, mock_col, _ = store
        mock_col.flush.reset_mock()
        s.embed_and_insert(_sample_chunks(2))
        mock_col.flush.assert_not_called()

    def test_empty_chunks_returns_zero(self, store):
        s, _, _ = store
        count = s.embed_and_insert([])
        assert count == 0

    def test_calls_dense_emb_embed(self, store):
        s, _, _ = store
        s.embed_and_insert(_sample_chunks(2))
        assert s._dense_emb.embed.call_count == 2


# ---------------------------------------------------------------------------
# Tests: Search
# ---------------------------------------------------------------------------
class TestSearch:
    def test_returns_results(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(5))
        results = s.search("topic", top_k=3)
        assert len(results) <= 3
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)

    def test_empty_collection_returns_empty(self, store):
        s, _, _ = store
        results = s.search("anything", top_k=5)
        assert results == []

    def test_calls_dense_emb_embed_for_query(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(2))
        s._dense_emb.embed.reset_mock()
        s.search("test query")
        s._dense_emb.embed.assert_called_once_with("test query")

    def test_calls_bm25_query_embed(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(2))
        s.search("test query")
        s._bm25_query.embed.assert_called()

    def test_uses_default_local_reranker_by_default(self, store):
        s, _, zvec_mod = store
        s.embed_and_upsert(_sample_chunks(2))
        s.search("test query")
        zvec_mod.DefaultLocalReRanker.assert_called()

    def test_result_has_expected_keys(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(2))
        results = s.search("test")
        expected_keys = {"content", "source", "heading", "heading_level",
                         "start_line", "end_line", "chunk_hash", "score"}
        assert set(results[0].keys()) == expected_keys

    def test_uses_hnsw_query_param(self, store):
        s, _, zvec_mod = store
        s.embed_and_upsert(_sample_chunks(2))
        s.search("test")
        zvec_mod.HnswQueryParam.assert_called_with(ef=300)


# ---------------------------------------------------------------------------
# Tests: Flush and Optimize
# ---------------------------------------------------------------------------
class TestFlushAndOptimize:
    def test_flush_calls_collection_flush(self, store):
        s, mock_col, _ = store
        mock_col.flush.reset_mock()
        s.flush()
        mock_col.flush.assert_called_once()

    def test_optimize_calls_collection_optimize(self, store):
        s, mock_col, _ = store
        s.optimize()
        mock_col.optimize.assert_called_once()

    def test_flush_no_collection_does_not_error(self, store):
        s, _, _ = store
        s._collection = None
        s.flush()  # should not raise

    def test_optimize_no_collection_does_not_error(self, store):
        s, _, _ = store
        s._collection = None
        s.optimize()  # should not raise


# ---------------------------------------------------------------------------
# Tests: Delete Operations
# ---------------------------------------------------------------------------
class TestDeleteOperations:
    def test_delete_by_source(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        s.delete_by_source("test.md")
        assert s.count() == 0

    def test_delete_by_source_does_not_flush(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        mock_col.flush.reset_mock()
        s.delete_by_source("test.md")
        mock_col.flush.assert_not_called()

    def test_delete_by_hashes(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        s.delete_by_hashes(["hash_0", "hash_1"])
        assert s.count() == 1

    def test_delete_by_hashes_empty(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        s.delete_by_hashes([])
        assert s.count() == 3
        mock_col.delete.assert_not_called()

    def test_delete_by_hashes_does_not_flush(self, store):
        s, mock_col, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        mock_col.flush.reset_mock()
        s.delete_by_hashes(["hash_0"])
        mock_col.flush.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Query and Count
# ---------------------------------------------------------------------------
class TestQueryAndCount:
    def test_count_returns_doc_count(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        assert s.count() == 3

    def test_hashes_by_source(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        hashes = s.hashes_by_source("test.md")
        assert hashes == {"hash_0", "hash_1", "hash_2"}

    def test_indexed_sources(self, store):
        s, _, _ = store
        chunks = _sample_chunks(2)
        chunks[1]["source"] = "other.md"
        s.embed_and_upsert(chunks)
        sources = s.indexed_sources()
        assert sources == {"test.md", "other.md"}

    def test_existing_hashes(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        found = s.existing_hashes(["hash_0", "hash_1", "nonexistent"])
        assert found == {"hash_0", "hash_1"}

    def test_existing_hashes_empty_input(self, store):
        s, _, _ = store
        found = s.existing_hashes([])
        assert found == set()

    def test_query_with_filter(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        results = s.query(filter_expr='source == "test.md"')
        assert len(results) == 3
        assert all("content" in r for r in results)

    def test_query_empty_filter(self, store):
        s, _, _ = store
        s.embed_and_upsert(_sample_chunks(3))
        results = s.query(filter_expr="")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: Drop and Close
# ---------------------------------------------------------------------------
class TestDropAndClose:
    def test_drop_destroys_collection(self, store):
        s, mock_col, _ = store
        s.drop()
        mock_col.destroy.assert_called_once()
        assert s._collection is None

    def test_close_flushes_collection(self, store):
        s, mock_col, _ = store
        mock_col.flush.reset_mock()
        s.close()
        mock_col.flush.assert_called_once()
        assert s._collection is None

    def test_drop_with_no_collection(self, store):
        s, _, _ = store
        s._collection = None
        s.drop()  # should not raise

    def test_close_with_no_collection(self, store):
        s, _, _ = store
        s._collection = None
        s.close()  # should not raise


# ---------------------------------------------------------------------------
# Tests: Status checking
# ---------------------------------------------------------------------------
class TestCheckStatuses:
    def test_ok_statuses_no_error(self, store):
        s, _, _ = store
        statuses = [FakeStatus(ok=True), FakeStatus(ok=True)]
        s._check_statuses(statuses, "test")  # should not raise

    def test_failed_status_logs_error(self, store, caplog):
        import logging
        s, _, _ = store
        statuses = [FakeStatus(ok=False, code=1, message="fail")]
        with caplog.at_level(logging.ERROR, logger="zvecsearch.store"):
            s._check_statuses(statuses, "upsert")
        assert "upsert failed" in caplog.text

    def test_none_statuses_no_error(self, store):
        s, _, _ = store
        s._check_statuses(None, "test")  # should not raise


# ---------------------------------------------------------------------------
# Tests: GeminiDenseEmbedding
# ---------------------------------------------------------------------------
class TestGeminiDenseEmbedding:
    """GeminiDenseEmbedding 단위 테스트 (google-genai는 mock)."""

    def _make_mock_genai(self, embedding_values=None):
        """google.genai mock을 만들어 반환 (google_mock, genai_mock) 튜플."""
        if embedding_values is None:
            embedding_values = [0.1] * 768

        mock_genai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = embedding_values
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_genai.Client.return_value.models.embed_content.return_value = mock_result

        # from google import genai → sys.modules["google"].genai
        mock_google = MagicMock()
        mock_google.genai = mock_genai
        return mock_google, mock_genai

    def test_requires_api_key(self):
        """API 키 없으면 ValueError."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("os.environ", {}, clear=True), \
             patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            with pytest.raises(ValueError, match="Google API key required"):
                GeminiDenseEmbedding(api_key=None)

    def test_accepts_explicit_api_key(self):
        """명시적 api_key 전달 시 정상 생성."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="test-key")
            assert emb.dimension == 768

    def test_reads_env_api_key(self):
        """GOOGLE_API_KEY 환경변수에서 키 읽기."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "env-key"}), \
             patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding()
            assert emb.dimension == 768

    def test_dimension_property(self):
        """dimension property가 설정값을 반환."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="k", dimension=512)
            assert emb.dimension == 512

    def test_embed_returns_vector(self):
        """embed()가 float 리스트를 반환."""
        expected = [0.5] * 768
        mock_google, mock_genai = self._make_mock_genai(embedding_values=expected)
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="k")
            result = emb.embed("테스트 문장")
            assert result == expected

    def test_embed_calls_genai_api(self):
        """embed()가 genai API를 올바른 파라미터로 호출."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(model="gemini-embedding-001", dimension=768, api_key="k")
            emb.embed("hello world")
            emb._client.models.embed_content.assert_called_once_with(
                model="gemini-embedding-001",
                contents=["hello world"],
                config={"output_dimensionality": 768},
            )

    def test_embed_rejects_non_string(self):
        """비문자열 입력 시 TypeError."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="k")
            with pytest.raises(TypeError, match="Expected 'input' to be str"):
                emb.embed(123)

    def test_embed_rejects_empty_string(self):
        """빈 문자열 입력 시 ValueError."""
        mock_google, mock_genai = self._make_mock_genai()
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="k")
            with pytest.raises(ValueError, match="empty or whitespace"):
                emb.embed("   ")

    def test_callable(self):
        """__call__이 embed()를 호출."""
        expected = [0.1] * 768
        mock_google, mock_genai = self._make_mock_genai(embedding_values=expected)
        with patch.dict("sys.modules", {"google": mock_google, "google.genai": mock_genai}):
            emb = GeminiDenseEmbedding(api_key="k")
            result = emb("테스트")
            assert result == expected

    def test_import_error_without_genai(self):
        """google-genai 미설치 시 ImportError."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "google" or name == "google.genai":
                raise ImportError("No module named 'google'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="google-genai"):
                GeminiDenseEmbedding(api_key="k")


class TestStoreGeminiProvider:
    """ZvecStore에서 provider='google' 선택 시 GeminiDenseEmbedding 사용 확인."""

    def test_google_provider_uses_gemini_embedding(self):
        """embedding_provider='google'일 때 GeminiDenseEmbedding 인스턴스 사용."""
        import zvecsearch.store as store_module
        store_module._zvec_initialized = False

        mock_collection = _make_mock_collection()
        mock_genai = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.values = [0.1] * 768
        mock_result = MagicMock()
        mock_result.embeddings = [mock_embedding]
        mock_genai.Client.return_value.models.embed_content.return_value = mock_result

        with patch("zvecsearch.store.zvec") as zvec_mod, \
             patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}), \
             patch.dict("sys.modules", {"google": MagicMock(), "google.genai": mock_genai}):
            zvec_mod.DataType = _zvec_stub.DataType
            zvec_mod.MetricType = _zvec_stub.MetricType
            zvec_mod.LogLevel = _zvec_stub.LogLevel
            zvec_mod.FieldSchema = MagicMock
            zvec_mod.VectorSchema = MagicMock
            zvec_mod.CollectionSchema = MagicMock
            zvec_mod.CollectionOption = MagicMock
            zvec_mod.HnswIndexParam = MagicMock
            zvec_mod.InvertIndexParam = MagicMock
            zvec_mod.VectorQuery = MagicMock
            zvec_mod.RrfReRanker = MagicMock()
            zvec_mod.WeightedReRanker = MagicMock()
            zvec_mod.HnswQueryParam = MagicMock()
            zvec_mod.Doc = FakeDoc

            bm25_doc = MagicMock()
            bm25_doc.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            bm25_query = MagicMock()
            bm25_query.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            _bm25_iter = iter([bm25_doc, bm25_query])
            zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(_bm25_iter))

            zvec_mod.create_and_open.return_value = mock_collection
            zvec_mod.open.return_value = mock_collection
            zvec_mod.init.return_value = None

            s = ZvecStore(
                path=str(TEST_DB),
                collection="test_gemini",
                embedding_provider="google",
                embedding_model="gemini-embedding-001",
            )

            assert isinstance(s._dense_emb, GeminiDenseEmbedding)
            assert s._dense_emb.dimension == 768

            # OpenAIDenseEmbedding은 호출되지 않아야 함
            zvec_mod.OpenAIDenseEmbedding.assert_not_called()


class TestStoreDefaultProvider:
    """ZvecStore에서 provider='default' 선택 시 DefaultLocalDenseEmbedding 사용 확인."""

    def test_default_provider_uses_local_embedding(self):
        """embedding_provider='default'일 때 DefaultLocalDenseEmbedding 사용."""
        import zvecsearch.store as store_module
        store_module._zvec_initialized = False

        mock_collection = _make_mock_collection()
        mock_default_emb = MagicMock()
        mock_default_emb.embed.side_effect = lambda text: [0.1] * 384
        mock_default_emb.dimension = 384

        with patch("zvecsearch.store.zvec") as zvec_mod:
            zvec_mod.DataType = _zvec_stub.DataType
            zvec_mod.MetricType = _zvec_stub.MetricType
            zvec_mod.LogLevel = _zvec_stub.LogLevel
            zvec_mod.FieldSchema = MagicMock
            zvec_mod.VectorSchema = MagicMock
            zvec_mod.CollectionSchema = MagicMock
            zvec_mod.CollectionOption = MagicMock
            zvec_mod.HnswIndexParam = MagicMock
            zvec_mod.InvertIndexParam = MagicMock
            zvec_mod.VectorQuery = MagicMock
            zvec_mod.RrfReRanker = MagicMock()
            zvec_mod.WeightedReRanker = MagicMock()
            zvec_mod.HnswQueryParam = MagicMock()
            zvec_mod.Doc = FakeDoc
            zvec_mod.DefaultLocalDenseEmbedding = MagicMock(return_value=mock_default_emb)

            bm25_doc = MagicMock()
            bm25_doc.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            bm25_query = MagicMock()
            bm25_query.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            _bm25_iter = iter([bm25_doc, bm25_query])
            zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(_bm25_iter))

            zvec_mod.create_and_open.return_value = mock_collection
            zvec_mod.open.return_value = mock_collection
            zvec_mod.init.return_value = None

            s = ZvecStore(
                path=str(TEST_DB),
                collection="test_default",
                embedding_provider="default",
            )

            # DefaultLocalDenseEmbedding이 호출되어야 함
            zvec_mod.DefaultLocalDenseEmbedding.assert_called_once()
            assert s._dense_emb is mock_default_emb

            # OpenAIDenseEmbedding과 GeminiDenseEmbedding은 호출되지 않아야 함
            zvec_mod.OpenAIDenseEmbedding.assert_not_called()

    def test_default_provider_embed_and_upsert(self):
        """embedding_provider='default'로 embed_and_upsert 동작 확인."""
        import zvecsearch.store as store_module
        store_module._zvec_initialized = False

        mock_collection = _make_mock_collection()
        mock_default_emb = MagicMock()
        mock_default_emb.embed.side_effect = lambda text: [0.1] * 384
        mock_default_emb.dimension = 384

        with patch("zvecsearch.store.zvec") as zvec_mod:
            zvec_mod.DataType = _zvec_stub.DataType
            zvec_mod.MetricType = _zvec_stub.MetricType
            zvec_mod.LogLevel = _zvec_stub.LogLevel
            zvec_mod.FieldSchema = MagicMock
            zvec_mod.VectorSchema = MagicMock
            zvec_mod.CollectionSchema = MagicMock
            zvec_mod.CollectionOption = MagicMock
            zvec_mod.HnswIndexParam = MagicMock
            zvec_mod.InvertIndexParam = MagicMock
            zvec_mod.VectorQuery = MagicMock
            zvec_mod.RrfReRanker = MagicMock()
            zvec_mod.WeightedReRanker = MagicMock()
            zvec_mod.HnswQueryParam = MagicMock()
            zvec_mod.Doc = FakeDoc
            zvec_mod.DefaultLocalDenseEmbedding = MagicMock(return_value=mock_default_emb)

            bm25_doc = MagicMock()
            bm25_doc.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            bm25_query = MagicMock()
            bm25_query.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            _bm25_iter = iter([bm25_doc, bm25_query])
            zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(_bm25_iter))

            zvec_mod.create_and_open.return_value = mock_collection
            zvec_mod.open.return_value = mock_collection
            zvec_mod.init.return_value = None

            s = ZvecStore(
                path=str(TEST_DB),
                collection="test_default_upsert",
                embedding_provider="default",
            )

            chunks = _sample_chunks(3)
            count = s.embed_and_upsert(chunks)
            assert count == 3
            assert mock_default_emb.embed.call_count == 3

    def test_default_provider_search(self):
        """embedding_provider='default'로 search 동작 확인."""
        import zvecsearch.store as store_module
        store_module._zvec_initialized = False

        mock_collection = _make_mock_collection()
        mock_default_emb = MagicMock()
        mock_default_emb.embed.side_effect = lambda text: [0.1] * 384
        mock_default_emb.dimension = 384

        with patch("zvecsearch.store.zvec") as zvec_mod:
            zvec_mod.DataType = _zvec_stub.DataType
            zvec_mod.MetricType = _zvec_stub.MetricType
            zvec_mod.LogLevel = _zvec_stub.LogLevel
            zvec_mod.FieldSchema = MagicMock
            zvec_mod.VectorSchema = MagicMock
            zvec_mod.CollectionSchema = MagicMock
            zvec_mod.CollectionOption = MagicMock
            zvec_mod.HnswIndexParam = MagicMock
            zvec_mod.InvertIndexParam = MagicMock
            zvec_mod.VectorQuery = MagicMock
            zvec_mod.RrfReRanker = MagicMock()
            zvec_mod.WeightedReRanker = MagicMock()
            zvec_mod.HnswQueryParam = MagicMock()
            zvec_mod.Doc = FakeDoc
            zvec_mod.DefaultLocalDenseEmbedding = MagicMock(return_value=mock_default_emb)

            bm25_doc = MagicMock()
            bm25_doc.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            bm25_query = MagicMock()
            bm25_query.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
            _bm25_iter = iter([bm25_doc, bm25_query])
            zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(_bm25_iter))

            zvec_mod.create_and_open.return_value = mock_collection
            zvec_mod.open.return_value = mock_collection
            zvec_mod.init.return_value = None

            s = ZvecStore(
                path=str(TEST_DB),
                collection="test_default_search",
                embedding_provider="default",
            )

            # Insert docs first
            s.embed_and_upsert(_sample_chunks(3))
            mock_default_emb.embed.reset_mock()

            # Search
            results = s.search("test query")
            mock_default_emb.embed.assert_called_once_with("test query")
            assert len(results) > 0
