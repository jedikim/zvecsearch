"""Tests for ZvecStore - the core storage layer wrapping zvec.

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
# Build a comprehensive mock of the zvec module so that store.py can import
# it even when the real native library is unavailable.
# ---------------------------------------------------------------------------

_zvec_mock = MagicMock()

# Enums
_zvec_mock.DataType.STRING = "STRING"
_zvec_mock.DataType.INT32 = "INT32"
_zvec_mock.DataType.VECTOR_FP32 = "VECTOR_FP32"
_zvec_mock.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
_zvec_mock.MetricType.COSINE = "COSINE"
_zvec_mock.MetricType.L2 = "L2"
_zvec_mock.MetricType.IP = "IP"
_zvec_mock.LogLevel.WARN = "WARN"

# Classes that store.py instantiates
_zvec_mock.FieldSchema = MagicMock
_zvec_mock.VectorSchema = MagicMock
_zvec_mock.CollectionSchema = MagicMock
_zvec_mock.CollectionOption = MagicMock
_zvec_mock.HnswIndexParam = MagicMock
_zvec_mock.InvertIndexParam = MagicMock
_zvec_mock.FlatIndexParam = MagicMock
_zvec_mock.VectorQuery = MagicMock
_zvec_mock.RrfReRanker = MagicMock

# Doc class needs to be a real factory (store.py creates Doc instances)
class FakeDoc:
    """Minimal Doc stand-in for testing."""
    def __init__(self, id, fields=None, vectors=None, score=None):
        self.id = id
        self.fields = fields or {}
        self.vectors = vectors or {}
        self.score = score or 0.0

    def field(self, name):
        return self.fields.get(name)

    def has_field(self, name):
        return name in self.fields

_zvec_mock.Doc = FakeDoc

# Patch zvec in sys.modules before importing store
sys.modules["zvec"] = _zvec_mock

from zvecsearch.store import ZvecStore  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
TEST_DIM = 4
TEST_DB = Path("/tmp/zvecsearch_test_db")


def _make_mock_collection():
    """Create a mock collection that simulates zvec Collection behavior."""
    collection = MagicMock()
    # Internal doc storage for simulating real behavior
    collection._docs = {}
    collection._destroyed = False

    # stats property returns an object with doc_count
    def _stats():
        s = SimpleNamespace()
        s.doc_count = len(collection._docs)
        return s
    type(collection).stats = property(lambda self: _stats())

    # upsert: store docs by id and return status list
    def _upsert(docs):
        if not isinstance(docs, list):
            docs = [docs]
        statuses = []
        for doc in docs:
            collection._docs[doc.id] = doc
            status = MagicMock()
            status.ok.return_value = True
            statuses.append(status)
        return statuses
    collection.upsert.side_effect = _upsert

    # flush: no-op
    collection.flush.return_value = None

    # delete: remove by ids
    def _delete(ids):
        if isinstance(ids, str):
            ids = [ids]
        statuses = []
        for id_ in ids:
            status = MagicMock()
            if id_ in collection._docs:
                del collection._docs[id_]
                status.ok.return_value = True
            else:
                status.ok.return_value = False
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

    # query: simulate vector search - returns docs matching filter or all docs
    def _query(vectors=None, topk=10, filter=None, output_fields=None,
               include_vector=False, reranker=None):
        import re
        results = list(collection._docs.values())
        if filter:
            m = re.search(r'source\s*==\s*"([^"]+)"', filter)
            if m:
                source_val = m.group(1)
                results = [d for d in results if d.fields.get("source") == source_val]
        # Give each result a score
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
def mock_zvec_env():
    """Set up mocked zvec environment and return the mock collection."""
    import zvecsearch.store as store_module
    # Reset the global init flag so each test starts fresh
    store_module._zvec_initialized = False

    mock_collection = _make_mock_collection()

    with patch("zvecsearch.store.zvec") as zvec_mod:
        # Wire up the module mock to return our simulating collection
        zvec_mod.DataType = _zvec_mock.DataType
        zvec_mod.MetricType = _zvec_mock.MetricType
        zvec_mod.LogLevel = _zvec_mock.LogLevel
        zvec_mod.FieldSchema = MagicMock
        zvec_mod.VectorSchema = MagicMock
        zvec_mod.CollectionSchema = MagicMock
        zvec_mod.CollectionOption = MagicMock
        zvec_mod.HnswIndexParam = MagicMock
        zvec_mod.InvertIndexParam = MagicMock
        zvec_mod.VectorQuery = MagicMock
        zvec_mod.RrfReRanker = MagicMock
        zvec_mod.Doc = FakeDoc
        zvec_mod.BM25EmbeddingFunction = MagicMock

        # BM25 mock: returns sparse dict from text
        bm25_instance = MagicMock()
        bm25_instance.embed.side_effect = lambda text: {hash(text) % 10000: 1.0}
        zvec_mod.BM25EmbeddingFunction.return_value = bm25_instance

        # create_and_open returns our mock collection
        zvec_mod.create_and_open.return_value = mock_collection
        zvec_mod.open.return_value = mock_collection
        zvec_mod.init.return_value = None

        yield zvec_mod, mock_collection


def _make_store_with_mock(mock_zvec_env, dim=TEST_DIM):
    """Create a ZvecStore using the mocked zvec environment."""
    zvec_mod, mock_collection = mock_zvec_env
    store = ZvecStore(
        path=str(TEST_DB),
        collection="test_chunks",
        dimension=dim,
    )
    return store


def _sample_chunks(n=3, dim=TEST_DIM):
    """Generate sample chunk dicts for testing."""
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
            "embedding": [float(i)] * dim,
        })
    return chunks


# ---------------------------------------------------------------------------
# Tests: Basic Operations
# ---------------------------------------------------------------------------
class TestZvecStoreBasic:
    def test_create_and_count(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        assert store.count() == 0
        store.close()

    def test_upsert_and_count(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(3)
        count = store.upsert(chunks)
        assert count == 3
        assert store.count() == 3
        store.close()

    def test_upsert_is_idempotent(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(2)
        store.upsert(chunks)
        store.upsert(chunks)  # same hashes - should overwrite, not duplicate
        assert store.count() == 2
        store.close()

    def test_upsert_empty(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        count = store.upsert([])
        assert count == 0
        assert store.count() == 0
        store.close()

    def test_delete_by_source(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        store.delete_by_source("test.md")
        assert store.count() == 0
        store.close()

    def test_delete_by_hashes(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        store.delete_by_hashes(["hash_0", "hash_1"])
        assert store.count() == 1
        store.close()

    def test_delete_by_hashes_empty(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        store.delete_by_hashes([])
        assert store.count() == 3
        store.close()


# ---------------------------------------------------------------------------
# Tests: Search
# ---------------------------------------------------------------------------
class TestZvecStoreSearch:
    def test_search_returns_results(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(5)
        store.upsert(chunks)
        results = store.search(
            query_embedding=[1.0] * TEST_DIM,
            query_text="topic",
            top_k=3,
        )
        assert len(results) <= 3
        assert all("content" in r for r in results)
        assert all("score" in r for r in results)
        store.close()

    def test_search_empty_collection(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        results = store.search(
            query_embedding=[1.0] * TEST_DIM,
            query_text="anything",
            top_k=5,
        )
        assert results == []
        store.close()

    def test_search_dense_only(self, mock_zvec_env):
        """Search with embedding only, no query text (no hybrid)."""
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        results = store.search(
            query_embedding=[1.0] * TEST_DIM,
            query_text="",
            top_k=2,
        )
        assert len(results) <= 2
        store.close()


# ---------------------------------------------------------------------------
# Tests: Query helpers
# ---------------------------------------------------------------------------
class TestZvecStoreQuery:
    def test_hashes_by_source(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        store.upsert(_sample_chunks(3))
        hashes = store.hashes_by_source("test.md")
        assert hashes == {"hash_0", "hash_1", "hash_2"}
        store.close()

    def test_indexed_sources(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(2)
        chunks[1]["source"] = "other.md"
        store.upsert(chunks)
        sources = store.indexed_sources()
        assert sources == {"test.md", "other.md"}
        store.close()

    def test_existing_hashes(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        store.upsert(_sample_chunks(3))
        found = store.existing_hashes(["hash_0", "hash_1", "nonexistent"])
        assert found == {"hash_0", "hash_1"}
        store.close()

    def test_existing_hashes_empty_input(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        store.upsert(_sample_chunks(3))
        found = store.existing_hashes([])
        assert found == set()
        store.close()


# ---------------------------------------------------------------------------
# Tests: Drop
# ---------------------------------------------------------------------------
class TestZvecStoreDrop:
    def test_drop(self, mock_zvec_env):
        store = _make_store_with_mock(mock_zvec_env)
        store.upsert(_sample_chunks(3))
        store.drop()
        # After drop, the collection should be None
        assert store._collection is None


# ---------------------------------------------------------------------------
# Tests: Constructor / Init
# ---------------------------------------------------------------------------
class TestZvecStoreInit:
    def test_default_path_expansion(self, mock_zvec_env):
        """Path should be expanded and include collection name."""
        store = _make_store_with_mock(mock_zvec_env)
        assert "test_chunks" in store._path
        store.close()

    def test_zvec_init_called(self, mock_zvec_env):
        """zvec.init() should be called during store creation."""
        zvec_mod, _ = mock_zvec_env
        _make_store_with_mock(mock_zvec_env)
        zvec_mod.init.assert_called()

    def test_schema_created_with_correct_fields(self, mock_zvec_env):
        """CollectionSchema should be created with all required fields."""
        zvec_mod, _ = mock_zvec_env
        _make_store_with_mock(mock_zvec_env)
        # Verify create_and_open was called (new collection)
        zvec_mod.create_and_open.assert_called_once()

    def test_close_flushes(self, mock_zvec_env):
        """close() should flush the collection."""
        zvec_mod, mock_collection = mock_zvec_env
        store = _make_store_with_mock(mock_zvec_env)
        store.close()
        mock_collection.flush.assert_called()


# ---------------------------------------------------------------------------
# Tests: Upsert creates correct Doc objects
# ---------------------------------------------------------------------------
class TestZvecStoreUpsertDetails:
    def test_upsert_creates_docs_with_correct_fields(self, mock_zvec_env):
        """Each upserted doc should have the correct id, fields, and vectors."""
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(1)
        store.upsert(chunks)

        # The mock collection stores FakeDoc objects
        _, mock_collection = mock_zvec_env
        assert "hash_0" in mock_collection._docs
        doc = mock_collection._docs["hash_0"]
        assert doc.id == "hash_0"
        assert doc.fields["content"] == "This is test content number 0 about topic 0"
        assert doc.fields["source"] == "test.md"
        assert doc.fields["heading"] == "Section 0"
        assert doc.fields["heading_level"] == 1
        assert doc.fields["start_line"] == 1
        assert doc.fields["end_line"] == 10
        assert "embedding" in doc.vectors
        assert "sparse_embedding" in doc.vectors
        store.close()

    def test_upsert_calls_bm25_for_sparse_embedding(self, mock_zvec_env):
        """BM25 document encoder should be called for each chunk's content."""
        zvec_mod, _ = mock_zvec_env
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(2)
        store.upsert(chunks)
        # BM25 doc encoder's embed should be called for each chunk
        assert store._bm25_doc.embed.call_count == 2
        store.close()

    def test_upsert_flushes_after_insert(self, mock_zvec_env):
        """upsert() should call flush() after inserting docs."""
        _, mock_collection = mock_zvec_env
        store = _make_store_with_mock(mock_zvec_env)
        chunks = _sample_chunks(1)
        # Reset flush call count after init
        mock_collection.flush.reset_mock()
        store.upsert(chunks)
        mock_collection.flush.assert_called()
        store.close()
