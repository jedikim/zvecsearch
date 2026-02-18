"""ZvecStore - Core storage layer using zvec embedded vector database.

Wraps zvec's Collection API to provide:
- Schema with dense + sparse (BM25) vectors for hybrid search
- Upsert/delete/search operations on markdown chunks
- HNSW indexing with COSINE metric for dense vectors
- BM25EmbeddingFunction for sparse vectors
- RrfReRanker for hybrid search fusion

The collection schema stores chunk metadata (hash, content, source, heading,
line numbers) alongside dense embeddings and BM25 sparse embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path
import zvec

logger = logging.getLogger(__name__)

_QUERY_FIELDS = [
    "content", "source", "heading", "chunk_hash",
    "heading_level", "start_line", "end_line",
]

# Track whether zvec.init() has been called globally
_zvec_initialized = False


class ZvecStore:
    """Vector storage layer using zvec embedded database.

    Creates and manages a zvec Collection with a fixed schema for storing
    markdown chunks with dense and sparse vector embeddings.

    Args:
        path: Base path for the database directory.
        collection: Name of the collection (also used as subdirectory).
        dimension: Dimensionality of dense embedding vectors.
        enable_mmap: Whether to enable memory-mapped I/O for the collection.
        index_metric: Distance metric for the HNSW index ('cosine', 'l2', 'ip').
        hnsw_ef: HNSW ef_construction parameter.
        hnsw_max_m: HNSW max_m parameter.
    """

    def __init__(
        self,
        path: str = "~/.zvecsearch/db",
        collection: str = "zvecsearch_chunks",
        dimension: int | None = 1536,
        enable_mmap: bool = True,
        index_metric: str = "cosine",
        hnsw_ef: int = 300,
        hnsw_max_m: int = 16,
    ):
        self._path = str(Path(path).expanduser() / collection)
        self._collection_name = collection
        self._dimension = dimension
        self._enable_mmap = enable_mmap
        self._index_metric = index_metric
        self._hnsw_ef = hnsw_ef
        self._hnsw_max_m = hnsw_max_m
        self._collection: zvec.Collection | None = None
        self._bm25_doc = None
        self._bm25_query = None

        self._init_zvec()
        self._ensure_collection()

    @staticmethod
    def _init_zvec() -> None:
        """Initialize zvec runtime (only once per process)."""
        global _zvec_initialized
        if not _zvec_initialized:
            try:
                zvec.init(log_level=zvec.LogLevel.WARN)
                _zvec_initialized = True
            except RuntimeError:
                # Already initialized by another caller
                _zvec_initialized = True

    def _get_metric_type(self):
        """Map string metric name to zvec MetricType enum."""
        return {
            "cosine": zvec.MetricType.COSINE,
            "l2": zvec.MetricType.L2,
            "ip": zvec.MetricType.IP,
        }.get(self._index_metric, zvec.MetricType.COSINE)

    def _ensure_collection(self) -> None:
        """Open existing collection or create a new one."""
        db_path = Path(self._path)
        if db_path.exists():
            option = zvec.CollectionOption()
            self._collection = zvec.open(self._path, option)
        else:
            schema = self._build_schema()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._collection = zvec.create_and_open(self._path, schema)
            # Indexes are declared inline with schema fields (HnswIndexParam, InvertIndexParam)

        # Initialize BM25 embedding functions for sparse vectors
        self._bm25_doc = zvec.BM25EmbeddingFunction(
            encoding_type="document", language="en"
        )
        self._bm25_query = zvec.BM25EmbeddingFunction(
            encoding_type="query", language="en"
        )

    def _build_schema(self) -> zvec.CollectionSchema:
        """Build the collection schema with scalar fields and vector fields."""
        fields = [
            zvec.FieldSchema("chunk_hash", zvec.DataType.STRING),
            zvec.FieldSchema("content", zvec.DataType.STRING),
            zvec.FieldSchema("source", zvec.DataType.STRING,
                             index_param=zvec.InvertIndexParam()),
            zvec.FieldSchema("heading", zvec.DataType.STRING),
            zvec.FieldSchema("heading_level", zvec.DataType.INT32),
            zvec.FieldSchema("start_line", zvec.DataType.INT32),
            zvec.FieldSchema("end_line", zvec.DataType.INT32),
        ]

        vectors = [
            zvec.VectorSchema(
                "embedding",
                zvec.DataType.VECTOR_FP32,
                dimension=self._dimension or 1536,
                index_param=zvec.HnswIndexParam(),
            ),
            zvec.VectorSchema(
                "sparse_embedding",
                zvec.DataType.SPARSE_VECTOR_FP32,
                dimension=0,
            ),
        ]

        return zvec.CollectionSchema(
            name=self._collection_name,
            fields=fields,
            vectors=vectors,
        )

    def upsert(self, chunks: list[dict]) -> int:
        """Insert or update chunks in the collection.

        Each chunk dict must contain: chunk_hash, content, source, heading,
        heading_level, start_line, end_line, embedding.

        The chunk_hash is used as the document ID (primary key) for upsert.
        BM25 sparse embeddings are automatically computed from content.

        Args:
            chunks: List of chunk dictionaries to upsert.

        Returns:
            Number of chunks upserted.
        """
        if not chunks:
            return 0

        docs = []
        for chunk in chunks:
            sparse_vec = self._bm25_doc.embed(chunk["content"])
            doc = zvec.Doc(
                id=chunk["chunk_hash"],
                fields={
                    "chunk_hash": chunk["chunk_hash"],
                    "content": chunk["content"],
                    "source": chunk["source"],
                    "heading": chunk["heading"],
                    "heading_level": chunk["heading_level"],
                    "start_line": chunk["start_line"],
                    "end_line": chunk["end_line"],
                },
                vectors={
                    "embedding": chunk["embedding"],
                    "sparse_embedding": sparse_vec,
                },
            )
            docs.append(doc)

        self._collection.upsert(docs)
        self._collection.flush()
        return len(docs)

    def search(
        self,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = 10,
        filter_expr: str = "",
    ) -> list[dict]:
        """Search for similar chunks using hybrid dense + sparse vector search.

        When query_text is provided, performs hybrid search combining dense
        embedding similarity with BM25 sparse matching, fused via RRF reranking.
        When query_text is empty, performs dense-only search.

        Args:
            query_embedding: Dense embedding vector for the query.
            query_text: Optional text for BM25 sparse matching.
            top_k: Maximum number of results to return.
            filter_expr: Optional filter expression (e.g., 'source == "foo.md"').

        Returns:
            List of result dicts with content, metadata, and score.
        """
        if self.count() == 0:
            return []

        queries = [
            zvec.VectorQuery("embedding", vector=query_embedding),
        ]

        reranker = None
        if query_text:
            sparse_vec = self._bm25_query.embed(query_text)
            queries.append(
                zvec.VectorQuery("sparse_embedding", vector=sparse_vec)
            )
            reranker = zvec.RrfReRanker(topn=top_k, rank_constant=60)

        results = self._collection.query(
            vectors=queries if len(queries) > 1 else queries[0],
            topk=top_k,
            filter=filter_expr or None,
            output_fields=_QUERY_FIELDS,
            reranker=reranker,
        )

        return [
            {
                "content": doc.field("content"),
                "source": doc.field("source"),
                "heading": doc.field("heading"),
                "heading_level": doc.field("heading_level"),
                "start_line": doc.field("start_line"),
                "end_line": doc.field("end_line"),
                "chunk_hash": doc.id,
                "score": doc.score or 0.0,
            }
            for doc in results
        ]

    def hashes_by_source(self, source: str) -> set[str]:
        """Get all chunk hashes for a given source file.

        Args:
            source: Source file path to query.

        Returns:
            Set of chunk hashes from that source.
        """
        results = self._collection.query(
            filter=f'source == "{source}"',
            output_fields=["chunk_hash"],
            topk=100000,
        )
        return {doc.field("chunk_hash") for doc in results}

    def indexed_sources(self) -> set[str]:
        """Get the set of all source files that have been indexed.

        Returns:
            Set of source file paths.
        """
        results = self._collection.query(
            output_fields=["source"],
            topk=100000,
        )
        return {doc.field("source") for doc in results}

    def existing_hashes(self, hashes: list[str]) -> set[str]:
        """Check which hashes already exist in the collection.

        Args:
            hashes: List of chunk hashes to check.

        Returns:
            Set of hashes that exist in the collection.
        """
        if not hashes:
            return set()
        fetched = self._collection.fetch(hashes)
        return set(fetched.keys())

    def delete_by_source(self, source: str) -> None:
        """Delete all chunks from a given source file.

        Args:
            source: Source file path whose chunks should be deleted.
        """
        self._collection.delete_by_filter(f'source == "{source}"')
        self._collection.flush()

    def delete_by_hashes(self, hashes: list[str]) -> None:
        """Delete chunks by their hash IDs.

        Args:
            hashes: List of chunk hashes to delete.
        """
        if hashes:
            self._collection.delete(hashes)
            self._collection.flush()

    def count(self) -> int:
        """Get the number of documents in the collection.

        Returns:
            Document count.
        """
        stats = self._collection.stats
        return stats.doc_count

    def drop(self) -> None:
        """Permanently destroy the collection and all its data."""
        if self._collection:
            self._collection.destroy()
            self._collection = None

    def query(self, filter_expr: str = "") -> list[dict]:
        """Query chunks by scalar filter expression (no vector search).

        Args:
            filter_expr: Filter expression (e.g., 'source == "foo.md"').

        Returns:
            List of matching chunk dicts.
        """
        if not filter_expr:
            return []
        results = self._collection.query(
            filter=filter_expr,
            output_fields=_QUERY_FIELDS,
            topk=100000,
        )
        return [
            {
                "content": doc.field("content"),
                "source": doc.field("source"),
                "heading": doc.field("heading"),
                "heading_level": doc.field("heading_level"),
                "start_line": doc.field("start_line"),
                "end_line": doc.field("end_line"),
                "chunk_hash": doc.id,
            }
            for doc in results
        ]

    def close(self) -> None:
        """Flush pending writes and release the collection handle."""
        if self._collection:
            self._collection.flush()
            self._collection = None
