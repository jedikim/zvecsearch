"""ZvecStore - zvec-native storage with embedded embedding, quantization, and optimize."""
from __future__ import annotations

import logging
from pathlib import Path
import zvec

logger = logging.getLogger(__name__)

_QUERY_FIELDS = [
    "content", "source", "heading", "chunk_hash",
    "heading_level", "start_line", "end_line",
]

_zvec_initialized = False

_METRIC_MAP = {
    "cosine": zvec.MetricType.COSINE,
    "l2": zvec.MetricType.L2,
    "ip": zvec.MetricType.IP,
}

_QUANTIZE_MAP = {
    "none": None,
    "int8": "INT8",
    "int4": "INT4",
    "fp16": "FP16",
}


class ZvecStore:
    def __init__(
        self,
        path: str = "~/.zvecsearch/db",
        collection: str = "zvecsearch_chunks",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        enable_mmap: bool = True,
        read_only: bool = False,
        hnsw_m: int = 16,
        hnsw_ef: int = 300,
        quantize_type: str = "int8",
        query_ef: int = 300,
        reranker: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 0.8,
    ):
        self._path = str(Path(path).expanduser() / collection)
        self._collection_name = collection
        self._enable_mmap = enable_mmap
        self._read_only = read_only
        self._hnsw_m = hnsw_m
        self._hnsw_ef = hnsw_ef
        self._quantize_type = quantize_type
        self._query_ef = query_ef
        self._reranker_type = reranker
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._collection = None

        # zvec built-in embedding functions
        self._dense_emb = zvec.OpenAIDenseEmbedding(model=embedding_model)
        self._bm25_doc = zvec.BM25EmbeddingFunction(encoding_type="document")
        self._bm25_query = zvec.BM25EmbeddingFunction(encoding_type="query")

        self._init_zvec()
        self._ensure_collection()

    @staticmethod
    def _init_zvec():
        global _zvec_initialized
        if not _zvec_initialized:
            try:
                zvec.init(log_level=zvec.LogLevel.WARN)
                _zvec_initialized = True
            except RuntimeError:
                _zvec_initialized = True

    def _ensure_collection(self):
        db_path = Path(self._path)
        if db_path.exists():
            option = zvec.CollectionOption(
                enable_mmap=self._enable_mmap,
                read_only=self._read_only,
            )
            self._collection = zvec.open(self._path, option)
        else:
            schema = self._build_schema()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._collection = zvec.create_and_open(self._path, schema)

    def _build_schema(self):
        dim = self._dense_emb.dim if hasattr(self._dense_emb, 'dim') else 1536
        hnsw_kwargs = {
            "metric_type": _METRIC_MAP.get("cosine", zvec.MetricType.COSINE),
            "m": self._hnsw_m,
            "ef_construction": self._hnsw_ef,
        }
        qt = _QUANTIZE_MAP.get(self._quantize_type)
        if qt is not None:
            hnsw_kwargs["quantize_type"] = qt

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
                "embedding", zvec.DataType.VECTOR_FP32,
                dimension=dim,
                index_param=zvec.HnswIndexParam(**hnsw_kwargs),
            ),
            zvec.VectorSchema(
                "sparse_embedding", zvec.DataType.SPARSE_VECTOR_FP32,
                dimension=0,
            ),
        ]
        return zvec.CollectionSchema(
            name=self._collection_name, fields=fields, vectors=vectors,
        )

    def _build_docs(self, chunks):
        docs = []
        for chunk in chunks:
            dense_vec = self._dense_emb.embed(chunk["content"])
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
                    "embedding": dense_vec,
                    "sparse_embedding": sparse_vec,
                },
            )
            docs.append(doc)
        return docs

    def _check_statuses(self, statuses, operation):
        if statuses is None:
            return
        for i, status in enumerate(statuses):
            if hasattr(status, 'ok') and callable(status.ok) and not status.ok():
                logger.error(
                    "%s failed for doc %d: code=%s msg=%s",
                    operation, i, status.code(), status.message(),
                )

    def embed_and_upsert(self, chunks):
        if not chunks:
            return 0
        docs = self._build_docs(chunks)
        statuses = self._collection.upsert(docs)
        self._check_statuses(statuses, "upsert")
        return len(docs)

    def embed_and_insert(self, chunks):
        if not chunks:
            return 0
        docs = self._build_docs(chunks)
        statuses = self._collection.insert(docs)
        self._check_statuses(statuses, "insert")
        return len(docs)

    def search(self, query_text, top_k=10):
        if self.count() == 0:
            return []
        dense_vec = self._dense_emb.embed(query_text)
        sparse_vec = self._bm25_query.embed(query_text)
        queries = [
            zvec.VectorQuery("embedding", vector=dense_vec),
            zvec.VectorQuery("sparse_embedding", vector=sparse_vec),
        ]
        if self._reranker_type == "weighted":
            reranker = zvec.WeightedReRanker(
                topn=top_k, weights=[self._dense_weight, self._sparse_weight],
            )
        else:
            reranker = zvec.RrfReRanker(topn=top_k, rank_constant=60)
        query_param = zvec.HnswQueryParam(ef=self._query_ef)
        results = self._collection.query(
            vectors=queries, topk=top_k, output_fields=_QUERY_FIELDS,
            reranker=reranker, query_param=query_param,
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

    def flush(self):
        if self._collection:
            self._collection.flush()

    def optimize(self):
        if self._collection:
            self._collection.optimize()

    def hashes_by_source(self, source):
        results = self._collection.query(
            filter=f'source == "{source}"', output_fields=["chunk_hash"],
        )
        return {doc.field("chunk_hash") for doc in results}

    def indexed_sources(self):
        results = self._collection.query(output_fields=["source"])
        return {doc.field("source") for doc in results}

    def existing_hashes(self, hashes):
        if not hashes:
            return set()
        fetched = self._collection.fetch(hashes)
        return set(fetched.keys())

    def delete_by_source(self, source):
        self._collection.delete_by_filter(f'source == "{source}"')

    def delete_by_hashes(self, hashes):
        if hashes:
            self._collection.delete(hashes)

    def count(self):
        stats = self._collection.stats
        return stats.doc_count

    def query(self, filter_expr=""):
        if not filter_expr:
            return []
        results = self._collection.query(
            filter=filter_expr, output_fields=_QUERY_FIELDS,
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

    def drop(self):
        if self._collection:
            self._collection.destroy()
            self._collection = None

    def close(self):
        if self._collection:
            self._collection.flush()
            self._collection = None
