# zvec-native Refactoring Design

**Date**: 2026-02-18
**Status**: Approved
**Approach**: B (zvec-all-in)

## Goal

Drop all memsearch legacy code. Rebuild zvecsearch as a zvec-native markdown search system where MD files are the source of truth and zvec handles embedding, indexing, and search.

## Architecture

```
MD files (SOT)
    |
    v
scanner.py (discover .md files)
    |
    v
chunker.py (heading-based markdown splitting)
    |
    v
store.py (zvec-native: embed + index + search)
    |--- zvec.OpenAIDenseEmbedding (dense)
    |--- zvec.BM25EmbeddingFunction (sparse)
    |--- zvec.HnswIndexParam + INT8 quantization
    |--- zvec.RrfReRanker / WeightedReRanker
    |
    v
cli.py (index, search, optimize, watch, compact)
```

## Key Decisions

1. **Remove `embeddings/` folder** (6 files). Use zvec's built-in: OpenAI, Local, Qwen.
2. **store.py owns embedding**. core.py passes text, store handles vectorization.
3. **INT8 quantization by default**. 4x memory reduction, minimal recall loss.
4. **Batch flush**. One flush after full indexing, not per-file.
5. **optimize() exposed**. CLI command + auto after indexing.
6. **HnswIndexParam/CollectionOption params actually wired**. Fix dead code.
7. **Status error handling**. Check upsert/insert return values.
8. **WeightedReRanker option**. Configurable dense/sparse weights.

## Files Changed

### Delete
- `src/zvecsearch/embeddings/__init__.py`
- `src/zvecsearch/embeddings/openai.py`
- `src/zvecsearch/embeddings/google.py`
- `src/zvecsearch/embeddings/voyage.py`
- `src/zvecsearch/embeddings/ollama.py`
- `src/zvecsearch/embeddings/local.py`

### Full Rewrite
- `src/zvecsearch/store.py` — zvec-native with embedded embedding, quantization, optimize, proper params
- `src/zvecsearch/core.py` — simplified, sync search, no manual embedding

### Modify
- `src/zvecsearch/cli.py` — add optimize command, quantize/reranker options
- `src/zvecsearch/config.py` — zvec-centric config schema
- `pyproject.toml` — remove per-provider optional deps, add zvec embedding deps

### Keep Unchanged
- `src/zvecsearch/chunker.py`
- `src/zvecsearch/scanner.py`
- `src/zvecsearch/watcher.py`
- `src/zvecsearch/compact.py`

### Tests
- Update `tests/test_store.py` — new store API
- Update `tests/test_core.py` — simplified core API
- Update `tests/test_cli.py` — new commands/options
- Update `tests/test_integration.py` — end-to-end with new flow
- Update `tests/test_embeddings.py` — remove or replace
- Update `tests/test_stress.py`, `tests/test_korean_stress.py` — adapt mocks

## store.py Design

### Constructor
```python
ZvecStore(
    path="~/.zvecsearch/db",
    collection="zvecsearch_chunks",
    embedding_provider="openai",
    embedding_model="text-embedding-3-small",
    enable_mmap=True,
    read_only=False,
    hnsw_m=16,
    hnsw_ef=300,
    quantize_type="int8",
    query_ef=300,
    reranker="rrf",
    dense_weight=1.0,
    sparse_weight=0.8,
)
```

### Methods
- `embed_and_insert(chunks)` — embed text via zvec, insert (for force=True)
- `embed_and_upsert(chunks)` — embed text via zvec, upsert (for incremental)
- `search(query_text, top_k)` — embed query + hybrid search
- `flush()` — explicit flush
- `optimize()` — segment merge + index rebuild
- `index_completeness()` — check build status
- `hashes_by_source(source)` — existing (fix topk=100000)
- `indexed_sources()` — existing (fix topk=100000)
- `existing_hashes(hashes)` — existing
- `delete_by_source(source)` — existing
- `delete_by_hashes(hashes)` — existing
- `count()` — existing
- `drop()` — existing
- `close()` — existing

### Embedding Integration
```python
# Dense: zvec built-in
self._dense_emb = zvec.OpenAIDenseEmbedding(model="text-embedding-3-small")

# Sparse: zvec BM25
self._bm25_doc = zvec.BM25EmbeddingFunction(encoding_type="document")
self._bm25_query = zvec.BM25EmbeddingFunction(encoding_type="query")

# On insert: store calls embed
dense_vec = self._dense_emb.embed(chunk["content"])
sparse_vec = self._bm25_doc.embed(chunk["content"])
```

## core.py Design

### Simplified Flow
```python
class ZvecSearch:
    def __init__(self, paths, **config):
        self._store = ZvecStore(**store_config)  # store owns embedding

    def index(self, force=False):
        files = scan_paths(self._paths)
        for f in files:
            self._index_file(f, force)
        self._store.flush()
        self._store.optimize()

    def search(self, query, top_k=10):
        return self._store.search(query, top_k)  # sync, no await needed
```

No async for index/search — zvec embedding is synchronous. Keep async only for compact (LLM calls).

## config.py Design

```toml
[zvec]
path = "~/.zvecsearch/db"
collection = "zvecsearch_chunks"
enable_mmap = true
quantize_type = "int8"

[zvec.hnsw]
m = 16
ef_construction = 300

[embedding]
provider = "openai"
model = "text-embedding-3-small"

[search]
query_ef = 300
reranker = "rrf"
dense_weight = 1.0
sparse_weight = 0.8

[chunking]
max_chunk_size = 1500
overlap_lines = 2

[watch]
debounce_ms = 500
```

## Risk Mitigation

1. **Lost embedding providers**: Google Gemini, Voyage, Ollama removed. Acceptable — OpenAI + Local + Qwen covers main use cases.
2. **Test breakage**: All tests use mocks (zvec stub), so internal changes don't affect test structure. Mock interfaces need updating.
3. **Async → Sync**: zvec embeddings are sync. Remove unnecessary async where possible, keep for LLM compact only.
