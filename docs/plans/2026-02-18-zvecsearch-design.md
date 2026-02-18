# ZvecSearch Design Document

## Overview

**zvecsearch** is a semantic memory search system that replaces memsearch's Milvus storage layer with zvec (Alibaba's embedded vector database). It retains all of memsearch's features while leveraging zvec's native capabilities for maximum performance.

## Architecture: zvec-Native Redesign (Approach C)

### Key Decision: Why zvec-Native?

- zvec provides embedded operation (no separate server needed)
- Native HNSW indexing with tunable parameters (ef, max_m)
- Built-in quantization (FP16, INT8, INT4) for memory efficiency
- Native BM25 sparse vector support
- RRF/Weighted reranker built-in
- Apache Arrow + RocksDB storage backend

### Project Structure

```
zvecsearch/
├── src/zvecsearch/
│   ├── __init__.py              # ZvecSearch class export
│   ├── core.py                  # ZvecSearch orchestrator
│   ├── store.py                 # ZvecStore - zvec native storage
│   ├── chunker.py               # Markdown chunking (from memsearch)
│   ├── scanner.py               # File scanner (from memsearch)
│   ├── watcher.py               # File watcher (from memsearch)
│   ├── config.py                # zvec-native config system
│   ├── compact.py               # LLM summarization (from memsearch)
│   ├── transcript.py            # JSONL transcript parser (from memsearch)
│   ├── cli.py                   # CLI interface
│   └── embeddings/              # 5 embedding providers (from memsearch)
│       ├── __init__.py          # Protocol + factory
│       ├── openai.py
│       ├── google.py
│       ├── voyage.py
│       ├── ollama.py
│       └── local.py
├── tests/                       # TDD test suite
│   ├── test_chunker.py
│   ├── test_scanner.py
│   ├── test_store.py
│   ├── test_core.py
│   ├── test_config.py
│   ├── test_watcher.py
│   └── test_compact.py
├── docs/plans/
├── pyproject.toml
├── CLAUDE.md
└── README.md
```

### Data Flow

```
Markdown Files → Scanner → Chunker → Embeddings → ZvecStore → zvec Collection
                                                      ↑
                                                  BM25 Sparse (zvec native)
                                                  HNSW Index (zvec native)
                                                  RRF Reranker (zvec native)
```

## Component Design

### 1. ZvecStore (store.py) - Core Change

Replaces MilvusStore entirely with zvec native APIs.

**Schema:**
```python
schema = zvec.CollectionSchema(
    name="zvecsearch_chunks",
    fields=[
        FieldSchema("chunk_hash", DataType.STRING),        # PK
        FieldSchema("content", DataType.STRING),
        FieldSchema("source", DataType.STRING),
        FieldSchema("heading", DataType.STRING),
        FieldSchema("heading_level", DataType.INT32),
        FieldSchema("start_line", DataType.INT32),
        FieldSchema("end_line", DataType.INT32),
    ],
    vectors=[
        VectorSchema("embedding", dim=<dynamic>, DataType.VECTOR_FP32),
    ]
)
```

**Index Configuration:**
- Dense: HNSW with COSINE metric, ef=300, max_m=16
- Quantization: FP16 by default (configurable)
- Optional: INT8/INT4 for memory-constrained environments

**Hybrid Search Strategy:**
- Dense vector search via zvec HNSW
- BM25 sparse search via zvec's BM25EmbeddingFunction + sparse vector field
- RRF reranking via zvec's RrfReRanker

**API Mapping (Milvus → zvec):**

| Milvus API | zvec API |
|---|---|
| `MilvusClient(uri=)` | `zvec.create_and_open(path, schema)` / `zvec.open(path)` |
| `client.upsert(data=)` | `collection.upsert(docs)` |
| `client.hybrid_search(reqs, ranker)` | `collection.query(VectorQuery) + RrfReRanker` |
| `client.query(filter=)` | `collection.fetch(filter=)` |
| `client.delete(filter=)` | `collection.delete_by_filter(filter)` |
| `client.delete(ids=)` | `collection.delete(keys)` |
| `client.drop_collection()` | `collection.destroy()` |
| `client.get_collection_stats()` | `collection.fetch()` count |
| BM25 Function (Milvus built-in) | `zvec.BM25EmbeddingFunction` |
| `RRFRanker(k=60)` | `zvec.RrfReRanker()` |

### 2. Config (config.py) - Modified

```toml
[zvec]
path = "~/.zvecsearch/db"           # Collection storage path
collection = "zvecsearch_chunks"
enable_mmap = true
max_buffer_size = 67108864           # 64MB

[index]
type = "hnsw"                        # hnsw | ivf | flat
metric = "cosine"                    # cosine | l2 | ip
quantize = "fp16"                    # fp16 | int8 | int4 | none
hnsw_ef = 300
hnsw_max_m = 16

[embedding]
provider = "openai"
model = ""

[compact]
llm_provider = "openai"
llm_model = ""
prompt_file = ""

[chunking]
max_chunk_size = 1500
overlap_lines = 2

[watch]
debounce_ms = 1500
```

### 3. Hybrid (Embedding Strategy)

- **Dense embeddings**: memsearch's 5 providers (OpenAI, Google, Voyage, Ollama, Local)
- **Sparse embeddings**: zvec's BM25EmbeddingFunction (native C++)
- **Reranking**: zvec's RrfReRanker (native C++)
- **Benefit**: Best of both worlds - flexible dense providers + optimized native sparse

### 4. Unchanged Components

These are ported from memsearch with minimal changes:
- `chunker.py` - Markdown chunking logic
- `scanner.py` - File discovery
- `watcher.py` - File system monitoring (watchdog)
- `compact.py` - LLM summarization
- `transcript.py` - JSONL parsing
- `embeddings/*` - All 5 providers
- `cli.py` - CLI structure (command names updated)

## Testing Strategy (TDD)

1. **Unit Tests First**: Write tests before implementation
2. **Store tests**: zvec CRUD, search, hybrid search, index management
3. **Integration tests**: Full pipeline (scan → chunk → embed → store → search)
4. **Config tests**: TOML layering, type coercion
5. **Watcher tests**: Debounce, event handling

## Dependencies

```toml
[dependencies]
zvec >= 0.1.0                        # Vector database engine
click >= 8.1                         # CLI
watchdog >= 4.0                      # File watching
tomli_w >= 1.0                       # TOML writing
openai >= 1.0                        # Default embeddings

[optional]
google-genai >= 1.0
voyageai >= 0.3
ollama >= 0.4
sentence-transformers >= 3.0
anthropic >= 0.40
```
