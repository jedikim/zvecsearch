# ZvecSearch

**v0.1.0**

Semantic memory search powered by [zvec](https://github.com/alibaba/zvec) (Alibaba's embedded vector database).

Index markdown files, embed them with OpenAI, Gemini, or local models, and perform hybrid search (dense + sparse) with no server required.

> Inspired by [memsearch](https://github.com/zilliztech/memsearch) and [OpenClaw](https://github.com/openclaw/openclaw)'s markdown-first memory architecture.

**[한국어 README](README.ko.md)**

## Installation

```bash
pip install -e ".[dev]"

# For Gemini embedding support
pip install -e ".[google]"

# For zvec default local embedding (no API key needed)
pip install sentence-transformers
```

### Requirements

- Python 3.10+
- zvec >= 0.2.0
- **Default local**: `sentence-transformers` (no API key, fully offline)
- **OpenAI**: `OPENAI_API_KEY`
- **Gemini**: `GOOGLE_API_KEY` + `pip install -e ".[google]"`

### zvec x86_64 Build Issue

The official zvec PyPI wheel is compiled with AVX-512 instructions, which causes `SIGILL` crashes on CPUs without AVX-512 support (most AMD CPUs, older Intel CPUs, and many VMs). See [alibaba/zvec#128](https://github.com/alibaba/zvec/issues/128).

A pre-built wheel compiled with `-march=x86-64-v2` (SSE4.2, compatible with most x86_64 systems) is included in the `dist/` directory:

```bash
pip install dist/zvec-0.2.1.dev0-cp312-cp312-linux_x86_64.whl
```

## Quick Start

### 3 Lines to Search

```python
from zvecsearch import ZvecSearch

zs = ZvecSearch(paths=["./docs"])
zs.index()                                    # index markdown files
results = zs.search("HNSW algorithm", top_k=5)  # semantic search
```

### CLI

```bash
zvecsearch index ./docs/           # index markdown files
zvecsearch search "HNSW algorithm" # semantic search
zvecsearch watch ./docs/           # watch for changes, auto-reindex
zvecsearch compact                 # LLM-based chunk summarization
```

## Usage

### Python API

```python
from zvecsearch import ZvecSearch

# Initialize with custom settings
zs = ZvecSearch(
    paths=["./docs", "./notes"],
    embedding_provider="openai",        # "default" (local), "openai", or "google"
    embedding_model="text-embedding-3-small",
    quantize_type="int8",               # "int8", "int4", "fp16", "none"
    reranker="rrf",                     # "rrf" or "weighted"
)

# Index all markdown files (incremental - only changed chunks are re-embedded)
zs.index()

# Force full re-index
zs.index(force=True)

# Index a single file
zs.index_file("./docs/new-note.md")

# Search
results = zs.search("vector similarity search", top_k=10)
for r in results:
    print(f"[{r['score']:.4f}] {r['source']}:{r['start_line']}-{r['end_line']}")
    print(f"  Heading: {r['heading']}")
    print(f"  {r['content'][:100]}...")
    print()

# Watch for file changes (auto-reindex on create/modify/delete)
watcher = zs.watch(debounce_ms=1500)
watcher.start()
# ... watcher runs in background ...
watcher.stop()

# LLM-based chunk summarization (async)
import asyncio
summary = asyncio.run(zs.compact(
    source="./docs/long-document.md",
    llm_provider="openai",
    output_dir="./output",
))

# Context manager support
with ZvecSearch(paths=["./docs"]) as zs:
    zs.index()
    results = zs.search("query")
```

### CLI Commands

```bash
# Index markdown files in one or more directories
zvecsearch index ./docs/
zvecsearch index ./docs/ ./notes/ --force    # force full re-index
zvecsearch index ./docs/ --provider google   # use Gemini embedding
zvecsearch index ./docs/ --provider default  # use local embedding (no API key)

# Semantic search
zvecsearch search "how does HNSW work"
zvecsearch search "query" --top-k 20 --json-output  # JSON output

# Watch for changes (auto-reindex)
zvecsearch watch ./docs/
zvecsearch watch ./docs/ --debounce-ms 3000

# LLM summarization
zvecsearch compact
zvecsearch compact --source ./docs/file.md

# Configuration
zvecsearch config list                       # show current config
zvecsearch config set embedding.provider google
zvecsearch config set embedding.model gemini-embedding-001
zvecsearch config set search.reranker weighted
zvecsearch config set zvec.quantize_type int4
```

### Configuration

Settings are resolved in priority order: **defaults** < **global config** < **project config** < **CLI flags**.

Global config: `~/.zvecsearch/config.toml`
Project config: `.zvecsearch.toml`

```toml
[zvec]
path = "~/.zvecsearch/db"          # database storage path
collection = "zvecsearch_chunks"   # collection name
enable_mmap = true                 # memory-mapped I/O
hnsw_m = 16                        # HNSW max connections per node
hnsw_ef = 300                      # HNSW ef_construction
quantize_type = "int8"             # "int8", "int4", "fp16", "none"

[embedding]
provider = "default"               # "default" (local), "openai", or "google"
model = ""                         # auto; or "text-embedding-3-small", "gemini-embedding-001"

[search]
top_k = 10
query_ef = 300                     # HNSW search-time ef
reranker = "default"               # "default" (local cross-encoder), "rrf", or "weighted"
dense_weight = 1.0                 # for weighted reranker
sparse_weight = 0.8                # for weighted reranker

[chunking]
max_chunk_size = 1500
overlap_lines = 2

[watch]
debounce_ms = 1500
```

## Architecture

### Hybrid Search

```
Query -> +-- Dense embedding (OpenAI/Gemini) -> HNSW cosine search --+
         +-- Sparse embedding (BM25)         -> Inverted index      --+
                                                                      |
                                                               RRF ReRanker -> Results
```

Every query runs **two parallel searches**:

1. **Dense search**: Query text is embedded (OpenAI or Gemini), then searched against HNSW index using cosine similarity.
2. **Sparse search**: BM25 keyword matching via zvec's native `BM25EmbeddingFunction`.

Results are fused by **RRF ReRanker** (default) or **Weighted ReRanker**, producing a single ranked list.

### zvec Default Local Providers

zvec's built-in default configuration uses all-local models that require **no API keys** and **no network access**:

| Component | Class | Model | Size |
|-----------|-------|-------|------|
| Dense embedding | `DefaultLocalDenseEmbedding` | all-MiniLM-L6-v2 (384 dim) | ~80MB |
| Sparse embedding | `DefaultLocalSparseEmbedding` | SPLADE | ~100MB |
| Reranker | `DefaultLocalReRanker` | cross-encoder/ms-marco-MiniLM-L6-v2 | ~80MB |

These `Default*` classes are available in zvec out of the box. Models are downloaded automatically on first use.

### Embedding Providers

| Provider | Model | Dimensions | Env Variable |
|----------|-------|-----------|--------------|
| zvec Default (local) | all-MiniLM-L6-v2 | 384 | None (local) |
| OpenAI | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| Gemini | gemini-embedding-001 | 768 | `GOOGLE_API_KEY` |

zvecsearch defaults to zvec's local providers (`DefaultLocalDenseEmbedding` + `DefaultLocalSparseEmbedding` + `DefaultLocalReRanker`) — no API key needed out of the box. OpenAI uses zvec's native `OpenAIDenseEmbedding`. Gemini is implemented as a custom `GeminiDenseEmbedding` class conforming to zvec's `DenseEmbeddingFunction` Protocol.

### Sparse Embedding

| Provider | Class | Description |
|----------|-------|-------------|
| BM25 (zvecsearch default) | `BM25EmbeddingFunction` | Keyword-based, local, no model download |
| SPLADE (zvec default) | `DefaultLocalSparseEmbedding` | Learned sparse, local, ~100MB model |

### Rerankers

| Reranker | Method | Description |
|----------|--------|-------------|
| **RRF** (default) | Rank fusion | Combines results by rank position. No tuning needed. |
| **Weighted** | Score fusion | Weighted sum of dense/sparse scores. Configurable weights. |
| **DefaultLocalReRanker** | Cross-encoder | ms-marco-MiniLM-L6-v2, higher accuracy, slower. Local, ~80MB. |
| **QwenReRanker** | Cross-encoder | Qwen-based reranker for Chinese/multilingual. |

### Storage

zvec is an **embedded** vector database — no server process needed.

- File-based storage at `~/.zvecsearch/db/` (configurable)
- HNSW index with COSINE metric (M=16, ef_construction=300)
- INT8 quantization by default (also supports INT4, FP16)
- Memory-mapped I/O for efficient large index loading
- Apache Arrow + RocksDB storage backend

### Incremental Indexing

Only changed content is re-embedded, saving API costs:

1. Each chunk gets a SHA-256 `chunk_hash` based on content, source, and line range
2. On re-index, existing hashes are compared — unchanged chunks are skipped
3. Stale chunks (deleted/modified content) are automatically removed

### Markdown Chunking

- Splits documents by headings (`#`, `##`, `###`, etc.)
- Each chunk carries metadata: `source`, `heading`, `heading_level`, `start_line`, `end_line`
- Configurable `max_chunk_size` with `overlap_lines` for context continuity

## Project Structure

```
zvecsearch/
├── src/zvecsearch/
│   ├── core.py        # ZvecSearch orchestrator (sync index/search/watch, async compact)
│   ├── store.py       # ZvecStore (zvec Collection wrapper + GeminiDenseEmbedding)
│   ├── chunker.py     # Markdown chunking (heading-based splitting)
│   ├── scanner.py     # File discovery (.md/.markdown)
│   ├── watcher.py     # File system monitoring (watchdog, debounce)
│   ├── config.py      # TOML config (global/project layered resolution)
│   ├── compact.py     # LLM-based chunk summarization (async)
│   ├── cli.py         # Click CLI interface
│   └── transcript.py  # Transcript utilities
├── tests/             # pytest unit tests (286)
├── benchmarks/        # 5-phase benchmarks (62)
├── scripts/           # Real API embedding test scripts
├── dist/              # Pre-built zvec wheel (x86-64-v2)
└── pyproject.toml
```

## Testing

```bash
# Unit tests (283 tests)
pytest tests/ -v

# Benchmarks (62 tests, no API key needed for Phase 1-4)
pytest benchmarks/ -v

# Phase 5: real embedding comparison (requires API keys)
OPENAI_API_KEY=... GOOGLE_API_KEY=... pytest benchmarks/test_phase5_embeddings.py -v -s

# Real API embedding scripts
OPENAI_API_KEY=... GOOGLE_API_KEY=... python scripts/test_gemini_embedding.py
OPENAI_API_KEY=... GOOGLE_API_KEY=... python scripts/test_zvecsearch_gemini.py

# Default local embedding test (no API key needed)
python scripts/test_default_local.py

# Lint
ruff check src/ tests/ benchmarks/
```

### Benchmark Results

| Phase | Metric | Result |
|-------|--------|--------|
| Phase 2 | Recall@5 | 0.8250 |
| Phase 2 | MRR | 0.8083 |
| Phase 2 | NDCG@5 | 0.7849 |
| Phase 3 | Faithfulness | 0.8667 |
| Phase 3 | Context Relevance | 0.7000 |
| Phase 4 | Chunking throughput | 42K docs/s |
| Phase 4 | Search QPS | 12K QPS |
| Phase 5 | Gemini Recall@5 | 1.0000 |
| Phase 5 | OpenAI Recall@5 | 1.0000 |
| Phase 5 | Keyword Recall@5 | 0.9333 |

Embedding-based search significantly outperforms keyword search on semantic queries (synonyms, paraphrases, cross-lingual).

## License

MIT
