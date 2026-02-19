# ZvecSearch

Semantic memory search system powered by zvec (Alibaba's embedded vector database).

## Architecture

- `src/zvecsearch/` - main package
  - `core.py` - ZvecSearch orchestrator (sync index/search/watch, async compact)
  - `store.py` - ZvecStore wrapping zvec Collection API
  - `chunker.py` - markdown chunking with heading-based splitting
  - `scanner.py` - file discovery for .md/.markdown files
  - `watcher.py` - file system monitoring with debounce
  - `config.py` - TOML config with layered resolution
  - `compact.py` - LLM-based chunk summarization
  - `cli.py` - Click CLI interface
  - `store.py` includes GeminiDenseEmbedding (OpenAI is zvec-native)
- `tests/` - pytest test suite (TDD)

## Key Commands

```bash
pip install -e ".[dev]"        # Install with dev deps
pytest tests/ -v               # Run all tests
pytest tests/test_store.py -v  # Run specific test
ruff check src/                # Lint
zvecsearch index ./memory/     # Index markdown files
zvecsearch search "query"      # Semantic search
```

## Design Decisions

- zvec-native storage (no Milvus dependency)
- HNSW index with COSINE metric for dense vectors
- BM25EmbeddingFunction for sparse vectors (zvec native)
- RrfReRanker for hybrid search fusion (zvec native)
- File-based storage at ~/.zvecsearch/db/ (embedded, no server)
