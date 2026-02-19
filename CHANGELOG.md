# Changelog

## v0.1.0

- Markdown chunking with heading-based splitting
- Hybrid search (dense + sparse) with RRF/Weighted/Cross-encoder reranking
- Three embedding providers: zvec Default (local), OpenAI, Gemini
- Incremental indexing with SHA-256 chunk hashing
- File watcher with debounce (inotify-based)
- LLM-based chunk summarization (compact)
- Click CLI with index/search/watch/compact/config commands
- TOML config with layered resolution (global/project/CLI)
- 286 unit tests + 62 benchmarks
