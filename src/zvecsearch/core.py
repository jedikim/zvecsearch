"""ZvecSearch - Core orchestrator for markdown semantic search.

Simplified flow: scan -> chunk -> store (store owns embedding).
index() and search() are synchronous — zvec embedding is sync.
compact() stays async for LLM calls.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Callable

from zvecsearch.chunker import chunk_markdown, compute_chunk_id
from zvecsearch.compact import compact_chunks
from zvecsearch.scanner import ScannedFile, scan_paths
from zvecsearch.store import ZvecStore
from zvecsearch.watcher import FileWatcher

logger = logging.getLogger(__name__)


class ZvecSearch:
    """Core orchestrator for semantic memory search.

    Ties together file scanning, markdown chunking, and zvec-native storage.
    Store owns embedding — callers pass text, store handles vectorization.

    Args:
        paths: Directories or files to scan for markdown content.
        zvec_path: Path to the zvec database directory.
        collection: Name of the zvec collection.
        embedding_provider: zvec embedding provider.
        embedding_model: Embedding model name.
        max_chunk_size: Maximum character size for a single chunk.
        overlap_lines: Number of overlap lines when splitting.
        enable_mmap: Enable memory-mapped I/O.
        hnsw_m: HNSW max connections per node.
        hnsw_ef: HNSW ef_construction parameter.
        quantize_type: Vector quantization type.
        query_ef: HNSW search-time ef.
        reranker: Reranking strategy ("rrf" or "weighted").
        dense_weight: Dense vector weight for weighted reranker.
        sparse_weight: Sparse vector weight for weighted reranker.
    """

    def __init__(
        self,
        paths: list[str | Path] | None = None,
        zvec_path: str = "~/.zvecsearch/db",
        collection: str = "zvecsearch_chunks",
        embedding_provider: str = "openai",
        embedding_model: str = "text-embedding-3-small",
        max_chunk_size: int = 1500,
        overlap_lines: int = 2,
        enable_mmap: bool = True,
        hnsw_m: int = 16,
        hnsw_ef: int = 300,
        quantize_type: str = "int8",
        query_ef: int = 300,
        reranker: str = "rrf",
        dense_weight: float = 1.0,
        sparse_weight: float = 0.8,
    ):
        self._paths = [Path(p) for p in (paths or [])]
        self._max_chunk_size = max_chunk_size
        self._overlap_lines = overlap_lines
        self._store = ZvecStore(
            path=zvec_path,
            collection=collection,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
            enable_mmap=enable_mmap,
            hnsw_m=hnsw_m,
            hnsw_ef=hnsw_ef,
            quantize_type=quantize_type,
            query_ef=query_ef,
            reranker=reranker,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight,
        )

    @property
    def store(self) -> ZvecStore:
        """Access the underlying ZvecStore instance."""
        return self._store

    def index(self, force: bool = False) -> int:
        """Scan and index markdown files. Synchronous — store handles embedding.

        Args:
            force: If True, re-embed all chunks (uses insert, faster).

        Returns:
            Total number of new/updated chunks indexed.
        """
        files = scan_paths(self._paths)
        total = 0
        for f in files:
            total += self._index_file(f, force=force)
        self._store.flush()
        self._store.optimize()
        return total

    def index_file(self, path: str | Path) -> int:
        """Index a single file by path.

        Args:
            path: Path to the markdown file to index.

        Returns:
            Number of chunks indexed from this file.
        """
        path = Path(path).resolve()
        st = path.stat()
        f = ScannedFile(path=path, mtime=st.st_mtime, size=st.st_size)
        n = self._index_file(f)
        self._store.flush()
        return n

    def _index_file(self, f: ScannedFile, force: bool = False) -> int:
        """Chunk, diff, and store a single file. Store handles embedding."""
        text = f.path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(
            text,
            source=str(f.path),
            max_chunk_size=self._max_chunk_size,
            overlap_lines=self._overlap_lines,
        )
        if not chunks:
            return 0

        chunk_dicts = [
            {
                "chunk_hash": compute_chunk_id(
                    c.source, c.start_line, c.end_line, c.content_hash, "zvec"
                ),
                "content": c.content,
                "source": c.source,
                "heading": c.heading,
                "heading_level": c.heading_level,
                "start_line": c.start_line,
                "end_line": c.end_line,
            }
            for c in chunks
        ]

        if force:
            self._store.delete_by_source(str(f.path))
            return self._store.embed_and_insert(chunk_dicts)

        # Incremental: remove stale, skip existing
        new_ids = {d["chunk_hash"] for d in chunk_dicts}
        old_hashes = self._store.hashes_by_source(str(f.path))
        stale = old_hashes - new_ids
        if stale:
            self._store.delete_by_hashes(list(stale))

        existing = self._store.existing_hashes(list(new_ids))
        to_store = [d for d in chunk_dicts if d["chunk_hash"] not in existing]

        if not to_store:
            return 0

        return self._store.embed_and_upsert(to_store)

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search — store handles embedding and hybrid search.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.

        Returns:
            List of result dicts with content, metadata, and score.
        """
        return self._store.search(query_text=query, top_k=top_k)

    async def compact(
        self,
        source: str | None = None,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        prompt_template: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Summarize indexed chunks using an LLM. Async — LLM calls are async."""
        if source:
            chunks = self._store.query(filter_expr=f'source == "{source}"')
        else:
            chunks = []

        if not chunks:
            return ""

        summary = await compact_chunks(
            chunks,
            llm_provider=llm_provider,
            model=llm_model,
            prompt_template=prompt_template,
        )

        if output_dir:
            memory_dir = Path(output_dir) / "memory"
            memory_dir.mkdir(parents=True, exist_ok=True)
            compact_file = memory_dir / f"{date.today()}.md"
            is_new = not compact_file.exists() or compact_file.stat().st_size == 0
            with open(compact_file, "a") as fh:
                if is_new:
                    fh.write(f"# {date.today()}\n")
                fh.write(f"\n\n## Memory Compact\n\n{summary}")
            self.index_file(compact_file)

        return summary

    def watch(
        self,
        on_event: Callable[[str, str, Path], None] | None = None,
        debounce_ms: int | None = None,
    ) -> FileWatcher:
        """Watch paths for markdown changes and auto-index."""
        def callback(event_type: str, path: Path) -> None:
            if event_type in ("created", "modified"):
                self.index_file(path)
            elif event_type == "deleted":
                self._store.delete_by_source(str(path))
            if on_event:
                on_event(event_type, f"{event_type}: {path.name}", path)

        kw = {}
        if debounce_ms is not None:
            kw["debounce_ms"] = debounce_ms

        return FileWatcher(self._paths, callback, **kw)

    def close(self) -> None:
        """Close the underlying store and release resources."""
        self._store.close()

    def __enter__(self) -> ZvecSearch:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
