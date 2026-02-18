"""ZvecSearch - Core orchestrator tying together scanning, chunking, embedding, and storage.

Provides the main ZvecSearch class that coordinates:
- File discovery via scanner.scan_paths
- Markdown chunking via chunker.chunk_markdown
- Embedding via configurable embedding providers
- Vector storage via ZvecStore
- Incremental indexing with stale chunk cleanup
- Hybrid search (dense + BM25)
- Memory compaction via LLM summarization
- File watching for automatic re-indexing
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path
from typing import Callable

from zvecsearch.chunker import Chunk, chunk_markdown, compute_chunk_id
from zvecsearch.compact import compact_chunks
from zvecsearch.embeddings import get_provider
from zvecsearch.scanner import ScannedFile, scan_paths
from zvecsearch.store import ZvecStore
from zvecsearch.watcher import FileWatcher

logger = logging.getLogger(__name__)


class ZvecSearch:
    """Core orchestrator for semantic memory search.

    Ties together file scanning, markdown chunking, embedding, and vector
    storage into a single high-level API.

    Args:
        paths: Directories or files to scan for markdown content.
        embedding_provider: Name of the embedding provider (e.g., "openai", "ollama").
        embedding_model: Model name override for the embedding provider.
        zvec_path: Path to the zvec database directory.
        collection: Name of the zvec collection.
        max_chunk_size: Maximum character size for a single chunk.
        overlap_lines: Number of overlap lines when splitting large sections.
        index_metric: Distance metric for HNSW index ("cosine", "l2", "ip").
        hnsw_ef: HNSW ef_construction parameter.
        hnsw_max_m: HNSW max_m parameter.
    """

    def __init__(
        self,
        paths: list[str | Path],
        embedding_provider: str = "openai",
        embedding_model: str | None = None,
        zvec_path: str = "~/.zvecsearch/db",
        collection: str = "zvecsearch_chunks",
        max_chunk_size: int = 1500,
        overlap_lines: int = 2,
        index_metric: str = "cosine",
        hnsw_ef: int = 300,
        hnsw_max_m: int = 16,
    ):
        self._paths = [Path(p) for p in paths]
        self._max_chunk_size = max_chunk_size
        self._overlap_lines = overlap_lines
        self._embedder = get_provider(embedding_provider, embedding_model)
        self._store = ZvecStore(
            path=zvec_path,
            collection=collection,
            dimension=self._embedder.dimension,
            index_metric=index_metric,
            hnsw_ef=hnsw_ef,
            hnsw_max_m=hnsw_max_m,
        )

    @property
    def store(self) -> ZvecStore:
        """Access the underlying ZvecStore instance."""
        return self._store

    async def index(self, force: bool = False) -> int:
        """Scan all configured paths and index discovered markdown files.

        Performs incremental indexing by default: only new or changed chunks
        are embedded and stored. Stale chunks (from modified/deleted content)
        are automatically cleaned up.

        Args:
            force: If True, re-embed all chunks regardless of existing state.

        Returns:
            Total number of new/updated chunks indexed.
        """
        files = scan_paths(self._paths)
        total = 0
        for f in files:
            total += await self._index_file(f, force=force)
        return total

    async def index_file(self, path: str | Path) -> int:
        """Index a single file by path.

        Args:
            path: Path to the markdown file to index.

        Returns:
            Number of chunks indexed from this file.
        """
        path = Path(path).resolve()
        st = path.stat()
        f = ScannedFile(path=path, mtime=st.st_mtime, size=st.st_size)
        return await self._index_file(f)

    async def _index_file(self, f: ScannedFile, force: bool = False) -> int:
        """Internal: chunk, diff, embed, and store a single scanned file.

        Args:
            f: ScannedFile to process.
            force: If True, skip diffing and re-embed everything.

        Returns:
            Number of chunks embedded and stored.
        """
        text = f.path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(
            text,
            source=str(f.path),
            max_chunk_size=self._max_chunk_size,
            overlap_lines=self._overlap_lines,
        )
        if not chunks:
            return 0

        model = self._embedder.model_name
        new_ids = {
            compute_chunk_id(c.source, c.start_line, c.end_line, c.content_hash, model): c
            for c in chunks
        }

        if not force:
            # Incremental: remove stale chunks and skip existing ones
            old_hashes = self._store.hashes_by_source(str(f.path))
            stale = old_hashes - set(new_ids.keys())
            if stale:
                self._store.delete_by_hashes(list(stale))

            existing = self._store.existing_hashes(list(new_ids.keys()))
            to_embed = {k: v for k, v in new_ids.items() if k not in existing}
        else:
            # Force: delete everything for this source and re-embed
            self._store.delete_by_source(str(f.path))
            to_embed = new_ids

        if not to_embed:
            return 0

        return await self._embed_and_store(list(to_embed.items()))

    async def _embed_and_store(self, items: list[tuple[str, Chunk]]) -> int:
        """Embed chunk texts and upsert the resulting records into the store.

        Args:
            items: List of (chunk_id, Chunk) tuples to embed and store.

        Returns:
            Number of records upserted.
        """
        texts = [c.content for _, c in items]
        embeddings = await self._embedder.embed(texts)

        records = []
        for (chunk_id, chunk), embedding in zip(items, embeddings):
            records.append({
                "chunk_hash": chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "heading": chunk.heading,
                "heading_level": chunk.heading_level,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "embedding": embedding,
            })

        return self._store.upsert(records)

    async def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search for chunks semantically similar to the query.

        Embeds the query text, then performs hybrid dense + BM25 search
        via the underlying ZvecStore.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.

        Returns:
            List of result dicts with content, metadata, and score.
        """
        embeddings = await self._embedder.embed([query])
        return self._store.search(
            query_embedding=embeddings[0],
            query_text=query,
            top_k=top_k,
        )

    async def compact(
        self,
        source: str | None = None,
        llm_provider: str = "openai",
        llm_model: str | None = None,
        prompt_template: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """Summarize indexed chunks using an LLM.

        Queries chunks (optionally filtered by source), sends them to the
        compact_chunks LLM summarizer, and optionally writes the result
        to a dated markdown file in output_dir/memory/.

        Args:
            source: Filter chunks by source file path.
            llm_provider: LLM provider for summarization ("openai", "anthropic", "gemini").
            llm_model: Model name override for the LLM provider.
            prompt_template: Custom prompt template (must contain {chunks} placeholder).
            output_dir: If set, write the summary to output_dir/memory/YYYY-MM-DD.md.

        Returns:
            The summarized text, or empty string if no chunks found.
        """
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
            await self.index_file(compact_file)

        return summary

    def watch(
        self,
        on_event: Callable[[str, str, Path], None] | None = None,
        debounce_ms: int | None = None,
    ) -> FileWatcher:
        """Start watching configured paths for markdown file changes.

        Creates a FileWatcher that automatically indexes new/modified files
        and removes deleted files from the store.

        Args:
            on_event: Optional callback(event_type, message, path) for notifications.
            debounce_ms: Debounce interval in milliseconds for file events.

        Returns:
            A FileWatcher instance (call .start() to begin watching).
        """
        def callback(event_type: str, path: Path) -> None:
            if event_type in ("created", "modified"):
                asyncio.run(self.index_file(path))
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
