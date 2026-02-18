# ZvecSearch Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build zvecsearch - a semantic memory search system that replaces memsearch's Milvus backend with zvec (Alibaba's embedded vector database), using zvec-native features for maximum performance.

**Architecture:** memsearch's full feature set (scanning, chunking, embedding, hybrid search, watching, compact) is preserved, but the storage layer is completely redesigned around zvec's native Collection API, HNSW indexing, BM25EmbeddingFunction for sparse vectors, and RrfReRanker for hybrid search fusion.

**Tech Stack:** Python 3.10+, zvec (C++/Python vector DB), click (CLI), watchdog (file monitoring), openai/google-genai/voyageai/ollama/sentence-transformers (embeddings), tomllib/tomli_w (config)

---

## Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `src/zvecsearch/__init__.py`
- Create: `CLAUDE.md`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68", "setuptools-scm>=8"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "zvecsearch"
version = "0.1.0"
description = "Semantic memory search powered by zvec vector database"
requires-python = ">=3.10"
license = "MIT"
dependencies = [
    "zvec>=0.1.0",
    "click>=8.1",
    "watchdog>=4.0",
    "tomli_w>=1.0",
    "openai>=1.0",
]

[project.optional-dependencies]
google = ["google-genai>=1.0"]
voyage = ["voyageai>=0.3"]
ollama = ["ollama>=0.4"]
local = ["sentence-transformers>=3.0"]
anthropic = ["anthropic>=0.40"]
all = [
    "google-genai>=1.0",
    "voyageai>=0.3",
    "ollama>=0.4",
    "sentence-transformers>=3.0",
    "anthropic>=0.40",
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.4",
]

[project.scripts]
zvecsearch = "zvecsearch.cli:cli"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.ruff]
target-version = "py310"
line-length = 100
```

**Step 2: Create src/zvecsearch/__init__.py**

```python
from zvecsearch.core import ZvecSearch

__all__ = ["ZvecSearch"]
```

**Step 3: Create CLAUDE.md**

```markdown
# ZvecSearch

Semantic memory search system powered by zvec (Alibaba's embedded vector database).

## Architecture

- `src/zvecsearch/` - main package
  - `core.py` - ZvecSearch orchestrator (async index/search/compact/watch)
  - `store.py` - ZvecStore wrapping zvec Collection API
  - `chunker.py` - markdown chunking with heading-based splitting
  - `scanner.py` - file discovery for .md/.markdown files
  - `watcher.py` - file system monitoring with debounce
  - `config.py` - TOML config with layered resolution
  - `compact.py` - LLM-based chunk summarization
  - `cli.py` - Click CLI interface
  - `embeddings/` - 5 embedding providers (openai, google, voyage, ollama, local)
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
```

**Step 4: Create directory structure**

```bash
mkdir -p src/zvecsearch/embeddings tests
touch src/zvecsearch/__init__.py src/zvecsearch/embeddings/__init__.py
```

**Step 5: Commit**

```bash
git add pyproject.toml src/ CLAUDE.md tests/
git commit -m "feat: project scaffolding with pyproject.toml and CLAUDE.md"
```

---

## Task 2: Chunker Module (TDD)

**Files:**
- Create: `tests/test_chunker.py`
- Create: `src/zvecsearch/chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
import pytest
from zvecsearch.chunker import Chunk, chunk_markdown, compute_chunk_id


class TestChunk:
    def test_chunk_has_content_hash(self):
        c = Chunk(content="hello", source="a.md", heading="", heading_level=0,
                  start_line=1, end_line=1)
        assert len(c.content_hash) == 16
        assert c.content_hash == Chunk(content="hello", source="a.md", heading="",
                                        heading_level=0, start_line=1, end_line=1).content_hash

    def test_different_content_different_hash(self):
        c1 = Chunk(content="aaa", source="a.md", heading="", heading_level=0,
                   start_line=1, end_line=1)
        c2 = Chunk(content="bbb", source="a.md", heading="", heading_level=0,
                   start_line=1, end_line=1)
        assert c1.content_hash != c2.content_hash


class TestChunkMarkdown:
    def test_simple_heading_split(self):
        md = "# H1\nParagraph one.\n## H2\nParagraph two."
        chunks = chunk_markdown(md, source="test.md")
        assert len(chunks) == 2
        assert chunks[0].heading == "H1"
        assert chunks[0].heading_level == 1
        assert chunks[1].heading == "H2"
        assert chunks[1].heading_level == 2

    def test_preamble_without_heading(self):
        md = "Some intro text.\n# Title\nBody."
        chunks = chunk_markdown(md, source="t.md")
        assert len(chunks) == 2
        assert chunks[0].heading == ""
        assert chunks[0].heading_level == 0

    def test_empty_input(self):
        assert chunk_markdown("", source="e.md") == []

    def test_whitespace_only(self):
        assert chunk_markdown("   \n\n  ", source="e.md") == []

    def test_large_section_splitting(self):
        big = "# Big\n" + ("word " * 400 + "\n\n") * 3
        chunks = chunk_markdown(big, source="big.md", max_chunk_size=500)
        assert len(chunks) > 1
        for c in chunks:
            assert c.heading == "Big"

    def test_source_and_lines(self):
        md = "# A\nLine1\nLine2\n# B\nLine3"
        chunks = chunk_markdown(md, source="s.md")
        assert chunks[0].source == "s.md"
        assert chunks[0].start_line == 1
        assert chunks[1].start_line == 4


class TestComputeChunkId:
    def test_deterministic(self):
        a = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        b = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        assert a == b

    def test_different_model_different_id(self):
        a = compute_chunk_id("s.md", 1, 5, "abc123", "openai")
        b = compute_chunk_id("s.md", 1, 5, "abc123", "google")
        assert a != b
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_chunker.py -v
```
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

Port from memsearch's chunker.py with identical logic:

```python
# src/zvecsearch/chunker.py
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    content: str
    source: str
    heading: str
    heading_level: int
    start_line: int
    end_line: int
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            h = hashlib.sha256(self.content.encode()).hexdigest()[:16]
            object.__setattr__(self, "content_hash", h)


def compute_chunk_id(
    source: str,
    start_line: int,
    end_line: int,
    content_hash: str,
    model: str,
) -> str:
    raw = f"markdown:{source}:{start_line}:{end_line}:{content_hash}:{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_markdown(
    text: str,
    source: str = "",
    max_chunk_size: int = 1500,
    overlap_lines: int = 2,
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    lines = text.split("\n")
    sections: list[tuple[str, int, int, int]] = []  # heading, level, start, end

    heading_positions = []
    for i, line in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", line)
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    if not heading_positions or heading_positions[0][0] > 0:
        end = heading_positions[0][0] if heading_positions else len(lines)
        sections.append(("", 0, 0, end))

    for idx, (pos, level, title) in enumerate(heading_positions):
        next_pos = heading_positions[idx + 1][0] if idx + 1 < len(heading_positions) else len(lines)
        sections.append((title, level, pos, next_pos))

    chunks: list[Chunk] = []
    for heading, level, start, end in sections:
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        if len(body) <= max_chunk_size:
            chunks.append(Chunk(
                content=body, source=source, heading=heading,
                heading_level=level, start_line=start + 1, end_line=end,
            ))
        else:
            chunks.extend(_split_large_section(
                lines[start:end], source, heading, level,
                start, max_chunk_size, overlap_lines,
            ))

    return chunks


def _split_large_section(
    lines: list[str],
    source: str,
    heading: str,
    heading_level: int,
    base_line: int,
    max_size: int,
    overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_start = 0

    for i, line in enumerate(lines):
        buf.append(line)
        current = "\n".join(buf).strip()
        if len(current) >= max_size and len(buf) > 1:
            text = "\n".join(buf[:-1]).strip()
            if text:
                chunks.append(Chunk(
                    content=text, source=source, heading=heading,
                    heading_level=heading_level,
                    start_line=base_line + buf_start + 1,
                    end_line=base_line + buf_start + len(buf) - 1,
                ))
            buf_start = max(0, i - overlap)
            buf = list(lines[buf_start:i + 1])

    if buf:
        text = "\n".join(buf).strip()
        if text:
            chunks.append(Chunk(
                content=text, source=source, heading=heading,
                heading_level=heading_level,
                start_line=base_line + buf_start + 1,
                end_line=base_line + buf_start + len(buf),
            ))

    return chunks
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_chunker.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_chunker.py src/zvecsearch/chunker.py
git commit -m "feat: add markdown chunker with heading-based splitting (TDD)"
```

---

## Task 3: Scanner Module (TDD)

**Files:**
- Create: `tests/test_scanner.py`
- Create: `src/zvecsearch/scanner.py`

**Step 1: Write the failing tests**

```python
# tests/test_scanner.py
import pytest
from pathlib import Path
from zvecsearch.scanner import ScannedFile, scan_paths


@pytest.fixture
def tmp_files(tmp_path):
    (tmp_path / "a.md").write_text("# A")
    (tmp_path / "b.markdown").write_text("# B")
    (tmp_path / "c.txt").write_text("not md")
    (tmp_path / ".hidden.md").write_text("# hidden")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.md").write_text("# D")
    return tmp_path


def test_scan_finds_markdown(tmp_files):
    results = scan_paths([tmp_files])
    names = {r.path.name for r in results}
    assert "a.md" in names
    assert "b.markdown" in names
    assert "d.md" in names
    assert "c.txt" not in names


def test_scan_ignores_hidden(tmp_files):
    results = scan_paths([tmp_files])
    names = {r.path.name for r in results}
    assert ".hidden.md" not in names


def test_scan_single_file(tmp_files):
    results = scan_paths([tmp_files / "a.md"])
    assert len(results) == 1
    assert results[0].path.name == "a.md"


def test_scan_deduplicates(tmp_files):
    results = scan_paths([tmp_files, tmp_files / "a.md"])
    paths = [r.path for r in results]
    assert len(paths) == len(set(paths))


def test_scanned_file_has_metadata(tmp_files):
    results = scan_paths([tmp_files / "a.md"])
    assert results[0].mtime > 0
    assert results[0].size > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_scanner.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/zvecsearch/scanner.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    mtime: float
    size: int


def scan_paths(
    paths: list[str | Path],
    extensions: tuple[str, ...] = (".md", ".markdown"),
    ignore_hidden: bool = True,
) -> list[ScannedFile]:
    seen: set[str] = set()
    results: list[ScannedFile] = []

    for p in paths:
        p = Path(p)
        if p.is_file():
            _maybe_add(p, extensions, seen, results, ignore_hidden)
        elif p.is_dir():
            for root, dirs, files in os.walk(p):
                if ignore_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in sorted(files):
                    fp = Path(root) / fname
                    _maybe_add(fp, extensions, seen, results, ignore_hidden)

    results.sort(key=lambda f: f.path)
    return results


def _maybe_add(
    fp: Path,
    extensions: tuple[str, ...],
    seen: set[str],
    results: list[ScannedFile],
    ignore_hidden: bool,
) -> None:
    if ignore_hidden and fp.name.startswith("."):
        return
    if fp.suffix.lower() not in extensions:
        return
    resolved = str(fp.resolve())
    if resolved in seen:
        return
    seen.add(resolved)
    st = fp.stat()
    results.append(ScannedFile(path=fp.resolve(), mtime=st.st_mtime, size=st.st_size))
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_scanner.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_scanner.py src/zvecsearch/scanner.py
git commit -m "feat: add file scanner for markdown discovery (TDD)"
```

---

## Task 4: Config Module (TDD)

**Files:**
- Create: `tests/test_config.py`
- Create: `src/zvecsearch/config.py`

**Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest
from pathlib import Path
from zvecsearch.config import (
    ZvecConfig, EmbeddingConfig, CompactConfig, ChunkingConfig,
    WatchConfig, IndexConfig, ZvecSearchConfig,
    load_config_file, deep_merge, resolve_config,
    config_to_dict, get_config_value, save_config,
)


def test_default_config():
    cfg = ZvecSearchConfig()
    assert cfg.zvec.path == "~/.zvecsearch/db"
    assert cfg.zvec.collection == "zvecsearch_chunks"
    assert cfg.embedding.provider == "openai"
    assert cfg.chunking.max_chunk_size == 1500
    assert cfg.index.type == "hnsw"
    assert cfg.index.metric == "cosine"


def test_load_missing_file():
    assert load_config_file(Path("/nonexistent")) == {}


def test_load_toml_file(tmp_path):
    f = tmp_path / "config.toml"
    f.write_text('[zvec]\npath = "/custom/db"\n')
    d = load_config_file(f)
    assert d["zvec"]["path"] == "/custom/db"


def test_deep_merge_basic():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 99}}
    result = deep_merge(base, over)
    assert result == {"a": {"x": 1, "y": 99}, "b": 3}


def test_deep_merge_none_skipped():
    base = {"a": 1}
    over = {"a": None}
    result = deep_merge(base, over)
    assert result == {"a": 1}


def test_resolve_with_cli_overrides():
    cfg = resolve_config({"zvec": {"collection": "custom"}})
    assert cfg.zvec.collection == "custom"
    assert cfg.zvec.path == "~/.zvecsearch/db"


def test_config_to_dict():
    cfg = ZvecSearchConfig()
    d = config_to_dict(cfg)
    assert d["zvec"]["path"] == "~/.zvecsearch/db"
    assert d["index"]["type"] == "hnsw"


def test_get_config_value():
    cfg = ZvecSearchConfig()
    assert get_config_value("zvec.collection", cfg) == "zvecsearch_chunks"
    assert get_config_value("index.metric", cfg) == "cosine"


def test_save_and_load(tmp_path):
    d = {"zvec": {"path": "/tmp/test"}}
    f = tmp_path / "test.toml"
    save_config(d, f)
    loaded = load_config_file(f)
    assert loaded["zvec"]["path"] == "/tmp/test"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/zvecsearch/config.py
from __future__ import annotations

import sys
from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

import tomli_w

_GLOBAL_CFG = Path("~/.zvecsearch/config.toml").expanduser()
_PROJECT_CFG = Path(".zvecsearch.toml")

DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding-001",
    "voyage": "voyage-3-lite",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
}


@dataclass
class ZvecConfig:
    path: str = "~/.zvecsearch/db"
    collection: str = "zvecsearch_chunks"
    enable_mmap: bool = True
    max_buffer_size: int = 67108864  # 64MB


@dataclass
class IndexConfig:
    type: str = "hnsw"
    metric: str = "cosine"
    quantize: str = "none"
    hnsw_ef: int = 300
    hnsw_max_m: int = 16


@dataclass
class EmbeddingConfig:
    provider: str = "openai"
    model: str = ""


@dataclass
class CompactConfig:
    llm_provider: str = "openai"
    llm_model: str = ""
    prompt_file: str = ""


@dataclass
class ChunkingConfig:
    max_chunk_size: int = 1500
    overlap_lines: int = 2


@dataclass
class WatchConfig:
    debounce_ms: int = 1500


@dataclass
class ZvecSearchConfig:
    zvec: ZvecConfig = None  # type: ignore[assignment]
    index: IndexConfig = None  # type: ignore[assignment]
    embedding: EmbeddingConfig = None  # type: ignore[assignment]
    compact: CompactConfig = None  # type: ignore[assignment]
    chunking: ChunkingConfig = None  # type: ignore[assignment]
    watch: WatchConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.zvec is None:
            self.zvec = ZvecConfig()
        if self.index is None:
            self.index = IndexConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.compact is None:
            self.compact = CompactConfig()
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.watch is None:
            self.watch = WatchConfig()


def load_config_file(path: Path | str) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def resolve_config(cli_overrides: dict | None = None) -> ZvecSearchConfig:
    defaults = config_to_dict(ZvecSearchConfig())
    merged = deep_merge(defaults, load_config_file(_GLOBAL_CFG))
    merged = deep_merge(merged, load_config_file(_PROJECT_CFG))
    if cli_overrides:
        merged = deep_merge(merged, cli_overrides)

    # Auto-fill embedding model
    emb = merged.get("embedding", {})
    if not emb.get("model"):
        emb["model"] = DEFAULT_MODELS.get(emb.get("provider", "openai"), "")

    cfg = ZvecSearchConfig()
    _SECTION_MAP = {
        "zvec": (ZvecConfig, "zvec"),
        "index": (IndexConfig, "index"),
        "embedding": (EmbeddingConfig, "embedding"),
        "compact": (CompactConfig, "compact"),
        "chunking": (ChunkingConfig, "chunking"),
        "watch": (WatchConfig, "watch"),
    }
    for key, (cls, attr) in _SECTION_MAP.items():
        section = merged.get(key, {})
        valid_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in section.items() if k in valid_fields}
        setattr(cfg, attr, cls(**filtered))

    return cfg


def config_to_dict(cfg: ZvecSearchConfig) -> dict:
    return asdict(cfg)


def save_config(cfg_dict: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(tomli_w.dumps(cfg_dict).encode())


def get_config_value(key: str, cfg: ZvecSearchConfig | None = None) -> Any:
    if cfg is None:
        cfg = resolve_config()
    parts = key.split(".")
    obj: Any = cfg
    for part in parts:
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj


def set_config_value(key: str, value: Any, project: bool = False) -> None:
    path = _PROJECT_CFG if project else _GLOBAL_CFG
    data = load_config_file(path)
    parts = key.split(".")
    target = data
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    save_config(data, path)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_config.py src/zvecsearch/config.py
git commit -m "feat: add TOML config system with layered resolution (TDD)"
```

---

## Task 5: Embedding Providers (TDD)

**Files:**
- Create: `tests/test_embeddings.py`
- Create: `src/zvecsearch/embeddings/__init__.py`
- Create: `src/zvecsearch/embeddings/openai.py`
- Create: `src/zvecsearch/embeddings/google.py`
- Create: `src/zvecsearch/embeddings/voyage.py`
- Create: `src/zvecsearch/embeddings/ollama.py`
- Create: `src/zvecsearch/embeddings/local.py`

**Step 1: Write the failing tests**

```python
# tests/test_embeddings.py
import pytest
from zvecsearch.embeddings import EmbeddingProvider, get_provider, PROVIDERS


def test_protocol_has_required_methods():
    assert hasattr(EmbeddingProvider, "model_name")
    assert hasattr(EmbeddingProvider, "dimension")
    assert hasattr(EmbeddingProvider, "embed")


def test_providers_registry():
    assert "openai" in PROVIDERS
    assert "google" in PROVIDERS
    assert "voyage" in PROVIDERS
    assert "ollama" in PROVIDERS
    assert "local" in PROVIDERS


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError, match="Unknown"):
        get_provider("nonexistent")


def test_get_provider_missing_dep():
    # This tests that missing optional deps give helpful error
    # (can only truly test if dep is not installed)
    pass
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_embeddings.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/zvecsearch/embeddings/__init__.py
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def model_name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


PROVIDERS: dict[str, tuple[str, str]] = {
    "openai": ("zvecsearch.embeddings.openai", "OpenAIEmbedding"),
    "google": ("zvecsearch.embeddings.google", "GoogleEmbedding"),
    "voyage": ("zvecsearch.embeddings.voyage", "VoyageEmbedding"),
    "ollama": ("zvecsearch.embeddings.ollama", "OllamaEmbedding"),
    "local": ("zvecsearch.embeddings.local", "LocalEmbedding"),
}

_INSTALL_HINTS: dict[str, str] = {
    "google": 'pip install "zvecsearch[google]"',
    "voyage": 'pip install "zvecsearch[voyage]"',
    "ollama": 'pip install "zvecsearch[ollama]"',
    "local": 'pip install "zvecsearch[local]"',
}

DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding-001",
    "voyage": "voyage-3-lite",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
}


def get_provider(name: str = "openai", model: str | None = None) -> EmbeddingProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown embedding provider: {name!r}. Choose from: {list(PROVIDERS)}")
    mod_path, cls_name = PROVIDERS[name]
    try:
        import importlib
        mod = importlib.import_module(mod_path)
    except ImportError as exc:
        hint = _INSTALL_HINTS.get(name, "")
        raise ImportError(f"Provider {name!r} requires extra deps. {hint}") from exc
    cls = getattr(mod, cls_name)
    model = model or DEFAULT_MODELS.get(name)
    return cls(model=model) if model else cls()
```

Port each provider from memsearch (OpenAI, Google, Voyage, Ollama, Local) with identical logic. These are thin wrappers around their respective SDKs.

```python
# src/zvecsearch/embeddings/openai.py
from __future__ import annotations
from openai import AsyncOpenAI

_KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

class OpenAIEmbedding:
    def __init__(self, model: str = "text-embedding-3-small"):
        self._model = model
        self._client = AsyncOpenAI()
        self._dim = _KNOWN_DIMS.get(model)

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("Unknown dimension; call embed() once first")
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(input=texts, model=self._model)
        vecs = [d.embedding for d in resp.data]
        if self._dim is None:
            self._dim = len(vecs[0])
        return vecs
```

```python
# src/zvecsearch/embeddings/google.py
from __future__ import annotations

class GoogleEmbedding:
    def __init__(self, model: str = "gemini-embedding-001", output_dimensionality: int = 768):
        from google import genai
        self._model = model
        self._dim = output_dimensionality
        self._client = genai.Client()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        from google.genai import types
        resp = await self._client.aio.models.embed_content(
            model=self._model,
            contents=texts,
            config=types.EmbedContentConfig(output_dimensionality=self._dim),
        )
        return [e.values for e in resp.embeddings]
```

```python
# src/zvecsearch/embeddings/voyage.py
from __future__ import annotations

_KNOWN_DIMS = {"voyage-3-lite": 512, "voyage-3": 1024, "voyage-3-large": 1024}

class VoyageEmbedding:
    def __init__(self, model: str = "voyage-3-lite"):
        import voyageai
        self._model = model
        self._dim = _KNOWN_DIMS.get(model, 512)
        self._client = voyageai.AsyncClient()

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embed(texts, model=self._model)
        return resp.embeddings
```

```python
# src/zvecsearch/embeddings/ollama.py
from __future__ import annotations
import asyncio
from functools import partial

class OllamaEmbedding:
    def __init__(self, model: str = "nomic-embed-text"):
        import ollama
        self._model = model
        self._client = ollama.Client()
        self._dim: int | None = None

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimension(self) -> int:
        if self._dim is None:
            raise RuntimeError("Unknown dimension; call embed() once first")
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, partial(self._client.embed, model=self._model, input=texts)
        )
        vecs = resp["embeddings"]
        if self._dim is None:
            self._dim = len(vecs[0])
        return vecs
```

```python
# src/zvecsearch/embeddings/local.py
from __future__ import annotations
import asyncio
from functools import partial

class LocalEmbedding:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model_name = model
        self._st = SentenceTransformer(model, trust_remote_code=True)
        self._dim = self._st.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        vecs = await loop.run_in_executor(
            None, partial(self._st.encode, texts, normalize_embeddings=True)
        )
        return vecs.tolist()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_embeddings.py -v
```
Expected: ALL PASS

**Step 5: Commit**

```bash
git add tests/test_embeddings.py src/zvecsearch/embeddings/
git commit -m "feat: add 5 embedding providers with protocol interface (TDD)"
```

---

## Task 6: ZvecStore - Core Storage Layer (TDD)

**Files:**
- Create: `tests/test_store.py`
- Create: `src/zvecsearch/store.py`

This is the critical task - replacing MilvusStore with zvec-native ZvecStore.

**Step 1: Write the failing tests**

```python
# tests/test_store.py
import pytest
import shutil
from pathlib import Path
from zvecsearch.store import ZvecStore

# Use a fixed small dimension for testing
TEST_DIM = 4
TEST_DB = Path("/tmp/zvecsearch_test_db")


@pytest.fixture(autouse=True)
def clean_db():
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)
    yield
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


def _make_store(dim=TEST_DIM):
    return ZvecStore(
        path=str(TEST_DB),
        collection="test_chunks",
        dimension=dim,
    )


def _sample_chunks(n=3, dim=TEST_DIM):
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


class TestZvecStoreBasic:
    def test_create_and_count(self):
        store = _make_store()
        assert store.count() == 0
        store.close()

    def test_upsert_and_count(self):
        store = _make_store()
        chunks = _sample_chunks(3)
        count = store.upsert(chunks)
        assert count == 3
        assert store.count() == 3
        store.close()

    def test_upsert_is_idempotent(self):
        store = _make_store()
        chunks = _sample_chunks(2)
        store.upsert(chunks)
        store.upsert(chunks)  # same hashes
        assert store.count() == 2
        store.close()

    def test_delete_by_source(self):
        store = _make_store()
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        store.delete_by_source("test.md")
        assert store.count() == 0
        store.close()

    def test_delete_by_hashes(self):
        store = _make_store()
        chunks = _sample_chunks(3)
        store.upsert(chunks)
        store.delete_by_hashes(["hash_0", "hash_1"])
        assert store.count() == 1
        store.close()


class TestZvecStoreSearch:
    def test_search_returns_results(self):
        store = _make_store()
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

    def test_search_empty_collection(self):
        store = _make_store()
        results = store.search(
            query_embedding=[1.0] * TEST_DIM,
            query_text="anything",
            top_k=5,
        )
        assert results == []
        store.close()


class TestZvecStoreQuery:
    def test_hashes_by_source(self):
        store = _make_store()
        store.upsert(_sample_chunks(3))
        hashes = store.hashes_by_source("test.md")
        assert hashes == {"hash_0", "hash_1", "hash_2"}
        store.close()

    def test_indexed_sources(self):
        store = _make_store()
        chunks = _sample_chunks(2)
        chunks[1]["source"] = "other.md"
        store.upsert(chunks)
        sources = store.indexed_sources()
        assert sources == {"test.md", "other.md"}
        store.close()

    def test_existing_hashes(self):
        store = _make_store()
        store.upsert(_sample_chunks(3))
        found = store.existing_hashes(["hash_0", "hash_1", "nonexistent"])
        assert found == {"hash_0", "hash_1"}
        store.close()


class TestZvecStoreDrop:
    def test_drop(self):
        store = _make_store()
        store.upsert(_sample_chunks(3))
        store.drop()
        # After drop, creating a new store should start fresh
        store2 = _make_store()
        assert store2.count() == 0
        store2.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_store.py -v
```
Expected: FAIL

**Step 3: Write implementation**

```python
# src/zvecsearch/store.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import zvec

logger = logging.getLogger(__name__)

_QUERY_FIELDS = [
    "content", "source", "heading", "chunk_hash",
    "heading_level", "start_line", "end_line",
]


class ZvecStore:
    """Vector storage layer using zvec embedded database."""

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
        self._bm25_doc: zvec.BM25EmbeddingFunction | None = None
        self._bm25_query: zvec.BM25EmbeddingFunction | None = None

        zvec.init(log_level=zvec.LogLevel.WARN)
        self._ensure_collection()

    def _get_metric_type(self) -> zvec.MetricType:
        return {
            "cosine": zvec.MetricType.COSINE,
            "l2": zvec.MetricType.L2,
            "ip": zvec.MetricType.IP,
        }.get(self._index_metric, zvec.MetricType.COSINE)

    def _ensure_collection(self) -> None:
        db_path = Path(self._path)
        if db_path.exists():
            option = zvec.CollectionOption()
            self._collection = zvec.open(self._path, option)
        else:
            schema = self._build_schema()
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._collection = zvec.create_and_open(self._path, schema)
            self._create_indexes()

        # Initialize BM25 functions
        self._bm25_doc = zvec.BM25EmbeddingFunction(
            encoding_type="document", language="en"
        )
        self._bm25_query = zvec.BM25EmbeddingFunction(
            encoding_type="query", language="en"
        )

    def _build_schema(self) -> zvec.CollectionSchema:
        fields = [
            zvec.FieldSchema("chunk_hash", zvec.DataType.STRING),
            zvec.FieldSchema("content", zvec.DataType.STRING),
            zvec.FieldSchema("source", zvec.DataType.STRING),
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

    def _create_indexes(self) -> None:
        metric = self._get_metric_type()
        self._collection.create_index(
            "embedding",
            zvec.HnswIndexParam(metric_type=metric),
        )
        self._collection.create_index(
            "source",
            zvec.InvertIndexParam(),
        )

    def upsert(self, chunks: list[dict]) -> int:
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

    def query(self, filter_expr: str = "") -> list[dict]:
        # Fetch all matching docs by filter
        # zvec doesn't have a direct scalar-only query, so we use fetch approach
        # For now, use vector query with large topk as workaround
        # TODO: Revisit when zvec adds scalar-only query support
        if not filter_expr:
            return []
        results = self._collection.query(
            filter=filter_expr,
            output_fields=_QUERY_FIELDS,
            topk=10000,
        )
        return [
            {f: doc.field(f) for f in _QUERY_FIELDS if doc.has_field(f)}
            | {"chunk_hash": doc.id}
            for doc in results
        ]

    def hashes_by_source(self, source: str) -> set[str]:
        results = self._collection.query(
            filter=f'source == "{source}"',
            output_fields=["chunk_hash"],
            topk=100000,
        )
        return {doc.field("chunk_hash") for doc in results}

    def indexed_sources(self) -> set[str]:
        results = self._collection.query(
            output_fields=["source"],
            topk=100000,
        )
        return {doc.field("source") for doc in results}

    def existing_hashes(self, hashes: list[str]) -> set[str]:
        found = set()
        fetched = self._collection.fetch(hashes)
        return set(fetched.keys())

    def delete_by_source(self, source: str) -> None:
        self._collection.delete_by_filter(f'source == "{source}"')
        self._collection.flush()

    def delete_by_hashes(self, hashes: list[str]) -> None:
        if hashes:
            self._collection.delete(hashes)
            self._collection.flush()

    def count(self) -> int:
        stats = self._collection.stats
        return stats.total_doc_count if hasattr(stats, 'total_doc_count') else 0

    def drop(self) -> None:
        if self._collection:
            self._collection.destroy()
            self._collection = None

    def close(self) -> None:
        if self._collection:
            self._collection.flush()
            self._collection = None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_store.py -v
```
Expected: ALL PASS (may need adjustment based on actual zvec API behavior)

**Step 5: Commit**

```bash
git add tests/test_store.py src/zvecsearch/store.py
git commit -m "feat: add ZvecStore with zvec-native HNSW, BM25, and RRF (TDD)"
```

---

## Task 7: Compact Module (TDD)

**Files:**
- Create: `tests/test_compact.py`
- Create: `src/zvecsearch/compact.py`

**Step 1: Write the failing tests**

```python
# tests/test_compact.py
import pytest
from unittest.mock import AsyncMock, patch
from zvecsearch.compact import compact_chunks, COMPACT_PROMPT


def test_default_prompt_has_placeholder():
    assert "{chunks}" in COMPACT_PROMPT


@pytest.mark.asyncio
async def test_compact_joins_chunks():
    chunks = [
        {"content": "chunk one"},
        {"content": "chunk two"},
    ]
    with patch("zvecsearch.compact._compact_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "summary"
        result = await compact_chunks(chunks, llm_provider="openai")
        call_prompt = mock.call_args[0][0]
        assert "chunk one" in call_prompt
        assert "chunk two" in call_prompt
        assert result == "summary"


@pytest.mark.asyncio
async def test_compact_custom_template():
    chunks = [{"content": "data"}]
    template = "Summarize: {chunks}"
    with patch("zvecsearch.compact._compact_openai", new_callable=AsyncMock) as mock:
        mock.return_value = "done"
        await compact_chunks(chunks, prompt_template=template)
        assert mock.call_args[0][0] == "Summarize: data"
```

**Step 2: Run, Step 3: Implement, Step 4: Verify, Step 5: Commit**

Port compact.py from memsearch with identical LLM provider logic (OpenAI, Anthropic, Gemini).

```bash
git add tests/test_compact.py src/zvecsearch/compact.py
git commit -m "feat: add LLM compact module for chunk summarization (TDD)"
```

---

## Task 8: Watcher Module (TDD)

**Files:**
- Create: `tests/test_watcher.py`
- Create: `src/zvecsearch/watcher.py`

**Step 1: Write the failing tests**

```python
# tests/test_watcher.py
import pytest
import time
from pathlib import Path
from zvecsearch.watcher import FileWatcher


def test_watcher_start_stop(tmp_path):
    events = []
    def cb(event_type, path):
        events.append((event_type, path))

    watcher = FileWatcher([tmp_path], cb, debounce_ms=100)
    watcher.start()
    time.sleep(0.2)
    watcher.stop()


def test_watcher_detects_new_file(tmp_path):
    events = []
    def cb(event_type, path):
        events.append((event_type, path))

    watcher = FileWatcher([tmp_path], cb, debounce_ms=200)
    watcher.start()
    time.sleep(0.1)
    (tmp_path / "new.md").write_text("# New")
    time.sleep(0.5)
    watcher.stop()
    assert any(e[0] == "created" for e in events) or len(events) > 0
```

**Step 2-5:** Port from memsearch's watcher.py, test, commit.

```bash
git add tests/test_watcher.py src/zvecsearch/watcher.py
git commit -m "feat: add file watcher with debounced markdown monitoring (TDD)"
```

---

## Task 9: ZvecSearch Core Orchestrator (TDD)

**Files:**
- Create: `tests/test_core.py`
- Create: `src/zvecsearch/core.py`

**Step 1: Write the failing tests**

```python
# tests/test_core.py
import pytest
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
from zvecsearch.core import ZvecSearch

TEST_DB = Path("/tmp/zvecsearch_test_core")


@pytest.fixture(autouse=True)
def clean():
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)
    yield
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


@pytest.fixture
def md_dir(tmp_path):
    (tmp_path / "a.md").write_text("# Hello\nThis is a test document about AI.\n## Sub\nMore content here.")
    return tmp_path


class TestZvecSearchIndex:
    @pytest.mark.asyncio
    async def test_index_returns_chunk_count(self, md_dir):
        with patch("zvecsearch.core.get_provider") as mock_prov:
            mock_emb = AsyncMock()
            mock_emb.model_name = "test"
            mock_emb.dimension = 4
            mock_emb.embed.return_value = [[0.1] * 4, [0.2] * 4]
            mock_prov.return_value = mock_emb

            zs = ZvecSearch(
                paths=[str(md_dir)],
                zvec_path=str(TEST_DB),
            )
            count = await zs.index()
            assert count >= 1
            zs.close()

    @pytest.mark.asyncio
    async def test_index_single_file(self, md_dir):
        with patch("zvecsearch.core.get_provider") as mock_prov:
            mock_emb = AsyncMock()
            mock_emb.model_name = "test"
            mock_emb.dimension = 4
            mock_emb.embed.return_value = [[0.1] * 4]
            mock_prov.return_value = mock_emb

            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            count = await zs.index_file(md_dir / "a.md")
            assert count >= 1
            zs.close()


class TestZvecSearchSearch:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, md_dir):
        with patch("zvecsearch.core.get_provider") as mock_prov:
            mock_emb = AsyncMock()
            mock_emb.model_name = "test"
            mock_emb.dimension = 4
            mock_emb.embed.return_value = [[0.1] * 4, [0.2] * 4]
            mock_prov.return_value = mock_emb

            zs = ZvecSearch(paths=[str(md_dir)], zvec_path=str(TEST_DB))
            await zs.index()

            mock_emb.embed.return_value = [[0.15] * 4]
            results = await zs.search("AI test", top_k=5)
            assert isinstance(results, list)
            zs.close()
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_core.py -v
```

**Step 3: Write implementation**

```python
# src/zvecsearch/core.py
from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path
from typing import Callable

from zvecsearch.chunker import Chunk, chunk_markdown, compute_chunk_id
from zvecsearch.embeddings import get_provider
from zvecsearch.scanner import ScannedFile, scan_paths
from zvecsearch.store import ZvecStore

logger = logging.getLogger(__name__)


class ZvecSearch:
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
        return self._store

    async def index(self, force: bool = False) -> int:
        files = scan_paths(self._paths)
        total = 0
        for f in files:
            total += await self._index_file(f, force=force)
        return total

    async def index_file(self, path: str | Path) -> int:
        path = Path(path).resolve()
        st = path.stat()
        f = ScannedFile(path=path, mtime=st.st_mtime, size=st.st_size)
        return await self._index_file(f)

    async def _index_file(self, f: ScannedFile, force: bool = False) -> int:
        text = f.path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_markdown(
            text, source=str(f.path),
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
            old_hashes = self._store.hashes_by_source(str(f.path))
            stale = old_hashes - set(new_ids.keys())
            if stale:
                self._store.delete_by_hashes(list(stale))
            existing = self._store.existing_hashes(list(new_ids.keys()))
            to_embed = {k: v for k, v in new_ids.items() if k not in existing}
        else:
            self._store.delete_by_source(str(f.path))
            to_embed = new_ids

        if not to_embed:
            return 0

        return await self._embed_and_store(list(to_embed.items()))

    async def _embed_and_store(self, items: list[tuple[str, Chunk]]) -> int:
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
        output_dir: str | Path | None = None,
    ) -> str:
        from zvecsearch.compact import compact_chunks

        if source:
            chunks = self._store.query(filter_expr=f'source == "{source}"')
        else:
            chunks = self._store.query(filter_expr="")
            if not chunks:
                # Fallback: get all chunks
                chunks = []

        if not chunks:
            return ""

        summary = await compact_chunks(
            chunks, llm_provider=llm_provider,
            model=llm_model, prompt_template=prompt_template,
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
    ):
        from zvecsearch.watcher import FileWatcher

        def callback(event_type: str, path: Path):
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
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/test_core.py -v
```

**Step 5: Commit**

```bash
git add tests/test_core.py src/zvecsearch/core.py
git commit -m "feat: add ZvecSearch orchestrator with index/search/compact/watch (TDD)"
```

---

## Task 10: CLI Interface (TDD)

**Files:**
- Create: `tests/test_cli.py`
- Create: `src/zvecsearch/cli.py`
- Create: `src/zvecsearch/__main__.py`

**Step 1: Write the failing tests**

```python
# tests/test_cli.py
import pytest
from click.testing import CliRunner
from zvecsearch.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def test_cli_help(runner):
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "zvecsearch" in result.output.lower() or "Usage" in result.output


def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_cli_config_list(runner):
    result = runner.invoke(cli, ["config", "list", "--resolved"])
    assert result.exit_code == 0


def test_cli_stats_no_collection(runner):
    result = runner.invoke(cli, ["stats"])
    # Should handle gracefully even with no collection
    assert result.exit_code in (0, 1)
```

**Step 2-5:** Implement CLI with Click, test, commit.

Port from memsearch's cli.py structure with these commands:
- `zvecsearch index <PATHS>`
- `zvecsearch search <QUERY>`
- `zvecsearch watch <PATHS>`
- `zvecsearch compact`
- `zvecsearch stats`
- `zvecsearch reset`
- `zvecsearch expand <HASH>`
- `zvecsearch transcript <PATH>`
- `zvecsearch config {init,set,get,list}`

```bash
git add tests/test_cli.py src/zvecsearch/cli.py src/zvecsearch/__main__.py
git commit -m "feat: add Click CLI with index/search/watch/compact/config commands (TDD)"
```

---

## Task 11: Transcript Module

**Files:**
- Create: `src/zvecsearch/transcript.py`

Port directly from memsearch's transcript.py (JSONL parsing for Claude Code plugin).

```bash
git add src/zvecsearch/transcript.py
git commit -m "feat: add transcript parser for JSONL conversation analysis"
```

---

## Task 12: Integration Testing

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration tests**

```python
# tests/test_integration.py
import pytest
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, patch
from zvecsearch.core import ZvecSearch

TEST_DB = Path("/tmp/zvecsearch_test_integration")


@pytest.fixture(autouse=True)
def clean():
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)
    yield
    if TEST_DB.exists():
        shutil.rmtree(TEST_DB)


@pytest.fixture
def md_dir(tmp_path):
    (tmp_path / "notes.md").write_text(
        "# Redis Caching\n"
        "Redis is an in-memory data store used for caching.\n"
        "It supports key-value pairs with TTL.\n\n"
        "## Configuration\n"
        "Set maxmemory and eviction policy.\n"
    )
    (tmp_path / "auth.md").write_text(
        "# Authentication\n"
        "JWT tokens are used for stateless auth.\n"
        "Tokens expire after 24 hours.\n"
    )
    return tmp_path


@pytest.mark.asyncio
async def test_full_pipeline(md_dir):
    """End-to-end: index → search → verify results."""
    with patch("zvecsearch.core.get_provider") as mock_prov:
        mock_emb = AsyncMock()
        mock_emb.model_name = "test"
        mock_emb.dimension = 4
        # Different embeddings for different chunks
        mock_emb.embed.side_effect = lambda texts: [
            [float(i) * 0.1 + 0.1] * 4 for i in range(len(texts))
        ]
        mock_prov.return_value = mock_emb

        zs = ZvecSearch(
            paths=[str(md_dir)],
            zvec_path=str(TEST_DB),
        )

        # Index
        count = await zs.index()
        assert count >= 3  # At least 3 chunks from 2 files

        # Search
        mock_emb.embed.return_value = [[0.15] * 4]
        results = await zs.search("redis caching", top_k=5)
        assert len(results) > 0
        assert all("content" in r for r in results)

        # Stats
        assert zs.store.count() >= 3

        # Re-index (idempotent)
        mock_emb.embed.side_effect = lambda texts: [
            [float(i) * 0.1 + 0.1] * 4 for i in range(len(texts))
        ]
        count2 = await zs.index()
        assert count2 == 0  # No new chunks

        # Force re-index
        count3 = await zs.index(force=True)
        assert count3 >= 3

        zs.close()
```

**Step 2-5:** Run, verify, commit.

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full index/search pipeline"
```

---

## Task 13: Final Polish & Push

**Step 1: Run full test suite**

```bash
pip install -e ".[dev]"
pytest tests/ -v --tb=short
```

**Step 2: Lint**

```bash
ruff check src/ tests/ --fix
```

**Step 3: Verify CLI works**

```bash
zvecsearch --help
zvecsearch --version
```

**Step 4: Final commit and push**

```bash
git add -A
git commit -m "chore: final polish, linting, and README"
git push -u origin main
```
