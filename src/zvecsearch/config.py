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
    max_buffer_size: int = 67108864


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
