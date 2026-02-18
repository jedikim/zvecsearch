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
