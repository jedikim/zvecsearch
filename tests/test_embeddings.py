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
