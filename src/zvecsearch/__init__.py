"""ZvecSearch - Semantic memory search powered by zvec vector database."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ZvecSearch":
        from zvecsearch.core import ZvecSearch
        return ZvecSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ZvecSearch"]
