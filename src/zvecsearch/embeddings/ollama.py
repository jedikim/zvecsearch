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
