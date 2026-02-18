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
