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
