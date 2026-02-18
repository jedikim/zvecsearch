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
