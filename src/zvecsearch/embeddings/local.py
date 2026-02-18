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
