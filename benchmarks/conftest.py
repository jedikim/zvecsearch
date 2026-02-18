"""벤치마크 공통 픽스처 — Ground Truth 로드, 청킹, 키워드/임베딩 검색."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# zvec 스텁 (네이티브 라이브러리 AVX-512 필요)
# ---------------------------------------------------------------------------
if "zvec" not in sys.modules:
    _zvec_stub = MagicMock()
    _zvec_stub.DataType.STRING = "STRING"
    _zvec_stub.DataType.INT32 = "INT32"
    _zvec_stub.DataType.VECTOR_FP32 = "VECTOR_FP32"
    _zvec_stub.DataType.SPARSE_VECTOR_FP32 = "SPARSE_VECTOR_FP32"
    _zvec_stub.MetricType.COSINE = "COSINE"
    _zvec_stub.MetricType.L2 = "L2"
    _zvec_stub.MetricType.IP = "IP"
    _zvec_stub.LogLevel.WARN = "WARN"
    _zvec_stub.FieldSchema = MagicMock
    _zvec_stub.VectorSchema = MagicMock
    _zvec_stub.CollectionSchema = MagicMock
    _zvec_stub.CollectionOption = MagicMock
    _zvec_stub.HnswIndexParam = MagicMock
    _zvec_stub.InvertIndexParam = MagicMock
    _zvec_stub.FlatIndexParam = MagicMock
    _zvec_stub.VectorQuery = MagicMock
    _zvec_stub.RrfReRanker = MagicMock
    _zvec_stub.BM25EmbeddingFunction = MagicMock
    _zvec_stub.Doc = MagicMock
    sys.modules["zvec"] = _zvec_stub

from zvecsearch.chunker import chunk_markdown  # noqa: E402

_DATA_DIR = Path(__file__).parent / "data"


# ---------------------------------------------------------------------------
# Ground Truth 데이터셋
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ground_truth() -> dict:
    """Ground truth JSON을 로드합니다."""
    gt_path = _DATA_DIR / "ground_truth.json"
    with open(gt_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def corpus_chunks(ground_truth) -> list[dict]:
    """모든 문서를 청킹하여 (chunk dict) 리스트를 반환합니다."""
    all_chunks = []
    for doc in ground_truth["documents"]:
        chunks = chunk_markdown(
            doc["content"],
            source=doc["filename"],
            max_chunk_size=1500,
            overlap_lines=2,
        )
        for c in chunks:
            all_chunks.append({
                "doc_id": doc["id"],
                "source": c.source,
                "content": c.content,
                "heading": c.heading,
                "heading_level": c.heading_level,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chunk_hash": c.content_hash,
            })
    return all_chunks


def keyword_search(
    query: str,
    chunks: list[dict],
    top_k: int = 10,
) -> list[dict]:
    """키워드 오버랩 기반 검색 시뮬레이션.

    쿼리의 키워드가 청크에 많이 포함될수록 높은 점수.
    실제 벡터 검색 없이 BM25 유사 순위를 근사합니다.
    """
    query_words = set(re.findall(r"[\w가-힣]+", query.lower()))
    query_words = {w for w in query_words if len(w) >= 2}

    scored = []
    for chunk in chunks:
        text = (chunk["content"] + " " + chunk["heading"]).lower()
        # TF-like score: 쿼리 단어가 텍스트에 나타나는 횟수
        score = 0.0
        for w in query_words:
            count = text.count(w)
            if count > 0:
                score += 1.0 + 0.5 * min(count, 5)  # 최대 5회까지 가중
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, chunk in scored[:top_k]:
        results.append({
            **chunk,
            "score": score,
        })
    return results


# ---------------------------------------------------------------------------
# Phase 5: 실제 임베딩 지원 — 캐시, 픽스처, 스킵 마커
# ---------------------------------------------------------------------------
_CACHE_DIR = _DATA_DIR / "embeddings_cache"

# Skip markers
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
requires_google = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set",
)
requires_both = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("GOOGLE_API_KEY"),
    reason="Both OPENAI_API_KEY and GOOGLE_API_KEY required",
)


class EmbeddingCache:
    """디스크 기반 JSON 캐시 — 임베딩 벡터를 저장하여 API 재호출 방지."""

    def __init__(self, provider: str, model: str):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = f"{provider}__{model}".replace("/", "_")
        self._path = _CACHE_DIR / f"{safe_name}.json"
        self._data: dict[str, list[float]] = {}
        if self._path.exists():
            self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> list[float] | None:
        return self._data.get(self._key(text))

    def put(self, text: str, embedding: list[float]) -> None:
        self._data[self._key(text)] = embedding

    def save(self) -> None:
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False),
            encoding="utf-8",
        )

    def has_all(self, texts: list[str]) -> bool:
        return all(self.get(t) is not None for t in texts)

    def get_many(self, texts: list[str]) -> list[list[float]]:
        return [self._data[self._key(t)] for t in texts]


async def embed_with_cache(
    provider,
    cache: EmbeddingCache,
    texts: list[str],
    batch_size: int = 32,
    delay: float = 0.5,
) -> list[list[float]]:
    """캐시된 임베딩 반환, 미캐시 텍스트만 API 호출."""
    results: dict[int, list[float]] = {}
    uncached: list[int] = []

    for i, text in enumerate(texts):
        cached = cache.get(text)
        if cached is not None:
            results[i] = cached
        else:
            uncached.append(i)

    for start in range(0, len(uncached), batch_size):
        batch_idx = uncached[start : start + batch_size]
        batch_texts = [texts[i] for i in batch_idx]
        embeddings = await provider.embed(batch_texts)
        for idx, emb in zip(batch_idx, embeddings):
            results[idx] = emb
            cache.put(texts[idx], emb)
        if start + batch_size < len(uncached):
            await asyncio.sleep(delay)

    cache.save()
    return [results[i] for i in range(len(texts))]


def _run_async(coro):
    """동기 컨텍스트에서 비동기 코루틴 실행."""
    return asyncio.run(coro)


@pytest.fixture(scope="session")
def corpus_texts(corpus_chunks) -> list[str]:
    """모든 청크의 content 문자열 리스트."""
    return [c["content"] for c in corpus_chunks]


@pytest.fixture(scope="session")
def semantic_queries(ground_truth) -> list[dict]:
    """시맨틱 쿼리 리스트 (키워드 검색이 실패하는 쿼리)."""
    return ground_truth.get("semantic_queries", [])


@pytest.fixture(scope="session")
def all_query_texts(ground_truth, semantic_queries) -> list[str]:
    """일반 30개 + 시맨틱 15개 쿼리 텍스트."""
    regular = [q["question"] for q in ground_truth["queries"]]
    semantic = [q["question"] for q in semantic_queries]
    return regular + semantic


def _make_embedding_fixtures(provider_name, model_name, env_var, factory):
    """OpenAI/Google 임베딩 픽스처 팩토리."""

    @pytest.fixture(scope="session")
    def corpus_embeddings(corpus_texts):
        cache = EmbeddingCache(provider_name, model_name)
        if cache.has_all(corpus_texts):
            return cache.get_many(corpus_texts)
        if not os.environ.get(env_var):
            pytest.skip(f"{env_var} not set and cache miss")
        provider = factory()
        return _run_async(embed_with_cache(provider, cache, corpus_texts))

    @pytest.fixture(scope="session")
    def query_embeddings(all_query_texts):
        cache = EmbeddingCache(provider_name, f"{model_name}__queries")
        if cache.has_all(all_query_texts):
            return cache.get_many(all_query_texts)
        if not os.environ.get(env_var):
            pytest.skip(f"{env_var} not set and cache miss")
        provider = factory()
        return _run_async(embed_with_cache(provider, cache, all_query_texts))

    return corpus_embeddings, query_embeddings


def _openai_factory():
    from zvecsearch.embeddings.openai import OpenAIEmbedding
    return OpenAIEmbedding(model="text-embedding-3-small")


def _google_factory():
    from zvecsearch.embeddings.google import GoogleEmbedding
    return GoogleEmbedding(model="gemini-embedding-001", output_dimensionality=768)


# OpenAI 임베딩 픽스처
openai_corpus_embeddings, openai_query_embeddings = _make_embedding_fixtures(
    "openai", "text-embedding-3-small", "OPENAI_API_KEY", _openai_factory,
)

# Google 임베딩 픽스처
google_corpus_embeddings, google_query_embeddings = _make_embedding_fixtures(
    "google", "gemini-embedding-001", "GOOGLE_API_KEY", _google_factory,
)
