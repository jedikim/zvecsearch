#!/usr/bin/env python3
"""GeminiDenseEmbedding 실제 API 테스트 + OpenAI 비교.

zvec 없이 독립 실행 가능. store.py의 GeminiDenseEmbedding을 복제하여 테스트.

Usage:
    GOOGLE_API_KEY=... OPENAI_API_KEY=... python scripts/test_gemini_embedding.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GeminiDenseEmbedding (store.py에서 복제 — zvec import 없이)
# ---------------------------------------------------------------------------
class GeminiDenseEmbedding:
    """Google Gemini dense embedding — zvec DenseEmbeddingFunction Protocol 호환."""

    def __init__(self, model="gemini-embedding-001", dimension=768, api_key=None):
        from google import genai
        resolved_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY required")
        self._model = model
        self._dimension = dimension
        self._client = genai.Client(api_key=resolved_key)

    @property
    def dimension(self):
        return self._dimension

    @lru_cache(maxsize=10)
    def embed(self, input: str) -> list[float]:
        input = input.strip()
        if not input:
            raise ValueError("empty input")
        result = self._client.models.embed_content(
            model=self._model,
            contents=[input],
            config={"output_dimensionality": self._dimension},
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 (개별 호출)."""
        return [self.embed(t) for t in texts]


# ---------------------------------------------------------------------------
# OpenAI embedding wrapper
# ---------------------------------------------------------------------------
class OpenAIDenseEmbedding:
    """OpenAI dense embedding (비교용)."""

    def __init__(self, model="text-embedding-3-small", api_key=None):
        import openai
        self._model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def dimension(self):
        return 1536

    def embed(self, input: str) -> list[float]:
        resp = self._client.embeddings.create(model=self._model, input=[input.strip()])
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(model=self._model, input=[t.strip() for t in texts])
        return [d.embedding for d in resp.data]


# ---------------------------------------------------------------------------
# 코사인 유사도 검색
# ---------------------------------------------------------------------------
def cosine_search(query_vec, corpus_vecs, corpus_texts, top_k=5):
    q = np.array(query_vec, dtype=np.float32)
    C = np.array(corpus_vecs, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    C_norms = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-10)
    scores = C_norms @ q_norm
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(corpus_texts[i], float(scores[i])) for i in top_idx]


# ---------------------------------------------------------------------------
# 테스트 데이터
# ---------------------------------------------------------------------------
CORPUS = [
    "코사인 유사도(cosine similarity)는 두 벡터 사이의 각도를 이용하여 유사도를 측정하는 방법이다.",
    "HNSW(Hierarchical Navigable Small World)는 근사 최근접 이웃 검색을 위한 그래프 기반 알고리즘이다.",
    "임베딩(embedding)은 텍스트나 이미지를 고차원 벡터 공간의 점으로 변환하는 기술이다.",
    "청킹(chunking)은 긴 문서를 검색에 적합한 작은 단위로 분할하는 과정이다.",
    "BM25는 TF-IDF를 개선한 확률적 정보 검색 모델로, 키워드 기반 검색에 널리 사용된다.",
    "트랜스포머(Transformer)의 셀프 어텐션 메커니즘은 입력 시퀀스의 각 위치가 다른 모든 위치를 참조할 수 있게 한다.",
    "양자화(quantization)는 벡터를 더 적은 비트로 표현하여 메모리 사용량을 줄이는 기법이다.",
    "하이브리드 검색(hybrid search)은 밀집 벡터 검색과 희소 벡터 검색을 결합하여 정확도를 높인다.",
    "증분 인덱싱(incremental indexing)은 변경된 문서만 다시 임베딩하여 API 비용을 절감하는 방식이다.",
    "한국어는 교착어로서, 형태소 분석 없이는 정확한 토큰화가 어렵다.",
]

QUERIES = [
    # 키워드 매칭 쿼리
    ("HNSW 알고리즘이란?", "HNSW"),
    ("BM25 검색 모델", "BM25"),
    # 동의어/패러프레이즈 쿼리 (시맨틱 필요)
    ("벡터 사이의 각도로 유사도를 재는 방식", "코사인"),
    ("문장을 숫자 배열로 변환하는 기술", "임베딩"),
    ("긴 글을 작은 단위로 나누는 방법", "청킹"),
    ("가까운 이웃을 빠르게 찾는 자료구조", "HNSW"),
    # 개념 수준 쿼리
    ("검색 정확도와 속도를 동시에 높이는 방법", "하이브리드"),
    ("대규모 벡터에서 메모리를 절약하는 기법", "양자화"),
    # 영한 혼합
    ("approximate nearest neighbor의 정확도", "HNSW"),
    ("transformer의 self-attention 동작 원리", "트랜스포머"),
]


def run_test():
    print("=" * 70)
    print("  GeminiDenseEmbedding 실제 API 테스트")
    print("=" * 70)

    # 1. 기본 동작 테스트
    print("\n[1] GeminiDenseEmbedding 초기화...")
    t0 = time.perf_counter()
    gemini = GeminiDenseEmbedding()
    print(f"    dimension: {gemini.dimension}")
    print(f"    model: {gemini._model}")

    print("\n[2] 단일 임베딩 테스트...")
    vec = gemini.embed("테스트 문장입니다")
    print(f"    벡터 길이: {len(vec)}")
    print(f"    벡터 샘플: {vec[:5]}")
    assert len(vec) == 768, f"Expected 768, got {len(vec)}"
    print("    ✓ PASS")

    # 2. OpenAI 초기화
    print("\n[3] OpenAI 임베딩 초기화...")
    openai_emb = OpenAIDenseEmbedding()
    vec2 = openai_emb.embed("테스트 문장입니다")
    print(f"    벡터 길이: {len(vec2)}")
    assert len(vec2) == 1536, f"Expected 1536, got {len(vec2)}"
    print("    ✓ PASS")

    # 3. 코퍼스 임베딩
    print("\n[4] 코퍼스 임베딩 (10개 문서)...")
    t1 = time.perf_counter()
    gemini_corpus = [gemini.embed(t) for t in CORPUS]
    gemini_time = time.perf_counter() - t1
    print(f"    Gemini: {gemini_time:.2f}초")

    t1 = time.perf_counter()
    openai_corpus = openai_emb.embed_batch(CORPUS)
    openai_time = time.perf_counter() - t1
    print(f"    OpenAI: {openai_time:.2f}초")

    # 4. 검색 비교
    print("\n[5] 검색 비교 (10개 쿼리, top_k=3)")
    print(f"    {'쿼리':<40} {'기대':<8} {'Gemini':>8} {'OpenAI':>8}")
    print("    " + "-" * 66)

    gemini_hits = 0
    openai_hits = 0

    for query, expected_keyword in QUERIES:
        # Gemini 검색
        q_vec_g = gemini.embed(query)
        results_g = cosine_search(q_vec_g, gemini_corpus, CORPUS, top_k=3)
        hit_g = any(expected_keyword in text for text, _ in results_g)
        if hit_g:
            gemini_hits += 1

        # OpenAI 검색
        q_vec_o = openai_emb.embed(query)
        results_o = cosine_search(q_vec_o, openai_corpus, CORPUS, top_k=3)
        hit_o = any(expected_keyword in text for text, _ in results_o)
        if hit_o:
            openai_hits += 1

        q_short = query[:38] + ".." if len(query) > 38 else query
        print(f"    {q_short:<40} {expected_keyword:<8} {'✓' if hit_g else '✗':>8} {'✓' if hit_o else '✗':>8}")

    total = len(QUERIES)
    print("    " + "-" * 66)
    print(f"    {'Hit Rate':<40} {'':8} {gemini_hits}/{total:>5} {openai_hits}/{total:>5}")

    # 5. 시맨틱 쿼리 상세
    print("\n[6] 시맨틱 쿼리 상세 (top-1 결과 비교)")
    semantic_queries = [
        "벡터 사이의 각도로 유사도를 재는 방식",
        "문장을 숫자 배열로 변환하는 기술",
        "검색 정확도와 속도를 동시에 높이는 방법",
    ]
    for q in semantic_queries:
        q_g = gemini.embed(q)
        q_o = openai_emb.embed(q)
        r_g = cosine_search(q_g, gemini_corpus, CORPUS, top_k=1)[0]
        r_o = cosine_search(q_o, openai_corpus, CORPUS, top_k=1)[0]
        print(f"\n    Q: {q}")
        print(f"    Gemini top-1 (score={r_g[1]:.4f}): {r_g[0][:60]}...")
        print(f"    OpenAI top-1 (score={r_o[1]:.4f}): {r_o[0][:60]}...")

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 70}")
    print(f"  총 소요시간: {elapsed:.2f}초")
    print(f"  Gemini Hit Rate: {gemini_hits}/{total} ({gemini_hits/total:.0%})")
    print(f"  OpenAI Hit Rate: {openai_hits}/{total} ({openai_hits/total:.0%})")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_test()
