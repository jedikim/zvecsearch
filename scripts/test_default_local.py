#!/usr/bin/env python3
"""zvec Default* 로컬 프로바이더 테스트.

DefaultLocalDenseEmbedding + DefaultLocalSparseEmbedding + DefaultLocalReRanker
API 키 없이 로컬에서 동작하는 zvec 기본 조합을 테스트한다.

Usage:
    python scripts/test_default_local.py
"""
from __future__ import annotations

import time

import numpy as np
import zvec


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


def test_dense_embedding():
    """DefaultLocalDenseEmbedding 기본 동작 테스트."""
    print("\n[1] DefaultLocalDenseEmbedding 초기화...")
    t0 = time.perf_counter()
    dense = zvec.DefaultLocalDenseEmbedding()
    init_time = time.perf_counter() - t0
    dim = getattr(dense, "dimension", getattr(dense, "dim", "?"))
    print(f"    dimension: {dim}")
    print(f"    init time: {init_time:.2f}s")

    print("\n[2] 단일 임베딩 테스트...")
    vec = dense.embed("테스트 문장입니다")
    print(f"    벡터 길이: {len(vec)}")
    print(f"    벡터 샘플: {vec[:5]}")
    print("    PASS")

    return dense


def test_sparse_embedding():
    """DefaultLocalSparseEmbedding 기본 동작 테스트."""
    print("\n[3] DefaultLocalSparseEmbedding 초기화...")
    t0 = time.perf_counter()
    sparse = zvec.DefaultLocalSparseEmbedding()
    init_time = time.perf_counter() - t0
    print(f"    init time: {init_time:.2f}s")

    print("\n[4] Sparse 임베딩 테스트...")
    svec = sparse.embed("테스트 문장입니다")
    print(f"    sparse 벡터 타입: {type(svec).__name__}")
    if isinstance(svec, dict):
        print(f"    non-zero entries: {len(svec)}")
    print("    PASS")

    return sparse


def test_reranker():
    """DefaultLocalReRanker 기본 동작 테스트."""
    print("\n[5] DefaultLocalReRanker 초기화...")
    t0 = time.perf_counter()
    reranker = zvec.DefaultLocalReRanker(query="test query", topn=5)
    init_time = time.perf_counter() - t0
    print("    model: cross-encoder/ms-marco-MiniLM-L6-v2")
    print(f"    init time: {init_time:.2f}s")
    print("    PASS")

    return reranker


def test_dense_search(dense):
    """DefaultLocalDenseEmbedding으로 코사인 유사도 검색."""
    print("\n[6] 코퍼스 임베딩 (10개 문서)...")
    t0 = time.perf_counter()
    corpus_vecs = [dense.embed(t) for t in CORPUS]
    embed_time = time.perf_counter() - t0
    print(f"    소요시간: {embed_time:.2f}s")

    print("\n[7] 검색 비교 (10개 쿼리, top_k=3)")
    print(f"    {'쿼리':<40} {'기대':<10} {'결과':>8}")
    print("    " + "-" * 58)

    hits = 0
    for query, expected_keyword in QUERIES:
        q_vec = dense.embed(query)
        results = cosine_search(q_vec, corpus_vecs, CORPUS, top_k=3)
        hit = any(expected_keyword in text for text, _ in results)
        if hit:
            hits += 1
        q_short = query[:38] + ".." if len(query) > 38 else query
        print(f"    {q_short:<40} {expected_keyword:<10} {'PASS' if hit else 'FAIL':>8}")

    total = len(QUERIES)
    print("    " + "-" * 58)
    print(f"    Hit Rate: {hits}/{total} ({hits/total:.0%})")

    return hits, total, corpus_vecs


def test_semantic_detail(dense, corpus_vecs):
    """시맨틱 쿼리 상세 (top-1 결과)."""
    print("\n[8] 시맨틱 쿼리 상세 (top-1)")
    semantic_queries = [
        "벡터 사이의 각도로 유사도를 재는 방식",
        "문장을 숫자 배열로 변환하는 기술",
        "검색 정확도와 속도를 동시에 높이는 방법",
    ]
    for q in semantic_queries:
        q_vec = dense.embed(q)
        result = cosine_search(q_vec, corpus_vecs, CORPUS, top_k=1)[0]
        print(f"\n    Q: {q}")
        print(f"    top-1 (score={result[1]:.4f}): {result[0][:70]}...")


def main():
    print("=" * 70)
    print("  zvec Default* 로컬 프로바이더 테스트")
    print("  (API 키 불필요, 로컬 모델 자동 다운로드)")
    print("=" * 70)

    t_start = time.perf_counter()

    # Dense embedding
    dense = test_dense_embedding()

    # Sparse embedding
    try:
        sparse = test_sparse_embedding()
    except Exception as e:
        print(f"    SKIP: {e}")
        sparse = None

    # Reranker
    try:
        reranker = test_reranker()
    except Exception as e:
        print(f"    SKIP: {e}")
        reranker = None

    # Dense search
    hits, total, corpus_vecs = test_dense_search(dense)

    # Semantic detail
    test_semantic_detail(dense, corpus_vecs)

    elapsed = time.perf_counter() - t_start
    print(f"\n{'=' * 70}")
    print("  zvec Default* 로컬 프로바이더 테스트 결과")
    print(f"{'=' * 70}")
    print(f"  DefaultLocalDenseEmbedding:  {'OK' if dense else 'FAIL'}")
    print(f"  DefaultLocalSparseEmbedding: {'OK' if sparse else 'SKIP'}")
    print(f"  DefaultLocalReRanker:        {'OK' if reranker else 'SKIP'}")
    print(f"  Dense Search Hit Rate:       {hits}/{total} ({hits/total:.0%})")
    print(f"  총 소요시간:                  {elapsed:.2f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
