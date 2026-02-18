"""검색 품질 및 콘텐츠 품질 측정을 위한 메트릭 함수.

구현된 메트릭:
- Recall@K: 상위 K개 결과에 관련 문서가 포함된 비율
- Precision@K: 상위 K개 결과 중 관련 문서 비율
- NDCG@K: 순위에 가중치를 둔 관련성 점수
- MRR: 첫 번째 관련 결과의 순위 역수 평균
- Keyword Recall: 기대 키워드가 검색 결과에 포함된 비율
- Heading Hit Rate: 기대 헤딩이 검색 결과에 포함된 비율
- Content Faithfulness: 답변이 검색 컨텍스트에 근거하는 정도
- Context Relevance: 검색된 컨텍스트의 관련성 비율
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence


# ---------------------------------------------------------------------------
# Phase 2: Retrieval Quality Metrics
# ---------------------------------------------------------------------------
def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """상위 K개 결과에 관련 문서가 포함된 비율.

    Returns: 0.0 ~ 1.0 (관련 문서 중 검색된 비율)
    """
    if not relevant_ids:
        return 1.0
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """상위 K개 결과 중 관련 문서 비율.

    Returns: 0.0 ~ 1.0
    """
    top_k = list(retrieved_ids[:k])
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    관련 문서에 이진 관련성(1 or 0)을 사용합니다.
    Returns: 0.0 ~ 1.0
    """
    top_k = list(retrieved_ids[:k])
    if not relevant_ids:
        return 1.0

    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG
    n_relevant_in_k = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant_in_k))

    return dcg / idcg if idcg > 0 else 0.0


def mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Mean Reciprocal Rank — 첫 번째 관련 결과의 순위 역수.

    Returns: 0.0 ~ 1.0
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """Average Precision — Precision@k의 관련 문서 위치별 평균.

    Returns: 0.0 ~ 1.0
    """
    if not relevant_ids:
        return 1.0
    hits = 0
    sum_precision = 0.0
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            hits += 1
            sum_precision += hits / (i + 1)
    return sum_precision / len(relevant_ids) if relevant_ids else 0.0


# ---------------------------------------------------------------------------
# Phase 3: Content Quality Metrics
# ---------------------------------------------------------------------------
def keyword_recall(
    retrieved_texts: Sequence[str],
    expected_keywords: Sequence[str],
) -> float:
    """기대 키워드가 검색 결과 텍스트에 포함된 비율.

    Returns: 0.0 ~ 1.0
    """
    if not expected_keywords:
        return 1.0
    combined = " ".join(retrieved_texts).lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


def heading_hit_rate(
    retrieved_headings: Sequence[str],
    expected_headings: Sequence[str],
) -> float:
    """기대 헤딩이 검색 결과 헤딩에 포함된 비율.

    부분 매칭: 기대 헤딩 문자열이 검색 결과 헤딩에 포함되면 적중.
    Returns: 0.0 ~ 1.0
    """
    if not expected_headings:
        return 1.0
    hits = 0
    for eh in expected_headings:
        for rh in retrieved_headings:
            if eh.lower() in rh.lower() or rh.lower() in eh.lower():
                hits += 1
                break
    return hits / len(expected_headings)


def content_faithfulness(
    answer: str,
    context_texts: Sequence[str],
) -> float:
    """답변의 핵심 구(phrase)가 검색 컨텍스트에 근거하는 비율.

    LLM 없이 키워드 오버랩 기반으로 근사 측정합니다.
    답변을 문장 단위로 분할하고, 각 문장의 핵심 명사가 컨텍스트에 있는지 확인.
    Returns: 0.0 ~ 1.0
    """
    if not answer.strip():
        return 0.0
    combined_context = " ".join(context_texts).lower()
    # 답변을 문장으로 분할
    sentences = re.split(r"[.。!?]\s*", answer)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return 0.0

    grounded = 0
    for sent in sentences:
        # 문장에서 2글자 이상 단어 추출
        words = [w for w in re.findall(r"[\w가-힣]+", sent.lower()) if len(w) >= 2]
        if not words:
            grounded += 1
            continue
        overlap = sum(1 for w in words if w in combined_context)
        if overlap / len(words) >= 0.3:  # 30% 이상 겹치면 근거 있음
            grounded += 1
    return grounded / len(sentences)


def context_relevance(
    query: str,
    retrieved_texts: Sequence[str],
) -> float:
    """검색된 컨텍스트 중 쿼리와 관련된 비율.

    쿼리 키워드가 1개 이상 포함된 청크의 비율로 근사 측정.
    Returns: 0.0 ~ 1.0
    """
    if not retrieved_texts:
        return 0.0
    query_words = set(re.findall(r"[\w가-힣]+", query.lower()))
    query_words = {w for w in query_words if len(w) >= 2}
    if not query_words:
        return 1.0

    relevant = 0
    for text in retrieved_texts:
        text_lower = text.lower()
        if any(w in text_lower for w in query_words):
            relevant += 1
    return relevant / len(retrieved_texts)


def answer_coverage(
    answer: str,
    expected_answer: str,
) -> float:
    """기대 답변의 핵심 내용이 실제 답변에 포함된 비율.

    기대 답변의 핵심 단어가 실제 답변에 얼마나 포함되는지 측정.
    Returns: 0.0 ~ 1.0
    """
    expected_words = set(re.findall(r"[\w가-힣]+", expected_answer.lower()))
    expected_words = {w for w in expected_words if len(w) >= 2}
    if not expected_words:
        return 1.0
    answer_text = answer.lower()
    found = sum(1 for w in expected_words if w in answer_text)
    return found / len(expected_words)


# ---------------------------------------------------------------------------
# Aggregate reporting
# ---------------------------------------------------------------------------
def compute_retrieval_report(
    queries: list[dict],
    results_per_query: list[list[dict]],
    k: int = 10,
) -> dict:
    """모든 쿼리에 대한 검색 품질 종합 리포트를 생성합니다."""
    metrics = {
        "recall@k": [],
        "precision@k": [],
        "ndcg@k": [],
        "mrr": [],
        "map": [],
        "keyword_recall": [],
        "heading_hit_rate": [],
    }

    for query, results in zip(queries, results_per_query):
        relevant_headings = set()
        for h in query.get("relevant_headings", []):
            relevant_headings.add(h.lower())

        # 검색된 청크의 헤딩 수집
        retrieved_headings = [r.get("heading", "") for r in results[:k]]
        retrieved_texts = [r.get("content", "") for r in results[:k]]

        # 관련 청크 ID 결정: 관련 헤딩을 포함하는 청크
        relevant_chunk_ids = set()
        retrieved_chunk_ids = []
        for r in results[:k]:
            chunk_id = r.get("chunk_hash", r.get("heading", ""))
            retrieved_chunk_ids.append(chunk_id)
            heading = r.get("heading", "").lower()
            for rh in relevant_headings:
                if rh in heading or heading in rh:
                    relevant_chunk_ids.add(chunk_id)

        if not relevant_chunk_ids:
            relevant_chunk_ids = {retrieved_chunk_ids[0]} if retrieved_chunk_ids else set()

        metrics["recall@k"].append(recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k))
        metrics["precision@k"].append(precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k))
        metrics["ndcg@k"].append(ndcg_at_k(retrieved_chunk_ids, relevant_chunk_ids, k))
        metrics["mrr"].append(mrr(retrieved_chunk_ids, relevant_chunk_ids))
        metrics["map"].append(average_precision(retrieved_chunk_ids, relevant_chunk_ids))
        metrics["keyword_recall"].append(
            keyword_recall(retrieved_texts, query.get("relevant_keywords", []))
        )
        metrics["heading_hit_rate"].append(
            heading_hit_rate(retrieved_headings, query.get("relevant_headings", []))
        )

    return {name: sum(vals) / len(vals) if vals else 0.0 for name, vals in metrics.items()}
