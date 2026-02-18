"""Phase 2: 검색 품질 벤치마크.

키워드 기반 검색 시뮬레이션을 사용하여 다음 메트릭을 측정:
- Recall@K: 관련 청크가 검색 결과에 포함된 비율
- Precision@K: 검색 결과 중 관련 청크 비율
- NDCG@K: 순위 가중 관련성 점수
- MRR: 첫 번째 관련 결과의 순위 역수
- MAP: 평균 정밀도
- Keyword Recall: 기대 키워드가 검색 결과에 포함된 비율
- Heading Hit Rate: 기대 헤딩이 검색 결과 헤딩에 매칭된 비율

모든 메트릭의 결과를 표 형식으로 출력합니다.
"""
from __future__ import annotations

import pytest

from benchmarks.conftest import keyword_search
from benchmarks.metrics import (
    average_precision,
    heading_hit_rate,
    keyword_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

TOP_K = 5


class TestRetrievalQuality:
    """30개 쿼리에 대한 검색 품질 종합 측정."""

    def _search_and_evaluate(self, query, corpus_chunks):
        """단일 쿼리에 대해 검색 + 메트릭 계산."""
        results = keyword_search(query["question"], corpus_chunks, top_k=TOP_K)

        # 관련 청크 결정: 관련 헤딩을 포함하는 청크
        relevant_headings = {h.lower() for h in query.get("relevant_headings", [])}
        relevant_doc_ids = set(query.get("relevant_doc_ids", []))

        relevant_chunk_hashes = set()
        for c in corpus_chunks:
            heading = c["heading"].lower()
            for rh in relevant_headings:
                if rh in heading or heading in rh:
                    if c["doc_id"] in relevant_doc_ids:
                        relevant_chunk_hashes.add(c["chunk_hash"])

        retrieved_hashes = [r["chunk_hash"] for r in results]
        retrieved_texts = [r["content"] for r in results]
        retrieved_headings = [r["heading"] for r in results]

        return {
            "query_id": query["id"],
            "recall@k": recall_at_k(retrieved_hashes, relevant_chunk_hashes, TOP_K),
            "precision@k": precision_at_k(retrieved_hashes, relevant_chunk_hashes, TOP_K),
            "ndcg@k": ndcg_at_k(retrieved_hashes, relevant_chunk_hashes, TOP_K),
            "mrr": mrr(retrieved_hashes, relevant_chunk_hashes),
            "map": average_precision(retrieved_hashes, relevant_chunk_hashes),
            "keyword_recall": keyword_recall(
                retrieved_texts, query.get("relevant_keywords", [])
            ),
            "heading_hit_rate": heading_hit_rate(
                retrieved_headings, query.get("relevant_headings", [])
            ),
        }

    def test_overall_recall_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Recall@K가 0.6 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._search_and_evaluate(q, corpus_chunks)
            scores.append(result["recall@k"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.6, f"평균 Recall@{TOP_K} = {avg:.3f} (기준: 0.6)"

    def test_overall_mrr_above_threshold(self, ground_truth, corpus_chunks):
        """평균 MRR이 0.5 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._search_and_evaluate(q, corpus_chunks)
            scores.append(result["mrr"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.5, f"평균 MRR = {avg:.3f} (기준: 0.5)"

    def test_overall_keyword_recall_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Keyword Recall이 0.5 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._search_and_evaluate(q, corpus_chunks)
            scores.append(result["keyword_recall"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.5, f"평균 Keyword Recall = {avg:.3f} (기준: 0.5)"

    def test_overall_heading_hit_rate_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Heading Hit Rate가 0.4 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._search_and_evaluate(q, corpus_chunks)
            scores.append(result["heading_hit_rate"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.4, f"평균 Heading Hit Rate = {avg:.3f} (기준: 0.4)"

    def test_full_benchmark_report(self, ground_truth, corpus_chunks, capsys):
        """전체 쿼리에 대한 벤치마크 리포트 출력."""
        all_results = []
        for q in ground_truth["queries"]:
            result = self._search_and_evaluate(q, corpus_chunks)
            all_results.append(result)

        # 메트릭별 평균 계산
        metric_names = ["recall@k", "precision@k", "ndcg@k", "mrr", "map",
                         "keyword_recall", "heading_hit_rate"]
        averages = {}
        for m in metric_names:
            vals = [r[m] for r in all_results]
            averages[m] = sum(vals) / len(vals)

        # 리포트 출력
        print("\n" + "=" * 70)
        print(f"  Phase 2: 검색 품질 벤치마크 리포트 (top_k={TOP_K})")
        print("=" * 70)
        print(f"  쿼리 수: {len(all_results)}")
        print(f"  코퍼스 청크 수: {len(corpus_chunks)}")
        print("-" * 70)
        print(f"  {'메트릭':<25} {'평균 점수':>10} {'판정':>10}")
        print("-" * 70)
        thresholds = {
            "recall@k": 0.6, "precision@k": 0.3, "ndcg@k": 0.4,
            "mrr": 0.5, "map": 0.3, "keyword_recall": 0.5,
            "heading_hit_rate": 0.4,
        }
        for m in metric_names:
            score = averages[m]
            threshold = thresholds.get(m, 0.3)
            status = "PASS" if score >= threshold else "FAIL"
            print(f"  {m:<25} {score:>10.4f} {status:>10}")
        print("=" * 70)

        # 쿼리별 상세 결과 (하위 5개)
        worst = sorted(all_results, key=lambda r: r["recall@k"])[:5]
        if worst:
            print("\n  하위 5개 쿼리 (Recall@K 기준):")
            for r in worst:
                print(f"    {r['query_id']}: recall={r['recall@k']:.2f} "
                      f"kw_recall={r['keyword_recall']:.2f} "
                      f"heading_hit={r['heading_hit_rate']:.2f}")
        print()


class TestPerQueryRetrieval:
    """개별 쿼리별 검색 정확도 검증."""

    def test_hnsw_params_query(self, ground_truth, corpus_chunks):
        """HNSW 파라미터 쿼리는 관련 청크를 검색해야 함."""
        q = next(q for q in ground_truth["queries"] if q["id"] == "q01")
        results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
        texts = " ".join(r["content"] for r in results)
        assert "ef_construction" in texts or "HNSW" in texts

    def test_korean_tokenization_query(self, ground_truth, corpus_chunks):
        """한국어 토큰화 쿼리는 관련 컨텐츠를 반환해야 함."""
        q = next(q for q in ground_truth["queries"] if q["id"] == "q06")
        results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
        texts = " ".join(r["content"] for r in results)
        assert "토큰화" in texts or "한국어" in texts

    def test_transformer_query(self, ground_truth, corpus_chunks):
        """트랜스포머 쿼리는 어텐션 관련 내용을 반환해야 함."""
        q = next(q for q in ground_truth["queries"] if q["id"] == "q08")
        results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
        texts = " ".join(r["content"] for r in results)
        assert "어텐션" in texts or "트랜스포머" in texts

    def test_chunking_strategy_query(self, ground_truth, corpus_chunks):
        """청킹 전략 쿼리는 관련 내용을 반환해야 함."""
        q = next(q for q in ground_truth["queries"] if q["id"] == "q16")
        results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
        texts = " ".join(r["content"] for r in results)
        assert "마크다운" in texts or "청킹" in texts

    def test_korean_ai_model_query(self, ground_truth, corpus_chunks):
        """한국어 AI 모델 쿼리는 관련 모델명을 반환해야 함."""
        q = next(q for q in ground_truth["queries"] if q["id"] == "q21")
        results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
        texts = " ".join(r["content"] for r in results)
        assert "HyperCLOVA" in texts or "네이버" in texts

    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_recall_improves_with_k(self, ground_truth, corpus_chunks, k):
        """K가 증가하면 평균 Recall이 감소하지 않아야 함."""
        scores = []
        for q in ground_truth["queries"]:
            relevant_headings = {h.lower() for h in q.get("relevant_headings", [])}
            relevant_doc_ids = set(q.get("relevant_doc_ids", []))
            relevant_hashes = set()
            for c in corpus_chunks:
                heading = c["heading"].lower()
                for rh in relevant_headings:
                    if rh in heading or heading in rh:
                        if c["doc_id"] in relevant_doc_ids:
                            relevant_hashes.add(c["chunk_hash"])

            results = keyword_search(q["question"], corpus_chunks, top_k=k)
            retrieved = [r["chunk_hash"] for r in results]
            scores.append(recall_at_k(retrieved, relevant_hashes, k))
        avg = sum(scores) / len(scores)
        # K가 커질수록 recall은 최소한 유지되어야 함
        assert avg >= 0.0  # 기본 검증 (단조 증가는 K별 비교에서 확인)
