"""Phase 3: 콘텐츠 품질 벤치마크 (RAGAS 스타일).

LLM 없이 키워드 오버랩 기반으로 근사 측정하는 메트릭:
- Faithfulness: 답변이 검색 컨텍스트에 근거하는 정도
- Context Relevance: 검색된 컨텍스트의 쿼리 관련성
- Answer Coverage: 기대 답변 키워드가 실제 검색 결과에 포함된 비율

실제 LLM 임베딩이 없으므로 키워드 검색 결과를 "검색 컨텍스트"로 사용하고,
기대 답변을 "생성된 답변"의 프록시로 사용합니다.
"""
from __future__ import annotations

from benchmarks.conftest import keyword_search
from benchmarks.metrics import (
    answer_coverage,
    content_faithfulness,
    context_relevance,
)

TOP_K = 5


class TestContentQuality:
    """검색 결과의 콘텐츠 품질 평가."""

    def _evaluate_query(self, query, corpus_chunks):
        """단일 쿼리에 대한 콘텐츠 품질 메트릭 계산."""
        results = keyword_search(query["question"], corpus_chunks, top_k=TOP_K)
        context_texts = [r["content"] for r in results]
        expected_answer = query["expected_answer"]

        return {
            "query_id": query["id"],
            "faithfulness": content_faithfulness(expected_answer, context_texts),
            "context_relevance": context_relevance(query["question"], context_texts),
            "answer_coverage": answer_coverage(
                " ".join(context_texts), expected_answer
            ),
        }

    def test_average_faithfulness_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Faithfulness가 0.5 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._evaluate_query(q, corpus_chunks)
            scores.append(result["faithfulness"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.5, f"평균 Faithfulness = {avg:.3f} (기준: 0.5)"

    def test_average_context_relevance_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Context Relevance가 0.5 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._evaluate_query(q, corpus_chunks)
            scores.append(result["context_relevance"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.5, f"평균 Context Relevance = {avg:.3f} (기준: 0.5)"

    def test_average_answer_coverage_above_threshold(self, ground_truth, corpus_chunks):
        """평균 Answer Coverage가 0.4 이상이어야 함."""
        scores = []
        for q in ground_truth["queries"]:
            result = self._evaluate_query(q, corpus_chunks)
            scores.append(result["answer_coverage"])
        avg = sum(scores) / len(scores)
        assert avg >= 0.4, f"평균 Answer Coverage = {avg:.3f} (기준: 0.4)"

    def test_zero_context_relevance_rare(self, ground_truth, corpus_chunks):
        """Context Relevance가 0인 쿼리가 10% 미만이어야 함.

        짧은 키워드(1글자) 쿼리는 2글자 최소 필터에 걸려 0이 될 수 있음.
        """
        zero_count = 0
        for q in ground_truth["queries"]:
            result = self._evaluate_query(q, corpus_chunks)
            if result["context_relevance"] == 0.0:
                zero_count += 1
        rate = zero_count / len(ground_truth["queries"])
        assert rate < 0.1, f"Context Relevance 0 비율: {rate:.1%} (기준: <10%)"

    def test_full_quality_report(self, ground_truth, corpus_chunks, capsys):
        """전체 쿼리에 대한 콘텐츠 품질 리포트 출력."""
        all_results = []
        for q in ground_truth["queries"]:
            result = self._evaluate_query(q, corpus_chunks)
            all_results.append(result)

        metric_names = ["faithfulness", "context_relevance", "answer_coverage"]
        averages = {}
        for m in metric_names:
            vals = [r[m] for r in all_results]
            averages[m] = sum(vals) / len(vals)

        print("\n" + "=" * 70)
        print("  Phase 3: 콘텐츠 품질 벤치마크 리포트 (RAGAS 스타일)")
        print("=" * 70)
        print(f"  쿼리 수: {len(all_results)}")
        print("-" * 70)
        print(f"  {'메트릭':<25} {'평균 점수':>10} {'판정':>10}")
        print("-" * 70)
        thresholds = {
            "faithfulness": 0.5,
            "context_relevance": 0.5,
            "answer_coverage": 0.4,
        }
        for m in metric_names:
            score = averages[m]
            threshold = thresholds[m]
            status = "PASS" if score >= threshold else "FAIL"
            print(f"  {m:<25} {score:>10.4f} {status:>10}")
        print("=" * 70)

        # 메트릭별 분포
        for m in metric_names:
            vals = sorted(r[m] for r in all_results)
            p25 = vals[len(vals) // 4]
            p50 = vals[len(vals) // 2]
            p75 = vals[3 * len(vals) // 4]
            print(f"  {m}: P25={p25:.2f}  P50={p50:.2f}  P75={p75:.2f}")
        print()

        # 하위 5개 쿼리
        worst = sorted(all_results, key=lambda r: r["answer_coverage"])[:5]
        if worst:
            print("  하위 5개 쿼리 (Answer Coverage 기준):")
            for r in worst:
                print(f"    {r['query_id']}: faith={r['faithfulness']:.2f} "
                      f"ctx_rel={r['context_relevance']:.2f} "
                      f"ans_cov={r['answer_coverage']:.2f}")
        print()


class TestFaithfulnessDetails:
    """Faithfulness 메트릭 상세 검증."""

    def test_answer_grounded_in_context(self, ground_truth, corpus_chunks):
        """기대 답변의 핵심 구문이 검색 컨텍스트에 근거해야 함."""
        grounded_count = 0
        total = len(ground_truth["queries"])
        for q in ground_truth["queries"]:
            results = keyword_search(q["question"], corpus_chunks, top_k=TOP_K)
            context = [r["content"] for r in results]
            faith = content_faithfulness(q["expected_answer"], context)
            if faith >= 0.5:
                grounded_count += 1
        grounded_rate = grounded_count / total
        assert grounded_rate >= 0.5, f"근거 비율 {grounded_rate:.1%} (기준: 50%)"

    def test_perfect_answer_has_high_faithfulness(self, corpus_chunks):
        """검색 컨텍스트의 내용을 그대로 답변하면 faithfulness가 높아야 함."""
        # 첫 번째 청크의 내용을 답변으로 사용
        if corpus_chunks:
            context = [corpus_chunks[0]["content"]]
            answer = corpus_chunks[0]["content"][:200]
            faith = content_faithfulness(answer, context)
            assert faith >= 0.8, f"동일 컨텍스트 faithfulness = {faith:.2f}"

    def test_irrelevant_answer_has_low_faithfulness(self, corpus_chunks):
        """무관한 답변은 faithfulness가 낮아야 함."""
        context = [c["content"] for c in corpus_chunks[:3]]
        irrelevant = "오늘 날씨가 매우 좋고 공원에서 산책하기 좋은 날입니다."
        faith = content_faithfulness(irrelevant, context)
        assert faith <= 0.5, f"무관한 답변 faithfulness = {faith:.2f}"
