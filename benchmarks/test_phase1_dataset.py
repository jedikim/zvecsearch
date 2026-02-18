"""Phase 1: Ground Truth 데이터셋 검증.

데이터셋의 완전성과 정확성을 확인합니다:
- 문서가 올바르게 청킹되는지
- 모든 쿼리에 대해 관련 청크가 실제로 존재하는지
- 기대 키워드가 관련 문서에 실제로 포함되는지
- 문서 간 크로스 레퍼런스 정확성
"""
from __future__ import annotations


class TestDatasetIntegrity:
    """Ground Truth 데이터셋 무결성 검증."""

    def test_documents_count(self, ground_truth):
        """최소 5개 문서가 포함되어야 함."""
        assert len(ground_truth["documents"]) >= 5

    def test_queries_count(self, ground_truth):
        """최소 30개 쿼리가 포함되어야 함."""
        assert len(ground_truth["queries"]) >= 30

    def test_all_documents_have_content(self, ground_truth):
        """모든 문서에 한국어 콘텐츠가 포함되어야 함."""
        for doc in ground_truth["documents"]:
            assert doc["content"].strip()
            has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in doc["content"])
            assert has_korean, f"문서 {doc['id']}에 한국어 없음"

    def test_all_queries_have_fields(self, ground_truth):
        """모든 쿼리에 필수 필드가 있어야 함."""
        required = {"id", "question", "expected_answer", "relevant_doc_ids",
                     "relevant_headings", "relevant_keywords"}
        for q in ground_truth["queries"]:
            missing = required - set(q.keys())
            assert not missing, f"쿼리 {q['id']}에 누락 필드: {missing}"

    def test_relevant_doc_ids_exist(self, ground_truth):
        """쿼리의 relevant_doc_ids가 실제 문서 ID와 일치해야 함."""
        doc_ids = {d["id"] for d in ground_truth["documents"]}
        for q in ground_truth["queries"]:
            for did in q["relevant_doc_ids"]:
                assert did in doc_ids, f"쿼리 {q['id']}: 문서 ID '{did}' 없음"


class TestChunkingCovers:
    """청킹 결과가 Ground Truth 기대값을 충족하는지 검증."""

    def test_total_chunks_reasonable(self, corpus_chunks):
        """5개 문서에서 최소 20개 이상 청크가 생성되어야 함."""
        assert len(corpus_chunks) >= 20

    def test_all_documents_chunked(self, ground_truth, corpus_chunks):
        """모든 문서가 최소 1개 이상의 청크를 생성해야 함."""
        chunked_docs = {c["doc_id"] for c in corpus_chunks}
        for doc in ground_truth["documents"]:
            assert doc["id"] in chunked_docs, f"문서 {doc['id']}의 청크 없음"

    def test_expected_headings_exist_in_chunks(self, ground_truth, corpus_chunks):
        """쿼리의 기대 헤딩이 실제 청크 헤딩에 존재해야 함."""
        all_headings = {c["heading"].lower() for c in corpus_chunks if c["heading"]}
        missing = []
        for q in ground_truth["queries"]:
            for eh in q["relevant_headings"]:
                found = any(
                    eh.lower() in h or h in eh.lower()
                    for h in all_headings
                )
                if not found:
                    missing.append((q["id"], eh))
        # 90% 이상 매칭되어야 함 (일부 헤딩은 부분 매칭 불가능할 수 있음)
        total = sum(len(q["relevant_headings"]) for q in ground_truth["queries"])
        hit_rate = 1.0 - len(missing) / total if total else 1.0
        assert hit_rate >= 0.9, f"헤딩 매칭률 {hit_rate:.1%}, 누락: {missing[:5]}"

    def test_expected_keywords_in_relevant_docs(self, ground_truth):
        """기대 키워드가 관련 문서에 실제로 존재해야 함."""
        docs_by_id = {d["id"]: d["content"].lower() for d in ground_truth["documents"]}
        missing = []
        for q in ground_truth["queries"]:
            for kw in q["relevant_keywords"]:
                found = any(
                    kw.lower() in docs_by_id.get(did, "")
                    for did in q["relevant_doc_ids"]
                )
                if not found:
                    missing.append((q["id"], kw))
        total = sum(len(q["relevant_keywords"]) for q in ground_truth["queries"])
        hit_rate = 1.0 - len(missing) / total if total else 1.0
        assert hit_rate >= 0.9, f"키워드 매칭률 {hit_rate:.1%}, 누락: {missing[:5]}"

    def test_all_chunks_have_korean_content(self, corpus_chunks):
        """모든 청크에 한국어 콘텐츠가 포함되어야 함."""
        for c in corpus_chunks:
            has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in c["content"])
            assert has_korean, f"한국어 없는 청크: {c['heading'][:30]}"
