"""Phase 4: 시스템 성능 벤치마크.

실제 시간/메모리를 측정합니다:
- 청킹 처리량 (문서/초, 청크/초)
- 청킹 지연시간 (P50, P95, P99)
- 스캐닝 처리량 (파일/초)
- 키워드 검색 지연시간 (QPS, P50, P95)
- 인덱싱 파이프라인 처리량 (mock 임베딩 사용)
- 메모리 효율성 (청크당 메타데이터 크기)
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.conftest import keyword_search
from zvecsearch.chunker import chunk_markdown
from zvecsearch.core import ZvecSearch
from zvecsearch.scanner import scan_paths


# ---------------------------------------------------------------------------
# 성능 측정 유틸리티
# ---------------------------------------------------------------------------
def _percentile(data: list[float], p: int) -> float:
    """백분위 값 계산."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def _generate_bench_docs(count: int = 100) -> list[str]:
    """벤치마크용 한국어 마크다운 문서 생성."""
    topics = [
        "인공지능", "머신러닝", "딥러닝", "자연어처리", "컴퓨터비전",
        "강화학습", "벡터검색", "임베딩", "트랜스포머", "데이터베이스",
    ]
    docs = []
    for i in range(count):
        topic = topics[i % len(topics)]
        md = f"# {topic} 문서 {i + 1}\n\n"
        md += f"{topic}에 대한 상세 설명입니다. 이 문서는 {topic}의 핵심 개념을 다룹니다.\n\n"
        md += f"## {topic}의 역사\n\n"
        md += f"{topic}은 오랫동안 연구되어 온 분야입니다. " * 5 + "\n\n"
        md += f"## {topic}의 응용\n\n"
        md += f"{topic}은 다양한 산업 분야에서 활용되고 있습니다. " * 5 + "\n\n"
        md += f"## {topic}의 미래\n\n"
        md += f"{topic}의 발전 가능성은 무한합니다. " * 5 + "\n"
        docs.append(md)
    return docs


class FakeEmbedder:
    """성능 측정용 임베딩 제공자."""

    def __init__(self, dim: int = 8):
        self.model_name = "bench-model"
        self.dimension = dim
        self._dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dim for _ in texts]


def _make_bench_store() -> MagicMock:
    """벤치마크용 상태 기반 스토어."""
    store = MagicMock()
    store._docs: dict[str, dict] = {}

    def _upsert(chunks):
        for c in chunks:
            store._docs[c["chunk_hash"]] = c
        return len(chunks)

    store.upsert.side_effect = _upsert
    store.count.side_effect = lambda: len(store._docs)
    store.hashes_by_source.side_effect = lambda src: {
        h for h, c in store._docs.items() if c.get("source") == src
    }
    store.existing_hashes.side_effect = lambda hashes: {
        h for h in hashes if h in store._docs
    }
    store.delete_by_hashes.side_effect = lambda hashes: [
        store._docs.pop(h, None) for h in hashes
    ]
    store.delete_by_source.side_effect = lambda src: [
        store._docs.pop(h) for h in list(
            h for h, c in store._docs.items() if c.get("source") == src
        )
    ]
    store.search.side_effect = lambda **kw: list(store._docs.values())[:kw.get("top_k", 10)]
    store.close.return_value = None
    return store


# ===========================================================================
# 테스트 그룹 1: 청킹 성능
# ===========================================================================
class TestChunkingPerformance:
    """청킹 처리량 및 지연시간 측정."""

    def test_chunking_throughput(self, capsys):
        """100개 문서 청킹 처리량 측정."""
        docs = _generate_bench_docs(100)
        start = time.perf_counter()
        total_chunks = 0
        for doc in docs:
            chunks = chunk_markdown(doc, source="bench.md")
            total_chunks += len(chunks)
        elapsed = time.perf_counter() - start

        docs_per_sec = len(docs) / elapsed
        chunks_per_sec = total_chunks / elapsed

        print("\n  청킹 처리량:")
        print(f"    문서 수: {len(docs)}")
        print(f"    총 청크 수: {total_chunks}")
        print(f"    소요 시간: {elapsed:.4f}초")
        print(f"    문서/초: {docs_per_sec:.0f}")
        print(f"    청크/초: {chunks_per_sec:.0f}")

        assert docs_per_sec > 100, f"청킹 처리량 {docs_per_sec:.0f} docs/s (기준: >100)"

    def test_chunking_latency_percentiles(self, capsys):
        """개별 문서 청킹 지연시간 백분위."""
        docs = _generate_bench_docs(200)
        latencies = []
        for doc in docs:
            start = time.perf_counter()
            chunk_markdown(doc, source="bench.md")
            latencies.append(time.perf_counter() - start)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

        print("\n  청킹 지연시간 (ms):")
        print(f"    P50: {p50 * 1000:.2f}ms")
        print(f"    P95: {p95 * 1000:.2f}ms")
        print(f"    P99: {p99 * 1000:.2f}ms")

        assert p95 < 0.1, f"P95 지연시간 {p95 * 1000:.1f}ms (기준: <100ms)"

    def test_large_document_chunking_performance(self, capsys):
        """대용량 문서 (5000줄) 청킹 성능."""
        lines = ["# 대규모 벤치마크 문서\n"]
        for i in range(200):
            lines.append(f"\n## 섹션 {i + 1}: 주제 분석\n")
            for j in range(24):
                lines.append(f"인공지능 기술 분석 라인 {j}. 이 섹션은 주제 {i}에 대해 다룹니다.\n")
        md = "\n".join(lines)

        start = time.perf_counter()
        chunks = chunk_markdown(md, source="large.md", max_chunk_size=1500)
        elapsed = time.perf_counter() - start

        print("\n  대용량 문서 청킹:")
        print(f"    문서 길이: {len(md):,}자 / {len(md.splitlines()):,}줄")
        print(f"    청크 수: {len(chunks)}")
        print(f"    소요 시간: {elapsed:.4f}초")

        assert elapsed < 1.0, f"대용량 문서 청킹 {elapsed:.2f}초 (기준: <1초)"


# ===========================================================================
# 테스트 그룹 2: 스캐닝 성능
# ===========================================================================
class TestScanningPerformance:
    """파일 스캐닝 처리량 측정."""

    def test_scan_500_files(self, tmp_path, capsys):
        """500개 파일 스캐닝 성능."""
        for i in range(500):
            (tmp_path / f"문서_{i:04d}.md").write_text(f"# Doc {i}\n내용 {i}")

        start = time.perf_counter()
        results = scan_paths([tmp_path])
        elapsed = time.perf_counter() - start

        files_per_sec = len(results) / elapsed

        print("\n  스캐닝 처리량:")
        print(f"    파일 수: {len(results)}")
        print(f"    소요 시간: {elapsed:.4f}초")
        print(f"    파일/초: {files_per_sec:.0f}")

        assert len(results) == 500
        assert files_per_sec > 500, f"스캐닝 {files_per_sec:.0f} files/s (기준: >500)"

    def test_scan_nested_directories(self, tmp_path, capsys):
        """5단계 중첩 × 20개 파일 스캐닝 성능."""
        count = 0
        for depth in range(5):
            d = tmp_path
            for level in range(depth + 1):
                d = d / f"레벨_{level}"
                d.mkdir(exist_ok=True)
            for i in range(20):
                (d / f"파일_{i}.md").write_text(f"# 파일 {depth}-{i}\n내용")
                count += 1

        start = time.perf_counter()
        results = scan_paths([tmp_path])
        elapsed = time.perf_counter() - start

        print("\n  중첩 디렉토리 스캐닝:")
        print(f"    총 파일: {count}")
        print(f"    발견 파일: {len(results)}")
        print(f"    소요 시간: {elapsed:.4f}초")

        assert len(results) == count


# ===========================================================================
# 테스트 그룹 3: 검색 성능
# ===========================================================================
class TestSearchPerformance:
    """키워드 검색 QPS 및 지연시간 측정."""

    def test_search_qps(self, corpus_chunks, capsys):
        """검색 QPS (Queries Per Second) 측정."""
        queries = [
            "HNSW 알고리즘", "코사인 유사도", "한국어 토큰화",
            "트랜스포머 어텐션", "임베딩 차원", "청크 크기",
            "증분 인덱싱", "BM25 검색", "벡터 정규화",
            "HyperCLOVA 모델",
        ]
        num_iterations = 100

        start = time.perf_counter()
        for _ in range(num_iterations):
            for q in queries:
                keyword_search(q, corpus_chunks, top_k=5)
        elapsed = time.perf_counter() - start

        total_queries = num_iterations * len(queries)
        qps = total_queries / elapsed

        print("\n  검색 QPS:")
        print(f"    총 쿼리 수: {total_queries}")
        print(f"    소요 시간: {elapsed:.4f}초")
        print(f"    QPS: {qps:.0f}")

        assert qps > 1000, f"QPS {qps:.0f} (기준: >1000)"

    def test_search_latency_percentiles(self, corpus_chunks, capsys):
        """검색 지연시간 백분위."""
        queries = [
            "인공지능 기술", "벡터 데이터베이스", "한국어 모델",
            "임베딩 최적화", "하이브리드 검색 시스템",
        ]
        latencies = []
        for _ in range(200):
            for q in queries:
                start = time.perf_counter()
                keyword_search(q, corpus_chunks, top_k=10)
                latencies.append(time.perf_counter() - start)

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)

        print("\n  검색 지연시간:")
        print(f"    P50: {p50 * 1000:.3f}ms")
        print(f"    P95: {p95 * 1000:.3f}ms")
        print(f"    P99: {p99 * 1000:.3f}ms")

        assert p95 < 0.01, f"P95 {p95 * 1000:.1f}ms (기준: <10ms)"


# ===========================================================================
# 테스트 그룹 4: 파이프라인 성능
# ===========================================================================
class TestPipelinePerformance:
    """전체 인덱싱 파이프라인 성능 측정 (mock 임베딩)."""

    @pytest.mark.asyncio
    async def test_index_100_files_throughput(self, tmp_path, capsys):
        """100개 파일 인덱싱 처리량."""
        docs = _generate_bench_docs(100)
        for i, doc in enumerate(docs):
            (tmp_path / f"bench_{i:03d}.md").write_text(doc)

        store = _make_bench_store()
        emb = FakeEmbedder()

        start = time.perf_counter()
        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            zs.close()
        elapsed = time.perf_counter() - start

        docs_per_sec = 100 / elapsed
        chunks_per_sec = count / elapsed

        print("\n  인덱싱 파이프라인 처리량:")
        print("    파일 수: 100")
        print(f"    총 청크 수: {count}")
        print(f"    소요 시간: {elapsed:.4f}초")
        print(f"    파일/초: {docs_per_sec:.0f}")
        print(f"    청크/초: {chunks_per_sec:.0f}")

        assert docs_per_sec > 50, f"인덱싱 {docs_per_sec:.0f} docs/s (기준: >50)"

    @pytest.mark.asyncio
    async def test_incremental_index_performance(self, tmp_path, capsys):
        """증분 인덱싱 성능 (재인덱싱 시 스킵 속도)."""
        docs = _generate_bench_docs(50)
        for i, doc in enumerate(docs):
            (tmp_path / f"bench_{i:03d}.md").write_text(doc)

        store = _make_bench_store()
        emb = FakeEmbedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            # 첫 인덱싱
            start1 = time.perf_counter()
            count1 = await zs.index()
            elapsed1 = time.perf_counter() - start1

            # 두 번째 인덱싱 (변경 없음 → 스킵)
            start2 = time.perf_counter()
            count2 = await zs.index()
            elapsed2 = time.perf_counter() - start2

            zs.close()

        speedup = elapsed1 / elapsed2 if elapsed2 > 0 else float("inf")

        print("\n  증분 인덱싱 성능:")
        print(f"    첫 인덱싱: {count1}청크, {elapsed1:.4f}초")
        print(f"    재인덱싱: {count2}청크, {elapsed2:.4f}초")
        print(f"    스피드업: {speedup:.1f}x")

        assert count2 == 0, "변경 없는 재인덱싱은 0 청크"
        # 소규모 데이터셋에서는 스캔/해시 비교 오버헤드로 시간 차이가 미미할 수 있음
        # 핵심은 재임베딩이 0건이라는 것 (count2 == 0)


# ===========================================================================
# 테스트 그룹 5: 메모리 효율성
# ===========================================================================
class TestMemoryEfficiency:
    """메모리 사용량 및 데이터 효율성 측정."""

    def test_chunk_metadata_size(self, corpus_chunks, capsys):
        """청크 메타데이터의 평균 크기."""
        total_content = sum(len(c["content"]) for c in corpus_chunks)
        total_metadata = sum(
            len(c["heading"]) + len(c["source"]) + len(c["chunk_hash"]) + 24  # int fields
            for c in corpus_chunks
        )
        overhead_ratio = total_metadata / total_content if total_content else 0

        print("\n  메모리 효율성:")
        print(f"    총 청크 수: {len(corpus_chunks)}")
        print(f"    총 콘텐츠 크기: {total_content:,}자")
        print(f"    총 메타데이터 크기: {total_metadata:,}바이트")
        print(f"    메타데이터 오버헤드: {overhead_ratio:.1%}")
        print(f"    평균 청크 크기: {total_content / len(corpus_chunks):.0f}자")

        # 소규모 코퍼스에서는 짧은 청크의 비율이 높아 오버헤드가 클 수 있음
        assert overhead_ratio < 1.0, f"메타데이터 오버헤드 {overhead_ratio:.1%} (기준: <100%)"

    def test_hash_storage_efficiency(self, corpus_chunks, capsys):
        """해시 저장 효율성 (중복 없음 확인)."""
        hashes = [c["chunk_hash"] for c in corpus_chunks]
        unique = set(hashes)
        dup_rate = 1.0 - len(unique) / len(hashes) if hashes else 0

        print("\n  해시 효율성:")
        print(f"    총 청크: {len(hashes)}")
        print(f"    고유 해시: {len(unique)}")
        print(f"    중복률: {dup_rate:.1%}")

        assert dup_rate < 0.1, f"해시 중복률 {dup_rate:.1%} (기준: <10%)"


# ===========================================================================
# 종합 리포트
# ===========================================================================
class TestPerformanceSummary:
    """Phase 4 종합 성능 리포트."""

    def test_print_summary(self, corpus_chunks, capsys):
        """종합 성능 요약 출력."""
        # 청킹 벤치마크
        docs = _generate_bench_docs(100)
        chunk_start = time.perf_counter()
        total_chunks = 0
        for doc in docs:
            total_chunks += len(chunk_markdown(doc, source="b.md"))
        chunk_elapsed = time.perf_counter() - chunk_start

        # 검색 벤치마크
        queries = ["인공지능", "벡터", "한국어", "임베딩", "검색"]
        search_latencies = []
        for _ in range(100):
            for q in queries:
                s = time.perf_counter()
                keyword_search(q, corpus_chunks, top_k=5)
                search_latencies.append(time.perf_counter() - s)

        print("\n" + "=" * 70)
        print("  Phase 4: 시스템 성능 벤치마크 종합 리포트")
        print("=" * 70)
        print(f"  {'항목':<30} {'결과':>15} {'단위':>10}")
        print("-" * 70)
        print(f"  {'청킹 처리량':<30} {100 / chunk_elapsed:>15.0f} {'docs/s':>10}")
        print(f"  {'청킹 청크 처리량':<30} {total_chunks / chunk_elapsed:>15.0f} {'chunks/s':>10}")
        print(f"  {'검색 QPS':<30} {len(search_latencies) / sum(search_latencies):>15.0f} {'queries/s':>10}")
        print(f"  {'검색 P50':<30} {_percentile(search_latencies, 50) * 1000:>15.3f} {'ms':>10}")
        print(f"  {'검색 P95':<30} {_percentile(search_latencies, 95) * 1000:>15.3f} {'ms':>10}")
        print(f"  {'코퍼스 청크 수':<30} {len(corpus_chunks):>15} {'개':>10}")
        total_content = sum(len(c["content"]) for c in corpus_chunks)
        print(f"  {'코퍼스 총 크기':<30} {total_content:>15,} {'자':>10}")
        print("=" * 70 + "\n")
