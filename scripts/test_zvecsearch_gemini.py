#!/usr/bin/env python3
"""zvecsearch 파이프라인을 통한 Gemini vs OpenAI 실제 API 통합 테스트.

zvec DB만 mock, 임베딩은 실제 API 호출.
ZvecSearch.index() → store.embed_and_upsert() → 실제 GeminiDenseEmbedding/OpenAI

Usage:
    GOOGLE_API_KEY=... OPENAI_API_KEY=... python scripts/test_zvecsearch_gemini.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# zvec 스텁 (AVX-512 없이 실행)
# ---------------------------------------------------------------------------
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
_zvec_stub.WeightedReRanker = MagicMock
_zvec_stub.HnswQueryParam = MagicMock
_zvec_stub.OpenAIDenseEmbedding = MagicMock
_zvec_stub.BM25EmbeddingFunction = MagicMock
sys.modules["zvec"] = _zvec_stub


class FakeDoc:
    """zvec.Doc 대체."""
    def __init__(self, id="", fields=None, vectors=None, score=0.0):
        self.id = id
        self.fields = fields or {}
        self.vectors = vectors or {}
        self.score = score

    def field(self, name):
        return self.fields.get(name)


_zvec_stub.Doc = FakeDoc


class RealVectorCollection:
    """실제 벡터를 저장하고, numpy 코사인 유사도로 검색하는 컬렉션."""

    def __init__(self):
        self._docs: dict[str, FakeDoc] = {}

    @property
    def stats(self):
        return SimpleNamespace(doc_count=len(self._docs))

    def upsert(self, docs):
        for d in (docs if isinstance(docs, list) else [docs]):
            self._docs[d.id] = d
        n = len(docs) if isinstance(docs, list) else 1
        return [SimpleNamespace(ok=lambda: True)] * n

    def insert(self, docs):
        return self.upsert(docs)

    def flush(self):
        pass

    def optimize(self):
        pass

    def fetch(self, ids):
        ids = ids if isinstance(ids, list) else [ids]
        return {i: self._docs[i] for i in ids if i in self._docs}

    def delete(self, ids):
        for i in (ids if isinstance(ids, list) else [ids]):
            self._docs.pop(i, None)

    def delete_by_filter(self, f):
        import re
        m = re.search(r'source\s*==\s*"([^"]+)"', f)
        if m:
            src = m.group(1)
            for k in [k for k, d in self._docs.items() if d.fields.get("source") == src]:
                del self._docs[k]

    def query(self, vectors=None, topk=10, filter=None,
              output_fields=None, reranker=None, query_param=None):
        """filter 쿼리 지원 (hashes_by_source, indexed_sources 등)."""
        import re
        docs = list(self._docs.values())
        if filter:
            m = re.search(r'source\s*==\s*"([^"]+)"', filter)
            if m:
                src = m.group(1)
                docs = [d for d in docs if d.fields.get("source") == src]
        return docs[:topk]

    def cosine_search(self, query_vec, topk=10):
        """numpy 코사인 유사도로 검색."""
        docs = list(self._docs.values())
        if not docs:
            return []

        q = np.array(query_vec, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        scored = []
        for d in docs:
            emb = d.vectors.get("embedding")
            if emb is None:
                continue
            c = np.array(emb, dtype=np.float32)
            c_norm = c / (np.linalg.norm(c) + 1e-10)
            score = float(np.dot(q_norm, c_norm))
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, d in scored[:topk]:
            d.score = score
            results.append(d)
        return results


# ---------------------------------------------------------------------------
# ZvecStore를 패치하여 실제 임베딩 + mock DB로 동작하게 함
# ---------------------------------------------------------------------------
from unittest.mock import patch  # noqa: E402

from zvecsearch.store import ZvecStore, GeminiDenseEmbedding  # noqa: E402
from zvecsearch.core import ZvecSearch  # noqa: E402


def make_patched_zvecsearch(provider, model, tmp_dir):
    """실제 임베딩 + 벡터 저장소로 ZvecSearch를 생성."""
    collection = RealVectorCollection()

    # BM25는 mock (sparse 검색은 테스트 대상이 아님)
    bm25_doc = MagicMock()
    bm25_doc.embed.side_effect = lambda t: {hash(t) % 10000: 1.0}
    bm25_query = MagicMock()
    bm25_query.embed.side_effect = lambda t: {hash(t) % 10000: 1.0}
    bm25_iter = iter([bm25_doc, bm25_query])

    with patch("zvecsearch.store.zvec") as zvec_mod:
        zvec_mod.DataType = _zvec_stub.DataType
        zvec_mod.MetricType = _zvec_stub.MetricType
        zvec_mod.LogLevel = _zvec_stub.LogLevel
        zvec_mod.FieldSchema = MagicMock
        zvec_mod.VectorSchema = MagicMock
        zvec_mod.CollectionSchema = MagicMock
        zvec_mod.CollectionOption = MagicMock
        zvec_mod.HnswIndexParam = MagicMock
        zvec_mod.InvertIndexParam = MagicMock
        zvec_mod.VectorQuery = MagicMock
        zvec_mod.RrfReRanker = MagicMock()
        zvec_mod.WeightedReRanker = MagicMock()
        zvec_mod.HnswQueryParam = MagicMock()
        zvec_mod.Doc = FakeDoc
        zvec_mod.BM25EmbeddingFunction = MagicMock(side_effect=lambda **kw: next(bm25_iter))
        zvec_mod.create_and_open.return_value = collection
        zvec_mod.open.return_value = collection
        zvec_mod.init.return_value = None

        # OpenAI: 실제 API wrapper
        if provider == "openai":
            import openai as openai_pkg
            client = openai_pkg.OpenAI()

            class RealOpenAIEmb:
                dimension = 1536
                dim = 1536

                def embed(self, text):
                    resp = client.embeddings.create(
                        model=model, input=[text.strip()]
                    )
                    return resp.data[0].embedding

            zvec_mod.OpenAIDenseEmbedding = MagicMock(return_value=RealOpenAIEmb())
        # Google: GeminiDenseEmbedding이 store.py에서 직접 생성됨

        import zvecsearch.store as store_mod
        store_mod._zvec_initialized = False

        zs = ZvecSearch(
            paths=[str(tmp_dir)],
            zvec_path="/tmp/fake_zvec_db",
            embedding_provider=provider,
            embedding_model=model,
        )

        # search()를 실제 코사인 유사도 검색으로 대체
        def real_search(query_text, top_k=10):
            dense_vec = zs._store._dense_emb.embed(query_text)
            results = collection.cosine_search(dense_vec, topk=top_k)
            return [
                {
                    "content": doc.field("content"),
                    "source": doc.field("source"),
                    "heading": doc.field("heading"),
                    "heading_level": doc.field("heading_level"),
                    "start_line": doc.field("start_line"),
                    "end_line": doc.field("end_line"),
                    "chunk_hash": doc.id,
                    "score": doc.score or 0.0,
                }
                for doc in results
            ]

        zs._store.search = real_search
        return zs, collection


# ---------------------------------------------------------------------------
# 테스트 실행
# ---------------------------------------------------------------------------
DOCUMENTS = {
    "cosine.md": """# 코사인 유사도

코사인 유사도(cosine similarity)는 두 벡터 사이의 각도를 이용하여 유사도를 측정하는 방법이다.
두 벡터의 내적을 각 벡터의 크기(norm)로 나누어 계산한다.
값의 범위는 -1에서 1까지이며, 1에 가까울수록 유사하다.
텍스트 검색에서 문서와 쿼리 벡터 사이의 유사도를 측정하는 데 널리 사용된다.
""",
    "hnsw.md": """# HNSW 알고리즘

HNSW(Hierarchical Navigable Small World)는 근사 최근접 이웃 검색(ANN)을 위한 그래프 기반 알고리즘이다.
다층 그래프 구조로 되어 있어 상위 레이어에서 빠르게 탐색 영역을 좁히고,
하위 레이어에서 정밀한 검색을 수행한다.
파라미터 M(최대 연결 수)과 ef(탐색 범위)로 정확도와 속도를 조절한다.
""",
    "embedding.md": """# 임베딩 기술

임베딩(embedding)은 텍스트, 이미지, 오디오 등을 고차원 벡터 공간의 점으로 변환하는 기술이다.
의미적으로 유사한 콘텐츠는 벡터 공간에서도 가까운 위치에 배치된다.
대표적인 모델로 OpenAI의 text-embedding-3-small, Google의 Gemini Embedding 등이 있다.
""",
    "chunking.md": """# 청킹 전략

청킹(chunking)은 긴 문서를 검색에 적합한 작은 단위로 분할하는 과정이다.
마크다운 헤딩 기반 분할, 고정 크기 분할, 의미 기반 분할 등의 방법이 있다.
적절한 청크 크기(1000-2000자)와 오버랩(2-3줄)이 검색 품질에 중요하다.
""",
    "hybrid.md": """# 하이브리드 검색

하이브리드 검색(hybrid search)은 밀집 벡터(dense)와 희소 벡터(sparse) 검색을 결합한다.
밀집 벡터는 시맨틱 유사도를, BM25 기반 희소 벡터는 키워드 매칭을 담당한다.
RRF(Reciprocal Rank Fusion)나 가중 결합으로 두 결과를 융합하여 정확도를 높인다.
""",
    "quantization.md": """# 벡터 양자화

양자화(quantization)는 부동소수점 벡터를 더 적은 비트로 표현하는 기법이다.
INT8 양자화는 메모리를 75% 절감하면서 정확도 손실은 1-3%에 불과하다.
대규모 벡터 컬렉션에서 메모리 효율과 검색 속도를 동시에 개선한다.
""",
}

QUERIES = [
    # 키워드 매칭 (쉬움)
    ("HNSW 알고리즘의 파라미터", "hnsw.md"),
    ("BM25 기반 하이브리드 검색", "hybrid.md"),
    # 동의어/패러프레이즈 (시맨틱 필요)
    ("벡터 사이의 각도로 유사도를 재는 방식", "cosine.md"),
    ("문장을 숫자 배열로 변환하는 기술", "embedding.md"),
    ("긴 글을 작은 단위로 나누는 방법", "chunking.md"),
    # 개념 수준
    ("검색 정확도와 속도를 동시에 높이는 방법", "hybrid.md"),
    ("대규모 벡터에서 메모리를 절약하는 기법", "quantization.md"),
    ("가까운 이웃을 빠르게 찾는 자료구조", "hnsw.md"),
    # 영한 혼합
    ("approximate nearest neighbor 정확도", "hnsw.md"),
    ("dense와 sparse 결합 검색", "hybrid.md"),
]


def run_pipeline(provider, model, tmp_dir):
    """ZvecSearch 파이프라인 실행: index → search."""
    print(f"\n  [{provider.upper()}] {model}")
    print(f"  {'─' * 50}")

    t0 = time.perf_counter()
    zs, col = make_patched_zvecsearch(provider, model, tmp_dir)

    # Index
    t1 = time.perf_counter()
    count = zs.index()
    index_time = time.perf_counter() - t1
    print(f"  index: {count} chunks in {index_time:.2f}s")

    # Search
    hits = 0
    results_detail = []
    for query, expected_source in QUERIES:
        t2 = time.perf_counter()
        results = zs.search(query, top_k=3)
        search_time = time.perf_counter() - t2

        top_source = results[0]["source"] if results else ""
        top_score = results[0]["score"] if results else 0.0
        hit = expected_source in top_source
        if hit:
            hits += 1
        results_detail.append((query, expected_source, top_source, top_score, hit, search_time))

    total_time = time.perf_counter() - t0
    zs.close()
    return hits, len(QUERIES), results_detail, total_time, count


def main():
    print("=" * 70)
    print("  zvecsearch 파이프라인 통합 테스트: Gemini vs OpenAI")
    print("  (ZvecSearch.index() → store.embed_and_upsert() → 실제 API)")
    print("=" * 70)

    # 마크다운 파일 생성
    tmp_dir = Path(tempfile.mkdtemp(prefix="zvecsearch_test_"))
    for name, content in DOCUMENTS.items():
        (tmp_dir / name).write_text(content, encoding="utf-8")
    print(f"\n  문서 디렉토리: {tmp_dir}")
    print(f"  문서 수: {len(DOCUMENTS)}")

    # Gemini 파이프라인
    # 각 provider를 순서대로 실행 (with 블록 내에서 완료해야 함)
    g_hits, g_total, g_detail, g_time, g_chunks = run_pipeline(
        "google", "gemini-embedding-001", tmp_dir,
    )

    o_hits, o_total, o_detail, o_time, o_chunks = run_pipeline(
        "openai", "text-embedding-3-small", tmp_dir,
    )

    # 상세 비교
    print(f"\n{'=' * 70}")
    print(f"  쿼리별 상세 비교 (top-1 소스)")
    print(f"{'=' * 70}")
    print(f"  {'쿼리':<35} {'기대':<16} {'Gemini':>8} {'OpenAI':>8}")
    print(f"  {'─' * 67}")
    for i in range(len(QUERIES)):
        q = QUERIES[i][0]
        expected = QUERIES[i][1].replace(".md", "")
        g_hit = "✓" if g_detail[i][4] else "✗"
        o_hit = "✓" if o_detail[i][4] else "✗"
        g_score = g_detail[i][3]
        o_score = o_detail[i][3]
        q_short = q[:33] + ".." if len(q) > 33 else q
        print(f"  {q_short:<35} {expected:<16} {g_hit} {g_score:.3f}  {o_hit} {o_score:.3f}")

    # 종합 리포트
    print(f"\n{'=' * 70}")
    print(f"  종합 리포트")
    print(f"{'=' * 70}")
    print(f"  {'항목':<30} {'Gemini':>15} {'OpenAI':>15}")
    print(f"  {'─' * 60}")
    print(f"  {'모델':<30} {'gemini-emb-001':>15} {'emb-3-small':>15}")
    print(f"  {'벡터 차원':<30} {'768':>15} {'1536':>15}")
    print(f"  {'인덱싱 청크 수':<30} {g_chunks:>15} {o_chunks:>15}")
    print(f"  {'Hit Rate (top-1)':<30} {g_hits}/{g_total:>12} {o_hits}/{o_total:>12}")
    print(f"  {'Hit Rate (%)':<30} {g_hits/g_total:>14.0%} {o_hits/o_total:>14.0%}")
    print(f"  {'총 소요시간':<30} {g_time:>14.2f}s {o_time:>14.2f}s")
    avg_g = sum(d[3] for d in g_detail) / len(g_detail)
    avg_o = sum(d[3] for d in o_detail) / len(o_detail)
    print(f"  {'평균 코사인 점수':<30} {avg_g:>15.4f} {avg_o:>15.4f}")
    print(f"{'=' * 70}")

    # cleanup
    import shutil
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
