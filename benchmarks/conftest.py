"""벤치마크 공통 픽스처 — Ground Truth 로드, 청킹, 키워드 기반 검색 시뮬레이션."""
from __future__ import annotations

import json
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
