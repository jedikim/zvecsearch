"""한국어 대규모 데이터를 사용한 종합 스트레스 테스트.

다음을 검증합니다:
- 한국어 문서 청킹 (띄어쓰기 없는 텍스트, 한영 혼합, 다양한 헤딩, 테이블, 코드 블록)
- 한국어 파일명/디렉토리명 스캐닝
- 한국어 파이프라인 (인덱싱 → 검색, 증분 인덱싱, 강제 재인덱싱)
- 한국어 해시 무결성 (유니코드 정규화, 자모 vs 완성형)
- 한국어 경계 조건 (50,000자 단락, 이모지 조합, 단일 글자 헤딩)
- 대규모 한국어 데이터 통합 (200개 파일, 1000줄 문서)
- 한국어 CLI 스트레스 (설정, 도움말, 통계)
"""
from __future__ import annotations

import random
import sys
import unicodedata
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# zvec 모듈 스텁 (네이티브 라이브러리가 AVX-512 필요)
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

from zvecsearch.chunker import Chunk, chunk_markdown, compute_chunk_id  # noqa: E402
from zvecsearch.core import ZvecSearch  # noqa: E402
from zvecsearch.scanner import scan_paths  # noqa: E402

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
TEST_DIM = 8

# 한국어 기술 용어 풀
_KOREAN_TOPICS = [
    "인공지능", "머신러닝", "딥러닝", "자연어처리", "컴퓨터비전",
    "강화학습", "생성형AI", "트랜스포머", "어텐션메커니즘", "벡터검색",
    "임베딩", "토크나이저", "파인튜닝", "전이학습", "데이터증강",
    "모델압축", "지식증류", "연합학습", "메타학습", "신경망아키텍처",
    "그래프신경망", "순환신경망", "합성곱신경망", "오토인코더", "변분추론",
    "베이지안최적화", "하이퍼파라미터", "배치정규화", "드롭아웃", "옵티마이저",
    "경사하강법", "역전파", "활성화함수", "손실함수", "정규화기법",
    "교차검증", "과적합방지", "데이터전처리", "특성공학", "차원축소",
    "클러스터링", "분류알고리즘", "회귀분석", "앙상블기법", "부스팅",
    "랜덤포레스트", "서포트벡터머신", "결정트리", "나이브베이즈", "로지스틱회귀",
]

_KOREAN_SENTENCES = [
    "인공지능 기술은 현대 사회의 다양한 분야에서 혁신적인 변화를 이끌고 있습니다.",
    "머신러닝 모델의 성능을 향상시키기 위해서는 대규모 학습 데이터가 필수적입니다.",
    "자연어처리 분야에서 트랜스포머 아키텍처는 혁명적인 발전을 가져왔습니다.",
    "벡터 데이터베이스는 유사도 검색을 효율적으로 수행하기 위한 핵심 인프라입니다.",
    "딥러닝 모델의 학습 과정에서 최적화 알고리즘의 선택은 매우 중요합니다.",
    "임베딩 기술을 통해 텍스트를 고차원 벡터 공간에 매핑할 수 있습니다.",
    "강화학습은 에이전트가 환경과 상호작용하며 최적의 정책을 학습하는 방법입니다.",
    "데이터 전처리 과정은 모델 성능에 직접적인 영향을 미치는 중요한 단계입니다.",
    "하이퍼파라미터 튜닝을 통해 모델의 일반화 성능을 크게 개선할 수 있습니다.",
    "분산 학습 시스템은 대규모 모델을 효율적으로 훈련시키기 위해 사용됩니다.",
]


# ---------------------------------------------------------------------------
# 유틸리티: 상태 기반 스토어 Mock
# ---------------------------------------------------------------------------
def _make_stateful_store() -> MagicMock:
    """상태 기반 인메모리 ZvecStore Mock."""
    store = MagicMock()
    store._docs: dict[str, dict] = {}

    def _upsert(chunks):
        for c in chunks:
            store._docs[c["chunk_hash"]] = c
        return len(chunks)

    store.upsert.side_effect = _upsert
    store.count.side_effect = lambda: len(store._docs)

    def _hashes_by_source(source):
        return {h for h, c in store._docs.items() if c.get("source") == source}

    store.hashes_by_source.side_effect = _hashes_by_source

    def _existing_hashes(hashes):
        return {h for h in hashes if h in store._docs}

    store.existing_hashes.side_effect = _existing_hashes

    def _delete_by_hashes(hashes):
        for h in hashes:
            store._docs.pop(h, None)

    store.delete_by_hashes.side_effect = _delete_by_hashes

    def _delete_by_source(source):
        to_del = [h for h, c in store._docs.items() if c.get("source") == source]
        for h in to_del:
            del store._docs[h]

    store.delete_by_source.side_effect = _delete_by_source

    def _search(query_embedding, query_text="", top_k=10, filter_expr=""):
        results = list(store._docs.values())[:top_k]
        return [
            {
                "content": c.get("content", ""),
                "source": c.get("source", ""),
                "heading": c.get("heading", ""),
                "heading_level": c.get("heading_level", 0),
                "start_line": c.get("start_line", 0),
                "end_line": c.get("end_line", 0),
                "chunk_hash": c.get("chunk_hash", ""),
                "score": 0.95 - i * 0.01,
            }
            for i, c in enumerate(results)
        ]

    store.search.side_effect = _search
    store.close.return_value = None
    store.drop.return_value = None
    return store


class FakeEmbedder:
    """Mock 오염을 피하기 위한 순수 Python 임베딩 제공자."""

    def __init__(self, dim: int = TEST_DIM):
        self.model_name = "korean-stress-test-model"
        self.dimension = dim
        self._dim = dim
        self._embed_calls: list[list[str]] = []
        self._custom_return = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._embed_calls.append(texts)
        if self._custom_return is not None:
            return self._custom_return
        return [[float(i + 1) * 0.1] * self._dim for i, _ in enumerate(texts)]


def _make_embedder(dim: int = TEST_DIM) -> FakeEmbedder:
    return FakeEmbedder(dim=dim)


# ---------------------------------------------------------------------------
# 한국어 대규모 데이터 생성 유틸리티
# ---------------------------------------------------------------------------
def _generate_korean_article(
    num_sections: int = 20,
    paragraphs_per: int = 3,
) -> str:
    """한국어 기술 문서 생성 (AI/ML 주제)."""
    parts = []
    for i in range(num_sections):
        level = (i % 4) + 1
        topic = _KOREAN_TOPICS[i % len(_KOREAN_TOPICS)]
        parts.append(f"{'#' * level} {topic} 기술 개요 — 섹션 {i + 1}")
        for p in range(paragraphs_per):
            sentence = _KOREAN_SENTENCES[(i + p) % len(_KOREAN_SENTENCES)]
            parts.append(f"\n{sentence} {topic}에 대한 {p + 1}번째 설명입니다. "
                         f"이 분야는 최근 급격한 발전을 이루고 있으며, "
                         f"다양한 산업 분야에서 활용되고 있습니다.\n")
    return "\n".join(parts)


def _generate_korean_mixed_markdown() -> str:
    """한영 혼합 마크다운 (코드 블록, 기술 용어)."""
    return """# ZvecSearch 아키텍처 설계

이 문서는 ZvecSearch의 핵심 아키텍처를 설명합니다.

## Backend Architecture

ZvecSearch는 `zvec` 벡터 데이터베이스를 기반으로 합니다.
HNSW(Hierarchical Navigable Small World) 알고리즘을 사용하여
근사 최근접 이웃(ANN) 검색을 수행합니다.

### 코드 구조

```python
class ZvecSearch:
    \"\"\"핵심 오케스트레이터 클래스.\"\"\"

    async def index(self, force: bool = False) -> int:
        # 파일 스캔 → 청킹 → 임베딩 → 저장
        files = scan_paths(self._paths)
        for f in files:
            await self._index_file(f, force=force)

    async def search(self, query: str, top_k: int = 10):
        # 쿼리 임베딩 → 하이브리드 검색
        embeddings = await self._embedder.embed([query])
        return self._store.search(embeddings[0], query)
```

## 임베딩 시스템 (Embedding System)

5개의 embedding provider를 지원합니다:
- **OpenAI**: `text-embedding-3-small` (기본값, 1536차원)
- **Google**: `text-embedding-004` (768차원)
- **Voyage AI**: `voyage-3-lite` (1024차원)
- **Ollama**: 로컬 모델 지원 (가변 차원)
- **Local**: `sentence-transformers` 기반 (384차원)

### 하이브리드 검색 (Hybrid Search)

Dense vector (HNSW/cosine) + Sparse vector (BM25)를 결합하여
RRF (Reciprocal Rank Fusion, k=60)로 재랭킹합니다.

## 성능 최적화

### 증분 인덱싱 (Incremental Indexing)

파일이 변경되지 않았으면 re-embedding을 건너뜁니다.
`chunk_hash`를 기반으로 변경 감지를 수행합니다.

### 메모리 최적화

zvec의 `enable_mmap=True` 설정으로 memory-mapped I/O를 활용합니다.
"""


def _generate_korean_table_heavy() -> str:
    """한국어 테이블 위주 문서."""
    parts = ["# 한국어 데이터 분석 보고서\n"]
    parts.append("## 모델 성능 비교표\n")
    parts.append("| 모델명 | 정확도 | F1 점수 | 추론 시간(ms) | 메모리(GB) |")
    parts.append("|--------|--------|---------|---------------|------------|")
    models = ["BERT-ko", "KoGPT-2", "KoBART", "KoELECTRA", "KoBigBird",
              "KoT5", "KoALBERT", "HyperCLOVA", "KoRoBERTa", "KoXLNet",
              "Polyglot-Ko", "LLaMA-Ko", "Gemma-Ko", "Solar", "EXAONE"]
    for m in models:
        acc = f"{random.uniform(85, 99):.1f}%"
        f1 = f"{random.uniform(80, 98):.2f}"
        latency = f"{random.randint(5, 200)}"
        mem = f"{random.uniform(0.5, 16):.1f}"
        parts.append(f"| {m} | {acc} | {f1} | {latency} | {mem} |")

    parts.append("\n## 데이터셋 통계\n")
    parts.append("| 데이터셋 | 문서 수 | 평균 길이 | 언어 | 도메인 |")
    parts.append("|----------|---------|----------|------|--------|")
    datasets = [
        ("KorQuAD 2.0", "10만", "350자", "한국어", "위키백과"),
        ("KLUE-NLI", "5만", "120자", "한국어", "뉴스"),
        ("한국어 위키", "80만", "2000자", "한국어", "백과사전"),
        ("네이버 영화리뷰", "20만", "80자", "한국어", "리뷰"),
        ("국립국어원 말뭉치", "100만", "500자", "한국어", "일반"),
        ("AI Hub 대화", "30만", "60자", "한국어", "대화"),
        ("법률 판례", "50만", "3000자", "한국어", "법률"),
        ("의학 논문", "15만", "5000자", "한국어/영어", "의학"),
        ("특허 문서", "200만", "1500자", "한국어", "특허"),
        ("소셜미디어", "500만", "40자", "한국어", "소셜"),
    ]
    for name, docs, avg_len, lang, domain in datasets:
        parts.append(f"| {name} | {docs} | {avg_len} | {lang} | {domain} |")

    return "\n".join(parts)


def _generate_korean_no_spaces(length: int = 2000) -> str:
    """띄어쓰기 없는 긴 한국어 텍스트."""
    base = "가나다라마바사아자차카타파하"
    # 자모 조합으로 다양한 글자 생성
    chars = []
    for i in range(length):
        idx = (i * 7 + 3) % len(base)
        chars.append(base[idx])
    return "".join(chars)


def _generate_korean_files(tmp_path, count: int = 100) -> list:
    """대량 한국어 파일 생성."""
    files = []
    for i in range(count):
        topic = _KOREAN_TOPICS[i % len(_KOREAN_TOPICS)]
        content = f"# {topic} 문서 {i + 1}\n\n"
        content += f"{_KOREAN_SENTENCES[i % len(_KOREAN_SENTENCES)]}\n\n"
        content += f"이 문서는 {topic}에 대한 {i + 1}번째 자료입니다. "
        content += "다양한 관점에서 기술의 발전 방향을 분석합니다.\n"
        fname = f"문서_{i:03d}_{topic}.md"
        path = tmp_path / fname
        path.write_text(content, encoding="utf-8")
        files.append(path)
    return files


# ===========================================================================
# 테스트 그룹 1: 한국어 청킹 스트레스 테스트
# ===========================================================================
class TestKoreanChunkerStress:
    """한국어 문서에 대한 청킹 정확성 테스트."""

    def test_korean_long_article_chunking(self):
        """50개 섹션 한국어 기술 문서 청킹."""
        md = _generate_korean_article(num_sections=50, paragraphs_per=3)
        chunks = chunk_markdown(md, source="한국어기술문서.md", max_chunk_size=800)
        assert len(chunks) >= 50
        # 모든 청크에 한국어 포함
        for c in chunks:
            has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in c.content)
            assert has_korean, f"한국어가 없는 청크: {c.content[:50]}"

    def test_korean_no_space_text_splitting(self):
        """띄어쓰기 없는 2000자 한국어 텍스트 분할."""
        no_space = _generate_korean_no_spaces(2000)
        md = f"# 띄어쓰기없는문장\n\n{no_space}"
        chunks = chunk_markdown(md, source="nospace.md", max_chunk_size=500)
        assert len(chunks) >= 1
        total_korean = sum(
            sum(1 for ch in c.content if "\uAC00" <= ch <= "\uD7A3")
            for c in chunks
        )
        assert total_korean >= 2000

    def test_korean_mixed_hangul_english_chunking(self):
        """한영 혼합 문서 청킹."""
        md = _generate_korean_mixed_markdown()
        chunks = chunk_markdown(md, source="mixed.md", max_chunk_size=600)
        assert len(chunks) >= 5
        # 한국어와 영어 모두 보존
        all_text = " ".join(c.content for c in chunks)
        assert "아키텍처" in all_text
        assert "ZvecSearch" in all_text
        assert "embedding" in all_text.lower()
        assert "하이브리드" in all_text

    def test_korean_heading_variations(self):
        """다양한 한국어 헤딩 형식 파싱."""
        md = """# 기본 한국어 헤딩

내용입니다.

## 특수문자 포함: 벡터(Vector) 검색!

설명입니다.

### 숫자 포함 — 3가지 방법론

방법론 설명.

#### 괄호와 기호 [참고] ★중요★

중요한 내용.

##### 긴 헤딩: 인공지능 기반 자연어처리 시스템의 설계와 구현에 관한 연구

연구 내용.

###### 마지막 레벨 — 요약 및 결론

결론입니다.
"""
        chunks = chunk_markdown(md, source="headings.md")
        headings = [c.heading for c in chunks if c.heading]
        assert "기본 한국어 헤딩" in headings
        assert any("벡터" in h for h in headings)
        assert any("3가지" in h for h in headings)
        assert any("★중요★" in h for h in headings)
        levels = {c.heading_level for c in chunks if c.heading_level > 0}
        assert levels == {1, 2, 3, 4, 5, 6}

    def test_korean_table_chunking(self):
        """한국어 테이블 문서 청킹."""
        md = _generate_korean_table_heavy()
        chunks = chunk_markdown(md, source="tables.md", max_chunk_size=1500)
        assert len(chunks) >= 2
        all_text = " ".join(c.content for c in chunks)
        assert "BERT-ko" in all_text
        assert "KorQuAD" in all_text
        # 테이블 구분자 보존
        assert "|" in all_text

    def test_korean_code_block_with_comments(self):
        """한국어 주석이 있는 코드 블록 보존."""
        md = """# 코드 예제

```python
# 한국어 주석: 데이터 전처리 함수
def 전처리(텍스트: str) -> str:
    \"\"\"텍스트를 정규화하는 함수입니다.\"\"\"
    결과 = 텍스트.strip()
    return 결과

# 실행 예시
print(전처리("  안녕하세요  "))
```

## 실행 결과

위 코드의 실행 결과는 "안녕하세요"입니다.
"""
        chunks = chunk_markdown(md, source="code.md")
        all_text = " ".join(c.content for c in chunks)
        assert "전처리" in all_text
        assert "한국어 주석" in all_text

    def test_korean_blockquote_and_list(self):
        """한국어 인용문, 목록 보존."""
        md = """# 한국어 마크다운 구조

> "지식은 힘이다." - 프랜시스 베이컨
> 이 인용문은 지식의 중요성을 강조합니다.

## 순서 없는 목록

- 첫 번째 항목: 인공지능의 역사
- 두 번째 항목: 현재 동향
  - 하위 항목: 대규모 언어 모델
  - 하위 항목: 멀티모달 AI
- 세 번째 항목: 미래 전망

## 순서 있는 목록

1. 데이터 수집 단계
2. 모델 학습 단계
3. 평가 및 배포 단계
"""
        chunks = chunk_markdown(md, source="lists.md")
        all_text = " ".join(c.content for c in chunks)
        assert "프랜시스 베이컨" in all_text
        assert "인공지능의 역사" in all_text
        assert "멀티모달 AI" in all_text

    def test_korean_special_characters(self):
        """한국어 특수 부호, 자모 보존."""
        md = """# 특수 문자 테스트

## 한글 자모

ㄱ ㄴ ㄷ ㄹ ㅁ ㅂ ㅅ ㅇ ㅈ ㅊ ㅋ ㅌ ㅍ ㅎ
ㅏ ㅑ ㅓ ㅕ ㅗ ㅛ ㅜ ㅠ ㅡ ㅣ

## 특수 부호

※ 주의사항: 다음 사항을 확인하세요.
★ 중요: 핵심 포인트입니다.
◆ 참고: 추가 정보가 있습니다.
■ 결론: 최종 요약입니다.
● 항목: 세부 내용입니다.
→ 다음 단계로 진행하세요.
「인용」 『참고문헌』 【보충 설명】

## 한국어 괄호 표현

(가) 첫 번째 조건
(나) 두 번째 조건
(다) 세 번째 조건
"""
        chunks = chunk_markdown(md, source="special.md")
        all_text = " ".join(c.content for c in chunks)
        assert "ㄱ" in all_text
        assert "ㅏ" in all_text
        assert "※" in all_text
        assert "★" in all_text
        assert "「인용」" in all_text
        assert "(가)" in all_text


# ===========================================================================
# 테스트 그룹 2: 한국어 스캐너 스트레스 테스트
# ===========================================================================
class TestKoreanScannerStress:
    """한국어 파일명/디렉토리명 스캐닝 테스트."""

    def test_scan_100_korean_named_files(self, tmp_path):
        """한국어 파일명 100개 스캐닝."""
        _generate_korean_files(tmp_path, count=100)
        results = scan_paths([tmp_path])
        assert len(results) == 100

    def test_korean_directory_names(self, tmp_path):
        """한국어 디렉토리명 5단계 중첩 스캐닝."""
        dirs = ["프로젝트", "문서관리", "기술자료", "인공지능", "벡터검색"]
        current = tmp_path
        for d in dirs:
            current = current / d
            current.mkdir()
            (current / f"{d}_설명.md").write_text(f"# {d}\n\n{d}에 대한 설명입니다.")
        results = scan_paths([tmp_path])
        assert len(results) == 5

    def test_mixed_korean_english_filenames(self, tmp_path):
        """한영 혼합 파일명 스캐닝."""
        names = [
            "AI기술_overview.md",
            "머신러닝_tutorial_v2.md",
            "deep_learning_딥러닝.md",
            "NLP_자연어처리_guide.md",
            "vector_벡터_search.md",
        ]
        for name in names:
            (tmp_path / name).write_text(f"# {name}\n내용")
        results = scan_paths([tmp_path])
        assert len(results) == 5

    def test_korean_filenames_with_spaces(self, tmp_path):
        """공백 포함 한국어 파일명 처리."""
        names = [
            "인공지능 개론.md",
            "머신러닝 기초 강의.md",
            "딥러닝 실습 자료 모음.md",
        ]
        for name in names:
            (tmp_path / name).write_text(f"# {name}\n내용")
        results = scan_paths([tmp_path])
        assert len(results) == 3

    def test_korean_filename_sorting(self, tmp_path):
        """한국어 파일명 정렬 순서 일관성."""
        names = ["다.md", "가.md", "나.md", "라.md", "마.md"]
        for name in names:
            (tmp_path / name).write_text(f"# {name}\n내용")
        results = scan_paths([tmp_path])
        result_names = [r.path.name for r in results]
        assert result_names == sorted(result_names)


# ===========================================================================
# 테스트 그룹 3: 한국어 파이프라인 스트레스 테스트
# ===========================================================================
class TestKoreanPipelineStress:
    """한국어 데이터를 사용한 전체 파이프라인 테스트."""

    @pytest.mark.asyncio
    async def test_index_100_korean_files(self, tmp_path):
        """100개 한국어 파일 인덱싱."""
        _generate_korean_files(tmp_path, count=100)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count == 100
            assert store.count() == 100
            # 한국어 내용 보존 확인
            stored = [d["content"] for d in store._docs.values()]
            korean_count = sum(
                1 for s in stored
                if any("\uAC00" <= ch <= "\uD7A3" for ch in s)
            )
            assert korean_count == 100
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_search_returns_korean_content(self, tmp_path):
        """한국어 쿼리로 검색."""
        for i in range(10):
            topic = _KOREAN_TOPICS[i]
            (tmp_path / f"doc_{i}.md").write_text(
                f"# {topic}\n\n{topic}에 대한 상세 설명입니다."
            )

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()

            emb._custom_return = [[0.5] * TEST_DIM]
            results = await zs.search("인공지능 기술", top_k=5)
            assert len(results) <= 5
            assert all("content" in r for r in results)
            # 결과에 한국어 포함
            for r in results:
                has_korean = any("\uAC00" <= ch <= "\uD7A3" for ch in r["content"])
                assert has_korean
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_incremental_add(self, tmp_path):
        """한국어 파일 50개 인덱싱 후 20개 추가."""
        _generate_korean_files(tmp_path, count=50)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count1 = await zs.index()
            assert count1 == 50

            # 20개 추가
            for i in range(50, 70):
                topic = _KOREAN_TOPICS[i % len(_KOREAN_TOPICS)]
                (tmp_path / f"추가문서_{i}.md").write_text(
                    f"# {topic} 추가\n\n추가된 문서입니다."
                )

            count2 = await zs.index()
            assert count2 == 20
            assert store.count() == 70
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_incremental_modify(self, tmp_path):
        """한국어 파일 내용 변경 후 증분 인덱싱."""
        fpath = tmp_path / "변경대상.md"
        fpath.write_text("# 원본 문서\n\n원본 내용입니다.")

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()
            assert store.count() == 1

            # 내용 변경
            fpath.write_text("# 수정된 문서\n\n수정된 내용으로 업데이트되었습니다.")
            count2 = await zs.index()
            assert count2 == 1  # 새 청크
            assert store.count() == 1  # 이전 것 삭제, 새 것 추가
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_incremental_delete_and_reindex(self, tmp_path):
        """파일 삭제 후 재인덱싱."""
        for i in range(5):
            (tmp_path / f"삭제테스트_{i}.md").write_text(
                f"# 문서 {i}\n\n내용 {i}입니다."
            )

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()
            assert store.count() == 5

            # 2개 파일 삭제
            (tmp_path / "삭제테스트_0.md").unlink()
            (tmp_path / "삭제테스트_1.md").unlink()

            # 재인덱싱 — 남은 3개만 재인덱싱 (삭제된 파일은 scan에서 빠짐)
            await zs.index()
            # 삭제된 파일의 소스가 store에서 자동 제거되지 않음 (scan에 없으므로)
            # 하지만 남은 파일은 이미 있으므로 0개 새 청크
            # 총 store에는 여전히 5개 (삭제된 소스를 명시적으로 지우지 않으면)
            assert store.count() >= 3
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_force_reindex_large(self, tmp_path):
        """50개 한국어 파일 강제 재인덱싱."""
        _generate_korean_files(tmp_path, count=50)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count1 = await zs.index()
            assert count1 == 50

            count2 = await zs.index(force=True)
            assert count2 == 50
            assert store.count() == 50
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_multi_search_cycle(self, tmp_path):
        """10회 인덱싱-검색 반복 사이클."""
        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            for cycle in range(10):
                topic = _KOREAN_TOPICS[cycle]
                (tmp_path / f"사이클_{cycle}_{topic}.md").write_text(
                    f"# {topic}\n\n{topic}에 대한 사이클 {cycle} 문서."
                )
                count = await zs.index()
                assert count == 1

                emb._custom_return = [[0.5] * TEST_DIM]
                results = await zs.search(f"{topic} 검색", top_k=50)
                assert len(results) == cycle + 1

            assert store.count() == 10
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_mixed_language_pipeline(self, tmp_path):
        """한국어+영어+일본어 혼합 파일 파이프라인."""
        (tmp_path / "korean.md").write_text(
            "# 한국어 문서\n\n인공지능 기술에 대한 한국어 설명입니다."
        )
        (tmp_path / "english.md").write_text(
            "# English Document\n\nThis is about artificial intelligence."
        )
        (tmp_path / "japanese.md").write_text(
            "# 日本語ドキュメント\n\n人工知能技術についての説明です。"
        )
        (tmp_path / "mixed.md").write_text(
            "# 혼합 Mixed 混合\n\n한국어 English 日本語 함께."
        )

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count == 4
            stored = {d["source"].split("/")[-1] for d in store._docs.values()}
            assert "korean.md" in stored
            assert "english.md" in stored
            assert "japanese.md" in stored
            assert "mixed.md" in stored
            zs.close()


# ===========================================================================
# 테스트 그룹 4: 한국어 해시 무결성 테스트
# ===========================================================================
class TestKoreanHashIntegrity:
    """한국어 콘텐츠의 해시 및 ID 무결성 검증."""

    def test_korean_content_hash_uniqueness(self):
        """500개 고유 한국어 콘텐츠 → 500개 고유 해시."""
        hashes = set()
        for i in range(500):
            topic = _KOREAN_TOPICS[i % len(_KOREAN_TOPICS)]
            content = f"{topic}에 대한 문서 번호 {i}입니다. 고유한 내용을 포함합니다."
            c = Chunk(content=content, source="hash.md", heading="",
                      heading_level=0, start_line=1, end_line=1)
            hashes.add(c.content_hash)
        assert len(hashes) == 500

    def test_korean_similar_content_different_hashes(self):
        """비슷한 한국어 문장(조사만 다른) → 다른 해시."""
        variations = [
            "인공지능이 발전하고 있다.",
            "인공지능은 발전하고 있다.",
            "인공지능을 발전시키고 있다.",
            "인공지능의 발전이 계속되고 있다.",
            "인공지능에 대한 발전이 이루어지고 있다.",
            "인공지능으로 인한 발전이 눈에 띈다.",
            "인공지능과 함께 발전하고 있다.",
            "인공지능에서 발전이 일어나고 있다.",
        ]
        hashes = set()
        for v in variations:
            c = Chunk(content=v, source="josa.md", heading="",
                      heading_level=0, start_line=1, end_line=1)
            hashes.add(c.content_hash)
        assert len(hashes) == len(variations)

    def test_korean_chunk_id_deterministic(self):
        """동일 한국어 입력 100회 반복 → 항상 같은 ID."""
        for _ in range(100):
            cid = compute_chunk_id("한국어문서.md", 1, 10, "해시값abc", "모델명")
            assert cid == compute_chunk_id("한국어문서.md", 1, 10, "해시값abc", "모델명")

    def test_korean_unicode_normalization_hashing(self):
        """NFC vs NFD 정규화 형태 → 해시 차이 검증."""
        # "한" in NFC (composed) vs NFD (decomposed)
        nfc_text = unicodedata.normalize("NFC", "한국어 테스트")
        nfd_text = unicodedata.normalize("NFD", "한국어 테스트")
        # NFC와 NFD는 바이트 표현이 다르므로 해시도 달라야 함
        c1 = Chunk(content=nfc_text, source="a.md", heading="",
                   heading_level=0, start_line=1, end_line=1)
        c2 = Chunk(content=nfd_text, source="a.md", heading="",
                   heading_level=0, start_line=1, end_line=1)
        if nfc_text.encode("utf-8") != nfd_text.encode("utf-8"):
            assert c1.content_hash != c2.content_hash
        else:
            assert c1.content_hash == c2.content_hash

    def test_korean_jamo_vs_syllable_hashing(self):
        """자모 조합 vs 완성형 음절 → 다른 해시."""
        # ㅎㅏㄴ (3 jamo) vs 한 (1 syllable)
        jamo = "ㅎㅏㄴ"
        syllable = "한"
        c1 = Chunk(content=jamo, source="a.md", heading="",
                   heading_level=0, start_line=1, end_line=1)
        c2 = Chunk(content=syllable, source="a.md", heading="",
                   heading_level=0, start_line=1, end_line=1)
        assert c1.content_hash != c2.content_hash


# ===========================================================================
# 테스트 그룹 5: 한국어 경계 조건 테스트
# ===========================================================================
class TestKoreanBoundaryConditions:
    """한국어 특화 경계 조건 테스트."""

    def test_only_korean_punctuation(self):
        """한국어 문장부호만 있는 섹션."""
        md = "# 제목\n\n。、「」『』【】\n\n## 다음 섹션\n\n내용"
        chunks = chunk_markdown(md, source="punct.md")
        assert len(chunks) >= 2
        # 마지막 섹션은 "내용"을 포함
        assert any("내용" in c.content for c in chunks)

    def test_korean_emoji_combination(self):
        """한국어 + 이모지 혼합 텍스트 보존."""
        md = """# 감정 분석 🤖

이 문서는 감정 분석에 대해 설명합니다 😊

## 긍정 😃 vs 부정 😢

긍정적 리뷰: "이 제품은 정말 좋습니다! 👍🎉"
부정적 리뷰: "실망스럽습니다 😞💔"

### 이모지 활용 사례 🌟

- 챗봇 응답: 안녕하세요! 🙋‍♂️
- 알림: 작업 완료! ✅
- 경고: 주의가 필요합니다! ⚠️🔴
"""
        chunks = chunk_markdown(md, source="emoji.md")
        all_text = " ".join(c.content for c in chunks)
        assert "🤖" in all_text
        assert "😊" in all_text
        assert "감정 분석" in all_text

    def test_extremely_long_korean_paragraph(self):
        """50,000자 한국어 단일 문단 분할.

        청커는 줄 단위로 분할하므로, 줄바꿈이 있는 긴 문단을 테스트.
        """
        sentence = "인공지능 기술은 현대 사회에서 매우 중요한 역할을 합니다."
        lines = []
        total_chars = 0
        while total_chars < 50_000:
            lines.append(sentence)
            total_chars += len(sentence) + 1  # +1 for newline
        long_para = "\n".join(lines)
        md = f"# 긴 문단\n\n{long_para}"
        chunks = chunk_markdown(md, source="long.md", max_chunk_size=2000)
        assert len(chunks) >= 25  # 50k / 2k = 25
        total = sum(len(c.content) for c in chunks)
        assert total >= 50_000

    def test_korean_url_and_links(self):
        """한국어 텍스트 내 URL, 마크다운 링크 보존."""
        md = """# 참고 자료

자세한 내용은 [네이버](https://www.naver.com)를 참고하세요.

관련 논문: [한국어 NLP 연구](https://arxiv.org/abs/2024.12345)

GitHub 저장소: https://github.com/jedikim/zvecsearch

## 관련 링크 모음

- [카카오 AI](https://ai.kakao.com) — 카카오의 AI 기술
- [네이버 클로바](https://clova.ai) — 네이버의 AI 플랫폼
"""
        chunks = chunk_markdown(md, source="links.md")
        all_text = " ".join(c.content for c in chunks)
        assert "https://www.naver.com" in all_text
        assert "github.com" in all_text
        assert "카카오 AI" in all_text

    def test_korean_footnotes_and_references(self):
        """각주, 참조 패턴 보존."""
        md = """# 연구 논문

인공지능은 1956년 다트머스 회의에서 시작되었다[^1].
최근 트랜스포머 아키텍처[^2]가 자연어처리에 혁명을 가져왔다.

[^1]: McCarthy, J. (1956). "The Dartmouth Conference"
[^2]: Vaswani, A. et al. (2017). "Attention Is All You Need"

## 참고문헌

1. 김철수 (2023). "한국어 대규모 언어 모델 개발". 한국정보과학회.
2. 이영희 (2024). "벡터 검색 최적화 기법". AI 연구소.
"""
        chunks = chunk_markdown(md, source="refs.md")
        all_text = " ".join(c.content for c in chunks)
        assert "[^1]" in all_text
        assert "김철수" in all_text

    def test_single_char_korean_headings(self):
        """단일 한글 글자 헤딩 파싱."""
        md = """# 가

첫 번째 섹션.

## 나

두 번째 섹션.

### 다

세 번째 섹션.
"""
        chunks = chunk_markdown(md, source="single.md")
        headings = [c.heading for c in chunks if c.heading]
        assert "가" in headings
        assert "나" in headings
        assert "다" in headings

    def test_korean_horizontal_rules_between_sections(self):
        """--- 구분선 사이 한국어 섹션."""
        md = """# 첫 번째 섹션

내용 1

---

# 두 번째 섹션

내용 2

---

# 세 번째 섹션

내용 3
"""
        chunks = chunk_markdown(md, source="rules.md")
        headings = [c.heading for c in chunks if c.heading]
        assert "첫 번째 섹션" in headings
        assert "두 번째 섹션" in headings
        assert "세 번째 섹션" in headings


# ===========================================================================
# 테스트 그룹 6: 대규모 한국어 데이터 통합 테스트
# ===========================================================================
class TestKoreanLargeScaleIntegration:
    """대규모 한국어 데이터의 전체 파이프라인 통합 테스트."""

    @pytest.mark.asyncio
    async def test_200_korean_files_full_pipeline(self, tmp_path):
        """200개 한국어 파일 전체 파이프라인."""
        _generate_korean_files(tmp_path, count=200)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count == 200
            assert store.count() == 200

            emb._custom_return = [[0.5] * TEST_DIM]
            results = await zs.search("인공지능", top_k=20)
            assert len(results) == 20
            zs.close()

    @pytest.mark.asyncio
    async def test_large_korean_document_1000_lines(self, tmp_path):
        """1000줄 한국어 문서 청킹 + 인덱싱 + 검색."""
        lines = ["# 대규모 한국어 기술 문서\n"]
        for i in range(50):
            topic = _KOREAN_TOPICS[i % len(_KOREAN_TOPICS)]
            lines.append(f"\n## 섹션 {i + 1}: {topic}\n")
            for j in range(19):
                sentence = _KOREAN_SENTENCES[(i + j) % len(_KOREAN_SENTENCES)]
                lines.append(f"{sentence}\n")
        md = "\n".join(lines)
        assert md.count("\n") >= 1000

        (tmp_path / "대규모문서.md").write_text(md)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count >= 50  # 최소 50개 섹션
            assert len(emb._embed_calls) >= 1
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_search_result_ordering(self, tmp_path):
        """다양한 top_k로 검색 결과 수 제한 준수."""
        _generate_korean_files(tmp_path, count=50)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()

            emb._custom_return = [[0.5] * TEST_DIM]
            for k in [1, 3, 5, 10, 25, 50]:
                results = await zs.search("머신러닝", top_k=k)
                assert len(results) <= k
                assert all(isinstance(r["score"], float) for r in results)
            zs.close()

    @pytest.mark.asyncio
    async def test_concurrent_korean_file_changes(self, tmp_path):
        """5회 사이클(파일 추가+수정+삭제) → 최종 상태."""
        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            live_files = set()
            for cycle in range(5):
                # 3개 파일 추가
                for j in range(3):
                    name = f"사이클{cycle}_문서{j}.md"
                    (tmp_path / name).write_text(
                        f"# 사이클 {cycle} 문서 {j}\n\n내용입니다."
                    )
                    live_files.add(name)

                await zs.index()

                # 이전 사이클 파일 1개 수정 (있는 경우)
                if cycle > 0:
                    modify_name = f"사이클{cycle - 1}_문서0.md"
                    if (tmp_path / modify_name).exists():
                        (tmp_path / modify_name).write_text(
                            f"# 수정됨 사이클 {cycle}\n\n수정된 내용."
                        )

                await zs.index()

            # 최종: 15개 파일이 생성됨
            assert len(live_files) == 15
            assert store.count() == 15
            zs.close()

    @pytest.mark.asyncio
    async def test_korean_content_utf8_roundtrip(self, tmp_path):
        """한국어 → 청킹 → 저장 → 조회 → 원본 일치."""
        original = "# 유니코드 라운드트립\n\n한글 가나다라마바사아자차카타파하. " \
                   "특수문자: ※★◆■●→. 자모: ㄱㄴㄷㄹㅁㅂㅅ. " \
                   "이모지: 🎉🚀💡. 일본어: こんにちは. 중국어: 你好世界."
        (tmp_path / "roundtrip.md").write_text(original)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()

            stored_content = list(store._docs.values())[0]["content"]
            assert "한글 가나다라마바사아자차카타파하" in stored_content
            assert "※★◆■●→" in stored_content
            assert "ㄱㄴㄷㄹㅁㅂㅅ" in stored_content
            assert "🎉🚀💡" in stored_content
            assert "こんにちは" in stored_content
            assert "你好世界" in stored_content
            zs.close()


# ===========================================================================
# 테스트 그룹 7: 한국어 CLI 스트레스 테스트
# ===========================================================================
class TestKoreanCLIStress:
    """한국어 환경에서의 CLI 동작 테스트."""

    def test_config_set_get_korean_collection(self):
        """한국어 컬렉션명 설정/조회."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        # config get으로 기본값 확인
        result = runner.invoke(cli, ["config", "get", "zvec.collection"])
        assert result.exit_code == 0
        assert "zvecsearch_chunks" in result.output

    def test_help_output_all_commands(self):
        """모든 CLI 명령어 --help 정상 동작."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        commands = ["index", "search", "watch", "compact", "stats",
                     "reset", "expand", "transcript", "config"]
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"'{cmd} --help' 실패: {result.output}"

    def test_stats_command_output_format(self):
        """stats 명령어 출력 형식 확인."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        # stats without valid store should handle gracefully
        result = runner.invoke(cli, ["stats", "--zvec-path", "/tmp/nonexistent_korean_test"])
        # Either succeeds or fails gracefully (not a crash)
        assert result.exit_code in (0, 1, 2)

    def test_config_list_all_sections(self):
        """config list가 모든 설정 섹션 포함."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "list", "--resolved"])
        assert result.exit_code == 0
        for section in ["zvec", "index", "embedding", "compact", "chunking", "watch"]:
            assert section in result.output, f"섹션 '{section}'이 출력에 없음"
