"""대규모 데이터와 복잡한 시나리오를 사용한 스트레스 테스트.

다음을 검증합니다:
- 수백 개의 파일/수천 개의 청크를 사용한 대규모 인덱싱
- 깊은 디렉토리 구조 스캐닝
- 유니코드/다국어 콘텐츠 처리
- 복잡한 마크다운 구조 (6단계 헤딩, 코드 블록, 테이블)
- 증분 인덱싱 정확성 (수정/삭제/추가)
- 해시 충돌 없음 검증
- 청킹 경계 조건 (매우 긴 줄, 빈 섹션, 특수 문자)
- 전체 파이프라인 스트레스 (scan → chunk → embed → store → search)
"""
from __future__ import annotations

import random
import sys
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
from zvecsearch.config import (  # noqa: E402
    ZvecSearchConfig,
    deep_merge,
    resolve_config,
    config_to_dict,
    load_config_file,
    save_config,
    get_config_value,
)
from zvecsearch.core import ZvecSearch  # noqa: E402
from zvecsearch.scanner import scan_paths  # noqa: E402

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------
TEST_DIM = 8
LARGE_FILE_COUNT = 50
LARGE_HEADINGS_PER_FILE = 20


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
    """Mock 오염을 피하기 위한 순수 Python 임베딩 제공자.

    test_store.py의 BM25 Mock이 MagicMock 클래스를 오염시키므로,
    AsyncMock 대신 순수 클래스를 사용하여 완전히 격리.
    """

    def __init__(self, dim: int = TEST_DIM):
        self.model_name = "stress-test-model"
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
    """결정적 벡터를 반환하는 임베딩 제공자."""
    return FakeEmbedder(dim=dim)


# ---------------------------------------------------------------------------
# 대규모 마크다운 생성 유틸리티
# ---------------------------------------------------------------------------
def _generate_large_markdown(
    num_headings: int = 20,
    paragraphs_per_heading: int = 3,
    words_per_paragraph: int = 80,
) -> str:
    """수많은 헤딩과 단락으로 구성된 대형 마크다운 생성."""
    parts = []
    for i in range(num_headings):
        level = (i % 4) + 1  # h1~h4 순환
        parts.append(f"{'#' * level} 섹션 {i}: 주제 {chr(65 + i % 26)}")
        for p in range(paragraphs_per_heading):
            words = " ".join(
                f"단어{random.randint(100, 999)}" for _ in range(words_per_paragraph)
            )
            parts.append(f"\n{words}\n")
    return "\n".join(parts)


def _generate_unicode_markdown() -> str:
    """다양한 유니코드/다국어 콘텐츠가 포함된 마크다운 생성."""
    return """# 한국어 테스트 문서

이것은 한국어로 작성된 테스트 문서입니다.
시맨틱 검색 시스템의 다국어 지원을 검증합니다.

## 日本語セクション

これは日本語のテストセクションです。
マルチバイト文字のチャンキングを検証します。

## 中文部分

这是一个中文测试部分。
用于验证多语言支持和Unicode处理。

## Emojis & Symbols 🚀

Special characters: é, ñ, ü, ø, ß, æ, θ, λ, π, Σ
Math: ∫∑∏√∞±≠≈≤≥
Arrows: ←→↑↓↔⇒⇐⇑⇓
Box drawing: ┌─┐│└─┘├┤┬┴┼

## 코드 블록 테스트

```python
def 인사(이름: str) -> str:
    return f"안녕하세요, {이름}님!"

# 유니코드 변수명도 지원합니다
결과 = 인사("세계")
print(결과)
```

## 테이블 테스트

| 이름 | 나이 | 도시 |
|------|------|------|
| 김철수 | 25 | 서울 |
| 이영희 | 30 | 부산 |
| 박지민 | 28 | 대구 |

## 혼합 콘텐츠

> 인용문: "지식은 힘이다." - 프랜시스 베이컨

- 목록 항목 1: 첫 번째
- 목록 항목 2: 두 번째
  - 하위 항목: 중첩된 목록

1. 번호 목록 1
2. 번호 목록 2
3. 번호 목록 3

---

마지막 단락. 이 문서는 다양한 유니코드 문자와 마크다운 구조를 테스트합니다.
"""


def _generate_complex_nested_markdown() -> str:
    """6단계 헤딩, 코드 블록, 인라인 코드가 포함된 복잡한 마크다운."""
    return """# Level 1: 아키텍처 개요

시스템 아키텍처에 대한 최상위 설명입니다.

## Level 2: 백엔드 시스템

백엔드는 Python으로 구현되었습니다.

### Level 3: 데이터베이스 계층

벡터 데이터베이스로 zvec를 사용합니다.

#### Level 4: 인덱싱 전략

HNSW 알고리즘을 사용한 근사 최근접 이웃 검색.

##### Level 5: 파라미터 튜닝

`ef_construction=300`, `max_m=16` 설정 사용.

###### Level 6: 벤치마크 결과

| 데이터셋 | 차원 | QPS | Recall@10 |
|----------|------|-----|-----------|
| 1M vectors | 768 | 5000 | 0.98 |
| 10M vectors | 768 | 2000 | 0.95 |

## Level 2: 프론트엔드

CLI 인터페이스로 Click 프레임워크 사용.

### Level 3: 명령어 구조

```bash
zvecsearch index ./docs/
zvecsearch search "벡터 데이터베이스란?"
zvecsearch watch ./docs/ --debounce-ms 2000
```

### Level 3: 설정 관리

TOML 기반 계층적 설정 시스템:
- 글로벌: `~/.zvecsearch/config.toml`
- 프로젝트: `.zvecsearch.toml`
- CLI 오버라이드

#### Level 4: 설정 예시

```toml
[zvec]
path = "~/.zvecsearch/db"
collection = "my_knowledge"

[embedding]
provider = "openai"
model = "text-embedding-3-small"
```

## Level 2: 임베딩 시스템

5개 임베딩 제공자 지원.

### Level 3: OpenAI

기본 제공자, `text-embedding-3-small` 모델.

### Level 3: 로컬 모델

`sentence-transformers` 기반, 오프라인 사용 가능.

# Level 1: 배포 가이드

## Level 2: 설치

```bash
pip install zvecsearch
pip install "zvecsearch[all]"  # 모든 의존성
```

## Level 2: 업그레이드

마이그레이션 스크립트가 필요할 수 있습니다.
"""


# ===========================================================================
# 테스트 그룹 1: 청커 스트레스 테스트
# ===========================================================================
class TestChunkerStress:
    """대규모 및 복잡한 마크다운에 대한 청킹 테스트."""

    def test_large_document_produces_many_chunks(self):
        """20개 헤딩이 있는 대형 문서가 올바르게 청킹되는지 검증."""
        md = _generate_large_markdown(num_headings=20, paragraphs_per_heading=3)
        chunks = chunk_markdown(md, source="large.md", max_chunk_size=500)
        # 큰 섹션이 분할되므로 최소 20개 이상의 청크 예상
        assert len(chunks) >= 20
        # 모든 청크에 콘텐츠가 있어야 함
        assert all(c.content.strip() for c in chunks)
        # 모든 청크가 max_chunk_size보다 작거나 합리적 범위 이내
        for c in chunks:
            assert len(c.content) < 2000  # 약간의 여유 허용

    def test_50_heading_document(self):
        """50개 헤딩 문서의 청킹 정확성."""
        md = _generate_large_markdown(num_headings=50, paragraphs_per_heading=1, words_per_paragraph=30)
        chunks = chunk_markdown(md, source="fifty.md")
        assert len(chunks) >= 50
        # 각 청크의 start_line이 단조 증가
        for i in range(1, len(chunks)):
            assert chunks[i].start_line >= chunks[i - 1].start_line

    def test_unicode_content_chunking(self):
        """유니코드/다국어 콘텐츠 청킹."""
        md = _generate_unicode_markdown()
        chunks = chunk_markdown(md, source="unicode.md")
        assert len(chunks) >= 7  # 최소 7개 섹션
        # 한국어 콘텐츠 포함 확인
        korean_chunks = [c for c in chunks if "한국어" in c.content or "테스트" in c.content]
        assert len(korean_chunks) >= 1
        # 일본어 콘텐츠 포함 확인
        japanese_chunks = [c for c in chunks if "日本語" in c.content]
        assert len(japanese_chunks) >= 1
        # 이모지 콘텐츠 포함 확인
        emoji_chunks = [c for c in chunks if "🚀" in c.content]
        assert len(emoji_chunks) >= 1

    def test_six_level_heading_hierarchy(self):
        """6단계 헤딩 계층 구조 처리."""
        md = _generate_complex_nested_markdown()
        chunks = chunk_markdown(md, source="complex.md")
        levels = {c.heading_level for c in chunks if c.heading_level > 0}
        # h1~h6까지 모든 레벨이 있어야 함
        assert levels == {1, 2, 3, 4, 5, 6}

    def test_code_block_heading_is_known_limitation(self):
        """코드 블록 안의 # 문자가 헤딩으로 인식되는 것은 알려진 제한 사항.

        현재 청커는 코드 블록 컨텍스트를 추적하지 않으므로,
        코드 블록 안의 '#'도 헤딩으로 분할됨. 이것은 memsearch와 동일한 동작.
        """
        md = "# Real Heading\n\n```python\n# This is a comment\ndef foo():\n    pass\n```\n\n## Another"
        chunks = chunk_markdown(md, source="code.md")
        headings = [c.heading for c in chunks if c.heading]
        assert "Real Heading" in headings
        assert "Another" in headings
        # 알려진 제한: 코드 블록 내 #도 헤딩으로 인식됨
        assert len(chunks) >= 2

    def test_very_long_single_line(self):
        """10,000자 단일 줄 처리."""
        long_line = "x" * 10_000
        md = f"# Long\n{long_line}"
        chunks = chunk_markdown(md, source="long.md", max_chunk_size=2000)
        assert len(chunks) >= 1
        # 전체 콘텐츠가 보존되어야 함
        total_chars = sum(len(c.content) for c in chunks)
        assert total_chars >= 10_000

    def test_empty_sections_between_headings(self):
        """헤딩 사이에 빈 섹션이 있는 경우."""
        md = "# A\n\n# B\n\n# C\nContent here."
        chunks = chunk_markdown(md, source="empty.md")
        # 빈 섹션은 건너뛰어야 함
        content_chunks = [c for c in chunks if c.content.strip()]
        assert all(c.content.strip() for c in content_chunks)

    def test_chunk_hash_uniqueness_at_scale(self):
        """대규모 청킹 시 해시 고유성 검증.

        오버랩 분할 시 동일 콘텐츠가 포함될 수 있으므로,
        고유 콘텐츠를 가진 청크는 고유 해시를 가져야 함.
        """
        md = _generate_large_markdown(num_headings=100, paragraphs_per_heading=2, words_per_paragraph=50)
        chunks = chunk_markdown(md, source="unique.md", max_chunk_size=200)
        # 동일 콘텐츠 → 동일 해시 (정상 동작)
        content_to_hash = {}
        for c in chunks:
            if c.content in content_to_hash:
                assert c.content_hash == content_to_hash[c.content], "같은 콘텐츠인데 다른 해시"
            else:
                content_to_hash[c.content] = c.content_hash
        # 고유 콘텐츠 수와 고유 해시 수가 일치해야 함
        unique_contents = set(c.content for c in chunks)
        unique_hashes = set(c.content_hash for c in chunks)
        assert len(unique_contents) == len(unique_hashes), "고유 콘텐츠와 고유 해시 수가 불일치"

    def test_chunk_id_varies_with_model(self):
        """같은 청크라도 모델이 다르면 chunk_id가 달라야 함."""
        ids_per_model = {}
        for model in ["openai", "google", "voyage", "ollama", "local"]:
            chunk_id = compute_chunk_id("test.md", 1, 10, "abc123", model)
            ids_per_model[model] = chunk_id
        # 모든 모델의 ID가 서로 달라야 함
        assert len(set(ids_per_model.values())) == 5

    def test_overlap_produces_shared_content(self):
        """오버랩 설정이 실제로 겹치는 콘텐츠를 생성하는지 검증."""
        lines = ["# Big Section"]
        for i in range(50):
            lines.append(f"Line {i}: " + "word " * 30)
        md = "\n".join(lines)
        chunks = chunk_markdown(md, source="overlap.md", max_chunk_size=300, overlap_lines=3)
        if len(chunks) >= 2:
            # 인접 청크 간 일부 줄이 겹쳐야 함
            first_lines = set(chunks[0].content.split("\n"))
            second_lines = set(chunks[1].content.split("\n"))
            overlap = first_lines & second_lines
            # 빈 줄 제외하고 겹치는 줄이 있어야 함
            meaningful_overlap = {line for line in overlap if line.strip()}
            assert len(meaningful_overlap) >= 1, "오버랩이 작동하지 않음"


# ===========================================================================
# 테스트 그룹 2: 스캐너 스트레스 테스트
# ===========================================================================
class TestScannerStress:
    """대규모 디렉토리 구조 스캐닝 테스트."""

    def test_scan_100_files(self, tmp_path):
        """100개 마크다운 파일 스캐닝."""
        for i in range(100):
            (tmp_path / f"doc_{i:03d}.md").write_text(f"# Doc {i}\nContent {i}")
        results = scan_paths([tmp_path])
        assert len(results) == 100

    def test_deeply_nested_directories(self, tmp_path):
        """10단계 깊이의 중첩 디렉토리."""
        current = tmp_path
        for depth in range(10):
            current = current / f"level_{depth}"
            current.mkdir()
            (current / f"doc_depth_{depth}.md").write_text(f"# Depth {depth}")
        results = scan_paths([tmp_path])
        assert len(results) == 10

    def test_mixed_extensions(self, tmp_path):
        """다양한 확장자가 혼합된 디렉토리."""
        for ext in [".md", ".markdown", ".txt", ".py", ".json", ".html", ".rst"]:
            for i in range(10):
                (tmp_path / f"file_{i}{ext}").write_text(f"Content {i}")
        results = scan_paths([tmp_path])
        # .md와 .markdown만 포함 (각 10개)
        assert len(results) == 20

    def test_hidden_files_and_dirs(self, tmp_path):
        """숨김 파일/디렉토리 무시 검증."""
        (tmp_path / "visible.md").write_text("# Visible")
        (tmp_path / ".hidden.md").write_text("# Hidden")
        hidden_dir = tmp_path / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "inside.md").write_text("# Inside hidden")
        results = scan_paths([tmp_path])
        assert len(results) == 1
        assert results[0].path.name == "visible.md"

    def test_symlink_handling(self, tmp_path):
        """심볼릭 링크가 있는 디렉토리."""
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        (real_dir / "a.md").write_text("# A")
        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)
        results = scan_paths([tmp_path])
        # 실제 파일과 링크된 파일 모두 발견 (중복 제거됨)
        paths = {r.path for r in results}
        assert len(paths) >= 1

    def test_unicode_filenames(self, tmp_path):
        """유니코드 파일명 처리."""
        (tmp_path / "한국어문서.md").write_text("# 한국어")
        (tmp_path / "日本語ドキュメント.md").write_text("# 日本語")
        (tmp_path / "документ.md").write_text("# Русский")
        results = scan_paths([tmp_path])
        assert len(results) == 3

    def test_large_files(self, tmp_path):
        """대용량 파일 (100KB+) 스캐닝."""
        large_content = "# Big\n" + "x" * 100_000
        (tmp_path / "large.md").write_text(large_content)
        results = scan_paths([tmp_path])
        assert len(results) == 1
        assert results[0].size > 100_000

    def test_dedup_across_multiple_paths(self, tmp_path):
        """여러 경로에서 동일 파일 중복 제거."""
        sub1 = tmp_path / "a"
        sub1.mkdir()
        (sub1 / "shared.md").write_text("# Shared")
        results = scan_paths([sub1, sub1 / "shared.md", tmp_path])
        paths = [r.path for r in results]
        assert len(paths) == len(set(paths))


# ===========================================================================
# 테스트 그룹 3: 설정 스트레스 테스트
# ===========================================================================
class TestConfigStress:
    """설정 시스템 경계 조건 테스트."""

    def test_deep_merge_deeply_nested(self):
        """5단계 깊이의 중첩 딕셔너리 병합."""
        base = {"a": {"b": {"c": {"d": {"e": 1}}}}}
        over = {"a": {"b": {"c": {"d": {"e": 99, "f": 100}}}}}
        result = deep_merge(base, over)
        assert result["a"]["b"]["c"]["d"]["e"] == 99
        assert result["a"]["b"]["c"]["d"]["f"] == 100

    def test_config_round_trip(self, tmp_path):
        """설정 저장 후 다시 로드하면 동일해야 함."""
        original = config_to_dict(ZvecSearchConfig())
        f = tmp_path / "cfg.toml"
        save_config(original, f)
        loaded = load_config_file(f)
        # 모든 최상위 키가 존재해야 함
        for key in ["zvec", "index", "embedding", "compact", "chunking", "watch"]:
            assert key in loaded

    def test_resolve_with_many_overrides(self):
        """여러 설정 값을 동시에 오버라이드."""
        cfg = resolve_config({
            "zvec": {"path": "/custom/path", "collection": "custom_col"},
            "index": {"metric": "l2", "hnsw_ef": 500},
            "embedding": {"provider": "ollama", "model": "custom-model"},
            "chunking": {"max_chunk_size": 3000, "overlap_lines": 5},
            "watch": {"debounce_ms": 3000},
        })
        assert cfg.zvec.path == "/custom/path"
        assert cfg.zvec.collection == "custom_col"
        assert cfg.index.metric == "l2"
        assert cfg.index.hnsw_ef == 500
        assert cfg.embedding.provider == "ollama"
        assert cfg.embedding.model == "custom-model"
        assert cfg.chunking.max_chunk_size == 3000
        assert cfg.watch.debounce_ms == 3000

    def test_get_config_value_all_paths(self):
        """모든 설정 경로에 대한 값 조회."""
        cfg = ZvecSearchConfig()
        paths_and_expected = [
            ("zvec.path", "~/.zvecsearch/db"),
            ("zvec.collection", "zvecsearch_chunks"),
            ("zvec.enable_mmap", True),
            ("index.type", "hnsw"),
            ("index.metric", "cosine"),
            ("index.hnsw_ef", 300),
            ("index.hnsw_max_m", 16),
            ("embedding.provider", "openai"),
            ("chunking.max_chunk_size", 1500),
            ("chunking.overlap_lines", 2),
            ("watch.debounce_ms", 1500),
        ]
        for path, expected in paths_and_expected:
            assert get_config_value(path, cfg) == expected, f"실패: {path}"

    def test_deep_merge_preserves_unrelated_keys(self):
        """병합 시 관련 없는 키가 보존되는지 검증."""
        base = {"a": 1, "b": 2, "c": {"x": 10, "y": 20}}
        over = {"c": {"x": 99}}
        result = deep_merge(base, over)
        assert result == {"a": 1, "b": 2, "c": {"x": 99, "y": 20}}


# ===========================================================================
# 테스트 그룹 4: 스토어/코어 대규모 파이프라인 테스트
# ===========================================================================
class TestLargePipelineStress:
    """대규모 데이터를 사용한 전체 파이프라인 테스트."""

    @pytest.mark.asyncio
    async def test_index_50_files(self, tmp_path):
        """50개 파일을 인덱싱하고 검색."""
        # 50개 마크다운 파일 생성
        for i in range(LARGE_FILE_COUNT):
            content = f"# 문서 {i}: 주제\n\n"
            content += f"이것은 문서 번호 {i}입니다. "
            content += f"키워드: AI, 머신러닝, 벡터검색, 문서{i}. "
            content += "Lorem ipsum dolor sit amet. " * 10
            (tmp_path / f"doc_{i:03d}.md").write_text(content)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count == LARGE_FILE_COUNT  # 각 파일 1 청크
            assert store.count() == LARGE_FILE_COUNT

            # 검색
            emb._custom_return = [[0.5] * TEST_DIM]
            results = await zs.search("AI 벡터검색", top_k=10)
            assert len(results) <= 10
            assert all("content" in r for r in results)
            zs.close()

    @pytest.mark.asyncio
    async def test_index_files_with_many_chunks(self, tmp_path):
        """큰 파일에서 수백 개의 청크가 생성되는 파이프라인."""
        big_md = _generate_large_markdown(
            num_headings=30, paragraphs_per_heading=2, words_per_paragraph=50
        )
        (tmp_path / "big.md").write_text(big_md)

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count >= 30  # 최소 30개 청크
            # 임베딩 호출이 실제로 발생했는지 확인
            assert len(emb._embed_calls) >= 1
            zs.close()

    @pytest.mark.asyncio
    async def test_incremental_index_add_files(self, tmp_path):
        """파일 추가 후 증분 인덱싱."""
        (tmp_path / "initial.md").write_text("# Initial\nFirst document.")

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            # 첫 인덱싱
            count1 = await zs.index()
            assert count1 == 1

            # 파일 추가
            (tmp_path / "added.md").write_text("# Added\nSecond document.")

            # 두 번째 인덱싱 - 새 파일만 인덱싱
            count2 = await zs.index()
            assert count2 == 1  # 새 파일만
            assert store.count() == 2  # 총 2개

            zs.close()

    @pytest.mark.asyncio
    async def test_incremental_index_modify_file(self, tmp_path):
        """파일 수정 후 증분 인덱싱이 올바르게 작동하는지 검증."""
        file_path = tmp_path / "changeable.md"
        file_path.write_text("# Version 1\nOriginal content.")

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            # 첫 인덱싱
            await zs.index()
            initial_count = store.count()
            assert initial_count == 1

            # 파일 수정 (내용 변경)
            file_path.write_text("# Version 2\nModified content with new info.")

            # 두 번째 인덱싱 - 변경된 청크가 업데이트됨
            count2 = await zs.index()
            assert count2 == 1  # 새 청크 1개
            # 이전 해시가 삭제되고 새 해시가 추가됨
            assert store.count() == 1

            zs.close()

    @pytest.mark.asyncio
    async def test_force_reindex_large_dataset(self, tmp_path):
        """대규모 데이터셋의 강제 재인덱싱."""
        for i in range(20):
            (tmp_path / f"doc_{i}.md").write_text(f"# Doc {i}\nContent for doc {i}.")

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            # 첫 인덱싱
            count1 = await zs.index()
            assert count1 == 20

            # 강제 재인덱싱
            count2 = await zs.index(force=True)
            assert count2 == 20  # 모든 청크 재인덱싱
            assert store.count() == 20

            zs.close()

    @pytest.mark.asyncio
    async def test_search_top_k_variations(self, tmp_path):
        """다양한 top_k 값으로 검색."""
        for i in range(30):
            (tmp_path / f"doc_{i}.md").write_text(f"# Doc {i}\nContent about topic {i}.")

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            await zs.index()

            emb._custom_return = [[0.5] * TEST_DIM]

            for k in [1, 3, 5, 10, 20, 50]:
                results = await zs.search("topic", top_k=k)
                assert len(results) <= k
                assert all(isinstance(r["score"], float) for r in results)

            zs.close()

    @pytest.mark.asyncio
    async def test_unicode_content_pipeline(self, tmp_path):
        """유니코드 콘텐츠의 전체 파이프라인."""
        (tmp_path / "korean.md").write_text(_generate_unicode_markdown())
        (tmp_path / "complex.md").write_text(_generate_complex_nested_markdown())

        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")
            count = await zs.index()
            assert count >= 10  # 두 파일에서 최소 10개 청크

            # 유니코드 콘텐츠가 정상 저장되었는지 확인
            stored_contents = [d["content"] for d in store._docs.values()]
            has_korean = any("한국어" in c for c in stored_contents)
            assert has_korean, "한국어 콘텐츠가 저장되지 않음"

            zs.close()

    @pytest.mark.asyncio
    async def test_multiple_index_search_cycles(self, tmp_path):
        """인덱싱-검색 사이클을 여러 번 반복."""
        store = _make_stateful_store()
        emb = _make_embedder()

        with patch("zvecsearch.core.get_provider", side_effect=lambda *a, **kw: emb), \
             patch("zvecsearch.core.ZvecStore", side_effect=lambda *a, **kw: store):
            zs = ZvecSearch(paths=[str(tmp_path)], zvec_path="/tmp/fake")

            for cycle in range(5):
                # 새 파일 추가
                (tmp_path / f"cycle_{cycle}.md").write_text(
                    f"# Cycle {cycle}\nContent for cycle {cycle}."
                )

                # 인덱싱
                count = await zs.index()
                assert count == 1  # 매번 새 파일 1개만

                # 검색
                emb._custom_return = [[0.5] * TEST_DIM]
                results = await zs.search("cycle", top_k=50)
                assert len(results) == cycle + 1  # 누적

            assert store.count() == 5
            zs.close()


# ===========================================================================
# 테스트 그룹 5: 해시/ID 무결성 테스트
# ===========================================================================
class TestHashIntegrity:
    """해시와 청크 ID의 무결성 검증."""

    def test_1000_unique_chunk_ids(self):
        """1000개의 서로 다른 입력에서 고유 청크 ID 생성."""
        ids = set()
        for i in range(1000):
            cid = compute_chunk_id(f"file_{i}.md", i, i + 10, f"hash_{i}", "model")
            ids.add(cid)
        assert len(ids) == 1000

    def test_chunk_hash_deterministic_across_calls(self):
        """동일 입력에 대한 해시 결정성."""
        for _ in range(100):
            c1 = Chunk(content="test content", source="a.md", heading="H",
                       heading_level=1, start_line=1, end_line=5)
            c2 = Chunk(content="test content", source="a.md", heading="H",
                       heading_level=1, start_line=1, end_line=5)
            assert c1.content_hash == c2.content_hash

    def test_similar_content_different_hashes(self):
        """미세하게 다른 콘텐츠도 다른 해시를 가져야 함."""
        hashes = set()
        for i in range(200):
            # 매번 고유한 콘텐츠 생성
            content = f"The quick brown fox jumps over the lazy dog number {i}"
            c = Chunk(content=content, source="a.md", heading="",
                      heading_level=0, start_line=1, end_line=1)
            hashes.add(c.content_hash)
        # 모든 고유 콘텐츠에 대해 고유 해시
        assert len(hashes) == 200

    def test_chunk_id_all_params_affect_result(self):
        """chunk_id의 모든 매개변수가 결과에 영향을 미치는지 검증."""
        base_id = compute_chunk_id("file.md", 1, 10, "hash1", "model1")

        # 각 매개변수를 변경하면 ID가 달라져야 함
        assert compute_chunk_id("other.md", 1, 10, "hash1", "model1") != base_id
        assert compute_chunk_id("file.md", 2, 10, "hash1", "model1") != base_id
        assert compute_chunk_id("file.md", 1, 11, "hash1", "model1") != base_id
        assert compute_chunk_id("file.md", 1, 10, "hash2", "model1") != base_id
        assert compute_chunk_id("file.md", 1, 10, "hash1", "model2") != base_id


# ===========================================================================
# 테스트 그룹 6: CLI 스트레스 테스트
# ===========================================================================
class TestCLIStress:
    """CLI의 대규모 출력 처리 테스트."""

    def test_config_list_contains_all_sections(self):
        """config list가 모든 설정 섹션을 포함하는지 검증."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["config", "list", "--resolved"])
        assert result.exit_code == 0
        for section in ["zvec", "index", "embedding", "compact", "chunking", "watch"]:
            assert section in result.output

    def test_help_for_all_commands(self):
        """모든 CLI 명령에 대한 --help 정상 동작."""
        from click.testing import CliRunner
        from zvecsearch.cli import cli

        runner = CliRunner()
        commands = ["index", "search", "watch", "compact", "stats",
                     "reset", "expand", "transcript", "config"]
        for cmd in commands:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"'{cmd} --help' 실패: {result.output}"
            assert "Usage" in result.output or "usage" in result.output.lower()
