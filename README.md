# ZvecSearch

zvec(Alibaba 임베디드 벡터 데이터베이스) 기반 시맨틱 메모리 검색 시스템.

마크다운 문서를 청킹하고 임베딩하여 하이브리드 검색(dense + sparse)을 수행한다.

## 설치

```bash
pip install -e ".[dev]"

# Gemini 임베딩 사용 시
pip install -e ".[google]"
```

### 요구사항

- Python 3.10+
- zvec 0.1.0+
- OpenAI API 키 (기본 임베딩) 또는 Google API 키 (Gemini 임베딩)

## 사용법

### CLI

```bash
# 마크다운 파일 인덱싱
zvecsearch index ./docs/

# 시맨틱 검색
zvecsearch search "벡터 검색 원리"

# 파일 변경 감시 (자동 재인덱싱)
zvecsearch watch ./docs/

# 인덱스 최적화 (컴팩션)
zvecsearch compact

# 설정 확인/변경
zvecsearch config show
zvecsearch config set embedding.provider google
zvecsearch config set embedding.model gemini-embedding-001
```

### Python API

```python
from zvecsearch import ZvecSearch

zs = ZvecSearch()

# 인덱싱
zs.index("./docs/")

# 검색
results = zs.search("HNSW 알고리즘이란?", top_k=5)
for r in results:
    print(f"[{r['score']:.4f}] {r['source']}:{r['start_line']}")
    print(f"  {r['content'][:80]}...")
```

### 설정

`~/.zvecsearch/config.toml` (글로벌) 또는 `.zvecsearch.toml` (프로젝트)에서 설정:

```toml
[zvec]
path = "~/.zvecsearch/db"
collection = "zvecsearch_chunks"
hnsw_m = 16
hnsw_ef = 300
quantize_type = "int8"

[embedding]
provider = "openai"          # "openai" 또는 "google"
model = "text-embedding-3-small"  # 또는 "gemini-embedding-001"

[search]
top_k = 10
reranker = "rrf"             # "rrf" 또는 "weighted"
dense_weight = 1.0
sparse_weight = 0.8
```

## 프로젝트 구조

```
zvecsearch/
├── src/zvecsearch/
│   ├── core.py        # ZvecSearch 오케스트레이터 (index/search/compact/watch)
│   ├── store.py       # ZvecStore — zvec Collection 래퍼 + GeminiDenseEmbedding
│   ├── chunker.py     # 마크다운 청킹 (헤딩 기반 분할)
│   ├── scanner.py     # .md/.markdown 파일 탐색
│   ├── watcher.py     # 파일 변경 감시 (watchdog, 디바운스)
│   ├── config.py      # TOML 설정 (글로벌/프로젝트 레이어)
│   ├── compact.py     # LLM 기반 청크 요약 (비동기)
│   ├── cli.py         # Click CLI 인터페이스
│   └── transcript.py  # 트랜스크립트 유틸리티
├── tests/             # pytest 테스트 (345개)
├── benchmarks/        # 5-Phase 벤치마크 (280개)
├── scripts/           # 실제 API 테스트 스크립트
└── pyproject.toml
```

## 아키텍처

### 하이브리드 검색

```
쿼리 → ┌─ Dense 임베딩 (OpenAI/Gemini) → HNSW 코사인 검색
       └─ Sparse 임베딩 (BM25)         → 역인덱스 검색
                                          ↓
                                    RRF ReRanker → 결과
```

- **Dense 벡터**: OpenAI `text-embedding-3-small` (1536차원) 또는 Gemini `gemini-embedding-001` (768차원)
- **Sparse 벡터**: zvec 네이티브 `BM25EmbeddingFunction` (document/query 분리)
- **리랭킹**: `RrfReRanker` (기본, rank_constant=60) 또는 `WeightedReRanker`
- **양자화**: INT8 (기본), INT4, FP16 지원 — 메모리 절감

### zvec 네이티브 스토리지

서버 없이 파일 기반으로 동작하는 임베디드 벡터 DB:

- HNSW 인덱스 (M=16, ef_construction=300)
- 코사인 유사도 메트릭
- mmap 지원으로 대용량 인덱스 효율적 로딩
- 증분 인덱싱: chunk_hash 기반으로 변경분만 재임베딩 → API 비용 절감

### 임베딩 프로바이더

| 프로바이더 | 모델 | 차원 | 설정 |
|-----------|------|------|------|
| OpenAI | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| Gemini | gemini-embedding-001 | 768 | `GOOGLE_API_KEY` |

Gemini는 zvec의 `DenseEmbeddingFunction` Protocol을 구현한 커스텀 클래스(`GeminiDenseEmbedding`)로 지원된다.

### 마크다운 청킹

- 헤딩(`#`, `##`, ...) 기반으로 문서를 의미 단위로 분할
- 각 청크에 source, heading, heading_level, start_line, end_line 메타데이터 부착
- SHA-256 chunk_hash로 중복/변경 감지

## 테스트

```bash
# 전체 테스트 (345개)
pytest tests/ -v

# 벤치마크 (280개, API 키 불필요)
pytest benchmarks/ -v

# 실제 임베딩 벤치마크 (API 키 필요)
OPENAI_API_KEY=... GOOGLE_API_KEY=... pytest benchmarks/test_phase5_embeddings.py -v -s

# 린트
ruff check src/ tests/ benchmarks/
```

### 벤치마크 결과 요약

| Phase | 항목 | 결과 |
|-------|------|------|
| Phase 2 | Recall@5 | 0.8250 |
| Phase 2 | MRR | 0.8083 |
| Phase 2 | NDCG@5 | 0.7849 |
| Phase 3 | Faithfulness | 0.8667 |
| Phase 3 | Context Relevance | 0.7000 |
| Phase 4 | Chunking 처리량 | 42K docs/s |
| Phase 4 | 검색 QPS | 12K QPS |
| Phase 5 | Gemini Recall@5 | 1.0000 |
| Phase 5 | OpenAI Recall@5 | 1.0000 |
| Phase 5 | Keyword Recall@5 | 0.9333 |

시맨틱 쿼리(동의어/패러프레이즈)에서 임베딩 검색이 키워드 검색 대비 확실한 우위를 보인다.

## 라이선스

MIT
