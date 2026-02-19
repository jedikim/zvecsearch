# ZvecSearch

[zvec](https://github.com/alibaba/zvec) (Alibaba 임베디드 벡터 데이터베이스) 기반 시맨틱 메모리 검색 시스템.

마크다운 문서를 청킹하고 임베딩(OpenAI, Gemini, 또는 로컬 모델)하여 하이브리드 검색(dense + sparse)을 수행한다. 서버 불필요.

> Inspired by [memsearch](https://github.com/zilliztech/memsearch) and [OpenClaw](https://github.com/openclaw/openclaw)'s markdown-first memory architecture.

**[English README](README.md)**

## 설치

```bash
pip install -e ".[dev]"

# Gemini 임베딩 사용 시
pip install -e ".[google]"

# zvec 기본 로컬 임베딩 사용 시 (API 키 불필요)
pip install sentence-transformers
```

### 요구사항

- Python 3.10+
- zvec >= 0.2.0
- **Default 로컬**: `sentence-transformers` (API 키 불필요, 완전 오프라인)
- **OpenAI**: `OPENAI_API_KEY`
- **Gemini**: `GOOGLE_API_KEY` + `pip install -e ".[google]"`

### zvec x86_64 빌드 이슈

공식 zvec PyPI 휠은 AVX-512 명령어로 컴파일되어 있어, AVX-512을 지원하지 않는 CPU(대부분의 AMD CPU, 구형 Intel CPU, 다수의 VM)에서 `SIGILL` 크래시가 발생한다. [alibaba/zvec#128](https://github.com/alibaba/zvec/issues/128) 참고.

`-march=x86-64-v2` (SSE4.2, 대부분의 x86_64 시스템과 호환)로 빌드된 휠을 `dist/` 디렉토리에 포함시켜 두었다:

```bash
pip install dist/zvec-0.2.1.dev0-cp312-cp312-linux_x86_64.whl
```

## 빠른 시작

### 3줄로 검색

```python
from zvecsearch import ZvecSearch

zs = ZvecSearch(paths=["./docs"])
zs.index()                                       # 마크다운 인덱싱
results = zs.search("HNSW 알고리즘이란?", top_k=5)  # 시맨틱 검색
```

### CLI

```bash
zvecsearch index ./docs/           # 마크다운 파일 인덱싱
zvecsearch search "HNSW 알고리즘"   # 시맨틱 검색
zvecsearch watch ./docs/           # 파일 변경 감시, 자동 재인덱싱
zvecsearch compact                 # LLM 기반 청크 요약
```

## 사용법

### Python API

```python
from zvecsearch import ZvecSearch

# 커스텀 설정으로 초기화
zs = ZvecSearch(
    paths=["./docs", "./notes"],
    embedding_provider="openai",        # "default" (로컬), "openai", 또는 "google"
    embedding_model="text-embedding-3-small",
    quantize_type="int8",               # "int8", "int4", "fp16", "none"
    reranker="rrf",                     # "rrf" 또는 "weighted"
)

# 마크다운 파일 인덱싱 (증분 — 변경된 청크만 재임베딩)
zs.index()

# 전체 재인덱싱
zs.index(force=True)

# 단일 파일 인덱싱
zs.index_file("./docs/new-note.md")

# 검색
results = zs.search("벡터 유사도 검색", top_k=10)
for r in results:
    print(f"[{r['score']:.4f}] {r['source']}:{r['start_line']}-{r['end_line']}")
    print(f"  제목: {r['heading']}")
    print(f"  {r['content'][:100]}...")
    print()

# 파일 변경 감시 (생성/수정/삭제 시 자동 재인덱싱)
watcher = zs.watch(debounce_ms=1500)
watcher.start()
# ... 백그라운드에서 감시 ...
watcher.stop()

# LLM 기반 청크 요약 (비동기)
import asyncio
summary = asyncio.run(zs.compact(
    source="./docs/long-document.md",
    llm_provider="openai",
    output_dir="./output",
))

# 컨텍스트 매니저 지원
with ZvecSearch(paths=["./docs"]) as zs:
    zs.index()
    results = zs.search("쿼리")
```

### CLI 명령어

```bash
# 마크다운 파일 인덱싱
zvecsearch index ./docs/
zvecsearch index ./docs/ ./notes/ --force    # 전체 재인덱싱
zvecsearch index ./docs/ --provider google   # Gemini 임베딩 사용
zvecsearch index ./docs/ --provider default  # 로컬 임베딩 (API 키 불필요)

# 시맨틱 검색
zvecsearch search "HNSW 동작 원리"
zvecsearch search "쿼리" --top-k 20 --json  # JSON 출력

# 파일 변경 감시 (자동 재인덱싱)
zvecsearch watch ./docs/
zvecsearch watch ./docs/ --debounce-ms 3000

# LLM 요약
zvecsearch compact
zvecsearch compact --source ./docs/file.md

# 설정
zvecsearch config show                       # 현재 설정 확인
zvecsearch config set embedding.provider google
zvecsearch config set embedding.model gemini-embedding-001
zvecsearch config set search.reranker weighted
zvecsearch config set zvec.quantize_type int4
```

### 설정

설정 우선순위: **기본값** < **글로벌 설정** < **프로젝트 설정** < **CLI 플래그**

글로벌 설정: `~/.zvecsearch/config.toml`
프로젝트 설정: `.zvecsearch.toml`

```toml
[zvec]
path = "~/.zvecsearch/db"          # DB 저장 경로
collection = "zvecsearch_chunks"   # 컬렉션 이름
enable_mmap = true                 # memory-mapped I/O
hnsw_m = 16                        # HNSW 노드당 최대 연결 수
hnsw_ef = 300                      # HNSW ef_construction
quantize_type = "int8"             # "int8", "int4", "fp16", "none"

[embedding]
provider = "default"               # "default" (로컬), "openai", 또는 "google"
model = ""                         # 자동; 또는 "text-embedding-3-small", "gemini-embedding-001"

[search]
top_k = 10
query_ef = 300                     # HNSW 검색 시 ef
reranker = "default"               # "default" (로컬 cross-encoder), "rrf", 또는 "weighted"
dense_weight = 1.0                 # weighted reranker용
sparse_weight = 0.8                # weighted reranker용

[chunking]
max_chunk_size = 1500
overlap_lines = 2

[watch]
debounce_ms = 1500
```

## 아키텍처

### 하이브리드 검색

```
Query -> +-- Dense embedding (OpenAI/Gemini) -> HNSW cosine search --+
         +-- Sparse embedding (BM25)         -> Inverted index      --+
                                                                      |
                                                               RRF ReRanker -> Results
```

모든 쿼리는 **두 가지 검색을 동시에** 수행:

1. **Dense 검색**: 쿼리를 임베딩(OpenAI 또는 Gemini)하여 HNSW 인덱스에서 코사인 유사도 검색.
2. **Sparse 검색**: zvec 네이티브 `BM25EmbeddingFunction`으로 키워드 매칭.

결과는 **RRF ReRanker** (기본) 또는 **Weighted ReRanker**가 합쳐서 최종 순위를 매긴다.

### zvec 기본 로컬 프로바이더

zvec의 기본 설정은 **API 키 없이** **네트워크 없이** 동작하는 로컬 모델 조합을 사용한다:

| 컴포넌트 | 클래스 | 모델 | 크기 |
|---------|-------|------|------|
| Dense 임베딩 | `DefaultLocalDenseEmbedding` | all-MiniLM-L6-v2 (384차원) | ~80MB |
| Sparse 임베딩 | `DefaultLocalSparseEmbedding` | SPLADE | ~100MB |
| 리랭커 | `DefaultLocalReRanker` | cross-encoder/ms-marco-MiniLM-L6-v2 | ~80MB |

이 `Default*` 클래스들은 zvec에 내장되어 있으며, 첫 사용 시 모델이 자동 다운로드된다.

### 임베딩 프로바이더

| 프로바이더 | 모델 | 차원 | 환경변수 |
|-----------|------|------|---------|
| zvec Default (로컬) | all-MiniLM-L6-v2 | 384 | 없음 (로컬) |
| OpenAI | text-embedding-3-small | 1536 | `OPENAI_API_KEY` |
| Gemini | gemini-embedding-001 | 768 | `GOOGLE_API_KEY` |

OpenAI는 zvec 네이티브 `OpenAIDenseEmbedding` 사용. Gemini는 zvec의 `DenseEmbeddingFunction` Protocol을 구현한 커스텀 `GeminiDenseEmbedding` 클래스로 지원. zvecsearch는 zvec의 로컬 프로바이더(`DefaultLocalDenseEmbedding` + `DefaultLocalSparseEmbedding` + `DefaultLocalReRanker`)를 기본값으로 사용한다 — API 키 없이 바로 동작한다. OpenAI는 zvec 네이티브 `OpenAIDenseEmbedding` 사용. Gemini는 zvec의 `DenseEmbeddingFunction` Protocol을 구현한 커스텀 `GeminiDenseEmbedding` 클래스로 지원.

### Sparse 임베딩

| 프로바이더 | 클래스 | 설명 |
|-----------|-------|------|
| BM25 (zvecsearch 기본) | `BM25EmbeddingFunction` | 키워드 기반, 로컬, 모델 다운로드 불필요 |
| SPLADE (zvec 기본) | `DefaultLocalSparseEmbedding` | 학습된 희소 임베딩, 로컬, ~100MB |

### 리랭커

| 리랭커 | 방식 | 설명 |
|-------|------|------|
| **RRF** (기본) | 순위 합산 | 순위 위치 기반 합산. 튜닝 불필요. |
| **Weighted** | 점수 합산 | dense/sparse 점수의 가중 합산. 비율 조절 가능. |
| **DefaultLocalReRanker** | Cross-encoder | ms-marco-MiniLM-L6-v2, 높은 정확도, 느림. 로컬, ~80MB. |
| **QwenReRanker** | Cross-encoder | Qwen 기반 리랭커, 중국어/다국어 지원. |

### 스토리지

zvec는 **임베디드** 벡터 DB — 서버 프로세스 불필요.

- 파일 기반 저장: `~/.zvecsearch/db/` (설정 가능)
- HNSW 인덱스, COSINE 메트릭 (M=16, ef_construction=300)
- INT8 양자화 기본 (INT4, FP16도 지원)
- mmap으로 대용량 인덱스 효율적 로딩
- Apache Arrow + RocksDB 스토리지 백엔드

### 증분 인덱싱

변경된 콘텐츠만 재임베딩하여 API 비용 절감:

1. 각 청크에 SHA-256 `chunk_hash` 부여 (콘텐츠, 소스, 라인 범위 기반)
2. 재인덱싱 시 기존 해시와 비교 — 변경 없는 청크는 스킵
3. 삭제/수정된 콘텐츠의 오래된 청크는 자동 제거

### 마크다운 청킹

- 헤딩(`#`, `##`, `###` 등) 기반으로 문서를 의미 단위로 분할
- 각 청크에 메타데이터 부착: `source`, `heading`, `heading_level`, `start_line`, `end_line`
- `max_chunk_size` 및 `overlap_lines` 설정으로 조절 가능

## 프로젝트 구조

```
zvecsearch/
├── src/zvecsearch/
│   ├── core.py        # ZvecSearch 오케스트레이터 (sync index/search/watch, async compact)
│   ├── store.py       # ZvecStore (zvec Collection 래퍼 + GeminiDenseEmbedding)
│   ├── chunker.py     # 마크다운 청킹 (헤딩 기반 분할)
│   ├── scanner.py     # 파일 탐색 (.md/.markdown)
│   ├── watcher.py     # 파일 변경 감시 (watchdog, 디바운스)
│   ├── config.py      # TOML 설정 (글로벌/프로젝트 레이어)
│   ├── compact.py     # LLM 기반 청크 요약 (비동기)
│   ├── cli.py         # Click CLI 인터페이스
│   └── transcript.py  # 트랜스크립트 유틸리티
├── tests/             # pytest 단위 테스트 (286개)
├── benchmarks/        # 5-Phase 벤치마크 (62개)
├── scripts/           # 실제 API 임베딩 테스트 스크립트
├── dist/              # 빌드된 zvec 휠 (x86-64-v2)
└── pyproject.toml
```

## 테스트

```bash
# 단위 테스트 (283개)
pytest tests/ -v

# 벤치마크 (62개, Phase 1-4는 API 키 불필요)
pytest benchmarks/ -v

# Phase 5: 실제 임베딩 비교 (API 키 필요)
OPENAI_API_KEY=... GOOGLE_API_KEY=... pytest benchmarks/test_phase5_embeddings.py -v -s

# 실제 API 임베딩 스크립트
OPENAI_API_KEY=... GOOGLE_API_KEY=... python scripts/test_gemini_embedding.py
OPENAI_API_KEY=... GOOGLE_API_KEY=... python scripts/test_zvecsearch_gemini.py

# 기본 로컬 임베딩 테스트 (API 키 불필요)
python scripts/test_default_local.py

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

시맨틱 쿼리(동의어, 패러프레이즈, 영한 혼합)에서 임베딩 검색이 키워드 검색 대비 확실한 우위를 보인다.

## 라이선스

MIT
