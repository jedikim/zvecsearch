# Scripts — 실제 API 임베딩 테스트

zvec 없이 독립 실행 가능한 실제 API 테스트 스크립트.

## 요구사항

```bash
pip install openai google-genai numpy
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="AIza..."
```

## 스크립트 목록

### test_gemini_embedding.py

Gemini vs OpenAI 임베딩 품질 직접 비교. numpy 코사인 유사도로 검색 수행.

```bash
GOOGLE_API_KEY=... OPENAI_API_KEY=... python scripts/test_gemini_embedding.py
```

**테스트 내용:**
- 단일 임베딩 벡터 생성 (Gemini 768차원, OpenAI 1536차원)
- 한국어 코퍼스 10개 문서 임베딩
- 10개 쿼리(키워드/동의어/개념/영한혼합) × top_k=3 Hit Rate 비교
- 시맨틱 쿼리 top-1 결과 상세 비교

**출력 예시:**
```
  쿼리                                     기대     Gemini   OpenAI
  ------------------------------------------------------------------
  HNSW 알고리즘이란?                        HNSW         ✓        ✓
  벡터 사이의 각도로 유사도를 재는 방식..     코사인       ✓        ✓
  ...
  Hit Rate                                          10/10     9/10
```

### test_zvecsearch_gemini.py

zvecsearch 파이프라인을 통한 실제 임베딩 테스트. `ZvecSearch.index()` → `store.embed_and_upsert()` → 코사인 검색.

```bash
GOOGLE_API_KEY=... OPENAI_API_KEY=... python scripts/test_zvecsearch_gemini.py
```

**테스트 내용:**
- zvecsearch 파이프라인으로 마크다운 문서 인덱싱
- 실제 API 임베딩 + numpy 코사인 유사도 검색 (zvec DB 대신)
- Gemini vs OpenAI Recall@5, 평균 유사도 점수 비교
- 카테고리별 분석: 키워드 매칭, 동의어/패러프레이즈, 개념 수준, 영한 혼합

**핵심 구현:**
- `RealVectorCollection` 클래스: zvec Collection API를 모방하되 내부적으로 numpy 코사인 검색 수행
- zvec DB 연산만 모킹, 임베딩 API는 실제 호출
- zvec가 설치되지 않은 환경에서도 임베딩 품질 평가 가능

**출력 예시:**
```
  Provider     Recall@5   Avg Score   Time
  ------------------------------------------------
  Gemini        10/10      0.762     12.3s
  OpenAI         9/10      0.460     8.7s
```

## 벤치마크 (benchmarks/)

`benchmarks/` 디렉토리에는 pytest 기반 5-Phase 벤치마크가 있다. API 키 없이도 Phase 1-4는 실행된다.

```bash
# Phase 1-4 (API 키 불필요, 키워드 기반)
pytest benchmarks/ -v -k "not phase5"

# Phase 5 (실제 임베딩, API 키 필요)
OPENAI_API_KEY=... GOOGLE_API_KEY=... pytest benchmarks/test_phase5_embeddings.py -v -s

# 전체
pytest benchmarks/ -v
```

| Phase | 목적 | 테스트 수 |
|-------|------|----------|
| Phase 1 | 데이터셋 검증 | ~20 |
| Phase 2 | 검색 품질 (Recall, MRR, NDCG) | ~60 |
| Phase 3 | 콘텐츠 품질 (RAGAS 메트릭) | ~40 |
| Phase 4 | 성능 (처리량, 지연시간) | ~80 |
| Phase 5 | 실제 임베딩 비교 (OpenAI vs Gemini vs Keyword) | ~80 |
