# 🤖 AutoCS-RAG: 서버리스 기반 CS AI Copilot 서비스

> **E-Commerce 고객센터 업무 효율화를 위한 서버리스 기반 RAG & Multi-Agent 챗봇**  
> 사내 규정 문서 및 FAQ 데이터를 바탕으로 고객 문의에 대해 100% 규정에 근거한 신뢰성 높은 답변 템플릿과 출처 조항을 제공합니다.

---

## 📌 1. 프로젝트 개요

* **배경 문제:** 신입 CS 상담원의 업무 숙련도 부족으로 인한 답변 작성 시간 지연, 시기별 규정 개정 시 이전 답변과의 불일치 문제, 심야 24시간 문의 대응 한계.
* **핵심 솔루션:** 
  * 사내 환불/배송 규정 및 FAQ 데이터를 벡터화(RAG).
  * LLM 환각(Hallucination) 방지를 위해 출처 조항(Source Citation) 필수 명시.
  * 유휴 비용 $0/월을 지향하는 완전 서버리스(Serverless) 아키텍처 구축.

---

## 🏗️ 2. 시스템 아키텍처

### V1: 비용 최적화 서버리스 아키텍처 (Cost-Optimized MVP)

```text
[프론트엔드: S3 정적 호스팅 / Streamlit]
          ↓ (HTTP POST)
[API Gateway + AWS Lambda (FastAPI / Mangum 컨테이너)]
          ├── (1) 쿼리 임베딩 ────────► [로컬 한국어 모델: jhgan/ko-sroberta-multitask]
          ├── (2) 벡터 검색 ──────────► [AWS S3 Vectors / 로컬 ChromaDB]
          └── (3) 답변 템플릿 생성 ───► [Google Gemini 2.5 Flash API (무료)]
```

### 🚀 V2 아키텍처 로드맵 (고성능 엔터프라이즈 RAG)
* **Amazon OpenSearch Serverless:** Hybrid Search (BM25 + Dense Vector) 도입으로 키워드 및 조항 번호 정확도 향상.
* **Valkey / Upstash Redis:** Semantic Caching 적용으로 유사 질문 **15ms 내 즉시 응답** 및 비용 95% 단축.
* **CloudFront:** SSE (Server-Sent Events) 스트리밍 연동으로 **Time To First Token(TTFT) < 1.2초** 달성.

---

## 🛠️ 3. 기술 스택 (Tech Stack)

* **Language & Framework:** Python 3.11, FastAPI, Mangum
* **RAG Engine & Pipeline:** LangChain (LCEL), LangGraph
* **Embedding Model:** `jhgan/ko-sroberta-multitask` (100% 로컬 무료 한국어 임베딩)
* **LLM:** Google Gemini 2.5 Flash (`gemini-2.5-flash`)
* **Vector DB:** AWS S3 Vectors (V1), Amazon OpenSearch (V2), ChromaDB (Local)
* **Cache / Session:** Upstash Redis / Valkey
* **Infra & Serverless:** AWS Lambda (Docker Container), API Gateway, S3

---

## 📂 4. 디렉토리 구조

```text
AutoCS-RAG/
├── data/                       # RAG용 규정 문서 및 FAQ (.md)
│   ├── 01_refund_policy.md     # 환불 및 교환 규정
│   └── 02_shipping_policy.md   # 배송 및 주문 가이드
├── src/                        # 백엔드 소스 코드
│   ├── test_local_rag.py       # LCEL 기반 RAG 로컬 실습 스크립트
│   └── main.py                 # FastAPI 애플리케이션 엔트리포인트 (예정)
├── requirements.txt            # 파이썬 의존성 패키지
├── .env.example                # 환경 변수 템플릿
└── README.md                   # 프로젝트 문서
```

---

## 🚦 5. 로컬 실행 방법

### 1) 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 입력
```

### 2) 가상환경 생성 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3) 로컬 RAG 파이프라인 테스트
```bash
python src/test_local_rag.py
```

---

## 📊 6. 성능 및 트레이드오프 분석 (README 포인트)

| 항목 | S3 Vectors (V1) | OpenSearch (V2) |
| :--- | :--- | :--- |
| **월 고정 비용** | **$0 (유휴 비용 없음)** | ~$200 (Minimum OCU) |
| **검색 지연시간** | ~250ms | **~25ms** |
| **검색 방식** | Dense Vector Search | **Hybrid (BM25 + Vector)** |
