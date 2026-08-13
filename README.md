# 🤖 AutoCS-RAG: 서버리스 기반 CS AI Copilot 서비스

> **E-Commerce 고객센터 업무 효율화를 위한 서버리스 기반 RAG & Multi-Agent 챗봇**  
> 사내 규정 문서 및 FAQ 데이터를 바탕으로 고객 문의에 대해 100% 규정에 근거한 신뢰성 높은 답변 템플릿과 출처 조항을 제공합니다.

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-LCEL-1C3C3C">
  <img alt="LLM" src="https://img.shields.io/badge/LLM-Gemini%202.5%20Flash-4285F4?logo=google&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue">
</p>

---

## 📑 목차
1. [프로젝트 개요](#-1-프로젝트-개요)
2. [시스템 아키텍처](#️-2-시스템-아키텍처)
3. [기술 스택](#️-3-기술-스택-tech-stack)
4. [디렉토리 구조](#-4-디렉토리-구조)
5. [로컬 실행 방법](#-5-로컬-실행-방법)
6. [실행 결과 예시](#-6-실행-결과-예시)
7. [성능 및 트레이드오프 분석](#-7-성능-및-트레이드오프-분석)
8. [라이선스](#-8-라이선스)

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

> ℹ️ 현재 저장소에는 **로컬 RAG 파이프라인 데모(V1 코어)** 가 구현되어 있으며, Lambda 배포 및 V2 구성 요소는 로드맵 단계입니다.

---

## 🛠️ 3. 기술 스택 (Tech Stack)

* **Language & Framework:** Python 3.11+, FastAPI, Mangum
* **RAG Engine & Pipeline:** LangChain (LCEL) — *LangGraph 기반 Multi-Agent는 로드맵*
* **Embedding Model:** `jhgan/ko-sroberta-multitask` (100% 로컬 무료 한국어 임베딩)
* **LLM:** Google Gemini 2.5 Flash (`gemini-2.5-flash`)
* **Vector DB:** AWS S3 Vectors (V1), Amazon OpenSearch (V2), ChromaDB (Local)
* **Cache / Session:** Upstash Redis / Valkey *(로드맵)*
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
├── LICENSE                     # Apache License 2.0
└── README.md                   # 프로젝트 문서
```

---

## 🚦 5. 로컬 실행 방법

### 사전 준비물
* Python **3.11 이상** (개발/검증 환경: 3.12)
* [Google AI Studio](https://aistudio.google.com/app/apikey)에서 발급받은 **Gemini API Key** (무료 티어 지원)
* 최초 실행 시 임베딩 모델(`ko-sroberta`, 약 440MB)이 자동 다운로드되며, 인터넷 연결이 필요합니다.

### 1) 환경 변수 설정
```bash
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 입력
```

### 2) 가상환경 생성 및 패키지 설치
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3) 로컬 RAG 파이프라인 테스트
```bash
python src/test_local_rag.py
```

---

## 📊 6. 실행 결과 예시

`test_local_rag.py`는 데이터 로딩 → 청킹(600자) → 로컬 임베딩 및 Chroma 색인 → Gemini 질의응답까지 4단계를 순차 실행합니다.

**입력 질문**
```
개봉해서 한 번 입어본 옷도 7일 이내면 단순 변심으로 환불되나요?
```

**출력 (요약)**
```text
🚀 [1/4] CS 데이터 문서 로딩 중...
✅ 총 2개 문서 로드 완료.
✂️ [2/4] 문서 청킹(Chunking) 진행 중...
🧠 [3/4] 로컬 한국어 임베딩 모델(ko-sroberta) 기반 Vector DB(Chroma) 구축 중...
💬 [4/4] Gemini LLM RAG 질의응답 테스트...

🤖 [AI CS 상담원 답변]:
결론적으로 단순 변심에 의한 환불이 어렵습니다. 착용 흔적이 있는 상품은
'상품의 가치가 현저히 감소한 경우'(제2조 3항)에 해당하여 청약철회가 제한됩니다...

📄 [검색된 참조 청크 내역]:
--- [청크 1] 01_refund_policy.md ---
...
```

> 답변은 항상 **결론 → 근거 조항 인용 → 참조 청크 출처** 순으로 제공되어, 환각을 방지하고 규정 근거를 추적할 수 있습니다.

---

## 📈 7. 성능 및 트레이드오프 분석

| 항목 | S3 Vectors (V1) | OpenSearch (V2) |
| :--- | :--- | :--- |
| **월 고정 비용** | **$0 (유휴 비용 없음)** | ~$200 (Minimum OCU) |
| **검색 지연시간** | ~250ms | **~25ms** |
| **검색 방식** | Dense Vector Search | **Hybrid (BM25 + Vector)** |

---

## 📄 8. 라이선스

본 프로젝트는 [Apache License 2.0](LICENSE) 하에 배포됩니다.
