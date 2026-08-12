import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. .env 환경 변수 로드
load_dotenv()

# Google API Key 설정 확인
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

def format_docs(docs):
    """검색된 문서들의 텍스트를 하나로 결합합니다."""
    return "\n\n".join(f"--- [근거 문서: {os.path.basename(doc.metadata.get('source', ''))}] ---\n{doc.page_content}" for doc in docs)

def run_local_rag_demo():
    print("🚀 [1/4] CS 데이터 문서 로딩 중...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    loader = DirectoryLoader(data_dir, glob="*.md", loader_cls=TextLoader)
    docs = loader.load()
    print(f"✅ 총 {len(docs)}개 문서 로드 완료.")

    print("\n✂️ [2/4] 문서 청킹(Chunking) 진행 중 (조항 보존을 위해 600자로 확장)...")
    # 조항 전체 문맥 유지를 위해 chunk_size를 600으로 조정
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    print(f"✅ 총 {len(splits)}개 텍스트 청크 생성 완료.")

    print("\n🧠 [3/4] 로컬 한국어 임베딩 모델(ko-sroberta) 기반 Vector DB(Chroma) 구축 중...")
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    print("✅ 한국어 임베딩 및 로컬 Chroma DB 저장 완료.")

    print("\n💬 [4/4] Gemini LLM RAG 질의응답 테스트...")
    
    # System Prompt 수정: 유연한 의미 추론 허용 및 친절한 CS 가이드
    system_prompt = (
        "너는 가상 의류 쇼핑몰 '스타일Hub'의 친절하고 정확한 전문 CS 상담원 AI야.\n"
        "제공된 [참고 문서]의 규정(예: 착용 흔적, 포장 훼손, 세탁, 가치 감소 등)에 근거하여 고객의 질문 상황을 유연하고 정확하게 해석해 줘.\n"
        "예를 들어 '한 번 입어본 옷'은 문서의 '착용 흔적'이나 '상품 가치 감소' 조항에 해당하는 것으로 적용해 판단해야 해.\n"
        "답변 시 결론(환불 가능 여부)을 먼저 명확히 안내하고, 관련 근거 조항(예: 제2조 3항 등)을 언급해 줘.\n\n"
        "[참고 문서]:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.1
    )
    
    # LCEL 파이프라인
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "input": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    # 테스트 질문
    user_query = "개봉해서 한 번 입어본 옷도 7일 이내면 단순 변심으로 환불되나요?"
    print(f"\n❓ [고객 질문]: {user_query}\n")

    response = rag_chain_with_source.invoke(user_query)

    print("🤖 [AI CS 상담원 답변]:")
    print(response["answer"])
    
    print("\n📄 [검색된 참조 청크 내역]:")
    for idx, doc in enumerate(response["context"]):
        print(f"\n--- [청크 {idx+1}] {os.path.basename(doc.metadata.get('source', ''))} ---")
        print(doc.page_content[:150] + "...")

if __name__ == "__main__":
    if not api_key:
        print("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해 주세요.")
    else:
        run_local_rag_demo()
