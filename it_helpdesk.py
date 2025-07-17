import streamlit as st
import json
import os
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from keywords import (
    GREETING_KEYWORDS,
    CASUAL_KEYWORDS,
    OFFTOPIC_KEYWORDS,
    WITTY_KEYWORDS,
)

load_dotenv()


def classify(user_input, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-mini")

    prompt = f"""당신은 IT 헬프데스크 상담사입니다.
다음 문의를 분석하여 적절한 분류를 결정하세요.

**분류 기준:**

1. **existing** - 일반적인 IT 문제 (기존 FAQ/매뉴얼 있음)
   - 네트워크, 계정, 이메일, 하드웨어, 소프트웨어 등 흔한 문제
   - 예: "와이파이 안됨", "비밀번호 잊음", "프린터 오류"

2. **new** - 새롭지만 가치 있는 IT 질문
   - 다른 직원들도 궁금해할 만한 새로운 기술/정책/도구 관련
   - 예: "새로운 협업도구 사용법", "최신 보안 정책", "신규 시스템"

3. **skip** - FAQ 가치 없는 질문
   - 인사, 감사, 일반 대화, IT 무관한 질문
   - 예: "안녕하세요", "감사합니다", "점심 메뉴", "회의실 예약"

**중요**: 다른 직원들도 궁금해할 만한 IT 질문인지 판단하세요.

문의: {user_input}

분류: """

    result = chat.invoke(
        [
            SystemMessage(
                content="당신은 IT 헬프데스크의 베테랑 상담사입니다. 정확한 문제 분류와 친절한 고객 응대를 동시에 수행합니다."
            ),
            HumanMessage(content=prompt),
        ]
    )

    classification = result.content.strip()

    # 분류 결과 정리
    if "existing" in classification:
        return "existing"
    elif "new" in classification:
        return "new"
    elif "skip" in classification:
        return "skip"
    else:
        return "skip"


def answer(user_input, chat_history, category, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-pro")
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")

    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # if db._collection.count() == 0:
    #     load_manual(db)

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 질문 재구성
    ctx_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "이전 대화를 참고해서 질문을 재구성하세요."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    hist_retriever = create_history_aware_retriever(chat, retriever, ctx_prompt)

    # 답변 생성
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""IT 헬프데스크입니다. 분류: {category}

        규칙:
        1. 매뉴얼 정보 활용
        2. 모르면 IT 관리자 연결
        3. 단계별 설명

        지원 영역: 네트워크, 계정, 이메일, 하드웨어, 소프트웨어, 보안

        참고: {{context}}""",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    qa_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(hist_retriever, qa_chain)

    # 히스토리 변환
    history = []
    for msg in chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    result = rag_chain.invoke({"input": user_input, "chat_history": history})
    return result["answer"]


def load_manual(db):
    with open("it_helpdesk_manual.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        doc = Document(
            page_content=item["text_content"],
            metadata={
                "id": item["id"],
                "category": item["metadata"]["category"],
                "scenario": item["metadata"]["scenario"],
                "keywords": ", ".join(item["metadata"]["keywords"]),
                "priority": item["metadata"]["priority"],
            },
        )
        docs.append(doc)

    db.add_documents(docs)
    print(f"{len(docs)}개 문서 벡터 DB에 추가 완료")


def get_response(user_input, chat_history, api_key):
    category = classify(user_input, api_key)
    response = answer(user_input, chat_history, category, api_key)
    return response


@st.cache_resource
def init_db():
    """앱 시작 시 한 번만 실행되는 DB 초기화"""
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("API KEY를 찾을 수 없습니다.")
        return None

    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    if db._collection.count() == 0:
        load_manual(db)
        print("✅ 매뉴얼 로딩 완료!")
    else:
        print("✅ 기존 매뉴얼 데이터 로딩 완료")

    return db


# UI 시작 부분
st.set_page_config(page_title="IT 헬프데스크", page_icon="💻")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("UPSTAGE_API_KEY")

if st.session_state.api_key:
    db = init_db()
    if db is None:
        st.error("매뉴얼 로딩 실패")

st.title("💻 IT 헬프데스크")

if not st.session_state.api_key:
    st.warning("API 키 필요")
    api_key = st.text_input("API 키:", type="password")
    if api_key:
        st.session_state.api_key = api_key
        st.rerun()

# 첫 인사 메시지
if not st.session_state.messages:
    greeting = """IT 헬프데스크입니다. 💻

    지원 가능:
    🌐 네트워크 (와이파이, VPN)
    🔐 계정/로그인
    📧 이메일/업무시스템
    🖨️ 하드웨어
    💿 소프트웨어
    🛡️ 보안

    문제를 설명해주세요."""

    st.session_state.messages.append({"role": "assistant", "content": greeting})

# 채팅 부분
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("문제 설명"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = get_response(
            prompt, st.session_state.messages[:-1], st.session_state.api_key
        )
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
