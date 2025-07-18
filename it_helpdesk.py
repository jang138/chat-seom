__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import streamlit as st
import json
import os
from dotenv import load_dotenv
from datetime import datetime
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from faq_manager import save_faq_candidates, load_faq_candidates, add_faq_candidate
from keywords import (
    GREETING_KEYWORDS,
    CASUAL_KEYWORDS,
    OFFTOPIC_KEYWORDS,
    CASUAL_CHAT_KEYWORDS,
)

load_dotenv()

# streamlit secrets 사용 시 활성화
# os.environ["UPSTAGE_API_KEY"] = st.secrets["UPSTAGE_API_KEY"]


def classify(user_input, api_key):

    # # 1. 먼저 ChromaDB에서 유사한 FAQ 검색
    # embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")
    # db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

    # results = db.similarity_search(user_input, k=1)

    # # 2. 유사도가 높으면 existing으로 분류
    # if results and results[0].metadata.get("type") in [
    #     "approved_faq",
    #     "user_generated_faq",
    # ]:
    #     return "existing"

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

    # 2. 승인된 FAQ도 로드
    if os.path.exists("approved_faqs.json"):
        try:
            with open("approved_faqs.json", "r", encoding="utf-8") as f:
                approved_faqs = json.load(f)

            for faq in approved_faqs:
                doc = Document(...)
                docs.append(doc)

            print(f"승인된 FAQ {len(approved_faqs)}개 추가")

        except Exception as e:
            print(f"승인된 FAQ 로드 실패: {e}")
    else:
        print("💡 승인된 FAQ 파일 없음")

    db.add_documents(docs)
    print(f"{len(docs)}개 문서 벡터 DB에 추가 완료")


def get_response(user_input, chat_history, api_key):
    classification = classify(user_input, api_key)

    if classification == "existing":
        return handle_existing(user_input, chat_history, api_key)
    elif classification == "new":
        return handle_new(user_input, chat_history, api_key)
    else:  # skip
        return handle_skip(user_input, api_key)


def handle_existing(user_input, chat_history, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-pro")
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")

    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

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
                """IT 헬프데스크입니다. 

                규칙:
                1. 매뉴얼 정보를 활용해서 단계별로 설명
                2. 해결 방법을 구체적으로 제시
                3. 모르면 IT 관리자 연결 안내 (내선 1004)
                4. 친절하고 도움이 되는 어조 유지

                지원 영역: 네트워크, 계정, 이메일, 하드웨어, 소프트웨어, 보안

                참고 매뉴얼: {context}""",
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


def handle_new(user_input, chat_history, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-pro")

    # 히스토리 변환, 4개까지만
    history = []
    for msg in chat_history[-4:]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))

    # 답변 생성 프롬프트
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """당신은 IT 헬프데스크의 베테랑 상담사입니다.

                이 질문은 새로운 유형의 IT 문의로, 기존 매뉴얼에 없는 내용입니다.

                답변 방식:
                1. 가능한 범위에서 도움이 될 만한 정보 제공
                2. 새로운 기술/정책이라 정보가 제한적일 수 있음을 안내
                3. 더 정확한 답변을 위해 IT 관리자 연결 필요함을 명시
                4. 친절하고 전문적인 어조 유지

                답변 마지막에 항상 다음 문구 포함:
                "더 정확한 정보는 IT 관리자(내선 1004)에게 문의해주세요."
                """,
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # 답변 생성
    chain = prompt | chat
    result = chain.invoke({"input": user_input, "chat_history": history})

    # FAQ 후보 생성
    faq_candidate = add_faq_candidate(user_input, result.content)

    # 세션 상태에 추가
    st.session_state.faq_candidates.append(faq_candidate)

    # 파일에 저장
    save_faq_candidates(st.session_state.faq_candidates)

    # # FAQ 후보 등록
    # faq_candidate = {
    #     "question": user_input,
    #     "generated_answer": result.content,
    #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    #     "status": "pending_review",
    # }

    # # 세션 상태에 FAQ 후보 저장
    # if "faq_candidates" not in st.session_state:
    #     st.session_state.faq_candidates = []

    # st.session_state.faq_candidates.append(faq_candidate)

    print(
        f"""
FAQ 후보 등록 완료! (총 {len(st.session_state.faq_candidates)}개)
1. 질문: {faq_candidate['question']}
2. 시간: {faq_candidate['timestamp']}
3. 답변: {faq_candidate['generated_answer'][:100]}...
        """
    )

    return result.content


def handle_skip(user_input, api_key):
    user_lower = user_input.lower()

    # 인사
    if any(keyword in user_lower for keyword in GREETING_KEYWORDS):
        return "안녕하세요! 😊 IT 헬프데스크입니다. 어떤 IT 문제로 도움이 필요하신가요?"

    # 일상
    elif any(keyword in user_lower for keyword in CASUAL_KEYWORDS):
        return handle_casual_chat(user_input)

    # 다른 업무
    elif any(keyword in user_lower for keyword in OFFTOPIC_KEYWORDS):
        return """죄송하지만 IT 관련 문의만 도와드릴 수 있어요. 😅
        
                해당 업무는 담당 부서에 문의해주세요!
                IT 관련 문제가 있으시면 언제든 말씀해주세요 💻"""

    # 매칭 실패
    else:
        return """IT 헬프데스크입니다! 😊 
        
                구체적인 IT 문제를 말씀해주시면 더 정확한 도움을 드릴 수 있습니다.
                🌐 네트워크, 🔐 계정, 📧 이메일, 🖨️ 하드웨어, 💿 소프트웨어"""


def handle_casual_chat(user_input):
    """일상 대화에 위트 있게 응답"""
    user_lower = user_input.lower()

    if any(word in user_lower for word in CASUAL_CHAT_KEYWORDS["weather"]):
        return "창밖 확인 못했어요 😅 대신 네트워크 연결 상태는 확인 가능해요!"

    elif any(word in user_lower for word in CASUAL_CHAT_KEYWORDS["food"]):
        return "저는 전기만 먹고 살아요 🔌 식사 드시고 IT 문의 있으시면 언제든지!"

    elif any(word in user_lower for word in CASUAL_CHAT_KEYWORDS["tired"]):
        return "힘드시겠어요! 간단한 IT 업무는 제가 도와드릴게요 💪"

    elif any(word in user_lower for word in CASUAL_CHAT_KEYWORDS["positive"]):
        return "IT 문제 해결도 재밌어요! 😄 무엇을 도와드릴까요?"

    elif any(word in user_lower for word in CASUAL_CHAT_KEYWORDS["thanks"]):
        return "천만에요! 😊 추가 IT 문의 있으시면 언제든지 말씀해주세요!"

    else:
        return "흥미로운 이야기네요! 😊 그런데 IT 관련 문제는 없으신가요?"


def init_faq_system():
    if "faq_candidates" not in st.session_state:
        st.session_state.faq_candidates = load_faq_candidates()
    return True


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
    init_faq_system()
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
