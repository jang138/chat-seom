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

load_dotenv()

def classify(user_input, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-mini")
    
    prompt = f"""다음 IT 문의를 카테고리로 분류하세요.
    옵션: 네트워크, 계정/인증, 이메일/업무시스템, 하드웨어, 소프트웨어, 보안, 기타

    문의: {user_input}
    카테고리:"""
    
    result = chat.invoke([SystemMessage(content="IT 문제 분류 전문가"), HumanMessage(content=prompt)])
    return result.content.strip()

def answer(user_input, chat_history, category, api_key):
    chat = ChatUpstage(api_key=api_key, model="solar-pro")
    embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")
    
    db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    
    if db._collection.count() == 0:
        load_manual(db)
    
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # 질문 재구성
    ctx_prompt = ChatPromptTemplate.from_messages([
        ("system", "이전 대화를 참고해서 질문을 재구성하세요."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    hist_retriever = create_history_aware_retriever(chat, retriever, ctx_prompt)
    
    # 답변 생성
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", f"""IT 헬프데스크입니다. 분류: {category}

        규칙:
        1. 매뉴얼 정보 활용
        2. 모르면 IT 관리자 연결
        3. 단계별 설명

        지원 영역: 네트워크, 계정, 이메일, 하드웨어, 소프트웨어, 보안

        참고: {{context}}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(chat, qa_prompt)
    rag_chain = create_retrieval_chain(hist_retriever, qa_chain)
    
    # 히스토리 변환
    history = []
    for msg in chat_history:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            history.append(AIMessage(content=msg["content"]))
    
    result = rag_chain.invoke({'input': user_input, 'chat_history': history})
    return result['answer']

def load_manual(db):
    with open("it_helpdesk_manual.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    with open("it_helpdesk_manual.json", 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for item in data:
        doc = Document(
            page_content=item['text_content'],
            metadata=item['metadata']
        )
        docs.append(doc)
    
    db.add_documents(docs)
    st.success(f"매뉴얼 {len(docs)}개 로딩완료")

def get_response(user_input, chat_history, api_key):
    category = classify(user_input, api_key)
    response = answer(user_input, chat_history, category, api_key)
    return response

# UI 시작 부분
st.set_page_config(page_title="IT 헬프데스크", page_icon="💻")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("UPSTAGE_API_KEY")

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
        response = get_response(prompt, st.session_state.messages[:-1], st.session_state.api_key)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})