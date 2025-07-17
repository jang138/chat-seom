import streamlit as st
import json
import os
from datetime import datetime
from faq_manager import load_faq_candidates, save_faq_candidates, clear_all_candidates
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

APPROVED_FAQS_FILE = "approved_faqs.json"


def load_approved_faqs():
    """승인된 FAQ 로드"""
    try:
        if os.path.exists(APPROVED_FAQS_FILE):
            with open(APPROVED_FAQS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        st.error(f"승인된 FAQ 로드 실패: {e}")
        return []


def save_approved_faqs(faqs):
    """승인된 FAQ 저장"""
    try:
        with open(APPROVED_FAQS_FILE, "w", encoding="utf-8") as f:
            json.dump(faqs, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"승인된 FAQ 저장 실패: {e}")
        return False


def approve_faq(candidate_index, edited_question, edited_answer):
    """FAQ 승인 처리"""
    try:
        # 1. FAQ 후보에서 승인 상태로 변경
        candidates = load_faq_candidates()
        if 0 <= candidate_index < len(candidates):
            candidates[candidate_index]["status"] = "approved"
            candidates[candidate_index]["edited_question"] = edited_question
            candidates[candidate_index]["edited_answer"] = edited_answer
            candidates[candidate_index]["approved_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # 2. 승인된 FAQ 목록에 추가
            approved_faqs = load_approved_faqs()
            new_faq = {
                "id": f"approved_{len(approved_faqs) + 1:03d}",
                "question": edited_question,
                "answer": edited_answer,
                "original_question": candidates[candidate_index]["question"],
                "approved_at": candidates[candidate_index]["approved_at"],
                "category": "user_generated",
            }
            approved_faqs.append(new_faq)

            # 3. 저장
            if save_faq_candidates(candidates) and save_approved_faqs(approved_faqs):
                # 4. ChromaDB에 추가
                add_to_chromadb(new_faq)
                return True
        return False
    except Exception as e:
        st.error(f"FAQ 승인 실패: {e}")
        return False


def reject_faq(candidate_index, reason=""):
    """FAQ 거절 처리"""
    try:
        candidates = load_faq_candidates()
        if 0 <= candidate_index < len(candidates):
            candidates[candidate_index]["status"] = "rejected"
            candidates[candidate_index]["rejection_reason"] = reason
            candidates[candidate_index]["rejected_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return save_faq_candidates(candidates)
        return False
    except Exception as e:
        st.error(f"FAQ 거절 실패: {e}")
        return False


def add_to_chromadb(faq):
    """승인된 FAQ를 ChromaDB에 추가"""
    try:
        api_key = os.getenv("UPSTAGE_API_KEY")
        embeddings = UpstageEmbeddings(api_key=api_key, model="solar-embedding-1-large")
        db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)

        # 새 FAQ를 Document로 변환
        doc = Document(
            page_content=f"질문: {faq['question']}\n답변: {faq['answer']}",
            metadata={
                "id": faq["id"],
                "category": faq["category"],
                "type": "user_generated_faq",
                "approved_at": faq["approved_at"],
            },
        )

        db.add_documents([doc])
        st.success("✅ ChromaDB에 FAQ 추가 완료!")
        return True
    except Exception as e:
        st.error(f"ChromaDB 추가 실패: {e}")
        return False


def main():
    st.set_page_config(page_title="FAQ 관리자", page_icon="👨‍💼", layout="wide")

    st.title("👨‍💼 FAQ 관리자 페이지")

    # 데이터를 한 번만 로드하고 session_state에 캐시
    if "admin_candidates" not in st.session_state:
        st.session_state.admin_candidates = load_faq_candidates()

    if "admin_approved_faqs" not in st.session_state:
        st.session_state.admin_approved_faqs = load_approved_faqs()

    # 새로고침 버튼
    if st.button("🔄 데이터 새로고침"):
        st.session_state.admin_candidates = load_faq_candidates()
        st.session_state.admin_approved_faqs = load_approved_faqs()
        st.success("데이터를 새로고침했습니다!")

    # 탭 생성
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📋 대기 중인 FAQ", "✅ 승인된 FAQ", "📊 통계", "🛠️ 관리"]
    )

    with tab1:
        st.header("📋 검토 대기 중인 FAQ 후보")

        # 캐시된 데이터 사용
        candidates = st.session_state.admin_candidates
        pending = [c for c in candidates if c.get("status") == "pending_review"]

        if not pending:
            st.info("현재 검토 대기 중인 FAQ가 없습니다.")
        else:
            st.write(f"총 {len(pending)}개의 FAQ가 검토를 기다리고 있습니다.")

            for i, candidate in enumerate(pending):
                with st.expander(f"FAQ 후보 #{i+1}: {candidate['question'][:50]}..."):
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.write("**원본 질문:**")
                        st.write(candidate["question"])

                        st.write("**AI 생성 답변:**")
                        st.write(candidate["generated_answer"])

                        st.write("**등록 시간:**", candidate["timestamp"])

                    with col2:
                        st.write("**편집**")

                        # 원본 찾기 (전체 candidates에서의 인덱스)
                        original_index = candidates.index(candidate)

                        edited_question = st.text_area(
                            "질문 편집:",
                            value=candidate["question"],
                            key=f"q_{original_index}",
                        )

                        edited_answer = st.text_area(
                            "답변 편집:",
                            value=candidate["generated_answer"],
                            height=150,
                            key=f"a_{original_index}",
                        )

                        col_approve, col_reject = st.columns(2)

                        with col_approve:
                            if st.button("✅ 승인", key=f"approve_{original_index}"):
                                if approve_faq(
                                    original_index, edited_question, edited_answer
                                ):
                                    st.success("FAQ 승인 완료!")
                                    # 캐시 업데이트
                                    st.session_state.admin_candidates = (
                                        load_faq_candidates()
                                    )
                                    st.session_state.admin_approved_faqs = (
                                        load_approved_faqs()
                                    )
                                    st.rerun()

                        with col_reject:
                            if st.button("❌ 거절", key=f"reject_{original_index}"):
                                reason = st.text_input(
                                    "거절 사유:", key=f"reason_{original_index}"
                                )
                                if reject_faq(original_index, reason):
                                    st.success("FAQ 거절 완료!")
                                    # 캐시 업데이트
                                    st.session_state.admin_candidates = (
                                        load_faq_candidates()
                                    )
                                    st.rerun()

    with tab2:
        st.header("✅ 승인된 FAQ 목록")

        # 캐시된 데이터 사용
        approved_faqs = st.session_state.admin_approved_faqs

        if not approved_faqs:
            st.info("아직 승인된 FAQ가 없습니다.")
        else:
            st.write(f"총 {len(approved_faqs)}개의 FAQ가 승인되었습니다.")

            for i, faq in enumerate(approved_faqs):
                with st.expander(f"FAQ #{i+1}: {faq['question'][:50]}..."):
                    st.write("**질문:**", faq["question"])
                    st.write("**답변:**", faq["answer"])
                    st.write("**승인 일시:**", faq["approved_at"])
                    st.write("**ID:**", faq["id"])

                    if "original_question" in faq:
                        st.write("**원본 질문:**", faq["original_question"])

    with tab3:
        st.header("📊 FAQ 관리 통계")

        # 캐시된 데이터 사용
        candidates = st.session_state.admin_candidates
        approved_faqs = st.session_state.admin_approved_faqs

        # 통계 계산
        total_candidates = len(candidates)
        pending_count = len(
            [c for c in candidates if c.get("status") == "pending_review"]
        )
        approved_count = len([c for c in candidates if c.get("status") == "approved"])
        rejected_count = len([c for c in candidates if c.get("status") == "rejected"])

        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("총 FAQ 후보", total_candidates)

        with col2:
            st.metric("대기 중", pending_count)

        with col3:
            st.metric("승인됨", approved_count)

        with col4:
            st.metric("거절됨", rejected_count)

        # 최근 활동
        st.subheader("최근 활동")
        recent_candidates = sorted(
            candidates, key=lambda x: x.get("timestamp", ""), reverse=True
        )[:5]

        for candidate in recent_candidates:
            status_emoji = {
                "pending_review": "⏳",
                "approved": "✅",
                "rejected": "❌",
            }.get(candidate.get("status"), "❓")

            st.write(
                f"{status_emoji} {candidate['question'][:60]}... ({candidate['timestamp']})"
            )

    with tab4:
        st.header("🛠️ 시스템 관리")

        st.subheader("위험한 작업")
        st.warning("⚠️ 아래 작업들은 되돌릴 수 없습니다!")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🗑️ 모든 FAQ 후보 삭제", type="secondary"):
                if st.checkbox("정말 삭제하시겠습니까?"):
                    if clear_all_candidates():
                        st.success("모든 FAQ 후보가 삭제되었습니다.")
                        # 캐시 초기화
                        st.session_state.admin_candidates = []
                        st.rerun()

        with col2:
            if st.button("🔄 ChromaDB 재구성", type="secondary"):
                st.info("이 기능은 아직 구현되지 않았습니다.")

        st.subheader("데이터 내보내기")

        # JSON 다운로드
        candidates = st.session_state.admin_candidates
        approved_faqs = st.session_state.admin_approved_faqs

        if st.button("📥 FAQ 후보 데이터 다운로드"):
            st.download_button(
                label="다운로드",
                data=json.dumps(candidates, ensure_ascii=False, indent=2),
                file_name=f"faq_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        if st.button("📥 승인된 FAQ 데이터 다운로드"):
            st.download_button(
                label="다운로드",
                data=json.dumps(approved_faqs, ensure_ascii=False, indent=2),
                file_name=f"approved_faqs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
