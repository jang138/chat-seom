import json
import os
from datetime import datetime

FAQ_CANDIDATES_FILE = "faq_candidates.json"


def save_faq_candidates(candidates):
    """FAQ 후보를 JSON 파일로 저장"""
    try:
        with open(FAQ_CANDIDATES_FILE, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        print(f"FAQ 후보 {len(candidates)}개 저장 완료: {FAQ_CANDIDATES_FILE}")
        return True
    except Exception as e:
        print(f"FAQ 후보 저장 실패: {e}")
        return False


def load_faq_candidates():
    """JSON 파일에서 FAQ 후보 로드"""
    try:
        if os.path.exists(FAQ_CANDIDATES_FILE):
            with open(FAQ_CANDIDATES_FILE, "r", encoding="utf-8") as f:
                candidates = json.load(f)
            print(f"FAQ 후보 {len(candidates)}개 로드 완료")
            return candidates
        else:
            print("FAQ 후보 파일이 없습니다. 빈 목록으로 시작합니다.")
            return []
    except Exception as e:
        print(f"FAQ 후보 로드 실패: {e}")
        return []


def add_faq_candidate(question, answer, status="pending_review"):
    """새로운 FAQ 후보 생성"""
    candidate = {
        "question": question,
        "generated_answer": answer,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
    }
    return candidate


def get_faq_candidates_count():
    """저장된 FAQ 후보 개수 반환"""
    candidates = load_faq_candidates()
    return len(candidates)


def get_pending_candidates():
    """검토 대기 중인 FAQ 후보만 반환"""
    candidates = load_faq_candidates()
    return [c for c in candidates if c.get("status") == "pending_review"]


def update_candidate_status(index, new_status):
    """FAQ 후보 상태 업데이트"""
    try:
        candidates = load_faq_candidates()
        if 0 <= index < len(candidates):
            candidates[index]["status"] = new_status
            candidates[index]["updated_at"] = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            return save_faq_candidates(candidates)
        else:
            print(f"잘못된 인덱스: {index}")
            return False
    except Exception as e:
        print(f"상태 업데이트 실패: {e}")
        return False


def clear_all_candidates():
    """모든 FAQ 후보 삭제 (테스트용)"""
    try:
        if os.path.exists(FAQ_CANDIDATES_FILE):
            os.remove(FAQ_CANDIDATES_FILE)
        print("모든 FAQ 후보 삭제 완료")
        return True
    except Exception as e:
        print(f"FAQ 후보 삭제 실패: {e}")
        return False
