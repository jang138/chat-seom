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
            return candidates
        else:
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
