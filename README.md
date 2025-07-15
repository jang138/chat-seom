# IT 헬프데스크 챗봇, CHAT SEOM

회사 IT 문제를 해결해주는 AI 기반 헬프데스크 시스템입니다.

## 📋 주요 기능

- **문제 자동 분류**: solar-mini 모델로 IT 문의를 카테고리별 분류
- **지능형 답변**: solar-pro 모델과 RAG 시스템으로 정확한 해결책 제공
- **멀티턴 대화**: 이전 대화 맥락을 고려한 자연스러운 상담
- **실시간 검색**: ChromaDB 벡터 데이터베이스 기반 매뉴얼 검색

## 🛠️ 지원 분야

- 🌐 **네트워크**: 와이파이, VPN 연결 문제
- 🔐 **계정/인증**: 비밀번호 재설정, 로그인 문제
- 📧 **이메일/업무시스템**: 아웃룩 설정, 메일 접속 불가
- 🖨️ **하드웨어**: 프린터, 주변기기 연결 문제
- 💿 **소프트웨어**: 설치 오류, 프로그램 문제
- 🛡️ **보안**: 바이러스, 악성코드 대응

## 🚀 실행 방법

### 1. 환경 설정

```bash
# 패키지 설치
pip install -r "requirements.txt"

# 환경 변수 설정 (.env 파일 생성)
UPSTAGE_API_KEY=your_api_key_here
```

### 2. 실행

```bash
streamlit run it_helpdesk.py
```

## 📊 기술 스택

- **LLM**: Upstage Solar API (mini + pro)
- **벡터 DB**: ChromaDB
- **임베딩**: solar-embedding-1-large
- **프레임워크**: LangChain
- **UI**: Streamlit

## 💻 사용 예시

```
사용자: "와이파이가 연결이 안 됩니다"
봇: 네트워크 연결 문제를 해결해드리겠습니다.
    1) 와이파이를 껐다 켜보세요
    2) 네트워크 설정에서 해당 네트워크를 삭제 후 재연결해보세요
    3) 공유기를 재시작해보세요
```

## 🔧 주요 구성 요소

### 1. 문제 분류 (`classify()`)

- solar-mini 모델 사용
- 사용자 문의를 7개 카테고리로 분류

### 2. 답변 생성 (`answer()`)

- solar-pro 모델 사용
- RAG 기반 매뉴얼 검색 및 답변 생성

### 3. 벡터 검색 (`load_manual()`)

- IT 매뉴얼을 ChromaDB에 저장
- 유사도 기반 관련 정보 검색

## 📄 매뉴얼 형식

```json
{
  "id": "wifi_001",
  "text_content": "와이파이 연결 문제 해결 방법...",
  "metadata": {
    "category": "네트워크",
    "scenario": "와이파이 연결 불가",
    "keywords": ["와이파이", "wifi", "연결"],
    "priority": "high"
  }
}
```
