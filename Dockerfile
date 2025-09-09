# audio-analysis/Dockerfile
FROM python:3.11-slim

# 시스템 라이브러리 (librosa/soundfile에 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    libsm6 \
    libxext6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스 & 정적 파일 포함
COPY app ./app
# COPY static ./static

# 환경 변수 (선택)
ENV PORT=8081 \
    BACKEND=mock \
    MEDIA_ROOT=/data

# 데이터 디렉터리
RUN mkdir -p /data
VOLUME ["/data"]

# 네트워크
EXPOSE 8081

# # 헬스체크 (선택: 필요한 경우 라우트 추가하거나 /static으로 체크)
# HEALTHCHECK --interval=300s --timeout=3s --start-period=10s --retries=3 \
#   CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8081/static/index.html').read()" || exit 1

# 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8081"]
