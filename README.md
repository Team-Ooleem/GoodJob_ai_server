## 실행 방법 (Docker)

본 서비스는 두 가지 기능을 제공합니다.

- 오디오 분석(librosa): `/audio/analyze`
- 아바타(모크) 렌더링: `/avatar/register-image`, `/avatar/render-sync`, `/avatar/idle/`

모크 렌더링은 ffmpeg로 "정지 이미지 + 오디오"를 합성하여 빠르게 MP4를 만듭니다.
실제 SadTalker 연동 전까지 프론트/백엔드 통합 검증용으로 사용하세요.

1. 이미지 빌드

```
docker build -t ai-server:latest .
```

2. 컨테이너 실행

```
docker run --rm -p 8081:8081 \
  -e BACKEND=mock \
  -e MEDIA_ROOT=/data \
  ai-server:latest
```

3. 헬스체크

```
curl http://localhost:8081/avatar/healthz
```

4. 아바타 이미지 등록

```
cd ./path
curl -F "file=@face.png" http://localhost:8081/avatar/register-image
# => {"avatar_id":"<UUID>", "path":"/data/avatar/<UUID>/image.png"}
생성된 UUID를 프론트엔드 리포지토리의 .env의 NEXT_PUBLIC_DEFAULT_AVATAR_ID로 넣어주세요.
```

백엔드는 위 `render-sync` 응답 바이트를 수신한 뒤 S3로 업로드하여 퍼블릭 URL을 프론트에 반환합니다.

참고

- ffmpeg는 Dockerfile에서 설치됩니다. 로컬 환경에서 직접 실행 시 ffmpeg가 PATH에 있어야 합니다.
- Docker 이미지 내 환경 변수
- `PORT`: 기본 8081
- `BACKEND`: `mock|cpu|gpu` 중 선택(현재는 mock만 구현)
- `MEDIA_ROOT`: 아바타 이미지 저장 경로(기본 `/data`)
