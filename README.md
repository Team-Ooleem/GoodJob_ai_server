## 실행 방법

따로 가상환경 설정할 필요 없이 아래 절차만 진행하시면 됩니다. (도커 이미지 안에서 requirements.txt 실행)

터미널에서 다음 명령어를 입력하여 Docker 이미지를 빌드합니다.

docker build -t audio-analysis:latest .

이미지 빌드 후 다음 명령어를 입력하여 이미지를 실행시킵니다.

docker run -p 8081:8081 audio-analysis:latest

만약 코드가 수정되었을 경우 위 빌드 명령어로 이미지를 다시 빌드한 후 실행시키면 됩니다.

(재빌드의 경우 시간이 오래 걸리지 않습니다)
