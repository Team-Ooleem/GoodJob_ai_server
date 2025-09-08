# audio-analysis/app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routers.audio_analysis import router as audio_router

origins = ["*"]  # 나중에 프론트엔드 서버로 변경해야함

app = FastAPI(title="Audio Analysis Service")
app.include_router(audio_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 허용할 origin
    allow_credentials=True,
    allow_methods=["*"],  # 모든 메서드 허용 (GET, POST 등)
    allow_headers=["*"],  # 모든 헤더 허용
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
