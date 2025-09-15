# audio-analysis/app/main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.routers.speaker_diarization import router as diarization_router

from app.routers.audio_analysis import router as audio_router
from app.routers.avatar import router as avatar_router

origins = [
    "https://good-job.shop",
    "https://ai.good-job.shop", 
    "https://api.good-job.shop",
    "https://localhost:3443",  # 개발 환경용
    "http://localhost:3000",   # 개발 환경용
    "http://localhost:8080",   # 개발 환경용
]

app = FastAPI(title="Audio Analysis Service")
app.include_router(audio_router)
app.include_router(avatar_router)
app.include_router(diarization_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
