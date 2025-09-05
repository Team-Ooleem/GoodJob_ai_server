# audio-analysis/app/routers/audio_analysis.py
from fastapi import APIRouter, UploadFile, File
import tempfile, shutil
from app.core.audio_features import analyze_audio

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        features = analyze_audio(tmp.name)
    return {"ok": True, "features": features}
