from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from app.core.speaker_diarization import process_audio_pipeline
import tempfile
import requests
import os
import base64
from app.core.speaker_diarization import SpeakerDiarizer, download_audio_from_gcs
from datetime import datetime

router = APIRouter(prefix="/diarization", tags=["diarization"])

class DiarizationRequest(BaseModel):
    audio_url: str
    min_speakers: int = 1
    max_speakers: int = 2

class DiarizationResponse(BaseModel):
    success: bool
    speaker_count: int
    total_duration: float
    segments: list
    timestamp: str
    processing_method: str

@router.get("/test")
async def test():
    return {"message": "화자분리 기능 준비 중"}

@router.post("/analyze", response_model=DiarizationResponse)
async def analyze_speakers(request: DiarizationRequest):
    """
    GCS URL에서 화자분리만 수행
    
    Args:
        request: 오디오 URL 및 설정
        
    Returns:
        화자분리 결과
    """
    try:
        # 1. GCS에서 오디오 다운로드
        print(f"🎵 GCS에서 오디오 다운로드: {request.audio_url}")
        response = requests.get(request.audio_url, timeout=300)
        response.raise_for_status()
        
        # 2. 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        try:
            # 3. Hugging Face 토큰
            token = os.getenv("HF_TOKEN", "your_token_here")
            
            # 4. 화자분리만 실행
            results = process_audio_pipeline(
                audio_path,
                token=token,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers
            )
            
            # 5. 응답 형식 맞추기
            if results.get("success", False):
                return DiarizationResponse(
                    success=True,
                    speaker_count=results.get("speaker_count", 0),
                    total_duration=results.get("total_duration", 0.0),
                    segments=results.get("segments", []),
                    timestamp=results.get("timestamp", ""),
                    processing_method=results.get("processing_method", "unknown")
                )
            else:
                raise HTTPException(status_code=500, detail=results.get("error", "화자분리 실패"))
            
        finally:
            # 6. 임시 파일 삭제
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"화자분리 처리 중 오류 발생: {str(e)}")


@router.post("/get-segments-from-gcs", response_model=DiarizationResponse)
async def get_segments_from_gcs(
    gcs_url: str = Form(...),
    token: str = Form(...),
    canvas_id: str = Form(...),
    mentor_idx: int = Form(...),
    mentee_idx: int = Form(...),
    session_start_offset: float = Form(default=0.0)
):
    """
    GCS URL → 화자분리 → 세그먼트 분리 → 버퍼 반환
    """
    try:
        # 1. GCS에서 오디오 다운로드
        audio_data = download_audio_from_gcs(gcs_url)
        
        # 2. 화자분리 수행
        diarizer = SpeakerDiarizer(token=token)
        segments = diarizer.diarize_audio(audio_data)  # ✅ 이제 메서드가 있음
        
        # 3. 세그먼트별로 오디오 분리하여 버퍼 반환
        segment_buffers = []
        for segment in segments:
            audio_buffer = diarizer.extract_audio_segment(
                audio_data, 
                segment['start_time'], 
                segment['end_time']
            )
            segment_buffers.append({
                'audioBuffer': base64.b64encode(audio_buffer).decode('utf-8'),
                'startTime': segment['start_time'],
                'endTime': segment['end_time'],
                'speakerTag': segment['speaker_tag']
            })
        
        # ✅ 모든 필수 필드 포함하여 반환
        return {
            "success": True,
            "speaker_count": len(set(seg['speakerTag'] for seg in segment_buffers)),
            "total_duration": max(seg['endTime'] for seg in segment_buffers) if segment_buffers else 0.0,
            "segments": segment_buffers,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "pynote_diarization"
        }
        
    except Exception as e:
        print(f"Error in get_segments_from_gcs: {e}")
        # ✅ 실패 시에도 모든 필수 필드 포함
        return {
            "success": False,
            "speaker_count": 0,
            "total_duration": 0.0,
            "segments": [],
            "timestamp": datetime.now().isoformat(),
            "processing_method": "pynote_diarization"
        }