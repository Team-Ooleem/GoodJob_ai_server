from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from app.core.speaker_diarization import process_audio_pipeline, SpeakerDiarizer, download_audio_from_gcs
import tempfile
import requests
import os
import base64
from datetime import datetime

router = APIRouter(prefix="/diarization", tags=["diarization"])

# 🚀 전역 diarizer 인스턴스 캐싱 (메모리 효율성)
_diarizer_instance = None

def get_diarizer(token: str = None) -> SpeakerDiarizer:
    """diarizer 인스턴스 재사용 (메모리 최적화)"""
    global _diarizer_instance
    
    if _diarizer_instance is None:
        token = token or os.getenv("HF_TOKEN", "")  # 🚀 간단하게
        _diarizer_instance = SpeakerDiarizer(token=token)
        print("🚀 새 diarizer 인스턴스 생성")
    else:
        print("♻️ 기존 diarizer 인스턴스 재사용")
    
    return _diarizer_instance

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
    """GCS URL에서 화자분리만 수행"""
    try:
        print(f"🎵 화자분리 분석 시작: {request.audio_url}")
        
        # 1. GCS에서 오디오 다운로드 (타임아웃 증가)
        response = requests.get(request.audio_url, timeout=600)  # 10분
        response.raise_for_status()
        
        # 2. 임시 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(response.content)
            tmp.flush()
            audio_path = tmp.name
        
        try:
            # 3. 환경변수에서 토큰 가져오기
            token = os.getenv("HF_TOKEN", "")
            if not token:
                print("⚠️ HF_TOKEN이 설정되지 않았습니다. fallback 모드로 실행됩니다.")
            
            # 4. 화자분리 실행
            results = process_audio_pipeline(
                audio_path,
                token=token,
                min_speakers=request.min_speakers,
                max_speakers=request.max_speakers
            )
            
            # 5. 응답 반환
            if results.get("success", False):
                return DiarizationResponse(**results)
            else:
                raise HTTPException(status_code=500, detail=results.get("error", "화자분리 실패"))
            
        finally:
            # 6. 임시 파일 정리
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        print(f"❌ 화자분리 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"화자분리 처리 중 오류 발생: {str(e)}")

@router.post("/get-segments-from-gcs", response_model=DiarizationResponse)
async def get_segments_from_gcs(
    gcs_url: str = Form(...),
    token: str = Form(...),
    canvas_id: str = Form(...),
    mentor_idx: int = Form(...),  # 101 (김코치)
    mentee_idx: int = Form(...),  # 202 (이멘티)
    session_start_offset: float = Form(default=0.0)
):
    try:
        print(f"🎯 pyannote GCS 세그먼트 분리 시작")
        print(f"📋 멘토 ID: {mentor_idx}, 멘티 ID: {mentee_idx}")
        
        # 1. GCS에서 오디오 다운로드
        audio_data = download_audio_from_gcs(gcs_url)
        print(f"✅ GCS 다운로드 완료: {len(audio_data) / (1024*1024):.1f}MB")
        
        # 2. 재사용 가능한 diarizer 인스턴스 사용
        diarizer = get_diarizer(token)
        segments = diarizer.diarize_audio(audio_data)
        print(f"🎤 화자분리 완료: {len(segments)}개 세그먼트")
        
        # 3. 멘토/멘티 매핑
        mapped_segments = diarizer.map_speakers_to_mentor_mentee(
            segments, mentor_idx, mentee_idx
        )
        print(f"🔄 스피커 매핑 완료: {len(mapped_segments)}개 세그먼트")
        
        # 4. 세그먼트별 오디오 추출
        segment_buffers = []
        total_segments = len(mapped_segments)
        
        for i, segment in enumerate(mapped_segments):
            try:
                audio_buffer = diarizer.extract_audio_segment(
                    audio_data, 
                    segment['start_time'], 
                    segment['end_time']
                )
                
                segment_buffers.append({
                    'audioBuffer': base64.b64encode(audio_buffer).decode('utf-8'),
                    'startTime': round(segment['start_time'], 2),  # 🚀 precision 최적화
                    'endTime': round(segment['end_time'], 2),
                    'speakerTag': segment['speaker_tag']  # 0=멘토(김코치), 1=멘티(이멘티)
                })
                
                # 🚀 진행률 표시 개선 (10% 단위)
                progress_step = max(1, total_segments // 10)
                if (i + 1) % progress_step == 0 or i == total_segments - 1:
                    print(f"📊 진행률: {i+1}/{total_segments} ({((i+1)/total_segments*100):.1f}%)")
                
            except Exception as segment_error:
                print(f"❌ 세그먼트 {i+1} 처리 실패: {segment_error}")
                continue
        
        # 🚀 메모리 정리
        del audio_data
        
        print(f"✅ 전체 처리 완료: {len(segment_buffers)}개 세그먼트")
        
        return {
            "success": True,
            "speaker_count": len(set(seg['speakerTag'] for seg in segment_buffers)),
            "total_duration": max(seg['endTime'] for seg in segment_buffers) if segment_buffers else 0.0,
            "segments": segment_buffers,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "optimized_mentor_mentee_mapping"
        }
        
    except Exception as e:
        print(f"❌ pyannote 처리 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "speaker_count": 0,
            "total_duration": 0.0,
            "segments": [],
            "timestamp": datetime.now().isoformat(),
            "processing_method": "mapping_failed"
        }