import numpy as np
import librosa
import torch
from typing import List, Dict, Tuple, Optional
import tempfile
import os
from pathlib import Path
import json
from datetime import datetime
import soundfile as sf
import io     
import base64  
import requests  # �� 추가
from fastapi import HTTPException  # �� 추가

try:
    from pyannote.audio import Pipeline
    from pyannote.core import Segment
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    print("Warning: pyannote-audio not available. Speaker diarization will use fallback method.")

from pydub import AudioSegment


class SpeakerDiarizer:
    """화자분리 전용 클래스"""
    
    def __init__(self, 
                 token: str = "",  # 🆕 기본값 추가
                 min_speakers: int = 1, 
                 max_speakers: int = 10,
                 use_pyannote: bool = True):
        self.token = token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_pyannote = use_pyannote and PYANNOTE_AVAILABLE
        
        # pyannote 파이프라인 초기화
        if self.use_pyannote:
            try:
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=self.token
                )
                print("✅ pyannote 파이프라인 로드 성공!")
            except Exception as e:
                print(f"❌ pyannote 파이프라인 로드 실패: {e}")
                self.use_pyannote = False
    
    def preprocess_audio(self, audio_path: str, target_sr: int = 16000) -> str:
        """오디오 전처리 (MP4, MP3, WAV 등 모든 형식 지원)"""
        try:
            # pydub로 먼저 로드 (MP4, MP3 등 지원)
            audio = AudioSegment.from_file(audio_path)
            
            # librosa가 처리할 수 있는 형식으로 변환
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                # WAV 형식으로 내보내기 (16kHz, 모노)
                audio.export(tmp_file.name, format="wav", parameters=["-ar", "16000", "-ac", "1"])
                
                # librosa로 다시 로드하여 정확한 처리를 위해
                y, sr = librosa.load(tmp_file.name, sr=target_sr, mono=True)
                
                # 최종 WAV 파일로 저장
                final_file = tmp_file.name.replace('.wav', '_final.wav')
                sf.write(final_file, y, target_sr)
                
                # 임시 파일 정리
                os.unlink(tmp_file.name)
                
                return final_file
                
        except Exception as e:
            print(f"❌ 오디오 전처리 실패: {e}")
            return audio_path
    
    def diarize_speakers(self, audio_path: str) -> List[Dict]:
        """화자분리 수행"""
        print(f"🎯 화자분리 시작: {audio_path}")
        
        processed_audio = self.preprocess_audio(audio_path)
        
        try:
            if self.use_pyannote:
                results = self.diarize_with_pyannote(processed_audio)
            else:
                results = self._fallback_diarization(processed_audio)
            
            print(f"✅ 화자분리 완료: {len(results)}개 구간 발견")
            return results
            
        finally:
            if os.path.exists(processed_audio) and processed_audio != audio_path:
                os.unlink(processed_audio)
    
    def diarize_with_pyannote(self, audio_path: str) -> List[Dict]:
        """pyannote를 사용한 화자분리"""
        if not self.use_pyannote:
            raise RuntimeError("pyannote not available")
        
        try:
            print("🔍 pyannote로 화자분리 수행 중...")
            
            diarization = self.pipeline(audio_path, 
                                      min_speakers=self.min_speakers,
                                      max_speakers=self.max_speakers)
            
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # 화자 ID를 숫자로 변환 (SPEAKER_00 -> 0, SPEAKER_01 -> 1)
                speaker_id_number = self._extract_speaker_number(speaker)
                
                results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                    "speaker": speaker,
                    "speaker_id": speaker_id_number  # 숫자로 변환된 화자 ID
                })
            
            return results
            
        except Exception as e:
            print(f"❌ pyannote 화자분리 실패: {e}")
            return self._fallback_diarization(audio_path)
    
    def _fallback_diarization(self, audio_path: str) -> List[Dict]:
        """fallback 화자분리"""
        print("�� fallback 화자분리 수행 중...")
        
        try:
            y, sr = librosa.load(audio_path, sr=16000, mono=True)
            intervals = librosa.effects.split(y, top_db=30)
            
            min_duration = 0.5
            filtered_intervals = []
            for start, end in intervals:
                duration = (end - start) / sr
                if duration >= min_duration:
                    filtered_intervals.append({
                        "start": start / sr,
                        "end": end / sr,
                        "duration": duration,
                        "speaker": "SPEAKER_00",
                        "speaker_id": 0  # fallback도 숫자로 변환
                    })
            
            return filtered_intervals
            
        except Exception as e:
            print(f"❌ fallback 화자분리 실패: {e}")
            return []
    
    def _extract_speaker_number(self, speaker_id: str) -> int:
        """화자 ID에서 숫자 추출"""
        try:
            if speaker_id.startswith("SPEAKER_"):
                return int(speaker_id.split("_")[1])
            return 0
        except:
            return 0

    def diarize_audio(self, audio_data: bytes) -> List[Dict]:
        """
        바이트 데이터로부터 화자분리 수행
        """
        try:
            # 1. 바이트 데이터를 임시 파일로 저장
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_data)
                tmp.flush()
                audio_path = tmp.name
            
            try:
                # 2. 기존 diarize_speakers 메서드 사용
                results = self.diarize_speakers(audio_path)
                
                # 3. 키 이름을 올바르게 변환
                converted_results = []
                for result in results:
                    converted_results.append({
                        'start_time': result['start'],
                        'end_time': result['end'],
                        'speaker_tag': result['speaker_id']
                    })
                
                return converted_results
                
            finally:
                # 4. 임시 파일 삭제
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            print(f"❌ diarize_audio 실패: {e}")
            return []

    def extract_audio_segment(self, audio_data: bytes, start_time: float, end_time: float) -> bytes:
        """
        오디오 데이터에서 특정 시간 구간을 추출하여 버퍼로 반환
        """
        try:
            # BytesIO로 오디오 데이터 로드
            audio_io = io.BytesIO(audio_data)
            
            # pydub으로 오디오 로드
            audio = AudioSegment.from_file(audio_io)
            
            # 시간을 밀리초로 변환
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # 세그먼트 추출
            segment = audio[start_ms:end_ms]
            
            # WAV 형식으로 변환하여 버퍼 반환
            output_io = io.BytesIO()
            segment.export(output_io, format="wav")
            
            return output_io.getvalue()
            
        except Exception as e:
            print(f"오디오 세그먼트 추출 실패: {e}")
            return b""


def process_audio_pipeline(audio_path: str, 
                          token: str,
                          min_speakers: int = 1, 
                          max_speakers: int = 5) -> Dict:
    """
    화자분리 전용 파이프라인
    
    Args:
        audio_path: 오디오 파일 경로
        token: Hugging Face 토큰
        min_speakers: 최소 화자 수
        max_speakers: 최대 화자 수
        
    Returns:
        화자분리 결과만
    """
    print("�� 화자분리 파이프라인 시작")
    
    try:
        # 화자분리기 초기화
        diarizer = SpeakerDiarizer(
            token=token,
            min_speakers=min_speakers, 
            max_speakers=max_speakers
        )
        
        # 화자분리만 수행
        diarization_results = diarizer.diarize_speakers(audio_path)
        
        # 결과 정리
        speaker_count = len(set(seg["speaker"] for seg in diarization_results)) if diarization_results else 0
        total_duration = sum(seg["duration"] for seg in diarization_results) if diarization_results else 0
        
        result = {
            "success": True,
            "speaker_count": speaker_count,
            "total_duration": total_duration,
            "segments": diarization_results,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "pyannote" if diarizer.use_pyannote else "fallback"
        }
        
        print(f"✅ 화자분리 완료: {speaker_count}명 화자, {total_duration:.1f}초")
        return result
        
    except Exception as e:
        print(f"❌ 화자분리 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def download_audio_from_gcs(gcs_url: str) -> bytes:
    """
    GCS URL에서 오디오 파일을 다운로드하여 바이트 데이터로 반환
    """
    try:
        print(f" GCS에서 오디오 다운로드 시작: {gcs_url}")
        
        # 타임아웃을 10분으로 증가
        response = requests.get(gcs_url, timeout=600)  # 10분 (600초)
        response.raise_for_status()
        
        # 파일 크기 체크 (500MB 제한)
        content_length = len(response.content)
        max_size = 500 * 1024 * 1024  # 500MB
        
        if content_length > max_size:
            raise HTTPException(
                status_code=400, 
                detail=f"오디오 파일이 너무 큽니다: {content_length / (1024*1024):.1f}MB (최대 500MB)"
            )
        
        print(f"✅ GCS 다운로드 완료: {content_length / (1024*1024):.1f}MB")
        return response.content
        
    except requests.exceptions.Timeout:
        print(f"❌ GCS 다운로드 타임아웃: {gcs_url}")
        raise HTTPException(status_code=408, detail="GCS 오디오 다운로드 타임아웃 (10분 초과)")
    except Exception as e:
        print(f"❌ GCS 오디오 다운로드 실패: {e}")
        raise HTTPException(status_code=400, detail=f"GCS 오디오 다운로드 실패: {e}")