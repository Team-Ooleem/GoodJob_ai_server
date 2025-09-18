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
import requests
from fastapi import HTTPException

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
                 token: str = "",
                 min_speakers: int = 1, 
                 max_speakers: int = 10,
                 use_pyannote: bool = True,
                 # 🔧 조절 파라미터 추가
                 min_duration_on: float = 1.0,      # 최소 발화 시간     
                 min_duration_off: float = 0.5,     # 최소 침묵 시간     
                 merge_threshold: float = 2.0):     # 짧은 세그먼트 병합 임계값

        self.token = token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_pyannote = use_pyannote and PYANNOTE_AVAILABLE
        
        # 🔧 조절 파라미터
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.merge_threshold = merge_threshold
        
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
        """오디오 전처리 (WAV 우선, MP4/MP3 지원)"""
        try:
            print(f" 오디오 전처리 시작: {audio_path}")
            
            # 파일 확장자 확인
            file_ext = os.path.splitext(audio_path)[1].lower()
            
            # WAV 파일인 경우 직접 처리
            if file_ext == '.wav':
                print("✅ WAV 파일 감지, 직접 처리")
                try:
                    # librosa로 직접 로드
                    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
                    
                    # 최종 WAV 파일로 저장
                    with tempfile.NamedTemporaryFile(suffix="_final.wav", delete=False) as tmp_file:
                        sf.write(tmp_file.name, y, target_sr)
                        print(f"✅ WAV 전처리 완료: {tmp_file.name}")
                        return tmp_file.name
                        
                except Exception as e:
                    print(f"⚠️ WAV 직접 처리 실패, pydub으로 재시도: {e}")
            
            # MP4/MP3 파일인 경우 pydub으로 처리
            print(f"🔄 {file_ext.upper()} 파일 처리 중...")
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
                
                print(f"✅ 오디오 전처리 완료: {final_file}")
                return final_file
                
        except Exception as e:
            print(f"❌ 오디오 전처리 실패: {e}")
            return audio_path
    
    def post_process_segments(self, segments: List[Dict]) -> List[Dict]:
        """세그먼트 후처리 - 짧은 세그먼트 병합"""
        if not segments:
            return segments
            
        filtered = []
        
        for segment in segments:
            duration = segment['end'] - segment['start']
            
            # 너무 짧은 세그먼트는 병합
            if duration < self.merge_threshold and filtered:
                # 이전 세그먼트와 같은 화자면 병합
                if filtered[-1]['speaker_id'] == segment['speaker_id']:      
                    filtered[-1]['end'] = segment['end']
                    filtered[-1]['duration'] = filtered[-1]['end'] - filtered[-1]['start']
                    continue
            
            filtered.append(segment)
        
        print(f" 세그먼트 후처리 완료: {len(segments)} → {len(filtered)}개 세그먼트")
        return filtered
    
    def diarize_speakers(self, audio_path: str) -> List[Dict]:
        """화자분리 수행"""
        print(f"🎯 화자분리 시작: {audio_path}")
        
        processed_audio = self.preprocess_audio(audio_path)
        
        try:
            if self.use_pyannote:
                results = self.diarize_with_pyannote(processed_audio)
            else:
                results = self._fallback_diarization(processed_audio)
            
            # 🔧 후처리 적용 - 짧은 세그먼트 병합
            results = self.post_process_segments(results)
            
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
            print(" pyannote로 화자분리 수행 중...")
            
            # ✅ 기본 파라미터만 사용
            diarization = self.pipeline(
                audio_path,
                min_speakers=self.min_speakers,
                max_speakers=self.max_speakers
            )
            
            results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # 화자 ID를 숫자로 변환 (SPEAKER_00 -> 0, SPEAKER_01 -> 1)
                speaker_id_number = self._extract_speaker_number(speaker)
                
                results.append({
                    "start": turn.start,
                    "end": turn.end,
                    "duration": turn.end - turn.start,
                    "speaker": speaker,
                    "speaker_id": speaker_id_number
                })
            
            return results
            
        except Exception as e:
            print(f"❌ pyannote 화자분리 실패: {e}")
            return self._fallback_diarization(audio_path)
    
    def _fallback_diarization(self, audio_path: str) -> List[Dict]:
        """fallback 화자분리"""
        print(" fallback 화자분리 수행 중...")
        
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
                        "speaker_id": 0
                    })
            
            return filtered_intervals
            
        except Exception as e:
            print(f"❌ 화자분리 실패: {e}")
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
        """바이트 데이터로부터 화자분리 수행"""
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
            print(f"❌ 화자분리 실패: {e}")
            return []

    def map_speakers_to_mentor_mentee(self, segments: List[Dict], mentor_idx: int, mentee_idx: int) -> List[Dict]:
        """화자 세그먼트를 멘토/멘티로 매핑"""
        try:
            if not segments:
                return []
            
            # 화자별 총 발화 시간 계산
            speaker_times = {}
            for segment in segments:
                speaker_id = segment.get('speaker_tag', 0)
                duration = segment.get('end_time', 0) - segment.get('start_time', 0)
                speaker_times[speaker_id] = speaker_times.get(speaker_id, 0) + duration
            
            # 더 많이 발화한 화자를 멘토로, 적게 발화한 화자를 멘티로 매핑
            sorted_speakers = sorted(speaker_times.items(), key=lambda x: x[1], reverse=True)
            
            if len(sorted_speakers) >= 2:
                mentor_speaker_id = sorted_speakers[0][0]  # 더 많이 발화
                mentee_speaker_id = sorted_speakers[1][0]  # 적게 발화
            else:
                # 화자가 1명인 경우
                mentor_speaker_id = sorted_speakers[0][0] if sorted_speakers else 0
                mentee_speaker_id = mentor_speaker_id
            
            # 세그먼트 매핑
            mapped_segments = []
            for segment in segments:
                speaker_id = segment.get('speaker_tag', 0)
                
                if speaker_id == mentor_speaker_id:
                    mapped_tag = 0  # 멘토
                elif speaker_id == mentee_speaker_id:
                    mapped_tag = 1  # 멘티
                else:
                    # 예상치 못한 화자 ID인 경우 기본값
                    mapped_tag = 0
                
                mapped_segments.append({
                    'start_time': segment.get('start_time', 0),
                    'end_time': segment.get('end_time', 0),
                    'speaker_tag': mapped_tag
                })
            
            print(f" 화자 매핑 완료: 멘토(화자 {mentor_speaker_id} → 태그 0), 멘티(화자 {mentee_speaker_id} → 태그 1)")
            return mapped_segments
            
        except Exception as e:
            print(f"❌ 화자 매핑 실패: {e}")
            # 실패 시 원본 세그먼트를 그대로 반환 (태그는 0으로)
            return [{
                'start_time': seg.get('start_time', 0),
                'end_time': seg.get('end_time', 0),
                'speaker_tag': 0
            } for seg in segments]

    def extract_audio_segment(self, audio_data: bytes, start_time: float, end_time: float) -> bytes:
        """오디오 데이터에서 특정 시간 구간을 추출하여 버퍼로 반환"""
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
            segment.export(output_io, format="wav", parameters=[
               "-ar", "16000",      # 16kHz 샘플레이트
               "-ac", "1",          # 모노 (1채널)
               "-sample_fmt", "s16" # 16비트 샘플 포맷
               ])
            
            return output_io.getvalue()
            
        except Exception as e:
            print(f"오디오 세그먼트 추출 실패: {e}")
            return b""


def download_audio_from_gcs(gcs_url: str) -> bytes:
    """GCS URL에서 오디오 파일을 다운로드하여 바이트 데이터로 반환"""
    try:
        print(f"📥 GCS에서 오디오 다운로드 시작: {gcs_url}")
        
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
        print(f"✅ GCS 다운로드 완료: {content_length} bytes")
        
        # 🔍 파일 형식 확인
        content_type = response.headers.get('content-type', 'unknown')
        print(f" Content-Type: {content_type}")
        
        # 파일 시그니처 확인 (첫 4바이트)
        if len(response.content) >= 4:
            file_signature = response.content[:4].hex()
            print(f"🔍 파일 시그니처: {file_signature}")
            
            # WAV 파일 확인
            if response.content[:4] == b'RIFF':
                print("✅ WAV 파일 감지!")
            elif response.content[:4] == b'\x00\x00\x00\x20':
                print("✅ MP4 파일 감지!")
        
        return response.content
        
    except requests.exceptions.Timeout:
        print(f"❌ GCS 다운로드 타임아웃: {gcs_url}")
        raise HTTPException(status_code=408, detail="GCS 오디오 다운로드 타임아웃 (10분 초과)")
    except Exception as e:
        print(f"❌ GCS 오디오 다운로드 실패: {e}")
        raise HTTPException(status_code=400, detail=f"GCS 오디오 다운로드 실패: {e}")

def process_audio_pipeline(audio_path: str, 
                          token: str = "", 
                          min_speakers: int = 1, 
                          max_speakers: int = 2) -> Dict:
    """
    오디오 파일에 대한 화자분리 파이프라인 처리
    
    Args:
        audio_path: 오디오 파일 경로
        token: HuggingFace 토큰
        min_speakers: 최소 화자 수
        max_speakers: 최대 화자 수
        
    Returns:
        화자분리 결과 딕셔너리
    """
    try:
        print(f" 화자분리 파이프라인 시작: {audio_path}")
        
        # SpeakerDiarizer 인스턴스 생성
        diarizer = SpeakerDiarizer(
            token=token,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # 화자분리 수행
        segments = diarizer.diarize_speakers(audio_path)
        
        if not segments:
            return {
                "success": False,
                "error": "화자분리 결과가 없습니다",
                "speaker_count": 0,
                "total_duration": 0.0,
                "segments": [],
                "timestamp": datetime.now().isoformat(),
                "processing_method": "failed"
            }
        
        # 결과 정리
        speaker_count = len(set(seg['speaker_id'] for seg in segments))
        total_duration = max(seg['end'] for seg in segments) if segments else 0.0
        
        # 세그먼트 형식 변환
        formatted_segments = []
        for seg in segments:
            formatted_segments.append({
                "start_time": seg['start'],
                "end_time": seg['end'],
                "duration": seg['duration'],
                "speaker_id": seg['speaker_id'],
                "speaker": seg['speaker']
            })
        
        print(f"✅ 화자분리 파이프라인 완료: {speaker_count}명, {len(segments)}개 세그먼트")
        
        return {
            "success": True,
            "speaker_count": speaker_count,
            "total_duration": total_duration,
            "segments": formatted_segments,
            "timestamp": datetime.now().isoformat(),
            "processing_method": "pyannote" if diarizer.use_pyannote else "fallback"
        }
        
    except Exception as e:
        print(f"❌ 화자분리 파이프라인 실패: {e}")
        return {
            "success": False,
            "error": str(e),
            "speaker_count": 0,
            "total_duration": 0.0,
            "segments": [],
            "timestamp": datetime.now().isoformat(),
            "processing_method": "error"
        }