import numpy as np
import librosa
import scipy.signal as sps


def _median_filt(x: np.ndarray, k: int = 7) -> np.ndarray:
    if x.size == 0:
        return x
    # 홀수 커널 보장
    k = max(3, k | 1)
    return sps.medfilt(x, kernel_size=k)


def analyze_audio(path: str, sr_target: int = 16000):
    y, sr = librosa.load(path, sr=sr_target, mono=True)

    # ---- 1) 견고한 f0 추정: pYIN + voicing ----
    # hop_length은 256(≈16kHz 기준 16ms), 필요시 320/256 조정
    hop_length = 256
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=50,
        fmax=400,
        sr=sr,
        hop_length=hop_length,
        frame_length=2048,
        fill_na=np.nan,
    )
    # 유성 프레임만
    mask = np.isfinite(f0) & (voiced_prob >= 0.1)
    f0_v = f0[mask]

    # 이상치/옥타브 점프 억제: (a) 미디안 필터 → (b) 프레임간 급격한 비율 변화 클리핑
    if f0_v.size > 0:
        f0_med = np.nanmedian(f0_v)
        f0_smooth = _median_filt(f0_v, 7)

        # 프레임 간 변화율 제한 (1프레임당 ±12세미톤 = 2배/0.5배를 상한/하한)
        # 세미톤 도메인에서 차분 클리핑
        st = 12.0 * np.log2(f0_smooth / (f0_med + 1e-8))
        dst = np.diff(st, prepend=st[0])
        max_step = 12.0  # 허용 최대 세미톤 변화(프레임 간)
        dst_clipped = np.clip(dst, -max_step, max_step)
        st_stable = np.cumsum(dst_clipped)
        # 다시 Hz로 복원할 필요 없이 st 도메인에서 지표 산출 가능

        # 강건 지표: 세미톤 MAD (중앙절대편차)
        st_med = np.nanmedian(st_stable)
        mad_st = np.nanmedian(np.abs(st_stable - st_med))
        # 참고로, 표준편차에 상응하는 강건 추정치 ≈ MAD * 1.4826
        f0_std_semitone = float(mad_st * 1.4826)

        # Hz 지표도 같이 계산
        f0_mean = float(np.nanmean(f0_smooth))
        f0_std = float(np.nanstd(f0_smooth))
        f0_cv = float(f0_std / (f0_mean + 1e-8)) if f0_mean > 0 else 0.0
    else:
        f0_mean = f0_std = f0_cv = 0.0
        f0_std_semitone = 0.0

    # ---- 2) 에너지/무성 ----
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_cv = float(rms_std / (rms_mean + 1e-8)) if rms_mean > 0 else 0.0

    # 준-지터/쉬머 (간이 프록시)
    if f0_v.size > 3:
        jitter_like = float(np.std(np.diff(f0_v)) / (np.mean(f0_v) + 1e-8))
    else:
        jitter_like = 0.0

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    if frames.size:
        peaks = np.max(np.abs(frames), axis=0)
        shimmer_like = (
            float(np.std(np.diff(peaks)) / (np.mean(peaks) + 1e-8))
            if peaks.size > 3
            else 0.0
        )
    else:
        peaks = np.array([])
        shimmer_like = 0.0

    # 침묵 비율
    intervals = librosa.effects.split(y, top_db=30)
    voiced_len = sum((e - s) for s, e in intervals)
    silence_ratio = float(1.0 - (voiced_len / max(1, len(y))))

    return {
        "f0_mean": float(f0_mean),
        "f0_std": float(f0_std),
        "f0_cv": float(f0_cv),
        "f0_std_semitone": float(f0_std_semitone),  # ← 강건(=MAD 기반) 세미톤 표준편차
        "rms_std": float(rms_std),
        "rms_cv": float(rms_cv),
        "jitter_like": float(jitter_like),
        "shimmer_like": float(shimmer_like),
        "silence_ratio": float(silence_ratio),
        "sr": int(sr),
    }
