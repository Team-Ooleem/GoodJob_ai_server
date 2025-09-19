import numpy as np
import librosa
import scipy.signal as sps
from scipy.ndimage import binary_closing


def _median_filt(x: np.ndarray, k: int = 7) -> np.ndarray:
    if x.size == 0:
        return x
    # 홀수 커널 보장
    k = max(3, k | 1)
    return sps.medfilt(x, kernel_size=k)


def _safe_number(x):
    """Return a JSON-safe number: float or int, or None if NaN/Inf."""
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, (float, np.floating)):
            v = float(x)
            return v if np.isfinite(v) else None
        return x
    except Exception:
        return None


def _intervals_to_frame_mask(n_frames: int, hop_length: int, intervals: np.ndarray, sr: int) -> np.ndarray:
    """Map sample-based non-silent intervals to a frame mask using frame centers.
    Returns a boolean array of length n_frames where True means frame center
    falls within any interval [start, end).
    """
    if n_frames <= 0:
        return np.zeros(0, dtype=bool)
    # frame center sample indices
    centers = (np.arange(n_frames) * hop_length) + (hop_length // 2)
    mask = np.zeros(n_frames, dtype=bool)
    for s, e in intervals:
        # mark frames whose centers fall inside [s, e)
        mask |= (centers >= s) & (centers < e)
    return mask


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
    # 유성 프레임만: OR 결합(voiced_flag) + 임계 완화(>=0.25) + 클로징 강화(5프레임)
    if isinstance(voiced_flag, np.ndarray) and isinstance(voiced_prob, np.ndarray):
        voicing = voiced_flag.astype(bool) | (voiced_prob >= 0.25)
    else:
        voicing = (voiced_prob >= 0.25)
    mask = np.isfinite(f0) & voicing
    if mask.size:
        mask = binary_closing(mask, structure=np.ones(5, dtype=bool))
    f0_v = f0[mask]

    # 이상치/옥타브 점프 억제: (a) 미디안 필터 → (b) 프레임간 급격한 비율 변화 클리핑
    if f0_v.size > 0:
        # f0 평활화(미디안 필터)
        f0_smooth = _median_filt(f0_v, 7)

        # 세미톤으로 직접 변환 후, 중앙값 기준 정규화
        f0_center = np.nanmedian(f0_smooth)
        if f0_center <= 0 or not np.isfinite(f0_center):
            f0_center = np.nanmean(f0_smooth) + 1e-8
        st_raw = 12.0 * np.log2((f0_smooth + 1e-8) / (f0_center + 1e-8))
        # 미디안 필터로 급격한 점프 완화
        st_filt = _median_filt(st_raw, 5)
        # 윈저라이즈(상하위 2.5% 절단)
        if st_filt.size >= 20:
            lo, hi = np.nanpercentile(st_filt, [2.5, 97.5])
            st_win = np.clip(st_filt, lo, hi)
        else:
            st_win = st_filt

        # 강건 지표: 세미톤 MAD -> 표준편차 상응치로 변환(≈1.4826)
        st_med = np.nanmedian(st_win)
        mad_st = np.nanmedian(np.abs(st_win - st_med))
        f0_std_semitone = float(mad_st * 1.4826)
        # robust 중심/분산(참고용)
        f0_median_hz = float(f0_center)
        f0_mad_semitone = float(mad_st)

        # Hz 지표(유성 구간 기준)
        f0_mean = float(np.nanmean(f0_smooth))
        f0_std = float(np.nanstd(f0_smooth))
        f0_cv = float(f0_std / (f0_mean + 1e-8)) if f0_mean > 0 else 0.0
    else:
        f0_mean = f0_std = f0_cv = 0.0
        f0_std_semitone = 0.0
        f0_median_hz = 0.0
        f0_mad_semitone = 0.0

    # ---- 2) 에너지/무성 ----
    frame_length = 1024
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms)) if rms.size else 0.0
    rms_std = float(np.std(rms)) if rms.size else 0.0
    rms_cv = float(rms_std / (rms_mean + 1e-8)) if rms_mean > 0 else 0.0

    # voiced-only RMS 지표 (같은 hop_length 정렬)
    rms_cv_voiced = 0.0
    rms_db_std_voiced = 0.0
    if rms.size and mask.size:
        n = int(min(len(rms), len(mask)))
        if n > 0:
            mask_frames = mask[:n]
            rms_frames = rms[:n]
            rms_voiced = rms_frames[mask_frames]
            if rms_voiced.size > 1:
                rv_mean = float(np.mean(rms_voiced))
                rv_std = float(np.std(rms_voiced))
                rms_cv_voiced = float(rv_std / (rv_mean + 1e-8)) if rv_mean > 0 else 0.0
                # dB 스케일 변동(권장 보조 지표)
                rms_db = librosa.amplitude_to_db(np.maximum(rms_voiced, 1e-10), ref=1.0)
                rms_db_std_voiced = float(np.std(rms_db))

    # 준-지터/쉬머 (간이 프록시)
    # jitter: 주기 T=1/f0 기반 정의에 근접
    jitter_like_mad = 0.0
    jitter_like_sigma_robust = 0.0
    if f0_v.size > 3:
        T = 1.0 / np.maximum(f0_v, 1e-8)
        jitter_like = float(np.std(np.diff(T)) / (np.mean(T) + 1e-8))
        # robust jitter (MAD 기반)
        try:
            T_med = float(np.nanmedian(T))
            D = np.diff(T)
            if D.size:
                D_med = float(np.nanmedian(D))
                mad_D = float(np.nanmedian(np.abs(D - D_med)))
                jitter_like_mad = float(mad_D / (T_med + 1e-8)) if T_med > 0 else 0.0
                jitter_like_sigma_robust = float(1.4826 * jitter_like_mad)
        except Exception:
            jitter_like_mad = 0.0
            jitter_like_sigma_robust = 0.0
    else:
        jitter_like = 0.0

    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    shimmer_like_mad = 0.0
    shimmer_like_sigma_robust = 0.0
    if frames.size:
        # shimmer: voiced-only RMS envelope 기반 변동(프록시)
        if rms.size and mask.size:
            n2 = int(min(len(rms), len(mask)))
            mask_frames2 = mask[:n2]
            rms_voiced2 = rms[:n2][mask_frames2]
            if rms_voiced2.size > 3:
                shimmer_like = float(
                    np.std(np.diff(rms_voiced2)) / (np.mean(rms_voiced2) + 1e-8)
                )
                # robust shimmer (voiced RMS 기반)
                try:
                    R = rms_voiced2
                    R_med = float(np.nanmedian(R))
                    DR = np.diff(R)
                    if DR.size:
                        DR_med = float(np.nanmedian(DR))
                        mad_DR = float(np.nanmedian(np.abs(DR - DR_med)))
                        shimmer_like_mad = float(mad_DR / (R_med + 1e-8)) if R_med > 0 else 0.0
                        shimmer_like_sigma_robust = float(1.4826 * shimmer_like_mad)
                except Exception:
                    shimmer_like_mad = 0.0
                    shimmer_like_sigma_robust = 0.0
            else:
                shimmer_like = 0.0
        else:
            # fallback: 전체 프레임 피크 기반
            peaks = np.max(np.abs(frames), axis=0)
            shimmer_like = (
                float(np.std(np.diff(peaks)) / (np.mean(peaks) + 1e-8))
                if peaks.size > 3
                else 0.0
            )
            # robust shimmer (피크 기반)
            try:
                P = peaks
                if P.size:
                    P_med = float(np.nanmedian(P))
                    DP = np.diff(P)
                    if DP.size:
                        DP_med = float(np.nanmedian(DP))
                        mad_DP = float(np.nanmedian(np.abs(DP - DP_med)))
                        shimmer_like_mad = float(mad_DP / (P_med + 1e-8)) if P_med > 0 else 0.0
                        shimmer_like_sigma_robust = float(1.4826 * shimmer_like_mad)
            except Exception:
                shimmer_like_mad = 0.0
                shimmer_like_sigma_robust = 0.0
    else:
        shimmer_like = 0.0
        shimmer_like_mad = 0.0
        shimmer_like_sigma_robust = 0.0

    # 침묵 비율 (두 임계값 비교: 30dB, 50dB)
    intervals_30 = librosa.effects.split(y, top_db=30)
    voiced_len_30 = sum((e - s) for s, e in intervals_30)
    silence_ratio = float(1.0 - (voiced_len_30 / max(1, len(y))))
    intervals_50 = librosa.effects.split(y, top_db=50)
    voiced_len_50 = sum((e - s) for s, e in intervals_50)
    silence_ratio_db50 = float(1.0 - (voiced_len_50 / max(1, len(y))))

    # 프레임 기반 신뢰도 표기
    total_frames = int(len(f0)) if isinstance(f0, np.ndarray) else 0
    voiced_frames = int(np.count_nonzero(mask)) if mask.size else 0
    voiced_ratio = float(voiced_frames / max(1, total_frames)) if total_frames else 0.0
    # 비침묵(speech) 프레임 대비 유성 비율
    speech_ratio = None
    speech_frames = None
    if total_frames:
        speech_mask_50 = _intervals_to_frame_mask(total_frames, hop_length, intervals_50, sr)
        speech_frames = int(np.count_nonzero(speech_mask_50))
        if speech_frames > 0:
            speech_ratio = float(np.count_nonzero(mask & speech_mask_50) / speech_frames)

    # ---- 3) Voicing 진단용 통계 ----
    voiced_prob_mean = None
    voiced_prob_median = None
    voiced_prob_p90 = None
    voiced_flag_ratio = None
    voiced_prob_ge_025_ratio = None
    voiced_prob_ge_035_ratio = None
    f0_valid_ratio = float(np.count_nonzero(np.isfinite(f0)) / max(1, total_frames)) if total_frames else 0.0

    if isinstance(voiced_prob, np.ndarray) and voiced_prob.size:
        finite_vp = voiced_prob[np.isfinite(voiced_prob)]
        if finite_vp.size:
            voiced_prob_mean = float(np.nanmean(finite_vp))
            voiced_prob_median = float(np.nanmedian(finite_vp))
            voiced_prob_p90 = float(np.nanpercentile(finite_vp, 90))
        voiced_prob_ge_025_ratio = float(np.count_nonzero(voiced_prob >= 0.25) / len(voiced_prob))
        voiced_prob_ge_035_ratio = float(np.count_nonzero(voiced_prob >= 0.35) / len(voiced_prob))
    if isinstance(voiced_flag, np.ndarray) and voiced_flag.size:
        voiced_flag_ratio = float(np.count_nonzero(voiced_flag) / len(voiced_flag))

    raw = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_cv": f0_cv,
        "f0_std_semitone": f0_std_semitone,  # ← 강건(=MAD 기반) 세미톤 표준편차
        # f0 robust 참고 지표
        "f0_median_hz": f0_median_hz,
        "f0_mad_semitone": f0_mad_semitone,
        "rms_std": rms_std,
        "rms_cv": rms_cv,
        "rms_cv_voiced": rms_cv_voiced,
        "rms_db_std_voiced": rms_db_std_voiced,
        "jitter_like": jitter_like,
        "jitter_like_mad": jitter_like_mad,
        "jitter_like_sigma_robust": jitter_like_sigma_robust,
        "shimmer_like": shimmer_like,
        "shimmer_like_mad": shimmer_like_mad,
        "shimmer_like_sigma_robust": shimmer_like_sigma_robust,
        "silence_ratio": silence_ratio,
        "silence_ratio_db50": silence_ratio_db50,
        "voiced_ratio": voiced_ratio,
        "voiced_ratio_speech": speech_ratio,
        "voiced_frames": int(voiced_frames),
        "speech_frames": int(speech_frames) if speech_frames is not None else None,
        "total_frames": int(total_frames),
        "sr": int(sr),
        # Diagnostics
        "voiced_prob_mean": voiced_prob_mean,
        "voiced_prob_median": voiced_prob_median,
        "voiced_prob_p90": voiced_prob_p90,
        "voiced_flag_ratio": voiced_flag_ratio,
        "voiced_prob_ge_025_ratio": voiced_prob_ge_025_ratio,
        "voiced_prob_ge_035_ratio": voiced_prob_ge_035_ratio,
        "f0_valid_ratio": f0_valid_ratio,
    }
    # NaN/Inf 방지: JSON 직렬화 안전화
    return {k: _safe_number(v) for k, v in raw.items()}
