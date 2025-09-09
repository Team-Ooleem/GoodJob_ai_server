import os
import uuid
import shutil
import tempfile
import subprocess
from typing import Optional

import numpy as np
import librosa
import soundfile as sf

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
import logging

router = APIRouter(prefix="/avatar", tags=["avatar"])


MEDIA_ROOT = os.environ.get("MEDIA_ROOT", os.path.join(os.getcwd(), "media"))
BACKEND_MODE = os.environ.get("BACKEND", "mock").lower()  # mock | gpu | cpu (here we implement mock)
logger = logging.getLogger("avatar")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _is_uuid(s: str) -> bool:
    try:
        uuid.UUID(str(s))
        return True
    except Exception:
        return False


def _find_avatar_image(avatar_id: str) -> tuple[str, Optional[str]]:
    """Return (avatar_dir, image_path or None)."""
    avatar_dir = os.path.join(MEDIA_ROOT, "avatar", avatar_id)
    if not os.path.isdir(avatar_dir):
        return avatar_dir, None
    image_path = None
    for fn in os.listdir(avatar_dir):
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            image_path = os.path.join(avatar_dir, fn)
            break
    return avatar_dir, image_path


def _load_and_prepare_audio_to_wav(src_path: str, sr_target: int = 22050, pre_ms: int = 200, post_ms: int = 200) -> tuple[str, float]:
    """Load any audio file to mono PCM16 WAV with target sr, add pre/post silence, save to temp WAV.
    Returns (wav_path, duration_seconds)
    """
    y, sr = librosa.load(src_path, sr=sr_target, mono=True)
    # padding
    pre = np.zeros(int(sr * pre_ms / 1000), dtype=y.dtype)
    post = np.zeros(int(sr * post_ms / 1000), dtype=y.dtype)
    y_pad = np.concatenate([pre, y, post]) if y.size else np.concatenate([pre, post])
    duration = float(len(y_pad) / sr)
    # write to temp PCM16 WAV
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    sf.write(tmp_wav.name, y_pad, sr, subtype="PCM_16")
    return tmp_wav.name, duration


def _ffmpeg_image_audio_to_mp4(image_path: str, audio_wav_path: str, resolution: int = 256) -> str:
    """Use ffmpeg to create a talking-head placeholder video from a still image + audio.
    Output: temp mp4 path.
    """
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg not found in PATH; required for mock render")

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # scale + pad to square resolution, keep aspect ratio
    vf = f"scale='min({resolution},iw)':'min({resolution},ih)':force_original_aspect_ratio=decrease,pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color=black"

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-i",
        image_path,
        "-i",
        audio_wav_path,
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        vf,
        "-r",
        "25",
        "-shortest",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _ffmpeg_idle_from_image(image_path: str, resolution: int, duration_sec: float) -> str:
    """Create a looped idle MP4 from a still image with subtle motion (mock).
    Adds a silent audio track to improve autoplay compatibility.
    Note: Using a conservative filter chain for portability.
    """
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=500, detail="ffmpeg not found in PATH; required for idle clip")

    out_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

    # We apply a very slow zoom-in/out effect using zoompan. For broader compatibility,
    # keep expressions simple and cap zoom range to ~1.02.
    # Frames = duration * 25 fps
    frames = max(1, int(duration_sec * 25))
    size = f"{resolution}x{resolution}"
    vf = (
        f"scale='min({resolution},iw)':'min({resolution},ih)':force_original_aspect_ratio=decrease,"
        f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2:color=black,"
        # gentle zoom-in
        f"zoompan=z='min(zoom+0.0002,1.02)':d=1:s={size},"
        # ensure yuv420p for broad compatibility
        f"fps=25,format=yuv420p"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-loop",
        "1",
        "-t",
        str(max(duration_sec, 0.1)),
        "-i",
        image_path,
        # add silent audio
        "-f",
        "lavfi",
        "-t",
        str(max(duration_sec, 0.1)),
        "-i",
        "anullsrc=cl=mono:r=22050",
        "-filter_complex",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


@router.get("/healthz")
def healthz():
    ffmpeg_ok = shutil.which("ffmpeg") is not None
    return {
        "ok": True,
        "backend": BACKEND_MODE,
        "cuda": False,  # this service doesn't check CUDA in mock mode
        "ffmpeg": ffmpeg_ok,
        "media_root": MEDIA_ROOT,
    }


@router.get("/debug/list")
def debug_list():
    """List available avatar directories and first image file, to debug 404 issues."""
    base = os.path.join(MEDIA_ROOT, "avatar")
    result = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            aid_path = os.path.join(base, name)
            if not os.path.isdir(aid_path):
                continue
            img = None
            for fn in os.listdir(aid_path):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                    img = fn
                    break
            result.append({"avatar_id": name, ":dir": aid_path, "image": img})
    return {"media_root": MEDIA_ROOT, "count": len(result), "items": result}


@router.get("/debug/resolve")
def debug_resolve(avatar_id: str = Query(...)):
    """Return exact path the server expects for a given avatar_id."""
    if not _is_uuid(avatar_id):
        raise HTTPException(status_code=400, detail="invalid avatar_id format (must be UUID)")
    avatar_dir, image_path = _find_avatar_image(avatar_id)
    return {
        "media_root": MEDIA_ROOT,
        "avatar_dir": avatar_dir,
        "exists": os.path.isdir(avatar_dir),
        "image_path": image_path,
        "image_exists": bool(image_path and os.path.isfile(image_path)),
    }


@router.get("/idle")
def idle_clip(
    avatar_id: str = Query(...),
    resolution: int = Query(256, ge=128, le=1024),
    duration: float = Query(10.0, gt=0.1, le=60.0),
):
    """Generate or serve a looped idle clip for the avatar (mock). Returns video/mp4.
    Caches the result under MEDIA_ROOT/avatar/{avatar_id}/idle_{resolution}_{duration}s.mp4
    """
    if not _is_uuid(avatar_id):
        raise HTTPException(status_code=400, detail="invalid avatar_id (must be UUID)")
    avatar_dir, image_path = _find_avatar_image(avatar_id)
    if not os.path.isdir(avatar_dir):
        raise HTTPException(status_code=404, detail=f"avatar not found at {avatar_dir}")
    if not image_path:
        raise HTTPException(status_code=400, detail=f"no avatar image present in {avatar_dir}")

    _ensure_dir(avatar_dir)
    # rounded duration to 1 decimal to keep cache names reasonable
    dur_tag = f"{duration:.1f}".replace(".", "p")
    cached = os.path.join(avatar_dir, f"idle_{resolution}_{dur_tag}s.mp4")
    if os.path.isfile(cached):
        def _iter_cached():
            with open(cached, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
        headers = {"X-Resolution": str(resolution), "X-Duration": f"{duration:.3f}", "X-Backend": BACKEND_MODE, "X-Cache": "HIT"}
        return StreamingResponse(_iter_cached(), media_type="video/mp4", headers=headers)

    # generate
    out = _ffmpeg_idle_from_image(image_path, resolution, duration)
    # move to cache path
    try:
        shutil.move(out, cached)
    except Exception:
        cached = out  # fallback: stream temp file

    def _iterfile():
        with open(cached, "rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    headers = {"X-Resolution": str(resolution), "X-Duration": f"{duration:.3f}", "X-Backend": BACKEND_MODE, "X-Cache": "MISS"}
    return StreamingResponse(_iterfile(), media_type="video/mp4", headers=headers)


@router.post("/register-image")
async def register_image(file: UploadFile = File(...)):
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="file is required")

    # generate avatar_id
    avatar_id = str(uuid.uuid4())
    avatar_dir = os.path.join(MEDIA_ROOT, "avatar", avatar_id)
    _ensure_dir(avatar_dir)

    # keep original extension if available
    name_lower = file.filename.lower()
    ext = ".jpg"
    for e in [".png", ".jpg", ".jpeg", ".webp"]:
        if name_lower.endswith(e):
            ext = e
            break

    dst_path = os.path.join(avatar_dir, f"image{ext}")
    with open(dst_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    # optional preview url (local file path)
    return {"avatar_id": avatar_id, "path": dst_path}


@router.post("/render-sync")
async def render_sync(
    avatar_id: str = Form(...),
    audio: UploadFile = File(...),
    resolution: Optional[int] = Form(256),
    still_mode: Optional[bool] = Form(True),
    pose_scale: Optional[float] = Form(0.2),
    expression_scale: Optional[float] = Form(0.2),
    enhance: Optional[bool] = Form(False),
):
    # basic validation
    if not _is_uuid(avatar_id):
        logger.warning("render-sync: invalid avatar_id format: %s", avatar_id)
        raise HTTPException(status_code=400, detail="invalid avatar_id (must be UUID)")
    if resolution not in (256, 512, None):
        raise HTTPException(status_code=400, detail="resolution must be 256 or 512")

    # locate avatar image
    avatar_dir, image_path = _find_avatar_image(avatar_id)
    if not os.path.isdir(avatar_dir):
        logger.warning("render-sync: avatar dir missing: dir=%s (MEDIA_ROOT=%s)", avatar_dir, MEDIA_ROOT)
        raise HTTPException(status_code=404, detail=f"avatar not found at {avatar_dir}; set MEDIA_ROOT or register-image first")
    if not image_path:
        logger.warning("render-sync: no image found in %s", avatar_dir)
        raise HTTPException(status_code=400, detail=f"no avatar image present in {avatar_dir}")

    # persist uploaded audio to temp file
    tmp_in = tempfile.NamedTemporaryFile(suffix=os.path.splitext(audio.filename or "_in")[1] or ".wav", delete=False)
    try:
        with tmp_in as f:
            shutil.copyfileobj(audio.file, f)
        wav_path, duration = _load_and_prepare_audio_to_wav(tmp_in.name)

        # mock backend: compose still image + audio into mp4 via ffmpeg
        try:
            out_mp4 = _ffmpeg_image_audio_to_mp4(image_path, wav_path, resolution or 256)
        finally:
            try:
                os.unlink(wav_path)
            except Exception:
                pass

        # stream mp4 back; backend will upload to S3
        def _iterfile():
            with open(out_mp4, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
            try:
                os.unlink(out_mp4)
            except Exception:
                pass

        headers = {
            "X-Resolution": str(resolution or 256),
            "X-Duration": f"{duration:.3f}",
            "X-Backend": BACKEND_MODE,
        }
        return StreamingResponse(_iterfile(), media_type="video/mp4", headers=headers)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")
    finally:
        try:
            os.unlink(tmp_in.name)
        except Exception:
            pass
