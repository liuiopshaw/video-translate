"""Node ④-alt: VoxCPM2 TTS via HuggingFace Gradio API.

No local GPU/MLX needed. Calls the hosted VoxCPM2 demo space.
~15s per segment. Free, no rate limit issues like Edge TTS.

Usage:
    export TTS_ENGINE=voxcpm
"""
import os
import subprocess
import time
import shutil
import tempfile
from state import PipelineState, TSeg, Error

VOXCPM_SPACE = os.environ.get("VOXCPM_SPACE", "OpenBMB/VoxCPM-Demo")
VOXCPM_VOICE = os.environ.get("VOXCPM_VOICE", "一位沉稳专业的男性播报员，声音清晰有力")
LOUDNORM_TARGET = "-16"

_client = None


def _get_client():
    """Lazy-load Gradio client (singleton, ~7s first connection)."""
    global _client
    if _client is None:
        from gradio_client import Client
        _client = Client(VOXCPM_SPACE)
        print("   ✅ VoxCPM2 Gradio API connected")
    return _client


def run_tts(state: PipelineState) -> dict:
    """Generate Chinese TTS using VoxCPM2 Gradio Space API.

    Same interface as Edge TTS node — drop-in replacement.
    """
    if state.get("tts_segments"):
        return {"stage": "synthesis"}

    cn_subs = state.get("subtitles_cn", [])
    if not cn_subs:
        return {
            "errors": [Error(
                stage="tts",
                message="No Chinese subtitles found. Run translate first.",
                retry_count=0,
            )],
            "stage": "tts",
        }

    work_dir = os.path.join(".video-translate", state["video_title"])
    tts_dir = os.path.join(work_dir, "tts_segments")
    norm_dir = os.path.join(work_dir, "tts_norm")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    segments = []
    errors = []

    print(f"   🎤 VoxCPM2: {len(cn_subs)} segments to generate")

    for sub in cn_subs:
        wav_path = os.path.join(tts_dir, f"{sub['index']:04d}.wav")
        norm_path = os.path.join(norm_dir, f"{sub['index']:04d}.wav")

        if os.path.exists(norm_path) and os.path.getsize(norm_path) >= 500:
            segments.append(TSeg(
                index=sub["index"], start=sub["start"], end=sub["end"],
                wav_path=norm_path,
            ))
            continue

        # Primary: VoxCPM2 API
        ok = _generate_voxcpm(sub["text"], wav_path)

        # Fallback: Edge TTS
        if not ok:
            errors.append(Error(
                stage="tts",
                message=f"VoxCPM2 failed for seg {sub['index']}, Edge TTS fallback",
                retry_count=0,
            ))
            time.sleep(1.0)
            from nodes.tts import _generate_tts
            ok = _generate_tts(sub["text"], wav_path, sub["index"])

        if not ok:
            continue

        # Normalize
        if not _loudnorm(wav_path, norm_path):
            os.replace(wav_path, norm_path)

        segments.append(TSeg(
            index=sub["index"], start=sub["start"], end=sub["end"],
            wav_path=norm_path,
        ))

        if len(segments) % 10 == 0:
            print(f"   📝 {len(segments)}/{len(cn_subs)}")

    if not segments:
        return {
            "errors": [Error(
                stage="tts",
                message="All TTS segments failed (VoxCPM2 + Edge TTS)",
                retry_count=0,
            )] + errors,
            "stage": "tts",
        }

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    from nodes.tts import _build_timeline_sequential
    _build_timeline_sequential(segments, cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": errors,
    }


def _generate_voxcpm(text: str, output: str) -> bool:
    """Generate TTS via VoxCPM2 Gradio API. Returns True on success.

    Uses VOXCPM_VOICE env var as control_instruction for consistent timbre.
    """
    for attempt in range(2):
        try:
            client = _get_client()
            result = client.predict(
                text_input=text,
                control_instruction=VOXCPM_VOICE,
                reference_wav_path_input=None,
                use_prompt_text=False,
                prompt_text_input="",
                cfg_value_input=2.0,
                do_normalize=False,
                denoise=False,
                api_name="/generate",
            )

            if isinstance(result, str) and os.path.exists(result):
                shutil.copy(result, output)
                if os.path.getsize(output) >= 500:
                    return True

            if os.path.exists(output):
                os.remove(output)

        except Exception:
            if attempt < 1:
                time.sleep(2.0)

    return False


def _loudnorm(input_path: str, output_path: str) -> bool:
    """Normalize loudness and downsample to 24kHz mono."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-af", f"loudnorm=I={LOUDNORM_TARGET}:TP=-1.5:LRA=11:linear=true",
         "-ar", "24000", "-ac", "1",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) >= 500
