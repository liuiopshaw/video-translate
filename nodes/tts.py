"""Node ④: Generate Chinese TTS audio using Edge TTS (full-text, no splitting).

Generates one continuous audio file — no per-sentence splitting, no timeline gaps.
The full audio is used directly; merge step handles length mismatch with -shortest.
"""
import os
import subprocess
import time
from state import PipelineState, Error

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("TTS_RATE", "+15%")
MIN_FILE_SIZE = 500


def run_tts(state: PipelineState) -> dict:
    """Generate one continuous Chinese voice track from the full translated text.

    Reads: state["subtitles_cn"], state["video_title"]
    Writes: state["cn_audio"], state["stage"], state["errors"]
    """
    if state.get("cn_audio") and os.path.exists(state["cn_audio"]):
        return {"stage": "synthesis"}

    cn_subs = state.get("subtitles_cn", [])
    if not cn_subs:
        return {
            "errors": [Error(
                stage="tts", message="No Chinese subtitles found.",
                retry_count=0,
            )],
            "stage": "tts",
        }

    work_dir = os.path.join(".video-translate", state["video_title"])
    os.makedirs(work_dir, exist_ok=True)

    # Build clean full text
    cleaned = []
    for sub in cn_subs:
        t = sub["text"].strip().rstrip("，。！？、；：,.!?;:")
        if t:
            cleaned.append(t)

    full_text = "。".join(cleaned) + "。"
    total = sum(len(s) for s in cleaned)
    print(f"   🎤 Edge TTS: generating full audio ({total} chars)...")

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    if not _generate_fulltext(full_text, cn_audio):
        return {
            "errors": [Error(
                stage="tts", message="Edge TTS full-text generation failed.",
                retry_count=0,
            )],
            "stage": "tts",
        }

    # Normalize loudness
    norm_audio = os.path.join(work_dir, "cn_audio_norm.wav")
    _loudnorm(cn_audio, norm_audio)

    return {
        "cn_audio": norm_audio,
        "stage": "synthesis",
        "errors": [],
    }


def _generate_fulltext(text: str, output: str) -> bool:
    """Generate one long audio file from entire Chinese text via Edge TTS."""
    text_file = output + ".txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["edge-tts", "--voice", VOICE, "--rate", RATE,
                 "--file", text_file, "--write-media", output],
                capture_output=True, text=True, timeout=300,
            )

            if (result.returncode == 0
                    and os.path.exists(output)
                    and os.path.getsize(output) >= MIN_FILE_SIZE):
                _safe_remove(text_file)
                return True

            if os.path.exists(output):
                os.remove(output)
            if attempt < 2:
                time.sleep(2.0 * (attempt + 1))

        except subprocess.TimeoutExpired:
            if attempt < 2:
                time.sleep(5.0)
        except Exception:
            if attempt < 2:
                time.sleep(2.0)

    _safe_remove(text_file)
    return False


def _loudnorm(input_path: str, output_path: str) -> None:
    """Normalize loudness and downsample to 24kHz mono."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true",
         "-ar", "24000", "-ac", "1",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=60,
    )


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
