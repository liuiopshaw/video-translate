"""Node ④: Generate Chinese TTS audio using Edge TTS (full-text pipeline)."""
import os
import subprocess
import time
from state import PipelineState, TSeg, Error
from nodes.tts_utils import fulltext_tts_pipeline

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("TTS_RATE", "+15%")
MIN_FILE_SIZE = 500


def run_tts(state: PipelineState) -> dict:
    """Generate Chinese voice audio — single API call for entire text.

    Reads: state["subtitles_cn"], state["video_title"]
    Writes: state["tts_segments"], state["cn_audio"], state["stage"], state["errors"]
    """
    if state.get("tts_segments"):
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

    segments, errors = fulltext_tts_pipeline(
        cn_subs, work_dir, _generate_fulltext, label="Edge TTS"
    )

    # Convert dict errors to Error TypedDicts
    typed_errors = [
        Error(stage=e["stage"], message=e["message"], retry_count=0)
        for e in errors
    ]

    if not segments:
        return {
            "errors": typed_errors,
            "stage": "tts",
        }

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    _build_timeline_sequential(segments, cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": typed_errors,
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


def _build_timeline_sequential(segments: list[dict], output_path: str) -> None:
    """Place segments on timeline via sequential 2-input amix."""
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    total_duration = max(s["end"] for s in sorted_segs) + 1

    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=24000:cl=mono:d={total_duration}",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True,
    )

    for seg in sorted_segs:
        delay_ms = int(seg["start"] * 1000)
        tmp = output_path + ".tmp.wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-i", seg["wav_path"],
             "-filter_complex",
             f"[0:a]volume=1[base];"
             f"[1:a]adelay={delay_ms}|{delay_ms}[spk];"
             f"[base][spk]amix=inputs=2:duration=longest:dropout_transition=0,volume=2",
             "-c:a", "pcm_s16le", tmp],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Mix failed at seg {seg['index']}: {result.stderr[:200]}")
        os.replace(tmp, output_path)


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
