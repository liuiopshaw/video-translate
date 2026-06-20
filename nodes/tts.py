"""Node ④: Generate Chinese TTS audio using Edge TTS."""
import os
import subprocess
import time
from state import PipelineState, TSeg, Error

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("TTS_RATE", "+15%")
LOUDNORM_TARGET = "-16"
MIN_FILE_SIZE = 500  # bytes — anything smaller is a failed generation


def run_tts(state: PipelineState) -> dict:
    """Generate Chinese voice audio for each subtitle segment.

    Reads: state["subtitles_cn"], state["video_title"]
    Writes: state["tts_segments"], state["cn_audio"], state["stage"], state["errors"]
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

    for sub in cn_subs:
        raw_path = os.path.join(tts_dir, f"{sub['index']:04d}.wav")
        norm_path = os.path.join(norm_dir, f"{sub['index']:04d}.wav")

        # Skip if already normalized
        if os.path.exists(norm_path):
            segments.append(TSeg(
                index=sub["index"], start=sub["start"], end=sub["end"],
                wav_path=norm_path,
            ))
            continue

        # Rate-limit: pause between sentences to avoid Edge TTS throttling
        if len(segments) > 0 and len(segments) % 10 == 0:
            time.sleep(2.0)  # longer pause every 10 sentences
        else:
            time.sleep(0.3)  # light gap between sentences

        ok = _generate_tts(sub["text"], raw_path, sub["index"])
        if not ok:
            errors.append(Error(
                stage="tts",
                message=f"TTS failed for segment {sub['index']}",
                retry_count=0,
            ))
            continue

        # Normalize volume to consistent loudness
        if not _loudnorm(raw_path, norm_path, sub["index"]):
            # Fall back to raw (unnormalized) file
            os.replace(raw_path, norm_path)

        segments.append(TSeg(
            index=sub["index"], start=sub["start"], end=sub["end"],
            wav_path=norm_path,
        ))

    if not segments:
        return {
            "errors": [Error(
                stage="tts",
                message="All TTS segments failed to generate",
                retry_count=0,
            )] + errors,
            "stage": "tts",
        }

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    _build_timeline_sequential(segments, cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": errors,
    }


def _generate_tts(text: str, output: str, idx: int) -> bool:
    """Generate TTS audio with retry logic using text-file input to avoid encoding issues."""
    # Write text to a temp file — avoids shell/encoding problems with long Chinese text
    text_file = output + ".txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["edge-tts", "--voice", VOICE, "--rate", RATE,
                 "--file", text_file, "--write-media", output],
                capture_output=True, text=True, timeout=60,
            )

            # Check for real success: file exists AND has content
            if (result.returncode == 0
                    and os.path.exists(output)
                    and os.path.getsize(output) >= MIN_FILE_SIZE):
                _safe_remove(text_file)
                return True

            # File too small or missing — clean up and retry
            if os.path.exists(output):
                os.remove(output)

            # Rate limiting — wait between retries
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

        except subprocess.TimeoutExpired:
            if attempt < 2:
                time.sleep(2.0)
        except Exception:
            if attempt < 2:
                time.sleep(1.0)

    _safe_remove(text_file)
    return False


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _loudnorm(input_path: str, output_path: str, idx: int) -> bool:
    """Normalize audio loudness to a consistent target (-16 LUFS)."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-af", f"loudnorm=I={LOUDNORM_TARGET}:TP=-1.5:LRA=11:linear=true",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0 and os.path.exists(output_path)


def _build_timeline_sequential(segments: list[dict], output_path: str) -> None:
    """Place each segment on the timeline by mixing one at a time.

    Starts with a silent base track of full duration, then iteratively mixes
    each segment in at its start time using 2-input amix. Since every segment
    is pre-normalized by loudnorm, all have consistent volume — no compensation
    needed.
    """
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    total_duration = max(s["end"] for s in sorted_segs) + 1

    # Step 1: Create a silent base track
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=24000:cl=mono:d={total_duration}",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True,
    )

    # Step 2: Mix in each segment one at a time
    for seg in sorted_segs:
        delay_ms = int(seg["start"] * 1000)
        tmp = output_path + ".tmp.wav"

        result = subprocess.run(
            ["ffmpeg", "-y",
             "-i", output_path,
             "-i", seg["wav_path"],
             "-filter_complex",
             f"[0:a]volume=1[base];"
             f"[1:a]adelay={delay_ms}|{delay_ms}[spk];"
             f"[base][spk]amix=inputs=2:duration=longest:dropout_transition=0,volume=2",
             "-c:a", "pcm_s16le", tmp],
            capture_output=True, text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Mix failed at seg {seg['index']}: {result.stderr[:200]}"
            )

        os.replace(tmp, output_path)
