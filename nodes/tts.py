"""Node ④: Generate Chinese TTS audio using Edge TTS."""
import os
import subprocess
import math
from state import PipelineState, TSeg, Error

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("TTS_RATE", "+15%")
BATCH_SIZE = 30  # Mix this many segments per ffmpeg call to avoid huge filter graphs


def run_tts(state: PipelineState) -> dict:
    """Generate Chinese voice audio for each subtitle segment.

    Reads: state["subtitles_cn"], state["video_title"]
    Writes: state["tts_segments"], state["cn_audio"], state["stage"], state["errors"]
    Skips if tts_segments already populated.
    Sentence-level fault tolerance: single failures don't block.
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
    os.makedirs(tts_dir, exist_ok=True)

    segments = []
    errors = []

    for sub in cn_subs:
        wav_path = os.path.join(tts_dir, f"{sub['index']:04d}.wav")

        if os.path.exists(wav_path):
            segments.append(TSeg(
                index=sub["index"],
                start=sub["start"],
                end=sub["end"],
                wav_path=wav_path,
            ))
            continue

        try:
            result = subprocess.run(
                [
                    "edge-tts",
                    "--voice", VOICE,
                    "--rate", RATE,
                    "--text", sub["text"],
                    "--write-media", wav_path,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0 and os.path.exists(wav_path):
                segments.append(TSeg(
                    index=sub["index"],
                    start=sub["start"],
                    end=sub["end"],
                    wav_path=wav_path,
                ))
            else:
                errors.append(Error(
                    stage="tts",
                    message=f"TTS failed for segment {sub['index']}: {result.stderr[:100]}",
                    retry_count=0,
                ))

        except FileNotFoundError:
            return {
                "errors": [Error(
                    stage="tts",
                    message="edge-tts not found. Install with: pip install edge-tts",
                    retry_count=0,
                )],
                "stage": "tts",
            }
        except subprocess.TimeoutExpired:
            errors.append(Error(
                stage="tts",
                message=f"TTS timed out for segment {sub['index']}",
                retry_count=0,
            ))
        except Exception as e:
            errors.append(Error(
                stage="tts",
                message=f"TTS error for segment {sub['index']}: {e}",
                retry_count=0,
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
    _place_on_timeline(segments, cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": errors,
    }


def _place_on_timeline(segments: list[dict], output_path: str) -> None:
    """Place TTS segments at their timestamps with correct volume.

    Processes segments in batches to keep ffmpeg filter graphs manageable.
    Each batch uses adelay + amix with volume compensation for N inputs.
    Batches are then mixed together sequentially.
    """
    sorted_segs = sorted(segments, key=lambda s: s["start"])
    total_duration = max(s["end"] for s in sorted_segs) + 1

    # Create a silent base track
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i",
         f"anullsrc=r=24000:cl=mono:d={total_duration}",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True,
    )

    # Process in batches
    for batch_start in range(0, len(sorted_segs), BATCH_SIZE):
        batch = sorted_segs[batch_start:batch_start + BATCH_SIZE]
        _mix_batch(batch, output_path)


def _mix_batch(segments: list[dict], current_path: str) -> None:
    """Mix a batch of segments into the current audio track."""
    n = len(segments)
    if n == 0:
        return

    # Input layout: [0] = base track, [1..n] = segment files
    filter_parts = ["[0:a]volume=1[base]"]

    for i, seg in enumerate(segments):
        delay_ms = int(seg["start"] * 1000)
        filter_parts.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[d{i}]")

    mix_inputs = "[base]" + "".join(f"[d{i}]" for i in range(n))
    total_inputs = n + 1
    filter_parts.append(
        f"{mix_inputs}amix=inputs={total_inputs}:duration=longest:"
        f"dropout_transition=0,volume={total_inputs}[out]"
    )

    filter_expr = ";".join(filter_parts)

    # Build command: base track + N segment files
    inputs = ["-i", current_path]
    for seg in segments:
        inputs += ["-i", seg["wav_path"]]

    tmp = current_path + ".tmp.wav"
    result = subprocess.run(
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_expr,
         "-map", "[out]", "-c:a", "pcm_s16le", tmp],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        # Show only the actual error, skip ffmpeg banner
        lines = result.stderr.split("\n")
        error_lines = [l for l in lines if "Error" in l or "Invalid" in l or "No such" in l or "failed" in l]
        if not error_lines:
            error_lines = lines[-5:]
        raise RuntimeError(f"Timeline mix failed: {' | '.join(error_lines)}")

    os.replace(tmp, current_path)
