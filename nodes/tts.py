"""Node ④: Generate Chinese TTS audio using Edge TTS."""
import os
import subprocess
from state import PipelineState, TSeg, Error

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = os.environ.get("TTS_RATE", "+15%")  # Faster for Chinese to keep pace with video


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

    # Place segments on timeline at their original timestamps
    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    _place_on_timeline(segments, cn_audio, state.get("metadata", {}))

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": errors,
    }


def _place_on_timeline(segments: list[dict], output_path: str, metadata: dict) -> None:
    """Place TTS audio segments at their original timestamps using ffmpeg adelay.

    This ensures audio stays in sync with video regardless of TTS speed.
    Each segment is delayed by its start time, then all are mixed together.
    The output duration matches the last segment's end time.
    """
    inputs = []
    for seg in segments:
        inputs += ["-i", seg["wav_path"]]

    n = len(segments)

    # Build adelay + amix filter chain
    # Each input gets delayed by its start_time (in milliseconds)
    delays = []
    for seg in segments:
        delay_ms = int(seg["start"] * 1000)
        delays.append(f"adelay={delay_ms}|{delay_ms}")

    # Mix all delayed streams together
    filter_parts = []
    for i in range(n):
        filter_parts.append(f"[{i}:a]{delays[i]}[d{i}]")

    mix_inputs = "".join(f"[d{i}]" for i in range(n))
    filter_parts.append(f"{mix_inputs}amix=inputs={n}:duration=longest:dropout_transition=0[out]")

    filter_expr = ";".join(filter_parts)

    result = subprocess.run(
        ["ffmpeg", "-y"] + inputs +
        ["-filter_complex", filter_expr,
         "-map", "[out]", output_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg timeline placement failed: {result.stderr[:200]}")
