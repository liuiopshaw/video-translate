"""Node ④: Generate Chinese TTS audio using Edge TTS."""
import os
import subprocess
from state import PipelineState, TSeg, Error

VOICE = os.environ.get("TTS_VOICE", "zh-CN-YunxiNeural")
RATE = "-15%"


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

    # Concat all into single cn_audio.wav
    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    _concat_wavs([s["wav_path"] for s in segments], cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": errors,
    }


def _concat_wavs(wav_paths: list[str], output_path: str) -> None:
    """Concatenate WAV files using ffmpeg concat demuxer."""
    concat_list = output_path + ".txt"
    with open(concat_list, "w") as f:
        for p in wav_paths:
            f.write(f"file '{p}'\n")

    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c", "copy", output_path],
        capture_output=True, text=True,
    )

    try:
        os.remove(concat_list)
    except OSError:
        pass

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[:200]}")
