"""Node ①: Extract audio track from video file using ffmpeg."""
import json
import os
import subprocess
from state import PipelineState, Error


def extract_audio(state: PipelineState) -> dict:
    """Extract 16kHz mono wav audio from the input video.

    Reads: state["input_video"], state["video_title"]
    Writes: state["audio_wav"], state["metadata"], state["stage"], state["errors"]
    Skips if audio_wav already exists.
    """
    if not state.get("input_video"):
        return {
            "errors": [Error(
                stage="extract",
                message="No input video available — did download fail?",
                retry_count=0,
            )],
            "metadata": {},
            "stage": "extract",
        }

    work_dir = os.path.join(".video-translate", state["video_title"])
    audio_wav = os.path.join(work_dir, "audio.wav")

    # Skip if already done
    if os.path.exists(audio_wav):
        return {
            "audio_wav": audio_wav,
            "metadata": {"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le", "format": "wav"},
            "stage": "asr",
        }

    os.makedirs(work_dir, exist_ok=True)

    input_video = state["input_video"]

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_video,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                audio_wav,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            return {
                "errors": [Error(
                    stage="extract",
                    message=f"ffmpeg error: {result.stderr[:200]}",
                    retry_count=0,
                )],
                "stage": "extract",
                "metadata": {},
            }

        return {
            "audio_wav": audio_wav,
            "stage": "asr",
            "metadata": {
                "sample_rate": 16000,
                "channels": 1,
                "codec": "pcm_s16le",
                "format": "wav",
            },
        }

    except FileNotFoundError:
        return {
            "errors": [Error(
                stage="extract",
                message="ffmpeg not found. Install with: brew install ffmpeg",
                retry_count=0,
            )],
            "stage": "extract",
            "metadata": {},
        }
    except subprocess.TimeoutExpired:
        return {
            "errors": [Error(
                stage="extract",
                message="ffmpeg timed out after 300s",
                retry_count=0,
            )],
            "stage": "extract",
            "metadata": {},
        }
