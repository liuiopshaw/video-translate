"""Node ②: Speech recognition using OpenAI Whisper."""
import os
from state import PipelineState, Sub, Error
from nodes.utils import save_srt


def run_asr(state: PipelineState) -> dict:
    """Run Whisper on extracted audio, producing timed subtitles.

    Reads: state["audio_wav"]
    Writes: state["subtitles_en"], state["stage"], state["errors"]
    Skips if subtitles_en already populated.
    """
    if state.get("subtitles_en"):
        return {"stage": "translate"}

    audio_wav = state.get("audio_wav", "")
    if not audio_wav:
        return {
            "errors": [Error(
                stage="asr",
                message="No audio_wav found in state. Run extract_audio first.",
                retry_count=0,
            )],
            "stage": "asr",
        }

    if not os.path.exists(audio_wav):
        return {
            "errors": [Error(
                stage="asr",
                message=f"Audio file not found: {audio_wav}",
                retry_count=0,
            )],
            "stage": "asr",
        }

    try:
        import whisper

        # Use tiny model for quick test, large-v3 for production
        model_name = os.environ.get("WHISPER_MODEL", "tiny")
        device = "cuda" if _cuda_available() else "cpu"

        model = whisper.load_model(model_name, device=device)
        result = model.transcribe(audio_wav, language="en")

        subtitles = []
        for i, seg in enumerate(result["segments"]):
            subtitles.append(Sub(
                index=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))

        # Save English SRT to disk
        work_dir = os.path.join(".video-translate", state["video_title"])
        save_srt(subtitles, os.path.join(work_dir, "subtitles_en.srt"))

        return {
            "subtitles_en": subtitles,
            "stage": "translate",
        }

    except ImportError:
        return {
            "errors": [Error(
                stage="asr",
                message="whisper not installed. Run: pip install openai-whisper",
                retry_count=0,
            )],
            "stage": "asr",
        }
    except Exception as e:
        return {
            "errors": [Error(
                stage="asr",
                message=str(e),
                retry_count=0,
            )],
            "stage": "asr",
        }


def _cuda_available() -> bool:
    """Check if CUDA GPU is available. Returns False on macOS/CPU-only."""
    # torch can segfault on bleeding-edge Python — skip for safety on darwin
    import sys
    if sys.platform == "darwin":
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
