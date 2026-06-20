"""Node ②: Speech recognition using WhisperX."""
import os
from state import PipelineState, Sub, Error


def run_asr(state: PipelineState) -> dict:
    """Run WhisperX large-v3 on extracted audio, producing timed subtitles.

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
        import whisperx

        device = "cuda" if _cuda_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        audio = whisperx.load_audio(audio_wav)
        result = model.transcribe(audio, batch_size=16)

        model_a, metadata = whisperx.load_align_model(
            language_code="en", device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )

        subtitles = []
        for i, seg in enumerate(result["segments"]):
            subtitles.append(Sub(
                index=i,
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))

        return {
            "subtitles_en": subtitles,
            "stage": "translate",
        }

    except ImportError:
        return {
            "errors": [Error(
                stage="asr",
                message="whisperx not installed. Run: pip install whisperx",
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
    """Check if CUDA GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
