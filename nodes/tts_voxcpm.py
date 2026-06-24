"""Node ④-alt: VoxCPM2 TTS via HuggingFace Gradio API (full-text pipeline).

One API call for the entire text → split by character ratio → place on timeline.
~15-30s total instead of N×15s.

Usage:
    export TTS_ENGINE=voxcpm
    export VOXCPM_VOICE="一位沉稳专业的男性播报员，声音清晰有力"
"""
import os
import subprocess
import shutil
from state import PipelineState, TSeg, Error
from nodes.tts_utils import fulltext_tts_pipeline

VOXCPM_SPACE = os.environ.get("VOXCPM_SPACE", "OpenBMB/VoxCPM-Demo")
VOXCPM_VOICE = os.environ.get("VOXCPM_VOICE", "一位沉稳专业的男性播报员，声音清晰有力")

_client = None


def _get_client():
    global _client
    if _client is None:
        from gradio_client import Client
        _client = Client(VOXCPM_SPACE)
        print("   ✅ VoxCPM2 Gradio API connected")
    return _client


def run_tts(state: PipelineState) -> dict:
    """Generate Chinese TTS via VoxCPM2 — single API call for entire text."""
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
        cn_subs, work_dir, _generate_fulltext_voxcpm, label="VoxCPM2"
    )

    typed_errors = [
        Error(stage=e["stage"], message=e["message"], retry_count=0)
        for e in errors
    ]

    if not segments:
        return {"errors": typed_errors, "stage": "tts"}

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    from nodes.tts import _build_timeline_sequential
    _build_timeline_sequential(segments, cn_audio)

    return {
        "tts_segments": segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": typed_errors,
    }


def _generate_fulltext_voxcpm(text: str, output: str) -> bool:
    """Generate full-text audio via VoxCPM2 Gradio API."""
    client = _get_client()

    for attempt in range(2):
        try:
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
                import time
                time.sleep(3.0)
    return False
