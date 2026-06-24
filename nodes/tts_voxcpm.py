"""Node ④-alt: VoxCPM2 TTS via HuggingFace Gradio API (batch pipeline).

Groups sentences into batches of ~10, sends each batch as one API call,
then splits proportionally and places on timeline. No Edge TTS fallback.

Usage:
    export TTS_ENGINE=voxcpm
"""
import os
import subprocess
import shutil
import time
from state import PipelineState, TSeg, Error
from nodes.tts_utils import fulltext_tts_pipeline

VOXCPM_SPACE = os.environ.get("VOXCPM_SPACE", "OpenBMB/VoxCPM-Demo")
VOXCPM_VOICE = os.environ.get(
    "VOXCPM_VOICE",
    "一位沉稳专业的男性播报员，声音清晰有力。"
    "语速均匀流畅，句间自然停顿，句内连贯无停顿无气口。"
)
BATCH_CHARS = int(os.environ.get("VOXCPM_BATCH_CHARS", "800"))

_client = None


def _get_client():
    """Lazy-load Gradio client with long HTTP timeout."""
    global _client
    if _client is None:
        from gradio_client import Client
        _client = Client(
            VOXCPM_SPACE,
            httpx_kwargs={"timeout": 180},  # 3 min — max our side can wait
        )
        print("   ✅ VoxCPM2 connected")
    return _client


def run_tts(state: PipelineState) -> dict:
    """Generate TTS via VoxCPM2 — batched fulltext (no Edge TTS fallback)."""
    if state.get("tts_segments"):
        return {"stage": "synthesis"}

    cn_subs = state.get("subtitles_cn", [])
    if not cn_subs:
        return {
            "errors": [Error(stage="tts",
                message="No Chinese subtitles found.", retry_count=0)],
            "stage": "tts",
        }

    work_dir = os.path.join(".video-translate", state["video_title"])

    # Group sentences into batches of ~500 chars
    all_segments = []
    all_errors = []

    batch = []
    batch_chars = 0

    for sub in cn_subs:
        batch.append(sub)
        batch_chars += len(sub["text"])

        if batch_chars >= BATCH_CHARS:
            segs, errs = fulltext_tts_pipeline(
                batch, work_dir, _generate_fulltext_voxcpm,
                label=f"VoxCPM2 batch {len(all_segments)//BATCH_CHARS + 1}"
            )
            all_segments.extend(segs)
            all_errors.extend(errs)
            batch = []
            batch_chars = 0

    # Last batch
    if batch:
        segs, errs = fulltext_tts_pipeline(
            batch, work_dir, _generate_fulltext_voxcpm, label="VoxCPM2 final"
        )
        all_segments.extend(segs)
        all_errors.extend(errs)

    typed_errors = [
        Error(stage=e["stage"], message=e["message"], retry_count=0)
        for e in all_errors
    ]

    if not all_segments:
        return {"errors": typed_errors, "stage": "tts"}

    cn_audio = os.path.join(work_dir, "cn_audio.wav")
    from nodes.tts import _build_timeline_sequential
    _build_timeline_sequential(all_segments, cn_audio)

    return {
        "tts_segments": all_segments,
        "cn_audio": cn_audio,
        "stage": "synthesis",
        "errors": typed_errors,
    }


def _generate_fulltext_voxcpm(text: str, output: str) -> bool:
    """Generate audio for batch of sentences via VoxCPM2 Gradio API.

    Falls back to per-sentence generation if batch is too long.
    """
    client = _get_client()

    for attempt in range(3):
        try:
            future = client.submit(
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
            result = future.result(timeout=180)  # wait up to 3 min
            if isinstance(result, str) and os.path.exists(result):
                shutil.copy(result, output)
                if os.path.getsize(output) >= 500:
                    return True
            if os.path.exists(output):
                os.remove(output)
            if attempt < 2:
                time.sleep(3.0)

        except Exception as e:
            print(f"   ⚠️ VoxCPM2 attempt {attempt+1}: {e}")
            if attempt < 2:
                time.sleep(5.0)

    # Batch failed — try per-sentence; this text IS a single sentence at this point
    # If the original text was already short, no point retrying
    if len(text) < 200:
        return False

    # Split on 。and try individually, then concat
    sentences = [s.strip() for s in text.split("。") if s.strip()]
    if len(sentences) <= 1:
        return False

    parts = []
    for i, s in enumerate(sentences):
        part_out = output + f".part{i}"
        if _generate_per_sentence(s + "。", part_out):
            parts.append(part_out)

    if not parts:
        return False

    # Concat parts
    concat_list = output + ".txt"
    with open(concat_list, "w") as f:
        for p in parts:
            f.write(f"file '{p}'\n")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list,
         "-c:a", "pcm_s16le", output],
        capture_output=True, text=True,
    )
    # Cleanup
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass
    try:
        os.remove(concat_list)
    except OSError:
        pass

    return os.path.exists(output) and os.path.getsize(output) >= 500


def _generate_per_sentence(text: str, output: str) -> bool:
    """Single-sentence VoxCPM2 call with long timeout."""
    client = _get_client()
    try:
        future = client.submit(
            text_input=text,
            control_instruction=VOXCPM_VOICE,
            reference_wav_path_input=None,
            use_prompt_text=False, prompt_text_input="",
            cfg_value_input=2.0, do_normalize=False, denoise=False,
            api_name="/generate",
        )
        result = future.result(timeout=180)
        if isinstance(result, str) and os.path.exists(result):
            shutil.copy(result, output)
            return os.path.getsize(output) >= 500
    except Exception:
        pass
    return False
