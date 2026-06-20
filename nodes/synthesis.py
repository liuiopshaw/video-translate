"""Node ⑤: Separate background music and mix with Chinese TTS audio."""
import os
import subprocess
import glob
from state import PipelineState, Error


def synthesize_audio(state: PipelineState) -> dict:
    """Separate background music from original audio, then mix with TTS.

    Reads: state["audio_wav"], state["tts_segments"], state["cn_audio"],
           state["keep_bgm"], state["video_title"]
    Writes: state["bgm_audio"], state["cn_audio_mixed"], state["stage"], state["errors"]
    Gracefully degrades: if UVR fails, produces output without BGM.
    """
    if state.get("cn_audio_mixed"):
        return {"stage": "merge"}

    tts_segments = state.get("tts_segments", [])
    if not tts_segments:
        return {
            "errors": [Error(
                stage="synthesis",
                message="No TTS segments found. Run TTS first.",
                retry_count=0,
            )],
            "stage": "synthesis",
        }

    work_dir = os.path.join(".video-translate", state["video_title"])
    bgm_audio = ""
    errors = []

    if state.get("keep_bgm", True) and state.get("audio_wav"):
        try:
            bgm_audio = _separate_background(state["audio_wav"], work_dir)
        except Exception as e:
            errors.append(Error(
                stage="synthesis",
                message=f"UVR separation failed, output will have no background music: {e}",
                retry_count=0,
            ))

    cn_audio = state.get("cn_audio", "")
    mixed_path = os.path.join(work_dir, "cn_audio_mixed.wav")

    if bgm_audio and os.path.exists(bgm_audio) and cn_audio:
        _mix_audio(cn_audio, bgm_audio, mixed_path)
    elif cn_audio:
        mixed_path = cn_audio
    else:
        return {
            "errors": [Error(
                stage="synthesis",
                message="No Chinese audio to output",
                retry_count=0,
            )] + errors,
            "stage": "synthesis",
        }

    return {
        "bgm_audio": bgm_audio,
        "cn_audio_mixed": mixed_path,
        "stage": "merge",
        "errors": errors,
    }


def _separate_background(audio_path: str, work_dir: str) -> str:
    """Use Demucs to separate vocals from background music. Returns instrumental path."""
    output_dir = os.path.join(work_dir, "uvr_output")
    os.makedirs(output_dir, exist_ok=True)

    result = subprocess.run(
        ["python", "-m", "demucs", "--two-stems", "vocals", "-o", output_dir, audio_path],
        capture_output=True, text=True, timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Demucs failed: {result.stderr[:200]}")

    basename = os.path.splitext(os.path.basename(audio_path))[0]
    # Demucs outputs to: output_dir/model_name/basename/no_vocals.wav
    for pattern in [
        os.path.join(output_dir, "*", basename, "no_vocals.wav"),
        os.path.join(output_dir, "htdemucs", basename, "no_vocals.wav"),
    ]:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    raise RuntimeError("Could not find Demucs output")


def _mix_audio(speech_path: str, bgm_path: str, output_path: str) -> None:
    """Mix speech (100%) with background music (30%) using ffmpeg."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", speech_path, "-i", bgm_path,
         "-filter_complex",
         "[0:a]volume=1.0[speech];[1:a]volume=0.3[bgm];[speech][bgm]amix=inputs=2:duration=first",
         output_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Audio mixing failed: {result.stderr[:200]}")
