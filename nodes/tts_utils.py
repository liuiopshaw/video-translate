"""Shared TTS utilities: full-text generation + proportional splitting."""
import os
import subprocess
from state import TSeg


def fulltext_tts_pipeline(
    cn_subs: list[dict],
    work_dir: str,
    generate_fn,  # callable(text, output_path) -> bool
    label: str = "TTS",
) -> tuple[list[dict], list[dict]]:
    """Single-call TTS pipeline: generate full audio, then split per sentence.

    Steps:
      1. Join all sentences into one text block.
      2. Generate one long audio file (1 API call instead of N).
      3. Split proportionally based on character counts.
      4. Normalize each segment.

    Returns (segments, errors).
    """
    errors = []

    # Step 1: Join all text
    full_text = "。".join(sub["text"] for sub in cn_subs) + "。"
    total_chars = sum(len(sub["text"]) for sub in cn_subs)

    # Step 2: Generate full audio
    full_audio = os.path.join(work_dir, "full_tts.wav")
    if not os.path.exists(full_audio) or os.path.getsize(full_audio) < 500:
        print(f"   🎤 {label}: generating full audio ({total_chars} chars)...")
        if not generate_fn(full_text, full_audio):
            return [], [{"stage": "tts", "message": f"{label} full-text generation failed"}]

    # Step 3: Get audio duration
    duration = _get_duration(full_audio)
    if duration <= 0:
        return [], [{"stage": "tts", "message": f"{label} audio has zero duration"}]

    # Step 4: Split proportionally
    tts_dir = os.path.join(work_dir, "tts_segments")
    norm_dir = os.path.join(work_dir, "tts_norm")
    os.makedirs(tts_dir, exist_ok=True)
    os.makedirs(norm_dir, exist_ok=True)

    segments = []
    cumulative = 0

    for sub in cn_subs:
        norm_path = os.path.join(norm_dir, f"{sub['index']:04d}.wav")

        if os.path.exists(norm_path) and os.path.getsize(norm_path) >= 500:
            segments.append(TSeg(
                index=sub["index"], start=sub["start"], end=sub["end"],
                wav_path=norm_path,
            ))
            cumulative += len(sub["text"])
            continue

        # Proportional split
        char_len = len(sub["text"])
        audio_start = (cumulative / total_chars) * duration if total_chars > 0 else 0
        audio_end = ((cumulative + char_len) / total_chars) * duration if total_chars > 0 else duration

        raw_path = os.path.join(tts_dir, f"{sub['index']:04d}.wav")
        _extract_segment(full_audio, raw_path, audio_start, audio_end)

        # Normalize
        if not _loudnorm(raw_path, norm_path):
            os.replace(raw_path, norm_path)

        segments.append(TSeg(
            index=sub["index"], start=sub["start"], end=sub["end"],
            wav_path=norm_path,
        ))
        cumulative += char_len

    return segments, errors


def _get_duration(path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=10,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return 0.0


def _extract_segment(src: str, dst: str, start: float, end: float) -> None:
    """Extract an audio segment by time range."""
    # Add small overlap for smooth transitions
    margin = 0.05
    start = max(0, start - margin)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src,
         "-ss", str(start), "-to", str(end + margin),
         "-c:a", "pcm_s16le", dst],
        capture_output=True, text=True, timeout=30,
    )


def _loudnorm(input_path: str, output_path: str) -> bool:
    """Normalize loudness and downsample to 24kHz mono."""
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path,
         "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:linear=true",
         "-ar", "24000", "-ac", "1",
         "-c:a", "pcm_s16le", output_path],
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) >= 500
