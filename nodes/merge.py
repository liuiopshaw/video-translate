"""Node ⑥: Merge Chinese audio with original video using ffmpeg."""
import os
import shutil
import subprocess
from state import PipelineState, Error


def merge_video(state: PipelineState) -> dict:
    """Combine original video stream with the mixed Chinese audio track.

    Reads: state["input_video"], state["cn_audio_mixed"], state["output_video"],
           state["video_title"], state["stage"]
    Writes: state["stage"], state["errors"]
    """
    if state.get("stage") == "done":
        return {"stage": "done"}

    cn_audio_mixed = state.get("cn_audio_mixed", "")
    if not cn_audio_mixed:
        return {
            "errors": [Error(
                stage="merge",
                message="No mixed Chinese audio found. Run synthesis first.",
                retry_count=0,
            )],
            "stage": "merge",
        }

    if not os.path.exists(cn_audio_mixed):
        return {
            "errors": [Error(
                stage="merge",
                message=f"Audio file not found: {cn_audio_mixed}",
                retry_count=0,
            )],
            "stage": "merge",
        }

    input_video = state["input_video"]
    output_video = state["output_video"]
    cn_srt_path = os.path.join(".video-translate", state["video_title"], "subtitles_cn.srt")

    try:
        cmd = ["ffmpeg", "-y", "-i", input_video, "-i", cn_audio_mixed]
        if os.path.exists(cn_srt_path):
            cmd += ["-i", cn_srt_path]
        cmd += ["-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0"]
        if os.path.exists(cn_srt_path):
            cmd += ["-map", "2:s:0", "-c:s", "mov_text"]
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest", output_video]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

        if result.returncode != 0:
            # Fallback: re-encode with libx264
            fallback_cmd = _build_fallback_cmd(input_video, cn_audio_mixed, cn_srt_path, output_video)
            result = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                return {
                    "errors": [Error(
                        stage="merge",
                        message=f"ffmpeg merge failed: {result.stderr[:200]}",
                        retry_count=0,
                    )],
                    "stage": "merge",
                }

        # Copy subtitle files alongside the output video
        _copy_subtitles(state)
        return {"stage": "done"}

    except FileNotFoundError:
        return {
            "errors": [Error(
                stage="merge",
                message="ffmpeg not found. Install with: brew install ffmpeg",
                retry_count=0,
            )],
            "stage": "merge",
        }
    except subprocess.TimeoutExpired:
        return {
            "errors": [Error(
                stage="merge",
                message="ffmpeg merge timed out after 600s",
                retry_count=0,
            )],
            "stage": "merge",
        }


def _copy_subtitles(state: PipelineState) -> None:
    """Copy SRT files from work dir to the output video's directory."""
    work_dir = os.path.join(".video-translate", state["video_title"])
    output_dir = os.path.dirname(os.path.abspath(state["output_video"]))
    if not output_dir:
        output_dir = "."

    for name in ["subtitles_en.srt", "subtitles_cn.srt", "subtitles_bilingual.srt"]:
        src = os.path.join(work_dir, name)
        if os.path.exists(src):
            # Derive output filename from video name
            base = os.path.splitext(os.path.basename(state["output_video"]))[0]
            dst = os.path.join(output_dir, f"{base}_{name}")
            shutil.copy2(src, dst)


def _build_fallback_cmd(input_video: str, audio: str, srt: str, output: str) -> list:
    """Build fallback ffmpeg command using libx264 encoding."""
    cmd = ["ffmpeg", "-y", "-i", input_video, "-i", audio]
    if srt and os.path.exists(srt):
        cmd += ["-i", srt]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-c:a", "aac", "-b:a", "192k",
            "-map", "0:v:0", "-map", "1:a:0"]
    if srt and os.path.exists(srt):
        cmd += ["-map", "2:s:0", "-c:s", "mov_text"]
    cmd += ["-shortest", output]
    return cmd
