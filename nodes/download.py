"""Node ⓪: Download video from URL using yt-dlp."""
import os
import re
import subprocess
import glob as glob_module
from state import PipelineState, Error


def download_video(state: PipelineState) -> dict:
    """Download video from a URL using yt-dlp.

    Reads: state["video_url"], state["video_title"]
    Writes: state["input_video"], state["video_title"], state["stage"], state["errors"]
    Skips if video_url is empty (local file mode).
    """
    video_url = state.get("video_url", "")
    if not video_url:
        return {"stage": "extract"}

    if state.get("input_video") and os.path.exists(state.get("input_video", "")):
        return {"stage": "extract"}

    safe_title = _sanitize_title(state["video_title"])
    work_dir = os.path.join(".video-translate", safe_title)
    download_dir = os.path.join(work_dir, "download")
    os.makedirs(download_dir, exist_ok=True)

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--no-playlist",
                "--restrict-filenames",
                "--js-runtimes", "node,deno",
                "--output", os.path.join(download_dir, "%(title)s.%(ext)s"),
                "--print", "title",
                video_url,
            ],
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            stderr = result.stderr
            msg = f"yt-dlp failed: {stderr[:200]}"
            if "JavaScript runtime" in stderr or "js-runtimes" in stderr:
                msg = ("yt-dlp 需要 JavaScript 运行时。\n"
                       "安装 Node.js: brew install node\n"
                       "原始错误: " + stderr[:150])
            return {
                "errors": [Error(
                    stage="download",
                    message=msg,
                    retry_count=0,
                )],
                "stage": "download",
            }

        output_lines = result.stdout.strip().split("\n")
        video_title = output_lines[0] if output_lines else state["video_title"]

        downloaded = glob_module.glob(os.path.join(download_dir, "*"))
        if not downloaded:
            return {
                "errors": [Error(
                    stage="download",
                    message="Download completed but no file found",
                    retry_count=0,
                )],
                "stage": "download",
            }

        downloaded.sort(key=os.path.getsize, reverse=True)
        input_video = downloaded[0]
        safe_title = _sanitize_title(video_title)

        return {
            "input_video": input_video,
            "video_title": safe_title,
            "stage": "extract",
        }

    except FileNotFoundError:
        return {
            "errors": [Error(
                stage="download",
                message="yt-dlp not found. Install with: pip install yt-dlp",
                retry_count=0,
            )],
            "stage": "download",
        }
    except subprocess.TimeoutExpired:
        return {
            "errors": [Error(
                stage="download",
                message="Download timed out after 30 minutes",
                retry_count=0,
            )],
            "stage": "download",
        }


def _sanitize_title(title: str) -> str:
    """Convert video title to a safe directory/filename prefix."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe = safe.strip().rstrip('.')
    if len(safe) > 80:
        safe = safe[:77] + "..."
    return safe if safe else "video"
