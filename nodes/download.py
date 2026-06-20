"""Node ⓪: Download video from URL using yt-dlp with multi-level fallback."""
import os
import re
import subprocess
import glob as glob_module
import shutil
from state import PipelineState, Error


def download_video(state: PipelineState) -> dict:
    """Download video from a URL using yt-dlp.

    Reads: state["video_url"], state["video_title"]
    Writes: state["input_video"], state["video_title"], state["stage"], state["errors"]

    Strategy (3 levels):
      1. yt-dlp with ejs + deno + node (most videos)
      2. yt-dlp + browser cookies (login-required / age-restricted)
      3. Clear error message with manual download instructions
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

    output_tmpl = os.path.join(download_dir, "%(title)s.%(ext)s")

    # ---- Level 1: Standard yt-dlp ----
    result = _try_ytdlp(video_url, output_tmpl)

    if result.get("file"):
        return _finalize(result, download_dir)

    stderr = result.get("stderr", "")

    # ---- Level 2: yt-dlp + browser cookies ----
    if _needs_login(stderr):
        cookies_file = os.path.join(work_dir, "cookies.txt")
        for browser in ["chrome", "safari", "firefox", "edge"]:
            result = _try_ytdlp(video_url, output_tmpl, cookies_browser=browser, cookies_file=cookies_file)
            if result.get("file"):
                return _finalize(result, download_dir)

        # Cookies didn't help — specific error
        return {
            "errors": [Error(
                stage="download",
                message=(
                    "YouTube 要求登录验证，浏览器 cookies 无法绕过。\n"
                    "请尝试:\n"
                    "  1. 在浏览器中登录 YouTube 并播放该视频确认可访问\n"
                    "  2. 确保视频不是私享/地区限制/年龄限制\n"
                    "  3. 手动下载后使用: python cli.py <本地文件>\n"
                    "  4. 升级 yt-dlp: pip install -U yt-dlp"
                ),
                retry_count=0,
            )],
            "stage": "download",
        }

    # ---- Level 3: Other yt-dlp error ----
    return {
        "errors": [Error(
            stage="download",
            message=_format_error(stderr),
            retry_count=0,
        )],
        "stage": "download",
    }


def _try_ytdlp(url: str, output: str, cookies_browser: str = "", cookies_file: str = "") -> dict:
    """Run yt-dlp and return result dict with 'file' key if successful."""
    node_path = shutil.which("node") or ""
    deno_path = shutil.which("deno") or ""

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--restrict-filenames",
        "--remote-components", "ejs:github",
        "-f", "best[height<=720]",
        "--output", output,
    ]

    # JS runtimes for YouTube sig challenges
    if node_path:
        cmd += ["--js-runtimes", f"node:{node_path}"]
    if deno_path:
        cmd += ["--js-runtimes", f"deno:{deno_path}"]
    else:
        cmd += ["--js-runtimes", "deno"]

    # Browser cookies for login-required videos
    if cookies_browser:
        cmd += ["--cookies-from-browser", cookies_browser]
    if cookies_file:
        cmd += ["--cookies", cookies_file]

    cmd.append(url)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0:
            return {"file": True, "stderr": result.stderr}
        return {"file": None, "stderr": result.stderr}
    except FileNotFoundError:
        return {"file": None, "stderr": "yt-dlp not found"}
    except subprocess.TimeoutExpired:
        return {"file": None, "stderr": "timeout"}


def _finalize(result: dict, download_dir: str) -> dict:
    """Extract the downloaded file info and build return state."""
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

    base = os.path.splitext(os.path.basename(input_video))[0]
    video_title = base.replace("_", " ")
    safe_title = _sanitize_title(video_title)

    return {
        "input_video": input_video,
        "video_title": safe_title,
        "stage": "extract",
    }


def _needs_login(stderr: str) -> bool:
    """Check if yt-dlp error requires browser login."""
    indicators = [
        "Sign in to confirm",
        "confirm you're not a bot",
        "age-restricted",
        "age restricted",
        "sign in",
        "login required",
        "This video is private",
        "Members-only",
    ]
    return any(ind.lower() in stderr.lower() for ind in indicators)


def _format_error(stderr: str) -> str:
    """Format yt-dlp error message for user display."""
    if "JavaScript runtime" in stderr or "js-runtimes" in stderr:
        return ("yt-dlp 需要 JavaScript 运行时。\n"
                "安装: brew install node deno")
    if "HTTP Error 403" in stderr:
        return "YouTube 拒绝访问 (403)。可能需要登录或视频不可用。"
    if "HTTP Error 404" in stderr:
        return "视频未找到 (404)。请检查 URL 是否正确。"
    return f"yt-dlp failed: {stderr[:200]}"


def _sanitize_title(title: str) -> str:
    """Convert video title to a safe directory/filename prefix."""
    safe = re.sub(r'[<>:"/\\|?*]', '_', title)
    safe = safe.strip().rstrip('.')
    if len(safe) > 80:
        safe = safe[:77] + "..."
    return safe if safe else "video"
