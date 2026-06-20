"""Pipeline state types for the video-translate agent."""
from typing import TypedDict


class Sub(TypedDict):
    """A single subtitle entry with timing."""
    index: int
    start: float
    end: float
    text: str


class TSeg(TypedDict):
    """A TTS audio segment with timing."""
    index: int
    start: float
    end: float
    wav_path: str


class Error(TypedDict):
    """An error record from pipeline execution."""
    stage: str
    message: str
    retry_count: int


class PipelineState(TypedDict, total=False):
    """Complete pipeline state passed between LangGraph nodes."""

    # === Input ===
    video_url: str
    input_video: str
    output_video: str
    keep_bgm: bool
    video_title: str

    # === Intermediate products ===
    audio_wav: str
    subtitles_en: list[Sub]
    subtitles_cn: list[Sub]
    tts_segments: list[TSeg]
    bgm_audio: str
    cn_audio: str
    cn_audio_mixed: str

    # === Metadata ===
    stage: str
    errors: list[Error]
    metadata: dict


def make_initial_state(
    input_path: str,
    output_path: str = "",
    resume: bool = False,
    force: bool = False,
    keep_bgm: bool = True,
) -> PipelineState:
    """Build the initial PipelineState from CLI arguments.

    Auto-detects whether input_path is a URL or local file.
    """
    import os

    is_url = input_path.startswith(("http://", "https://"))

    if is_url:
        video_url = input_path
        input_video = ""
        video_title = "video"
    else:
        video_url = ""
        input_video = input_path
        basename = os.path.splitext(os.path.basename(input_path))[0]
        video_title = basename

    if not output_path:
        output_path = f"{video_title}_cn.mp4"

    return PipelineState(
        video_url=video_url,
        input_video=input_video,
        output_video=output_path,
        keep_bgm=keep_bgm,
        video_title=video_title,
        # Intermediate products — empty initially
        audio_wav="",
        subtitles_en=[],
        subtitles_cn=[],
        tts_segments=[],
        bgm_audio="",
        cn_audio="",
        cn_audio_mixed="",
        # Metadata
        stage="download",
        errors=[],
        metadata={},
    )
