"""Node ③: Translate English subtitles to Chinese using DeepSeek API."""
import json
import logging
import os
import re
from state import PipelineState, Sub, Error
from nodes.utils import save_srt, save_bilingual_srt

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL_NAME = os.environ.get("TRANSLATE_MODEL", "deepseek-v4-pro")

TRANSLATE_SYSTEM_PROMPT = """You are a professional translator specializing in educational content.
Translate English subtitles to fluent Chinese (Simplified).
Rules:
- Use spoken/conversational Chinese suitable for a lecture audience
- Keep technical terms accurate and consistent
- Preserve original timing metadata unchanged
- Output ONLY valid JSON with the exact structure shown

Example input:
{"subtitles": [{"index": 0, "start": 0.0, "end": 2.5, "text": "Hello everyone"}]}

Example output:
{"subtitles": [{"index": 0, "start": 0.0, "end": 2.5, "text": "大家好"}]}
"""

MAX_RETRIES = 3


def translate(state: PipelineState, llm=None) -> dict:
    """Translate all English subtitles to Chinese in one API call.

    Reads: state["subtitles_en"]
    Writes: state["subtitles_cn"], state["stage"], state["errors"]

    Args:
        state: PipelineState with subtitles_en populated.
        llm: Optional pre-configured LLM instance (for testing).

    Skips if subtitles_cn already populated.
    Retries up to 3 times on format errors.
    """
    if state.get("subtitles_cn"):
        return {"stage": "tts"}

    en_subs = state.get("subtitles_en", [])
    if not en_subs:
        return {
            "errors": [Error(
                stage="translate",
                message="No English subtitles found. Run ASR first.",
                retry_count=0,
            )],
            "stage": "translate",
        }

    input_json = json.dumps({"subtitles": en_subs}, ensure_ascii=False)

    if llm is None:
        if not DEEPSEEK_API_KEY:
            return {
                "errors": [Error(
                    stage="translate",
                    message="DEEPSEEK_API_KEY not set. Export it or set in environment.",
                    retry_count=0,
                )],
                "stage": "translate",
            }
        if ChatOpenAI is None:
            return {
                "errors": [Error(
                    stage="translate",
                    message="langchain_openai is not installed. Run: pip install langchain-openai",
                    retry_count=0,
                )],
                "stage": "translate",
            }
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=0,
        )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = llm.invoke([
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": input_json},
            ])

            parsed = _parse_translation_response(response.content)

            if parsed and len(parsed) == len(en_subs):
                # Save Chinese and bilingual SRT to disk
                work_dir = os.path.join(".video-translate", state["video_title"])
                save_srt(parsed, os.path.join(work_dir, "subtitles_cn.srt"))
                save_bilingual_srt(en_subs, parsed, os.path.join(work_dir, "subtitles_bilingual.srt"))

                return {
                    "subtitles_cn": parsed,
                    "stage": "tts",
                }

            if attempt < MAX_RETRIES:
                logger.warning("Translate attempt %d/%d failed (format/parse error), retrying...", attempt, MAX_RETRIES)
                continue

        except Exception as e:
            if attempt < MAX_RETRIES and _is_retryable(e):
                logger.warning("Translate attempt %d/%d failed, retrying...", attempt, MAX_RETRIES)
                continue
            return {
                "errors": [Error(
                    stage="translate",
                    message=f"Translation failed after {attempt} attempt(s): {e}",
                    retry_count=attempt,
                )],
                "stage": "translate",
            }

    return {
        "errors": [Error(
            stage="translate",
            message=f"Translation failed: output count mismatch after {MAX_RETRIES} retries",
            retry_count=MAX_RETRIES,
        )],
        "stage": "translate",
    }


def _is_retryable(e: Exception) -> bool:
    """Check if an exception is a transient error worth retrying.

    Retryable: connection errors, timeouts, rate limits, server errors.
    Non-retryable: programming errors like TypeError, ValueError, ImportError.
    """
    retryable_substrings = (
        "timeout",
        "timed out",
        "connection",
        "reset by peer",
        "broken pipe",
        "rate limit",
        "too many requests",
        "429",
        "503",
        "502",
        "500",
        "server error",
        "service unavailable",
        "overloaded",
    )
    msg = str(e).lower()
    for substr in retryable_substrings:
        if substr in msg:
            return True
    # Also retry on generic OSError subclasses that indicate network issues
    if isinstance(e, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


def _parse_translation_response(content: str) -> list[Sub] | None:
    """Extract subtitle list from LLM response text."""
    try:
        data = json.loads(content)
        return data.get("subtitles", [])
    except json.JSONDecodeError:
        pass

    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(1))
            return data.get("subtitles", [])
        except json.JSONDecodeError:
            pass

    match = re.search(r'\{.*"subtitles".*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("subtitles", [])
        except json.JSONDecodeError:
            pass

    return None
