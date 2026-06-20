"""Node ③: Translate English subtitles to Chinese using DeepSeek API."""
import json
import os
import re
from state import PipelineState, Sub, Error

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "REPLACED_API_KEY")
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
        from langchain_openai import ChatOpenAI
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
                return {
                    "subtitles_cn": parsed,
                    "stage": "tts",
                }

            if attempt < MAX_RETRIES:
                continue

        except Exception as e:
            if attempt >= MAX_RETRIES:
                return {
                    "errors": [Error(
                        stage="translate",
                        message=f"Translation failed after {MAX_RETRIES} attempts: {e}",
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
