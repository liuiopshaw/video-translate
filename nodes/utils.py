"""Shared utilities for pipeline nodes."""
import os


def save_srt(subtitles: list[dict], path: str) -> None:
    """Save subtitle entries to an SRT file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sub in subtitles:
            f.write(f"{sub['index'] + 1}\n")
            f.write(f"{_fmt_time(sub['start'])} --> {_fmt_time(sub['end'])}\n")
            f.write(f"{sub['text']}\n\n")


def save_bilingual_srt(en: list[dict], cn: list[dict], path: str) -> None:
    """Save bilingual (EN+CN) subtitle entries to an SRT file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e, c in zip(en, cn):
            idx = e.get("index", 0) + 1
            start = _fmt_time(e.get("start", c.get("start", 0)))
            end = _fmt_time(e.get("end", c.get("end", 0)))
            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{e['text']}\n{c['text']}\n\n")


def _fmt_time(seconds: float) -> str:
    """Format seconds to SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
