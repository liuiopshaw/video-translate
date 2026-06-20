#!/usr/bin/env python3
"""CLI entry point for video-translate.

Usage:
    python cli.py lecture.mp4
    python cli.py "https://youtube.com/watch?v=xxx"
    python cli.py lecture.mp4 --resume
    python cli.py lecture.mp4 --no-bgm
"""

import argparse
import sys
from state import make_initial_state
from graph import build_graph


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="video-translate",
        description="Convert English lecture videos to Chinese-dubbed videos",
    )
    parser.add_argument(
        "input",
        help="Video file path or URL (YouTube, Bilibili, Coursera, etc.)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output video path (default: {title}_cn.mp4)",
        default=None,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run all stages",
    )
    parser.add_argument(
        "--no-bgm",
        action="store_true",
        help="Skip background music separation",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point. Returns exit code."""
    args = parse_args(argv)

    state = make_initial_state(
        input_path=args.input,
        output_path=args.output or "",
        keep_bgm=not args.no_bgm,
    )

    graph = build_graph()

    is_url = args.input.startswith(("http://", "https://"))
    config = {"configurable": {"thread_id": state["video_title"]}}

    print(f"🎬 Video-Translate Pipeline")
    print(f"   Input: {args.input}")
    print(f"   Output: {state['output_video']}")
    print(f"   Mode: {'URL download' if is_url else 'Local file'}")
    print(f"   BGM: {'Yes' if state['keep_bgm'] else 'No'}")
    print()

    try:
        if args.resume:
            snapshot = graph.get_state(config)
            if snapshot and snapshot.values:
                print("📂 Resuming from checkpoint...")
                result = graph.invoke(None, config)
            else:
                print("⚠️  No checkpoint found, starting fresh...")
                result = graph.invoke(dict(state), config)
        else:
            result = graph.invoke(dict(state), config)

        errors = result.get("errors", [])
        stage = result.get("stage", "unknown")

        if stage == "done":
            print(f"\n✅ Done! Output: {result['output_video']}")
        else:
            print(f"\n⚠️  Pipeline stopped at stage: {stage}")

        if errors:
            print(f"\n📋 {len(errors)} error(s):")
            for e in errors:
                print(f"   [{e['stage']}] {e['message']}")

        return 0 if stage == "done" else 1

    except KeyboardInterrupt:
        print("\n⏸️  Interrupted. Run with --resume to continue.")
        return 130
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
