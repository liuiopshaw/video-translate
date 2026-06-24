"""LangGraph StateGraph definition for the video-translate pipeline.

Graph structure:

    download ──→ extract_audio ──→ asr ──→ translate ──→ tts ──→ synthesis ──→ merge ──→ END
        │              │              │         │            │         │             │
        └── conditional entry: skip if no URL
                       conditional exit: stop on hard error
"""

import os
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from state import PipelineState
from nodes.download import download_video
from nodes.extract_audio import extract_audio
from nodes.asr import run_asr
from nodes.translate import translate
from nodes.synthesis import synthesize_audio
from nodes.merge import merge_video

# Auto-select TTS engine
_TTS_ENGINE = os.environ.get("TTS_ENGINE", "edge")
if _TTS_ENGINE == "voxcpm":
    from nodes.tts_voxcpm import run_tts
else:
    from nodes.tts import run_tts


def build_graph() -> StateGraph:
    """Build and compile the video-translate StateGraph."""

    graph = StateGraph(PipelineState)

    # Add all nodes
    graph.add_node("download", download_video)
    graph.add_node("extract_audio", extract_audio)
    graph.add_node("asr", run_asr)
    graph.add_node("translate", translate)
    graph.add_node("tts", run_tts)
    graph.add_node("synthesis", synthesize_audio)
    graph.add_node("merge", merge_video)

    # Set entry point
    graph.set_entry_point("download")

    # Map from node name to the stage value it returns on success
    _expected_stages = {
        "download": "extract",
        "extract_audio": "asr",
        "asr": "translate",
        "translate": "tts",
        "tts": "synthesis",
        "synthesis": "merge",
        "merge": "done",
    }

    stages = ["download", "extract_audio", "asr", "translate", "tts", "synthesis", "merge"]
    next_stages = stages[1:] + ["done"]

    for stage, next_stage in zip(stages, next_stages):
        expected_output = _expected_stages[stage]
        target = END if next_stage == "done" else next_stage
        graph.add_conditional_edges(
            stage,
            lambda s, t=target, exp=expected_output: _route(s, t, exp),
            {"next": target, "__end__": END}
        )

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


def _route(state: PipelineState, target: str, expected_output: str) -> str:
    """Route to next stage or end on hard errors.

    Args:
        state: Current pipeline state.
        target: The next node name (or END for the final stage).
        expected_output: The stage value the current node sets on success.

    Returns:
        "next" to proceed to target, "__end__" to halt on error.
    """
    stage = state.get("stage", "")

    # Hard errors in early stages are fatal
    errors = state.get("errors", [])
    if errors and errors[-1]["stage"] in {"extract", "download"}:
        return "__end__"

    # Check if this node's output stage matches what we expect
    if stage == expected_output or (target == "__end__" and stage == "done"):
        return "next"

    return "__end__"
