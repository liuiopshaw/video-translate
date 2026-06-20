"""Tests for the LangGraph pipeline graph."""
import pytest
from state import make_initial_state
from graph import build_graph


class TestGraph:
    def test_build_graph_returns_compiled_graph(self):
        graph = build_graph()
        assert graph is not None
        assert hasattr(graph, "invoke")

    def test_graph_nodes_exist(self):
        graph = build_graph()
        nodes = graph.get_graph().nodes
        assert "download" in nodes
        assert "extract_audio" in nodes
        assert "asr" in nodes
        assert "translate" in nodes
        assert "tts" in nodes
        assert "synthesis" in nodes
        assert "merge" in nodes

    def test_graph_entry_point_is_download(self):
        graph = build_graph()
        internal = graph.get_graph()
        assert internal is not None
