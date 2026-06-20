"""Tests for state types and serialization."""
import json
from state import Sub, TSeg, Error, PipelineState, make_initial_state


class TestSub:
    def test_sub_creation(self):
        sub = {"index": 0, "start": 1.5, "end": 4.2, "text": "Hello world"}
        assert sub["index"] == 0
        assert sub["start"] == 1.5
        assert sub["end"] == 4.2
        assert sub["text"] == "Hello world"

    def test_sub_defaults(self):
        # Sub has no defaults — all fields required
        pass


class TestPipelineState:
    def test_make_initial_state_with_local_file(self):
        state = make_initial_state(input_path="lecture.mp4")
        assert state["input_video"] == "lecture.mp4"
        assert state["video_url"] == ""
        assert state["video_title"] == "lecture"
        assert state["output_video"] == "lecture_cn.mp4"
        assert state["keep_bgm"] is True
        assert state["stage"] == "download"
        assert state["errors"] == []
        assert state["subtitles_en"] == []
        assert state["subtitles_cn"] == []
        assert state["tts_segments"] == []

    def test_make_initial_state_with_url(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc123")
        assert state["video_url"] == "https://youtube.com/watch?v=abc123"
        assert state["input_video"] == ""
        assert state["video_title"] == "video"
        assert state["stage"] == "download"

    def test_make_initial_state_custom_output(self):
        state = make_initial_state(input_path="lecture.mp4", output_path="custom.mp4", keep_bgm=False)
        assert state["output_video"] == "custom.mp4"
        assert state["keep_bgm"] is False

    def test_make_initial_state_with_mp4_extension(self):
        state = make_initial_state(input_path="/Users/foo/videos/my.lecture.mp4")
        assert state["video_title"] == "my.lecture"
        assert state["output_video"] == "my.lecture_cn.mp4"

    def test_state_is_serializable(self):
        state = make_initial_state(input_path="lecture.mp4")
        dumped = json.dumps(state, default=str)
        loaded = json.loads(dumped)
        assert loaded["input_video"] == "lecture.mp4"
        assert loaded["keep_bgm"] is True
        assert loaded["subtitles_en"] == []
