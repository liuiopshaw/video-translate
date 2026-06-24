"""Integration test: full pipeline with mocked externals."""
import sys
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from state import make_initial_state
from nodes.download import download_video
from nodes.extract_audio import extract_audio
from nodes.asr import run_asr
from nodes.translate import translate
from nodes.tts import run_tts
from nodes.synthesis import synthesize_audio
from nodes.merge import merge_video


class TestIntegration:
    def test_cli_module_imports(self):
        """Verify all modules import cleanly."""
        from state import PipelineState, make_initial_state, Sub, TSeg, Error
        from nodes.download import download_video
        from nodes.extract_audio import extract_audio
        from nodes.asr import run_asr
        from nodes.translate import translate
        from nodes.tts import run_tts
        from nodes.synthesis import synthesize_audio
        from nodes.merge import merge_video
        from graph import build_graph
        from cli import parse_args, main

    def test_parse_and_build_state(self):
        """Verify argument -> state -> graph pipeline."""
        from cli import parse_args
        from state import make_initial_state
        from graph import build_graph

        args = parse_args(["/tmp/test.mp4"])
        state = make_initial_state(input_path=args.input)
        assert state["video_url"] == ""
        assert state["input_video"] == "/tmp/test.mp4"

    def test_conditional_download_local_file(self):
        """Verify download node skips for local files."""
        state = make_initial_state(input_path="/tmp/test.mp4")
        result = download_video(state)
        assert result == {"stage": "extract"}

    def test_full_state_pipeline(self):
        """Walk through all 7 nodes with mocked externals."""
        state = make_initial_state(input_path="/tmp/test.mp4")

        # Step 0: download (skips for local file)
        state.update(download_video(state))
        assert state["stage"] == "extract"

        # Step 1: extract_audio
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            state.update(extract_audio(state))
        assert state["audio_wav"] != ""

        # Step 2: ASR
        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            with patch("os.path.exists", return_value=True), \
                 patch("whisper.load_model") as mock_model:
                mock_model.return_value.transcribe.return_value = {
                    "segments": [{"start": 0.0, "end": 2.0, "text": "Hello world"}]
                }
                state.update(run_asr(state))
        assert len(state["subtitles_en"]) == 1

        # Step 3: translate
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"subtitles": [{"index": 0, "start": 0.0, "end": 2.0, "text": "你好世界"}]}'
        )
        state.update(translate(state, llm=mock_llm))
        assert len(state["subtitles_cn"]) == 1

        # Step 4: TTS (fulltext pipeline)
        with patch("nodes.tts._generate_fulltext", return_value=True), \
             patch("nodes.tts_utils._get_duration", return_value=10.0), \
             patch("nodes.tts_utils._extract_segment"), \
             patch("nodes.tts_utils._loudnorm", return_value=True), \
             patch("nodes.tts._build_timeline_sequential"), \
             patch("os.makedirs"):
            state.update(run_tts(state))
        assert len(state["tts_segments"]) == 1

        # Step 5: synthesis
        with patch("subprocess.run") as mock:
            mock.return_value = MagicMock(returncode=0)
            state.update(synthesize_audio(state))
        assert state["cn_audio_mixed"] != ""

        # Step 6: merge
        with patch("subprocess.run") as mock, \
             patch("os.path.exists", return_value=True):
            mock.return_value = MagicMock(returncode=0)
            state.update(merge_video(state))
        assert state["stage"] == "done"

        print("\nFull pipeline walkthrough passed!")
