"""Unit tests for pipeline nodes."""
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from state import PipelineState, make_initial_state, Sub, TSeg, Error
from nodes.extract_audio import extract_audio


class TestExtractAudio:
    def test_extract_audio_returns_state_with_audio_path(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            result = extract_audio(state)

            assert result["audio_wav"] != ""
            assert result["audio_wav"].endswith(".wav")
            assert result["stage"] == "asr"

    def test_extract_audio_calls_ffmpeg_with_correct_args(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            extract_audio(state)

            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "ffmpeg"
            assert "-ar" in call_args
            assert "16000" in call_args
            assert "-ac" in call_args
            assert "1" in call_args
            assert "-acodec" in call_args
            assert "pcm_s16le" in call_args
            assert "-vn" in call_args

    def test_extract_audio_file_not_found_raises(self):
        state = make_initial_state(input_path="/nonexistent/video.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            result = extract_audio(state)

            assert len(result["errors"]) == 1
            assert result["errors"][0]["stage"] == "extract"

    def test_extract_audio_ffmpeg_error_raises(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error")
            result = extract_audio(state)

            assert len(result["errors"]) == 1
            assert result["errors"][0]["stage"] == "extract"

    def test_extract_audio_skips_if_audio_wav_exists_and_not_force(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/tmp/existing.wav"

        with patch("os.path.exists", return_value=True), \
             patch("subprocess.run") as mock_run:
            result = extract_audio(state)

            assert result["audio_wav"] == "/tmp/existing.wav"
            mock_run.assert_not_called()

    def test_extract_audio_timeout_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)
            result = extract_audio(state)
            assert len(result["errors"]) == 1
            assert result["errors"][0]["stage"] == "extract"
