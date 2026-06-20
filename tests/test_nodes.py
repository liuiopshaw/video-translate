"""Unit tests for pipeline nodes."""
import sys
import pytest
import subprocess
from unittest.mock import patch, MagicMock
from state import PipelineState, make_initial_state, Sub, TSeg, Error
from nodes.extract_audio import extract_audio
from nodes.asr import run_asr
from nodes.translate import translate
from nodes.tts import run_tts
from nodes.synthesis import synthesize_audio
from nodes.download import download_video
from nodes.merge import merge_video


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

    def test_extract_audio_file_not_found_returns_error(self):
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

    def test_extract_audio_skips_if_audio_wav_exists(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        with patch("os.path.exists", return_value=True), \
             patch("subprocess.run") as mock_run:
            result = extract_audio(state)

            assert result["audio_wav"] == ".video-translate/lecture/audio.wav"
            mock_run.assert_not_called()

    def test_extract_audio_timeout_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)
            result = extract_audio(state)
            assert len(result["errors"]) == 1
            assert result["errors"][0]["stage"] == "extract"


class TestAsr:
    def test_run_asr_returns_subtitles(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/tmp/audio.wav"

        mock_whisper = MagicMock()
        with patch.dict(sys.modules, {"whisper": mock_whisper}):
            with patch("os.path.exists", return_value=True), \
                 patch("whisper.load_model") as mock_model:

                mock_segments = [
                    {"start": 0.0, "end": 2.5, "text": "Hello world"},
                    {"start": 2.5, "end": 5.0, "text": "Welcome to the lecture"},
                ]
                mock_model.return_value.transcribe.return_value = {"segments": mock_segments}

                result = run_asr(state)

        assert len(result["subtitles_en"]) == 2
        assert result["subtitles_en"][0]["text"] == "Hello world"
        assert result["subtitles_en"][0]["index"] == 0
        assert result["subtitles_en"][1]["text"] == "Welcome to the lecture"
        assert result["stage"] == "translate"

    def test_run_asr_skips_if_subtitles_exist(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/tmp/audio.wav"
        state["subtitles_en"] = [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "Existing"}
        ]

        result = run_asr(state)
        assert result == {"stage": "translate"}

    def test_run_asr_no_audio_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        result = run_asr(state)
        assert len(result["errors"]) == 1
        assert result["errors"][0]["stage"] == "asr"

    def test_run_asr_audio_not_found_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/nonexistent/audio.wav"

        with patch("os.path.exists", return_value=False):
            result = run_asr(state)
        assert len(result["errors"]) == 1


class TestTranslate:
    def test_translate_returns_chinese_subtitles(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_en"] = [
            {"index": 0, "start": 0.0, "end": 2.5, "text": "Hello world"},
            {"index": 1, "start": 2.5, "end": 5.0, "text": "Machine learning is fascinating"},
        ]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = (
            '{"subtitles": ['
            '{"index": 0, "start": 0.0, "end": 2.5, "text": "大家好"},'
            '{"index": 1, "start": 2.5, "end": 5.0, "text": "机器学习非常迷人"}'
            ']}'
        )
        mock_llm.invoke.return_value = mock_response

        result = translate(state, llm=mock_llm)

        assert len(result["subtitles_cn"]) == 2
        assert result["subtitles_cn"][0]["text"] == "大家好"
        assert result["subtitles_cn"][1]["text"] == "机器学习非常迷人"
        assert result["stage"] == "tts"

    def test_translate_skips_if_cn_subtitles_exist(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_en"] = [{"index": 0, "start": 0.0, "end": 1.0, "text": "Hello"}]
        state["subtitles_cn"] = [{"index": 0, "start": 0.0, "end": 1.0, "text": "你好"}]

        mock_llm = MagicMock()
        result = translate(state, llm=mock_llm)
        assert result == {"stage": "tts"}

    def test_translate_empty_en_subtitles_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        mock_llm = MagicMock()
        result = translate(state, llm=mock_llm)
        assert len(result["errors"]) == 1
        assert result["errors"][0]["stage"] == "translate"

    def test_translate_llm_retry_on_format_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_en"] = [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "Hello"},
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            MagicMock(content="not valid json"),
            MagicMock(content='{"subtitles": [{"index": 0, "start": 0.0, "end": 1.0, "text": "你好"}]}'),
        ]

        result = translate(state, llm=mock_llm)
        assert len(result["subtitles_cn"]) == 1
        assert mock_llm.invoke.call_count == 2


class TestTts:
    def test_run_tts_generates_segments(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_cn"] = [
            {"index": 0, "start": 0.0, "end": 2.5, "text": "大家好"},
            {"index": 1, "start": 2.5, "end": 5.0, "text": "欢迎来到课程"},
        ]

        with patch("subprocess.run") as mock_run, \
             patch("os.makedirs"), \
             patch("os.path.exists", side_effect=[False, True, False, True]), \
             patch("nodes.tts._place_on_timeline"):
            mock_run.return_value = MagicMock(returncode=0)

            result = run_tts(state)

            assert len(result["tts_segments"]) == 2
            assert result["tts_segments"][0]["index"] == 0
            assert result["tts_segments"][0]["start"] == 0.0
            assert result["tts_segments"][0]["end"] == 2.5
            assert result["tts_segments"][0]["wav_path"].endswith(".wav")
            assert result["stage"] == "synthesis"
            assert mock_run.call_count == 2  # one per segment

    def test_run_tts_skips_if_tts_segments_exist(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_cn"] = [{"index": 0, "start": 0.0, "end": 1.0, "text": "你好"}]
        state["tts_segments"] = [{"index": 0, "start": 0.0, "end": 1.0, "wav_path": "/tmp/0.wav"}]

        result = run_tts(state)
        assert result == {"stage": "synthesis"}

    def test_run_tts_skip_on_failure_continues(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_cn"] = [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "第一句"},
            {"index": 1, "start": 1.0, "end": 2.0, "text": "第二句"},
        ]

        with patch("subprocess.run") as mock_run, \
             patch("os.path.exists", side_effect=[False, False, False, True]) as mock_exists, \
             patch("nodes.tts._place_on_timeline"):
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr="Error"),
                MagicMock(returncode=0),
            ]

            result = run_tts(state)

            assert len(result["tts_segments"]) == 1
            assert result["tts_segments"][0]["index"] == 1
            assert len(result["errors"]) == 1

    def test_run_tts_empty_cn_subtitles_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        result = run_tts(state)
        assert len(result["errors"]) == 1

    def test_run_tts_uses_chinese_voice(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["subtitles_cn"] = [
            {"index": 0, "start": 0.0, "end": 1.0, "text": "测试"},
        ]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            run_tts(state)

            call_args = mock_run.call_args[0][0]
            assert "--voice" in call_args
            voice_idx = call_args.index("--voice")
            assert "zh-CN" in call_args[voice_idx + 1]
            assert "--rate" in call_args
            rate_idx = call_args.index("--rate")
            assert "+15%" in call_args[rate_idx + 1]


class TestSynthesis:
    def test_synthesize_mixes_bgm_with_tts(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/tmp/audio.wav"
        state["tts_segments"] = [
            {"index": 0, "start": 0.0, "end": 2.0, "wav_path": "/tmp/tts/0000.wav"},
        ]
        state["cn_audio"] = "/tmp/cn_audio.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = synthesize_audio(state)

            assert "cn_audio_mixed" in result
            assert result["stage"] == "merge"

    def test_synthesize_no_bgm_flag_skips_separation(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4", keep_bgm=False)
        state["audio_wav"] = "/tmp/audio.wav"
        state["tts_segments"] = [
            {"index": 0, "start": 0.0, "end": 2.0, "wav_path": "/tmp/tts/0000.wav"},
        ]
        state["cn_audio"] = "/tmp/cn_audio.wav"

        result = synthesize_audio(state)

        assert result["bgm_audio"] == ""
        assert result["cn_audio_mixed"] == state["cn_audio"]

    def test_synthesize_uvr_failure_degrades_gracefully(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["audio_wav"] = "/tmp/audio.wav"
        state["tts_segments"] = [
            {"index": 0, "start": 0.0, "end": 2.0, "wav_path": "/tmp/tts/0000.wav"},
        ]
        state["cn_audio"] = "/tmp/cn_audio.wav"

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = [
                MagicMock(returncode=1, stderr="UVR error"),
                MagicMock(returncode=0),
            ]
            result = synthesize_audio(state)

            assert "cn_audio_mixed" in result
            assert result["errors"][0]["stage"] == "synthesis"

    def test_synthesize_skips_if_already_done(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["tts_segments"] = [{"index": 0, "start": 0.0, "end": 1.0, "wav_path": "/tmp/0.wav"}]
        state["cn_audio_mixed"] = "/tmp/already_mixed.wav"

        result = synthesize_audio(state)
        assert result == {"stage": "merge"}

    def test_synthesize_no_tts_segments_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")

        result = synthesize_audio(state)
        assert len(result["errors"]) == 1


class TestMerge:
    def test_merge_produces_output_video(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["cn_audio_mixed"] = "/tmp/cn_audio_mixed.wav"

        with patch("subprocess.run") as mock_run, \
             patch("os.path.exists", side_effect=lambda p: p in ("/tmp/cn_audio_mixed.wav",)):
            mock_run.return_value = MagicMock(returncode=0)
            result = merge_video(state)

            assert result["stage"] == "done"
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "ffmpeg"
            assert "-map" in call_args

    def test_merge_no_audio_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        result = merge_video(state)
        assert len(result["errors"]) == 1

    def test_merge_ffmpeg_timeout_returns_error(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["cn_audio_mixed"] = "/tmp/cn_audio_mixed.wav"

        with patch("subprocess.run") as mock_run, \
             patch("os.path.exists", return_value=True):
            mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 600)
            result = merge_video(state)
            assert len(result["errors"]) == 1

    def test_merge_skips_if_done(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        state["cn_audio_mixed"] = "/tmp/mixed.wav"
        state["stage"] = "done"
        result = merge_video(state)
        assert result == {"stage": "done"}


class TestDownload:
    def test_download_from_url(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc123")
        state["video_title"] = "video"

        with patch("subprocess.run") as mock_run, \
             patch("glob.glob", return_value=["/tmp/My_Lecture_Title.mp4"]), \
             patch("os.path.getsize", return_value=1024):
            mock_run.return_value = MagicMock(returncode=0)
            result = download_video(state)

            assert result["input_video"] == "/tmp/My_Lecture_Title.mp4"
            assert result["video_title"] != "video"
            assert result["stage"] == "extract"

    def test_download_skips_if_url_empty(self):
        state = make_initial_state(input_path="/tmp/lecture.mp4")
        assert state["video_url"] == ""
        result = download_video(state)
        assert result == {"stage": "extract"}

    def test_download_skips_if_already_downloaded(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc")
        state["input_video"] = "/tmp/existing.mp4"
        with patch("os.path.exists", return_value=True):
            result = download_video(state)
            assert result == {"stage": "extract"}

    def test_download_failure_returns_error(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Video unavailable")
            result = download_video(state)
            assert len(result["errors"]) == 1

    def test_download_output_dir_created(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc123")
        with patch("subprocess.run") as mock_run, \
             patch("os.makedirs") as mock_mkdir, \
             patch("glob.glob", return_value=["/tmp/video.mp4"]), \
             patch("os.path.getsize", return_value=1024):
            mock_run.return_value = MagicMock(returncode=0)
            download_video(state)
            mock_mkdir.assert_called()

    def test_download_login_required_fallback_fails(self):
        state = make_initial_state(input_path="https://youtube.com/watch?v=abc")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="Sign in to confirm you're not a bot"
            )
            result = download_video(state)
            assert len(result["errors"]) == 1
            assert "登录" in result["errors"][0]["message"]
