"""Tests for CLI argument parsing."""
import pytest
from cli import parse_args


class TestCli:
    def test_parse_args_local_file(self):
        args = parse_args(["lecture.mp4"])
        assert args.input == "lecture.mp4"
        assert args.output is None
        assert args.resume is False
        assert args.force is False
        assert args.no_bgm is False

    def test_parse_args_url(self):
        args = parse_args(["https://youtube.com/watch?v=abc"])
        assert args.input == "https://youtube.com/watch?v=abc"

    def test_parse_args_with_output(self):
        args = parse_args(["lecture.mp4", "-o", "output.mp4"])
        assert args.output == "output.mp4"

    def test_parse_args_resume(self):
        args = parse_args(["lecture.mp4", "--resume"])
        assert args.resume is True

    def test_parse_args_force(self):
        args = parse_args(["lecture.mp4", "--force"])
        assert args.force is True

    def test_parse_args_no_bgm(self):
        args = parse_args(["lecture.mp4", "--no-bgm"])
        assert args.no_bgm is True

    def test_parse_args_missing_input_exits(self):
        with pytest.raises(SystemExit):
            parse_args([])
