# Video-Translate

一键将英文单人讲课视频转换为中文配音视频（保留背景音）。

## 安装

```bash
# 安装 ffmpeg
brew install ffmpeg

# 安装 Python 依赖
pip install -r requirements.txt
```

## 用法

输出默认保存到 `~/Movies/`，命名规则为 `{原标题}_cn.mp4`。

```bash
# 本地文件 → ~/Movies/lecture_cn.mp4
python cli.py lecture.mp4

# URL 下载 → ~/Movies/{标题}_cn.mp4
python cli.py "https://youtube.com/watch?v=xxx"

# 指定输出路径
python cli.py lecture.mp4 -o ~/Desktop/result.mp4

# 断点续传
python cli.py lecture.mp4 --resume

# 不保留背景音
python cli.py lecture.mp4 --no-bgm
```
