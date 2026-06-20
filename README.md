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

```bash
# 本地文件
python cli.py lecture.mp4

# URL 下载
python cli.py "https://youtube.com/watch?v=xxx"

# 断点续传
python cli.py lecture.mp4 --resume

# 不保留背景音
python cli.py lecture.mp4 --no-bgm
```
