#!/usr/bin/env python3
"""Video-Translate Web GUI — launch with: python webui.py"""
import json
import sys
import os
import subprocess
import threading
import queue
import time
from flask import Flask, request, jsonify, render_template_string
from state import make_initial_state

app = Flask(__name__)
_pipeline_queue = queue.Queue()
_pipeline_running = False
_pipeline_state = {"stage": "idle", "logs": [], "errors": []}

HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Video-Translate</title>
<style>
  :root {
    --pink: #EC4899;
    --blue: #3B82F6;
    --bg: #0F172A;
    --surface: #1E293B;
    --surface2: #334155;
    --text: #F1F5F9;
    --text2: #94A3B8;
    --green: #22C55E;
    --red: #EF4444;
    --amber: #F59E0B;
    --radius: 12px;
  }
  * { box-sizing:border-box;margin:0;padding:0 }
  body {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    padding: 24px;
  }
  .app { width: 100%; max-width: 720px; }
  header {
    text-align: center;
    padding: 32px 0 24px;
  }
  h1 {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    background: linear-gradient(135deg, var(--pink), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
  }
  .subtitle { font-size: 14px; color: var(--text2); }
  .card {
    background: var(--surface);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: var(--radius);
    padding: 24px;
    margin-bottom: 16px;
  }
  .card h2 {
    font-size: 15px; font-weight: 600;
    margin-bottom: 16px;
    color: var(--text);
    display: flex; align-items: center; gap: 8px;
  }
  .card h2 .dot { width:8px;height:8px;border-radius:50%;background:var(--pink) }

  .form-row { display:flex; gap:12px; margin-bottom:12px; }
  .form-row:last-child { margin-bottom:0 }
  @media (max-width:500px) { .form-row { flex-direction:column } }

  input[type="text"], input[type="url"] {
    flex:1; padding: 10px 14px;
    background: var(--surface2);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    color: var(--text);
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border .2s;
  }
  input[type="text"]:focus, input[type="url"]:focus {
    border-color: var(--blue);
  }
  input::placeholder { color: var(--text2); opacity: .6 }
  .hint { font-size: 11px; color: var(--text2); margin-top: 4px; }

  .toggle-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0;
    font-size: 13px; color: var(--text2);
    cursor: pointer; user-select: none;
  }
  .toggle {
    width:40px;height:22px;border-radius:11px;
    background: var(--surface2);
    position:relative; cursor:pointer;
    transition: background .2s;
    flex-shrink: 0;
  }
  .toggle.on { background: var(--blue) }
  .toggle::after {
    content:''; position:absolute;
    top:2px;left:2px;
    width:18px;height:18px;border-radius:50%;
    background:#fff;
    transition: transform .2s;
  }
  .toggle.on::after { transform: translateX(18px) }

  .checkbox-row {
    display:flex;align-items:center;gap:10px;
    font-size:13px;color:var(--text2);cursor:pointer;user-select:none;
  }
  .checkbox-row input { display:none }
  .checkmark {
    width:18px;height:18px;border-radius:4px;
    border:2px solid var(--surface2);
    display:flex;align-items:center;justify-content:center;
    transition: all .2s; flex-shrink:0;
  }
  .checkbox-row input:checked + .checkmark {
    background: var(--blue); border-color: var(--blue);
  }
  .checkbox-row input:checked + .checkmark::after {
    content:'✓';color:#fff;font-size:11px;font-weight:700;
  }

  .btn {
    width:100%; padding:12px 24px;
    border:none; border-radius: 8px;
    font-size:15px;font-weight:600;font-family:inherit;
    cursor:pointer;
    transition: all .2s;
    letter-spacing: -0.2px;
  }
  .btn-primary {
    background: linear-gradient(135deg, var(--pink), var(--blue));
    color:#fff;
  }
  .btn-primary:hover { opacity: .9; transform: translateY(-1px); }
  .btn-primary:active { transform: translateY(0) }
  .btn-primary:disabled {
    opacity:.5; cursor:not-allowed; transform:none;
  }
  .btn-secondary {
    background: var(--surface2); color: var(--text);
    margin-top: 8px;
  }

  .pipeline { display: flex; gap: 6px; margin: 16px 0; flex-wrap: wrap; }
  .stage-pill {
    padding: 6px 12px; border-radius: 20px;
    font-size: 11px; font-weight: 600;
    background: var(--surface2);
    color: var(--text2);
    white-space: nowrap;
    transition: all .3s;
  }
  .stage-pill.active { background: var(--blue); color: #fff }
  .stage-pill.done { background: #166534; color: var(--green) }
  .stage-pill.error { background: #7F1D1D; color: var(--red) }

  .log-area {
    background: #0B0F19;
    border-radius: 8px;
    padding: 16px;
    max-height: 300px;
    overflow-y: auto;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.7;
    color: var(--text2);
    display: none;
  }
  .log-area.visible { display: block }
  .log-line { padding: 1px 0 }
  .log-line.info { color: var(--text2) }
  .log-line.stage { color: var(--blue) }
  .log-line.done { color: var(--green) }
  .log-line.error { color: var(--red) }
  .log-line.warn { color: var(--amber) }

  .result-banner {
    display: none;
    padding: 16px; border-radius: 8px;
    margin-top: 12px;
    font-size: 14px; font-weight: 600;
  }
  .result-banner.success { display:block; background:#166534; color:var(--green) }
  .result-banner.error { display:block; background:#7F1D1D; color:var(--red) }

  footer {
    text-align:center; padding: 24px 0;
    font-size:12px; color:var(--text2); opacity:.5;
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <h1>&#x1f3ac; Video-Translate</h1>
    <p class="subtitle">英文视频 → 中文配音 · 7 阶段 AI 管线 · 一键完成</p>
  </header>

  <div class="card">
    <h2><span class="dot"></span>输入</h2>
    <div class="form-row">
      <input type="url" id="input" placeholder="本地文件路径 或 https://youtube.com/watch?v=..." autofocus>
    </div>
    <div class="form-row">
      <input type="text" id="output" placeholder="输出文件名（可选，默认 {标题}_cn.mp4）">
    </div>
    <div class="hint">支持: YouTube / Bilibili / Coursera 等数百个站点 · 本地 MP4/MKV</div>
  </div>

  <div class="card">
    <h2><span class="dot"></span>选项</h2>
    <div class="toggle-row" onclick="document.getElementById('no_bgm').click()">
      <label style="flex:1">保留背景音（BGM）</label>
      <div class="toggle on" id="bgm_toggle"></div>
      <input type="checkbox" id="no_bgm" onchange="toggleBgm()" checked hidden>
    </div>
    <div class="checkbox-row" onclick="document.getElementById('resume').click()">
      <input type="checkbox" id="resume">
      <div class="checkmark"></div>
      <label>从上次中断恢复（--resume）</label>
    </div>
    <div class="checkbox-row" onclick="document.getElementById('force').click()">
      <input type="checkbox" id="force">
      <div class="checkmark"></div>
      <label>强制重跑所有阶段（--force）</label>
    </div>
  </div>

  <button class="btn btn-primary" id="startBtn" onclick="startPipeline()">
    &#9654; 开始转换
  </button>

  <div class="card" id="progressCard" style="display:none">
    <h2><span class="dot"></span>管线状态</h2>
    <div class="pipeline">
      <span class="stage-pill" id="s0">&#x2193; 下载</span>
      <span class="stage-pill" id="s1">&#x2460; 提取音频</span>
      <span class="stage-pill" id="s2">&#x2461; 语音识别</span>
      <span class="stage-pill" id="s3">&#x2462; 翻译字幕</span>
      <span class="stage-pill" id="s4">&#x2463; TTS配音</span>
      <span class="stage-pill" id="s5">&#x2464; 背景音混音</span>
      <span class="stage-pill" id="s6">&#x2465; 合成视频</span>
    </div>
    <div class="log-area visible" id="logArea"></div>
    <div class="result-banner" id="resultBanner"></div>
  </div>

  <footer>
    LangGraph · WhisperX · DeepSeek · Edge TTS · UVR · yt-dlp · ffmpeg
  </footer>
</div>

<script>
const STAGES = ['download','extract','asr','translate','tts','synthesis','merge','done'];
const STAGE_NAMES = ['下载视频','提取音频','语音识别','翻译字幕','TTS配音','混音','合成视频','完成'];

function toggleBgm() {
  const c = document.getElementById('no_bgm');
  c.checked = !c.checked;
  document.getElementById('bgm_toggle').classList.toggle('on', c.checked);
}

async function startPipeline() {
  const input = document.getElementById('input').value.trim();
  if (!input) return alert('请输入视频路径或URL');

  const btn = document.getElementById('startBtn');
  btn.disabled = true;
  btn.textContent = '运行中...';

  const card = document.getElementById('progressCard');
  card.style.display = 'block';
  document.getElementById('resultBanner').className = 'result-banner';
  document.getElementById('resultBanner').textContent = '';

  const logArea = document.getElementById('logArea');
  logArea.innerHTML = '';

  // Reset stage pills
  STAGES.forEach((s,i) => {
    const el = document.getElementById('s'+i);
    if (el) { el.className='stage-pill'; }
  });

  const body = {
    input: input,
    output: document.getElementById('output').value.trim() || '',
    no_bgm: !document.getElementById('no_bgm').checked,
    resume: document.getElementById('resume').checked,
    force: document.getElementById('force').checked,
  };

  function log(level, msg) {
    const cls = level==='stage'?'stage':level==='done'?'done':level==='error'?'error':level==='warn'?'warn':'info';
    logArea.innerHTML += `<div class="log-line ${cls}">${msg}</div>`;
    logArea.scrollTop = logArea.scrollHeight;
  }

  function setStage(name) {
    const idx = STAGES.indexOf(name);
    if (idx < 0) return;
    STAGES.forEach((s,i) => {
      const el = document.getElementById('s'+i);
      if (!el) return;
      if (i < idx) el.className = 'stage-pill done';
      else if (i === idx) el.className = 'stage-pill active';
      else el.className = 'stage-pill';
    });
  }

  try {
    // Initialize pipeline
    log('info', '&#x1f680; 初始化管线...');
    const initRes = await fetch('/api/init', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    });
    const initData = await initRes.json();
    if (initData.error) { log('error', initData.error); throw new Error(initData.error); }
    log('info', '&#x2705; 管线就绪: ' + initData.mode);
    log('info', '&#x1f4fa; 输出: ' + initData.output);

    // Stream stages
    const streamRes = await fetch('/api/run', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({state:initData.state})
    });

    const reader = streamRes.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const evt = JSON.parse(line);
          if (evt.stage) {
            setStage(evt.stage);
            const idx = STAGES.indexOf(evt.stage);
            const name = idx>=0 ? STAGE_NAMES[idx] : evt.stage;
            log('stage', '&#x25b6; ' + name);
          }
          if (evt.log) log(evt.level||'info', evt.log);
          if (evt.done) {
            const banner = document.getElementById('resultBanner');
            if (evt.success) {
              banner.className = 'result-banner success';
              banner.textContent = '&#x2705; 完成! 输出: ' + evt.output;
              log('done', '&#x2705; 转换完成! ' + evt.output);
            } else {
              banner.className = 'result-banner error';
              banner.textContent = '&#x26a0; 管线在 ' + evt.stage + ' 阶段停止';
              log('error', '&#x26a0; 管线在 ' + evt.stage + ' 阶段停止');
            }
          }
        } catch(e) { /* skip malformed line */ }
      }
    }
  } catch(e) {
    log('error', '&#x274c; ' + e.message);
    document.getElementById('resultBanner').className = 'result-banner error';
    document.getElementById('resultBanner').textContent = '&#x274c; 错误: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = '&#9654; 重新开始';
  }
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/init", methods=["POST"])
def api_init():
    """Initialize pipeline state and return it."""
    data = request.get_json()
    input_path = data.get("input", "")
    if not input_path:
        return jsonify({"error": "请输入视频路径或URL"}), 400

    state = make_initial_state(
        input_path=input_path,
        output_path=data.get("output", ""),
        keep_bgm=data.get("no_bgm", True),
    )

    is_url = input_path.startswith(("http://", "https://"))
    mode = "URL 下载模式" if is_url else "本地文件模式"

    return jsonify({
        "mode": mode,
        "output": state["output_video"],
        "state": dict(state),
    })


@app.route("/api/run", methods=["POST"])
def api_run():
    """Run the pipeline and stream progress."""
    data = request.get_json()
    raw_state = data.get("state", {})

    def generate():
        from graph import build_graph

        graph = build_graph()
        config = {"configurable": {"thread_id": raw_state.get("video_title", "webui")}}

        stage_order = ["download", "extract", "asr", "translate", "tts", "synthesis", "merge"]
        stage_names = {
            "download": "下载视频",
            "extract": "提取音频",
            "asr": "语音识别",
            "translate": "翻译字幕",
            "tts": "TTS配音",
            "synthesis": "混音",
            "merge": "合成视频",
        }

        try:
            current_state = dict(raw_state)
            for stage in stage_order:
                # Yield stage start
                yield json.dumps({
                    "stage": stage,
                    "log": f"▶ {stage_names.get(stage, stage)}...",
                    "level": "stage",
                }, default=str) + "\n"

                try:
                    result = graph.invoke(current_state, config)

                    # Update current state
                    current_state.update(result)

                    # Check for errors
                    errors = result.get("errors", [])
                    new_errors = [e for e in errors if isinstance(e, dict)]

                    if new_errors:
                        for e in new_errors:
                            yield json.dumps({
                                "log": f"⚠ [{e.get('stage','?')}] {e.get('message','')}",
                                "level": "warn",
                            }, default=str) + "\n"

                    # Check if stage completed
                    result_stage = result.get("stage", "")
                    if result_stage == "done":
                        yield json.dumps({
                            "log": "✅ 全部完成!",
                            "level": "done",
                        }, default=str) + "\n"
                        yield json.dumps({
                            "done": True,
                            "success": True,
                            "output": result.get("output_video", raw_state.get("output_video", "")),
                        }, default=str) + "\n"
                        return

                except Exception as node_error:
                    yield json.dumps({
                        "log": f"❌ {stage} 阶段失败: {node_error}",
                        "level": "error",
                    }, default=str) + "\n"
                    yield json.dumps({
                        "done": True,
                        "success": False,
                        "stage": stage,
                    }, default=str) + "\n"
                    return

            # If all stages ran but not "done"
            yield json.dumps({
                "log": "⚠ 管线未到达完成状态",
                "level": "warn",
            }, default=str) + "\n"
            yield json.dumps({
                "done": True,
                "success": result.get("stage") == "done",
                "output": result.get("output_video", ""),
            }, default=str) + "\n"

        except Exception as e:
            yield json.dumps({
                "log": f"❌ 管线异常: {e}",
                "level": "error",
            }, default=str) + "\n"
            yield json.dumps({
                "done": True,
                "success": False,
            }, default=str) + "\n"

    return app.response_class(
        generate(),
        mimetype="text/plain",
        headers={"X-Accel-Buffering": "no"},
    )


def main():
    """Start the web server and open browser."""
    import webbrowser
    port = 8080

    print(f"""
╔══════════════════════════════════════════════════╗
║  🎬  Video-Translate Web GUI                    ║
║                                                  ║
║  打开浏览器访问: http://localhost:{port}             ║
║  按 Ctrl+C 停止服务器                             ║
╚══════════════════════════════════════════════════╝
    """)

    # Open browser after a small delay
    threading.Timer(1.0, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
