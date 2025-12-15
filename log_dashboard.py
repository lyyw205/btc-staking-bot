from __future__ import annotations

import re
import subprocess
from datetime import datetime
from typing import List, Dict

from flask import Flask, jsonify, request, Response

APP_HOST = "127.0.0.1"
APP_PORT = 8080

SERVICE_NAME = "btc-bot"  # journalctl -u <service>

# "거래 관련"으로 보고 싶은 키워드(너 로그 스타일에 맞춰 추가/삭제 가능)
TRADE_RE = re.compile(r"\b(BUY|SELL|LONG|SHORT|ENTRY|EXIT|TP|SL)\b", re.IGNORECASE)

# "에러"로 보고 싶은 키워드
ERROR_RE = re.compile(
    r"(ERROR|Traceback|Exception|ModuleNotFoundError|Failed with result|status=\d+/FAILURE)",
    re.IGNORECASE,
)

app = Flask(__name__)

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>btc-bot dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 14px; }
    .row { display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
    input, select, button { padding:8px 10px; border-radius:10px; border:1px solid #3333; }
    button { cursor:pointer; }
    .meta { opacity:0.8; font-size: 13px; }
    pre { border:1px solid #3333; border-radius:12px; padding:12px; height: 82vh; overflow:auto; background:#0b0b0b; color:#e8e8e8; }
    .line { white-space: pre-wrap; word-break: break-word; }
    .trade { color: #8ef; }
    .err { color: #ff7b7b; font-weight: 600; }
    .dim { opacity: 0.75; }
  </style>
</head>
<body>
  <div class="row">
    <strong>btc-bot logs</strong>
    <span class="meta" id="meta">-</span>
  </div>

  <div class="row">
    <label>Lines
      <input id="lines" type="number" min="50" max="5000" value="500" style="width:90px;">
    </label>

    <label>Mode
      <select id="mode">
        <option value="all">All</option>
        <option value="trade">BUY/SELL only</option>
        <option value="error">Errors only</option>
      </select>
    </label>

    <label>Search
      <input id="q" placeholder="text contains..." style="width:260px;">
    </label>

    <label>Refresh
      <select id="refresh">
        <option value="2">2s</option>
        <option value="5" selected>5s</option>
        <option value="10">10s</option>
        <option value="30">30s</option>
        <option value="0">Off</option>
      </select>
    </label>

    <button onclick="loadLogs()">Refresh now</button>
  </div>

  <pre id="out"><span class="dim">Loading...</span></pre>

<script>
let timer = null;

function escHtml(s){
  return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

function setTimer(){
  if (timer) clearInterval(timer);
  const sec = parseInt(document.getElementById('refresh').value, 10);
  if (sec > 0) timer = setInterval(loadLogs, sec * 1000);
}

async function loadLogs(){
  const lines = document.getElementById('lines').value || 500;
  const mode  = document.getElementById('mode').value;
  const q     = document.getElementById('q').value || "";
  const url = `/api/logs?lines=${encodeURIComponent(lines)}&mode=${encodeURIComponent(mode)}&q=${encodeURIComponent(q)}`;

  const r = await fetch(url);
  const j = await r.json();

  document.getElementById('meta').textContent =
    `updated=${j.updated_at} | mode=${j.mode} | lines=${j.lines} | matched=${j.matched}`;

  const out = document.getElementById('out');
  out.innerHTML = j.items.map(it => {
    const cls = it.tag === "error" ? "line err" : (it.tag === "trade" ? "line trade" : "line");
    return `<div class="${cls}">${escHtml(it.text)}</div>`;
  }).join("");
}

document.getElementById('mode').addEventListener('change', loadLogs);
document.getElementById('refresh').addEventListener('change', () => { setTimer(); loadLogs(); });
document.getElementById('q').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadLogs(); });

setTimer();
loadLogs();
</script>
</body>
</html>
"""

def _journalctl(lines: int) -> str:
    cmd = [
        "journalctl",
        "-u", SERVICE_NAME,
        "-b",
        "-n", str(lines),
        "--no-pager",
        "-l",
        "-o", "short-iso",
    ]
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)

def _tag_line(s: str) -> str:
    if ERROR_RE.search(s):
        return "error"
    if TRADE_RE.search(s):
        return "trade"
    return "normal"

@app.get("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.get("/api/logs")
def api_logs():
    lines = int(request.args.get("lines", "500"))
    mode  = request.args.get("mode", "all").lower()
    q     = (request.args.get("q", "") or "").strip().lower()

    raw = _journalctl(lines)
    rows = [r for r in raw.splitlines() if r.strip()]

    items: List[Dict[str, str]] = []
    for r in rows:
        tag = _tag_line(r)

        if mode == "trade" and tag != "trade":
            continue
        if mode == "error" and tag != "error":
            continue
        if q and q not in r.lower():
            continue

        items.append({"text": r, "tag": tag})

    return jsonify({
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": mode,
        "lines": lines,
        "matched": len(items),
        "items": items,
    })

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
