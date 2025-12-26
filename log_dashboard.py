from __future__ import annotations

import re
import subprocess
from datetime import datetime
from typing import List, Dict

from flask import Flask, jsonify, request, Response

from btc_config import BTCConfig
from btc_db_manager import BTCDB

# .env 자동 로딩
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

APP_HOST = "127.0.0.1"
APP_PORT = 8080

SERVICE_NAME = "btc-stacking-bot"  # journalctl -u <service>

# "거래 관련"으로 보고 싶은 키워드(너 로그 스타일에 맞춰 추가/삭제 가능)
TRADE_RE = re.compile(r"\b(BUY|SELL|LONG|SHORT|ENTRY|EXIT|TP|SL)\b", re.IGNORECASE)

# "에러"로 보고 싶은 키워드
ERROR_RE = re.compile(
    r"(ERROR|Traceback|Exception|ModuleNotFoundError|Failed with result|status=\d+/FAILURE)",
    re.IGNORECASE,
)

app = Flask(__name__)

TUNE_FIELDS = {
    "grid_step_pct": {
        "type": "float",
        "min": 0.002,
        "max": 0.05,
        "step": 0.001,
        "title": "추가매수 그리드(grid_step_pct)",
        "desc": "추가 매수 간격 설정(%)",
    },
    "take_profit_pct": {
        "type": "float",
        "min": 0.002,
        "max": 0.05,
        "step": 0.001,
        "title": "익절 비율(take_profit_pct)",
        "desc": "익절 비율(%)",
    },
    "buy_quote_usdt": {
        "type": "float",
        "min": 5.0,
        "max": 250.0,
        "step": 1.0,
        "title": "1회 추가 매수액(buy_quote_usdt)",
        "desc": "1회 추가 매수 금액(USDT)",
    },
    "tp_refresh_sec": {
        "type": "int",
        "min": 5,
        "max": 300,
        "step": 5,
        "title": "익절 주문 갱신(tp_refresh_sec)",
        "desc": "익절 범위 업데이트 주기",
    },
    "trailing_activate_pct": {
        "type": "float",
        "min": 0.005,
        "max": 0.2,
        "step": 0.001,
        "title": "트레일링 시작 기준(trailing_activate_pct)",
        "desc": "매수가 대비 이익이 이 비율 이상이면 트레일링 시작",
    },
    "trailing_ratio": {
        "type": "float",
        "min": 0.5,
        "max": 0.95,
        "step": 0.01,
        "title": "트레일링 되돌림 비율(trailing_ratio)",
        "desc": "상승폭 대비 몇 % 되돌림에서 매도",
    },
    "sell_fraction_on_tp": {
        "type": "float",
        "min": 0.01,
        "max": 1.0,
        "step": 0.01,
        "title": "트레일링 익절 비율(sell_fraction_on_tp)",
        "desc": "TP도달/ 체결 시 보유 btc의 몇%를 팔지",
    },
    "min_trade_usdt": {
        "type": "float",
        "min": 0.0,
        "max": 50.0,
        "step": 0.5,
        "title": "최소 주문 금액(min_trade_usdt)",
        "desc": "이 값보다 낮은 주문은 실행하지 않음",
    },
    "buy_min_usdt": {
        "type": "float",
        "min": 1.0,
        "max": 50.0,
        "step": 0.5,
        "title": "동적 최소 매수 금액(buy_min_usdt)",
        "desc": "매수 금액 변동성 True 시 최소 매수 금액",
    },
    "buy_max_usdt": {
        "type": "float",
        "min": 5.0,
        "max": 300.0,
        "step": 1.0,
        "title": "동적 최대 매수 금액(buy_max_usdt)",
        "desc": "매수 금액 변동성 True 시 최대 매수 금액",
    },
    "recenter_threshold_pct": {
        "type": "float",
        "min": 0.005,
        "max": 0.1,
        "step": 0.001,
        "title": "기준가를 갱신 조건(recenter_threshold_pct)",
        "desc": "기준가와 현재가의 괴리가 2% 이상 벌어지면 → 기준가를 다시 잡는다.",
    },
    "recenter_cooldown_sec": {
        "type": "int",
        "min": 10,
        "max": 600,
        "step": 10,
        "title": "갱신 쿨타임(recenter_cooldown_sec)",
        "desc": "기준가를 갱신한 뒤 최소 몇 초 동안은 다시 갱신하지 않도록 막는 쿨다운",
    },
    "price_vol_window": {
        "type": "int",
        "min": 5,
        "max": 200,
        "step": 1,
        "title": "변동성 계산 구간(price_vol_window)",
        "desc": "변동성 계산에 쓰는 최근 N개 구간 길이",
    },
    "vol_low": {
        "type": "float",
        "min": 0.0001,
        "max": 0.05,
        "step": 0.0001,
        "title": "변동성 낮음 기준(vol_low)",
        "desc": "변동성이 낮다 판단 기준",
    },
    "vol_high": {
        "type": "float",
        "min": 0.0001,
        "max": 0.1,
        "step": 0.0001,
        "title": "변동성 높음 기준(vol_high)",
        "desc": "변동성이 높다 판단 기준",
    },
    "vol_boost_max": {
        "type": "float",
        "min": 0.5,
        "max": 3.0,
        "step": 0.01,
        "title": "변동성 증폭 상한(vol_boost_max)",
        "desc": "변동성이 특정 조건일 때 매수 금액을 최대 몇 배까지 늘릴지(상한)",
    },
    "vol_cut_min": {
        "type": "float",
        "min": 0.1,
        "max": 1.0,
        "step": 0.01,
        "title": "변동성 축소 하한(vol_cut_min)",
        "desc": "변동성이 특정 조건일 때 매수 금액을 최소 몇 배까지 줄일지(하한)",
    },
    "crash_drop_pct": {
        "type": "float",
        "min": 0.01,
        "max": 0.2,
        "step": 0.001,
        "title": "급락 기준폭(crash_drop_pct)",
        "desc": "급락 판단 기준 하락폭",
    },
    "loop_interval_sec": {
        "type": "int",
        "min": 5,
        "max": 600,
        "step": 5,
        "title": "메인 루프 주기(loop_interval_sec)",
        "desc": "메인 루프가 몇 초마다 한 번씩 실행되는지",
    },
    "trade_cap_ratio": {
        "type": "float",
        "min": 0.05,
        "max": 1.0,
        "step": 0.01,
        "title": "트레이드캡(trade_cap_ratio)",
        "desc": "전체 USDT 중에서 이 전략이 실제로 굴리는 운용 자금 비율",
    },
}

TOGGLE_FIELDS = {
    "dynamic_buy": {
        "title": "매수 금액 변동성(dynamic_buy)",
        "desc": "매수 금액을 고정이 아니라 상황(노출/변동성 등)에 따라 가변으로 할지.",
    },
    "trailing_sell_enable": {
        "title": "트레일링 매도 사용(trailing_sell_enable)",
        "desc": "ON이면 트레일링 매도만 사용하고 TP 리밋은 비활성화",
    },
    "verbose": {
        "title": "로깅(verbose)",
        "desc": "로그를 상세히 찍을지 여부",
    },
}

_db: BTCDB | None = None

HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>btc-bot dashboard</title>
  <style>
    :root {
      --bg: #0f1214;
      --panel: #151a1f;
      --panel-2: #1b2229;
      --text: #f2f1ea;
      --muted: #9aa3ad;
      --accent: #e9b44c;
      --accent-2: #4cc9f0;
      --danger: #ff7b7b;
      --border: #2b323a;
    }
    * { box-sizing: border-box; }
    body {
      font-family: "Trebuchet MS", "Lucida Sans Unicode", "Verdana", sans-serif;
      margin: 0;
      color: var(--text);
      background: radial-gradient(circle at 20% 20%, #1a2128 0%, #0f1214 45%, #0b0e11 100%);
    }
    .wrap { padding: 18px; max-width: 1200px; margin: 0 auto; }
    .title { display:flex; align-items:center; gap:12px; margin-bottom:16px; }
    .badge { background: linear-gradient(120deg, var(--accent), #f7d488); color:#1a1303; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 700; letter-spacing: 0.6px; }
    .row { display:flex; gap:12px; align-items:center; flex-wrap:wrap; margin-bottom:12px; }
    .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 16px; margin-top: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
    .panel.half { flex: 1; min-width: 320px; }
    .label { font-size: 13px; color: var(--muted); letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px; display:block; }
    input, select, button {
      padding: 8px 10px; border-radius: 10px; border:1px solid var(--border);
      background: var(--panel-2); color: var(--text);
    }
    button { cursor:pointer; }
    button.primary { background: linear-gradient(120deg, var(--accent), #f7d488); color: #1a1303; border: none; font-weight: 700; }
    button.secondary { border: 1px solid var(--border); }
    .meta { opacity:0.8; font-size: 13px; color: var(--muted); }
    .grid { display:grid; gap: 14px; grid-template-columns: repeat(2, minmax(260px, 1fr)); }
    .section { margin-top: 12px; }
    .section-title { font-size: 14px; color: var(--muted); letter-spacing: 0.5px; text-transform: uppercase; margin: 20px 0 10px 5px; }
    .tune-row { display:flex; gap:10px; align-items:center; }
    .tune-row input[type="number"] { width: 110px; }
    .tune-row input[type="range"] { flex: 1; accent-color: var(--accent); }
    .unit-wrap { position: relative; display: inline-flex; align-items: center; }
    .unit-wrap input { padding-right: 34px; }
    .unit { position: absolute; right: 10px; font-size: 11px; color: var(--muted); pointer-events: none; }
    .hint { font-size: 12px; color: var(--muted); }
    .status { font-size: 13px; color: var(--accent-2); }
    .logs { margin-top: 16px; }
    .stats-grid { display:grid; gap: 12px; grid-template-columns: repeat(4, minmax(160px, 1fr)); }
    .stat { background: var(--panel-2); border: 1px solid var(--border); border-radius: 12px; padding: 10px; }
    .stat .val { font-size: 18px; font-weight: 700; }
    .stat .cap { font-size: 11px; color: var(--muted); letter-spacing: 0.4px; text-transform: uppercase; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 8px 6px; border-bottom: 1px solid var(--border); text-align: left; }
    th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 11px; letter-spacing: 0.4px; }
    tbody tr:hover { background: #141a20; }
    pre { border:1px solid var(--border); border-radius:12px; padding:12px; height: 45vh; overflow:auto; background:#0b0b0b; color:#e8e8e8; }
    .line { white-space: pre-wrap; word-break: break-word; }
    .trade { color: #8ef; }
    .err { color: var(--danger); font-weight: 600; }
    .dim { opacity: 0.75; }
    .switch {
      position: relative; display: inline-block; width: 52px; height: 28px; margin-top: 10px;
    }
    .switch input { opacity: 0; width: 0; height: 0; }
    .slider {
      position: absolute; cursor: pointer; inset: 0; background-color: #3a424c; transition: .2s; border-radius: 999px;
    }
    .slider:before {
      position: absolute; content: ""; height: 22px; width: 22px; left: 3px; top: 3px; background-color: #fff; transition: .2s; border-radius: 50%;
    }
    input:checked + .slider { background-color: #4cc9f0; }
    input:checked + .slider:before { transform: translateX(24px); }
    .modal {
      position: fixed; inset: 0; display: none; align-items: center; justify-content: center;
      background: rgba(5, 7, 10, 0.7); z-index: 999;
    }
    .modal.show { display: flex; }
    .modal-card {
      background: var(--panel); border: 1px solid var(--border); border-radius: 16px;
      padding: 18px; min-width: 280px; text-align: center; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .modal-card strong { display: block; margin-bottom: 6px; }
    .modal-card button { margin-top: 12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="title">
      <span class="badge">LIVE</span>
      <strong>btc-bot dashboard</strong>
      <span class="meta" id="meta">-</span>
    </div>

    <div class="panel logs">
      <div class="row">
        <strong>Trades</strong>
        <span class="meta" id="trade-meta">-</span>
      </div>
      <div class="stats-grid" id="trade-stats"></div>
      <div class="row" style="margin-top: 10px;">
        <label class="label">Rows
          <input id="trade-lines" type="number" min="10" max="300" value="80" style="width:90px;">
        </label>
        <button onclick="loadTrades()">Refresh trades</button>
      </div>
      <div style="overflow:auto; max-height: 36vh;">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Side</th>
              <th>Price</th>
              <th>Qty</th>
              <th>Quote</th>
              <th>Fee</th>
            </tr>
          </thead>
          <tbody id="trade-out">
            <tr><td colspan="6" class="dim">Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="panel logs">
      <div class="row">
        <strong>Logs</strong>
        <span class="meta" id="log-meta">-</span>
      </div>

      <div class="row">
        <label class="label">Lines
          <input id="lines" type="number" min="50" max="5000" value="500" style="width:90px;">
        </label>

        <label class="label">Mode
          <select id="mode">
            <option value="all">All</option>
            <option value="trade">BUY/SELL only</option>
            <option value="error">Errors only</option>
          </select>
        </label>

        <label class="label">Search
          <input id="q" placeholder="text contains..." style="width:260px;">
        </label>

        <label class="label">Refresh
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
    </div>

    <div class="panel">
      <div class="row" style="justify-content: space-between;">
        <div>
          <div class="label">Tune Controls</div>
          <div class="hint">Saved values are applied on the running bot without restart.</div>
        </div>
        <div class="status" id="save-status">-</div>
      </div>
      <div class="section">
        <div class="section-title">매수 조건</div>
        <div class="grid" id="tune-buy"></div>
      </div>
      <div class="section">
        <div class="section-title">매도 조건</div>
        <div class="grid" id="tune-sell"></div>
      </div>
      <div class="section">
        <div class="section-title">기준가 조건</div>
        <div class="grid" id="tune-base"></div>
      </div>
      <div class="section">
        <div class="section-title">변동성 조건</div>
        <div class="grid" id="tune-vol"></div>
      </div>
      <div class="section">
        <div class="section-title">기타 조건</div>
        <div class="grid" id="tune-etc"></div>
      </div>
      <div class="row" style="margin-top: 20px; justify-content: flex-end;">
        <button class="primary" onclick="saveTune()">Save</button>
        <button class="secondary" onclick="loadTune()">Reload</button>
      </div>
    </div>
  </div>

  <div class="modal" id="save-modal">
    <div class="modal-card">
      <strong>Saved</strong>
      <div id="save-modal-text" class="hint">Tune values updated.</div>
      <button class="primary" onclick="closeSaveModal()">OK</button>
    </div>
  </div>

<script>
let timer = null;
const tuneFields = [
  {key: "grid_step_pct", title: "추가매수 그리드(grid_step_pct) ⭐", desc: "추가 매수 간격 설정(%)", min: 0.002, max: 0.05, step: 0.001, unit: "%"},
  {key: "take_profit_pct", title: "익절 비율(take_profit_pct) ⭐", desc: "익절 비율(%)", min: 0.002, max: 0.05, step: 0.001, unit: "%"},
  {key: "buy_quote_usdt", title: "1회 추가 매수액(buy_quote_usdt) ⭐", desc: "1회 추가 매수 금액(USDT)", min: 5, max: 250, step: 1, unit: "USDT"},
  {key: "tp_refresh_sec", title: "익절 주문 갱신(tp_refresh_sec)", desc: "익절 범위 업데이트 주기", min: 5, max: 300, step: 5, unit: "s"},
  {key: "trailing_activate_pct", title: "트레일링 시작 기준(trailing_activate_pct)", desc: "매수가 대비 이익이 이 비율 이상이면 트레일링 시작", min: 0.005, max: 0.2, step: 0.001, unit: "%"},
  {key: "trailing_ratio", title: "트레일링 되돌림 비율(trailing_ratio)", desc: "상승폭 대비 몇 % 되돌림에서 매도", min: 0.5, max: 0.95, step: 0.01, unit: "%"},
  {key: "sell_fraction_on_tp", title: "익절 비율(sell_fraction_on_tp) ⭐", desc: "TP도달/ 체결 시 보유 btc의 몇%를 팔지", min: 0.01, max: 1.0, step: 0.01, unit: "%"},
  {key: "min_trade_usdt", title: "최소 주문 금액(min_trade_usdt) ⭐", desc: "이 값보다 낮은 주문은 실행하지 않음", min: 0, max: 50, step: 0.5, unit: "USDT"},
  {key: "buy_min_usdt", title: "동적 최소 매수 금액(buy_min_usdt)", desc: "매수 금액 변동성 True 시 최소 매수 금액", min: 1, max: 50, step: 0.5, unit: "USDT"},
  {key: "buy_max_usdt", title: "동적 최대 매수 금액(buy_max_usdt)", desc: "매수 금액 변동성 True 시 최대 매수 금액", min: 5, max: 300, step: 1, unit: "USDT"},
  {key: "recenter_threshold_pct", title: "기준가를 갱신 조건(recenter_threshold_pct)", desc: "기준가와 현재가의 괴리가 2% 이상 벌어지면 → 기준가를 다시 잡는다.", min: 0.005, max: 0.1, step: 0.001, unit: "%"},
  {key: "recenter_cooldown_sec", title: "갱신 쿨타임(recenter_cooldown_sec)", desc: "기준가를 갱신한 뒤 최소 몇 초 동안은 다시 갱신하지 않도록 막는 쿨다운", min: 10, max: 600, step: 10, unit: "s"},
  {key: "price_vol_window", title: "변동성 계산 구간(price_vol_window)", desc: "변동성 계산에 쓰는 최근 N개 구간 길이", min: 5, max: 200, step: 1, unit: "n"},
  {key: "vol_low", title: "변동성 낮음 기준(vol_low)", desc: "변동성이 낮다 판단 기준", min: 0.0001, max: 0.05, step: 0.0001, unit: ""},
  {key: "vol_high", title: "변동성 높음 기준(vol_high)", desc: "변동성이 높다 판단 기준", min: 0.0001, max: 0.1, step: 0.0001, unit: ""},
  {key: "vol_boost_max", title: "변동성 증폭 상한(vol_boost_max)", desc: "변동성이 특정 조건일 때 매수 금액을 최대 몇 배까지 늘릴지(상한)", min: 0.5, max: 3.0, step: 0.01, unit: "x"},
  {key: "vol_cut_min", title: "변동성 축소 하한(vol_cut_min)", desc: "변동성이 특정 조건일 때 매수 금액을 최소 몇 배까지 줄일지(하한)", min: 0.1, max: 1.0, step: 0.01, unit: "x"},
  {key: "crash_drop_pct", title: "급락 기준폭(crash_drop_pct)", desc: "급락 판단 기준 하락폭", min: 0.01, max: 0.2, step: 0.001, unit: "%"},
  {key: "loop_interval_sec", title: "메인 루프 주기(loop_interval_sec)", desc: "메인 루프가 몇 초마다 한 번씩 실행되는지", min: 5, max: 600, step: 5, unit: "s"},
  {key: "trade_cap_ratio", title: "트레이드캡(trade_cap_ratio)", desc: "전체 USDT 중에서 이 전략이 실제로 굴리는 운용 자금 비율", min: 0.05, max: 1.0, step: 0.01, unit: "%"},
];

const tuneGroups = {
  buy: ["grid_step_pct", "buy_quote_usdt", "min_trade_usdt", "buy_min_usdt", "buy_max_usdt"],
  sell: ["take_profit_pct", "tp_refresh_sec", "trailing_activate_pct", "trailing_ratio", "sell_fraction_on_tp"],
  base: ["recenter_threshold_pct", "recenter_cooldown_sec"],
  vol: ["price_vol_window", "vol_low", "vol_high", "vol_boost_max", "vol_cut_min"],
  etc: ["crash_drop_pct", "loop_interval_sec", "trade_cap_ratio"],
};

const toggleFields = [
  {key: "dynamic_buy", title: "매수 금액 변동성(dynamic_buy)", desc: "매수 금액을 고정이 아니라 상황(노출/변동성 등)에 따라 가변으로 할지."},
  {key: "trailing_sell_enable", title: "트레일링 매도 사용(trailing_sell_enable)", desc: "ON이면 트레일링 매도만 사용하고 TP 리밋은 비활성화"},
  {key: "verbose", title: "로깅(verbose)", desc: "로그를 상세히 찍을지 여부"},
];

const toggleGroups = {
  buy: ["dynamic_buy"],
  sell: ["trailing_sell_enable"],
  etc: ["verbose"],
};

function escHtml(s){
  return s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
}

function setTimer(){
  if (timer) clearInterval(timer);
  const sec = parseInt(document.getElementById('refresh').value, 10);
  if (sec > 0) timer = setInterval(() => { loadLogs(); loadTrades(); }, sec * 1000);
}

async function loadLogs(){
  const lines = document.getElementById('lines').value || 500;
  const mode  = document.getElementById('mode').value;
  const q     = document.getElementById('q').value || "";
  const url = `/api/logs?lines=${encodeURIComponent(lines)}&mode=${encodeURIComponent(mode)}&q=${encodeURIComponent(q)}`;

  const r = await fetch(url);
  const j = await r.json();

  document.getElementById('log-meta').textContent =
    `updated=${j.updated_at} | mode=${j.mode} | lines=${j.lines} | matched=${j.matched}`;

  const out = document.getElementById('out');
  out.innerHTML = j.items.map(it => {
    const cls = it.tag === "error" ? "line err" : (it.tag === "trade" ? "line trade" : "line");
    return `<div class="${cls}">${escHtml(it.text)}</div>`;
  }).join("");
}

function buildTuneUI(values){
  const gridMap = {
    buy: document.getElementById('tune-buy'),
    sell: document.getElementById('tune-sell'),
    base: document.getElementById('tune-base'),
    vol: document.getElementById('tune-vol'),
    etc: document.getElementById('tune-etc'),
  };
  Object.values(gridMap).forEach(g => { g.innerHTML = ""; });

  const fieldMap = {};
  tuneFields.forEach(f => { fieldMap[f.key] = f; });
  const toggleMap = {};
  toggleFields.forEach(f => { toggleMap[f.key] = f; });

  const addField = (grid, f) => {
    const val = values[f.key];
    const title = f.title || f.key;
    const desc = f.desc || "";
    const listId = `dl-${f.key}`;
    const row = document.createElement('div');
    row.className = "panel";
    const isFloat = String(f.step).includes(".");
    const ticks = 5;
    let options = "";
    for (let i = 0; i <= ticks; i++) {
      const v = f.min + ((f.max - f.min) * i / ticks);
      const txt = isFloat ? v.toFixed(3) : Math.round(v).toString();
      options += `<option value="${txt}"></option>`;
    }
    row.innerHTML = `
      <div class="label"><strong>${title}</strong></div>
      <div class="hint">${desc}</div>
      <div class="tune-row">
        <input type="range" min="${f.min}" max="${f.max}" step="${f.step}" value="${val}" id="r-${f.key}" list="${listId}">
        <div class="unit-wrap">
          <input type="number" min="${f.min}" max="${f.max}" step="${f.step}" value="${val}" id="n-${f.key}">
          <span class="unit">${f.unit || ""}</span>
        </div>
      </div>
      <datalist id="${listId}">${options}</datalist>
      <div class="hint">min ${f.min} / max ${f.max}</div>
    `;
    grid.appendChild(row);

    const range = row.querySelector(`#r-${f.key}`);
    const num = row.querySelector(`#n-${f.key}`);
    range.addEventListener('input', () => { num.value = range.value; });
    num.addEventListener('input', () => { range.value = num.value; });
  };

  const addToggle = (grid, f) => {
    const val = !!values[f.key];
    const row = document.createElement('div');
    row.className = "panel";
    row.innerHTML = `
      <div class="label"><strong>${f.title}</strong></div>
      <div class="hint">${f.desc}</div>
      <label class="switch">
        <input type="checkbox" id="b-${f.key}" ${val ? "checked" : ""}>
        <span class="slider"></span>
      </label>
    `;
    grid.appendChild(row);
  };

  Object.entries(tuneGroups).forEach(([group, keys]) => {
    const grid = gridMap[group];
    const toggles = toggleGroups[group] || [];
    toggles.forEach(k => {
      const f = toggleMap[k];
      if (f) addToggle(grid, f);
    });
    keys.forEach(k => {
      const f = fieldMap[k];
      if (f) addField(grid, f);
    });
  });
}

function formatNum(x, digits = 2){
  const n = Number(x);
  if (!Number.isFinite(n)) return "-";
  return n.toFixed(digits);
}

async function loadTrades(){
  const lines = document.getElementById('trade-lines').value || 80;
  const url = `/api/trades?lines=${encodeURIComponent(lines)}`;
  const r = await fetch(url);
  const j = await r.json();
  if (!r.ok) {
    document.getElementById('trade-meta').textContent = j.error || "load failed";
    return;
  }

  document.getElementById('trade-meta').textContent =
    `updated=${j.updated_at} | rows=${j.items.length}`;

  const stats = j.stats || {};
  const statsEl = document.getElementById('trade-stats');
  statsEl.innerHTML = `
    <div class="stat"><div class="cap">core added (btc)</div><div class="val">${formatNum(stats.core_added_btc_total, 8)}</div></div>
    <div class="stat"><div class="cap">last core add (btc)</div><div class="val">${formatNum(stats.core_added_btc_last, 8)}</div></div>
    <div class="stat"><div class="cap">realized profit (btc eq)</div><div class="val">${formatNum(stats.profit_realized_btc_equiv, 8)}</div></div>
    <div class="stat"><div class="cap">pending profit (btc eq)</div><div class="val">${formatNum(stats.profit_pool_btc_equiv, 8)}</div></div>
  `;

  const out = document.getElementById('trade-out');
  out.innerHTML = j.items.map(it => {
    return `
      <tr>
        <td>${it.time}</td>
        <td>${it.side}</td>
        <td>${formatNum(it.price, 2)}</td>
        <td>${formatNum(it.qty, 8)}</td>
        <td>${formatNum(it.quote_qty, 2)}</td>
        <td>${formatNum(it.commission, 6)} ${it.commission_asset || ""}</td>
      </tr>
    `;
  }).join("");
}

async function loadTune(){
  const r = await fetch("/api/tune");
  const j = await r.json();
  if (!r.ok) {
    document.getElementById('save-status').textContent = j.error || "load failed";
    return;
  }
  buildTuneUI(j.values);
  document.getElementById('save-status').textContent = `loaded at ${j.updated_at}`;
  document.getElementById('meta').textContent = `tune=${j.updated_at}`;
}

async function saveTune(){
  const payload = {};
  tuneFields.forEach(f => {
    const v = document.getElementById(`n-${f.key}`).value;
    payload[f.key] = v;
  });
  toggleFields.forEach(f => {
    const v = document.getElementById(`b-${f.key}`).checked;
    payload[f.key] = v;
  });
  const r = await fetch("/api/tune", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const j = await r.json();
  if (!r.ok) {
    document.getElementById('save-status').textContent = j.error || "save failed";
    openSaveModal("저장 실패: " + (j.error || "unknown"));
    return;
  }
  buildTuneUI(j.values);
  document.getElementById('save-status').textContent = `saved at ${j.updated_at}`;
  document.getElementById('meta').textContent = `tune=${j.updated_at}`;
  openSaveModal("저장되었습니다.");
}

function openSaveModal(msg){
  document.getElementById('save-modal-text').textContent = msg;
  document.getElementById('save-modal').classList.add('show');
}

function closeSaveModal(){
  document.getElementById('save-modal').classList.remove('show');
}

document.getElementById('mode').addEventListener('change', loadLogs);
document.getElementById('refresh').addEventListener('change', () => { setTimer(); loadLogs(); loadTrades(); });
document.getElementById('q').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadLogs(); });

setTimer();
loadLogs();
loadTune();
loadTrades();
</script>
</body>
</html>
"""


def _get_db() -> BTCDB:
    global _db
    if _db is not None:
        return _db
    cfg = BTCConfig()
    db_url = cfg.build_db_url()
    if not db_url:
        raise RuntimeError("BTC_DB_URL or BTC_DB_* is required for dashboard tuning")
    _db = BTCDB(db_url=db_url, verbose=False)
    return _db

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


@app.get("/api/trades")
def api_trades():
    try:
        lines = int(request.args.get("lines", "80"))
        lines = max(10, min(300, lines))
        cfg = BTCConfig()
        prefix = str(getattr(cfg, "client_order_prefix", "BTCSTACK_"))
        symbol = cfg.symbol

        sql = """
            SELECT f.trade_id, f.side, f.price, f.qty, f.quote_qty,
                   f.commission, f.commission_asset, f.trade_time_ms
            FROM btc_fills f
            JOIN btc_orders o ON o.order_id = f.order_id
            WHERE f.symbol = %s
            AND o.client_order_id LIKE %s
            ORDER BY COALESCE(f.trade_time_ms, 0) DESC, f.trade_id DESC
            LIMIT %s
        """
        items = []
        last_price = None
        conn = _get_db().get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (symbol, f"{prefix}%", lines))
                rows = cur.fetchall() or []
        finally:
            conn.close()

        for r in rows:
            ts = int(r.get("trade_time_ms") or 0)
            dt = datetime.fromtimestamp(ts / 1000.0) if ts > 0 else None
            price = float(r.get("price") or 0.0)
            if last_price is None and price > 0:
                last_price = price
            items.append(
                {
                    "trade_id": int(r.get("trade_id") or 0),
                    "side": r.get("side") or "",
                    "price": price,
                    "qty": float(r.get("qty") or 0.0),
                    "quote_qty": float(r.get("quote_qty") or 0.0),
                    "commission": float(r.get("commission") or 0.0),
                    "commission_asset": r.get("commission_asset") or "",
                    "time": dt.strftime("%m-%d %H:%M:%S") if dt else "-",
                }
            )

        profit_pool_usdt = float(_get_db().get_setting("profit_pool_usdt", 0.0) or 0.0)
        profit_realized_usdt = float(_get_db().get_setting("profit_realized_usdt", 0.0) or 0.0)
        core_added_btc_total = float(_get_db().get_setting("profit_core_added_btc_total", 0.0) or 0.0)
        core_added_btc_last = float(_get_db().get_setting("profit_core_added_btc_last", 0.0) or 0.0)
        profit_pool_btc_equiv = (profit_pool_usdt / last_price) if last_price else 0.0
        profit_realized_btc_equiv = (profit_realized_usdt / last_price) if last_price else 0.0

        return jsonify({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
            "stats": {
                "profit_pool_usdt": profit_pool_usdt,
                "profit_realized_usdt": profit_realized_usdt,
                "profit_pool_btc_equiv": profit_pool_btc_equiv,
                "profit_realized_btc_equiv": profit_realized_btc_equiv,
                "core_added_btc_total": core_added_btc_total,
                "core_added_btc_last": core_added_btc_last,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/api/tune")
def api_get_tune():
    try:
        cfg = BTCConfig()
        values: Dict[str, float | int | bool] = {}
        for key, spec in TUNE_FIELDS.items():
            raw = _get_db().get_setting(f"tune.{key}", None)
            if raw is None or str(raw).strip() == "":
                val = getattr(cfg, key)
            else:
                try:
                    val = float(raw) if spec["type"] == "float" else int(float(raw))
                except Exception:
                    val = getattr(cfg, key)
            values[key] = val
        for key in TOGGLE_FIELDS.keys():
            raw = _get_db().get_setting(f"tune.{key}", None)
            if raw is None or str(raw).strip() == "":
                values[key] = bool(getattr(cfg, key))
            else:
                s = str(raw).strip().lower()
                values[key] = s in ("1", "true", "yes", "on")

        return jsonify({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "values": values,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/tune")
def api_set_tune():
    try:
        data = request.get_json(silent=True) or {}
        cfg = BTCConfig()
        values: Dict[str, float | int | bool] = {}
        for key, spec in TUNE_FIELDS.items():
            if key not in data:
                continue
            raw = data.get(key)
            try:
                val = float(raw) if spec["type"] == "float" else int(float(raw))
            except Exception:
                return jsonify({"error": f"invalid value for {key}"}), 400

            if val < spec["min"] or val > spec["max"]:
                return jsonify({"error": f"{key} out of range"}), 400

            _get_db().set_setting(f"tune.{key}", val)
            values[key] = val

        for key in TOGGLE_FIELDS.keys():
            if key not in data:
                continue
            raw = data.get(key)
            if isinstance(raw, bool):
                val = raw
            else:
                s = str(raw).strip().lower()
                if s in ("1", "true", "yes", "on"):
                    val = True
                elif s in ("0", "false", "no", "off"):
                    val = False
                else:
                    return jsonify({"error": f"invalid value for {key}"}), 400
            _get_db().set_setting(f"tune.{key}", val)
            values[key] = val

        if not values:
            return jsonify({"error": "no values provided"}), 400

        response_vals = {}
        for key, spec in TUNE_FIELDS.items():
            raw = _get_db().get_setting(f"tune.{key}", None)
            if raw is None or str(raw).strip() == "":
                response_vals[key] = getattr(cfg, key)
            else:
                response_vals[key] = float(raw) if spec["type"] == "float" else int(float(raw))
        for key in TOGGLE_FIELDS.keys():
            raw = _get_db().get_setting(f"tune.{key}", None)
            if raw is None or str(raw).strip() == "":
                response_vals[key] = bool(getattr(cfg, key))
            else:
                response_vals[key] = str(raw).strip().lower() in ("1", "true", "yes", "on")

        return jsonify({
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "values": response_vals,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
