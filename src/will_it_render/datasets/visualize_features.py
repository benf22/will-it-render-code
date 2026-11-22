"""Interactive attention-map visualization via a local Flask web server.

Loads per-head safetensor attention maps from extraction directories and
renders animated GIFs for a user-selected denoising step and transformer layer.

Usage:
    # From a labels CSV (has extraction_dir, object_name, label columns):
    python -m labeling.visualize_features labels.csv

    # From an extraction root directory only (no labels CSV needed):
    python -m labeling.visualize_features --root /path/to/extraction_dir

    # From a prompts CSV + extraction root (CSV has no extraction_dir column):
    python -m labeling.visualize_features prompts.csv --root /path/to/extraction_dir

    # With custom port/host:
    python -m labeling.visualize_features --root /path/to/extraction_dir --port 8080 --host 0.0.0.0

Access over SSH:
    ssh -L 8080:localhost:8080 <remote-host>
    Then open http://localhost:8080 in your local browser.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import socket
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from flask import Flask, jsonify, render_template_string, request
from PIL import Image
from safetensors import safe_open

from will_it_render.common.aggregator import _resolve_spatial


# ---------------------------------------------------------------------------
# HTML / JS template
# ---------------------------------------------------------------------------

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Feature Map Visualizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html { font-size: 17px; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; min-height: 100vh; }

  #topbar {
    position: sticky; top: 0; z-index: 10;
    background: #1a1a1a; border-bottom: 1px solid #333;
    padding: 14px 24px;
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  }
  #progress-bar-wrap { position: absolute; bottom: 0; left: 0; right: 0; height: 3px; background: #333; }
  #progress-bar { height: 3px; background: #4a9eff; transition: width 0.2s; }

  #counter { font-size: 1em; color: #888; white-space: nowrap; }
  #object-name { font-weight: 700; font-size: 1.2em; }

  #filter-btns { display: flex; gap: 8px; }
  .fbtn {
    background: #2a2a2a; color: #aaa; border: 1px solid #444;
    padding: 6px 16px; border-radius: 20px; cursor: pointer; font-size: 0.95em; font-weight: 600;
    transition: background 0.15s, color 0.15s;
  }
  .fbtn:hover { background: #383838; }
  .fbtn.active-all  { background: #2a2a3a; color: #8ab4ff; border-color: #4a6aaa; }
  .fbtn.active-pos  { background: #1a4a1a; color: #5dc75d; border-color: #3a7a3a; }
  .fbtn.active-neg  { background: #4a1a1a; color: #e05555; border-color: #7a3a3a; }
  .fbtn.active-soft { background: #3a3a1a; color: #c8b840; border-color: #6a6a2a; }

  #label-badge {
    padding: 4px 14px; border-radius: 14px; font-size: 1em; font-weight: 700;
  }
  .label-pos  { background: #1a4a1a; color: #5dc75d; border: 1px solid #3a7a3a; }
  .label-neg  { background: #4a1a1a; color: #e05555; border: 1px solid #7a3a3a; }
  .label-soft { background: #3a3a1a; color: #c8b840; border: 1px solid #6a6a2a; }

  #dir-path { font-size: 0.75em; color: #555; word-break: break-all; flex: 1 1 100%; margin-top: 2px; }

  #nav { display: flex; gap: 8px; }
  #jump-wrap { display: flex; gap: 6px; align-items: center; margin-left: auto; }
  #jump-input {
    width: 90px; padding: 6px 10px; border-radius: 6px; font-size: 1em;
    background: #2a2a2a; color: #eee; border: 1px solid #444;
  }
  #jump-input::placeholder { color: #555; }

  button {
    background: #2a2a2a; color: #ccc; border: 1px solid #444;
    padding: 8px 22px; border-radius: 6px; cursor: pointer; font-size: 1em;
    transition: background 0.15s;
  }
  button:hover:not(:disabled) { background: #3a3a3a; }
  button:disabled { opacity: 0.3; cursor: default; }

  #controls-bar {
    background: #161616; border-bottom: 1px solid #222;
    padding: 10px 24px;
    display: flex; align-items: center; gap: 20px; flex-wrap: wrap;
  }
  .ctrl-group { display: flex; align-items: center; gap: 8px; }
  .ctrl-label { font-size: 0.9em; color: #888; white-space: nowrap; }
  .ctrl-input {
    width: 80px; padding: 5px 10px; border-radius: 6px; font-size: 1em;
    background: #2a2a2a; color: #eee; border: 1px solid #444;
  }
  #load-btn { padding: 7px 28px; }
  #steps-hint { font-size: 0.78em; color: #555; }

  #main { padding: 24px; }

  #loading-msg { text-align: center; color: #555; margin: 40px 0; font-size: 1.1em; }
  #error-msg   { color: #e05555; margin: 20px 0; font-size: 1em; }

  #prompt-card {
    background: #181818; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 14px 18px; margin-bottom: 18px;
  }
  #prompt-text { color: #ccc; font-size: 0.95em; line-height: 1.5; margin-top: 4px; }
  .info-row { display: flex; align-items: center; gap: 16px; margin-top: 10px; flex-wrap: wrap; }
  .info-key { font-size: 0.8em; color: #666; }
  .info-val { font-size: 0.9em; color: #aaa; font-weight: 600; }
  #info-label-badge { padding: 2px 12px; border-radius: 10px; font-size: 0.85em; font-weight: 700; }

  #ref-section {
    display: flex; align-items: flex-start; gap: 16px;
    background: #181818; border: 1px solid #2a2a2a; border-radius: 8px;
    padding: 10px 14px; margin-bottom: 18px;
  }
  #ref-section img { display: block; border-radius: 4px; max-height: 160px; width: auto; }
  #ref-img { image-rendering: pixelated; }
  .ref-caption { font-size: 0.72em; color: #666; margin-top: 5px; }

  #heads-section-label {
    font-size: 0.85em; color: #666; margin-bottom: 10px;
  }
  #heads-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 8px;
  }
  .head-card {
    background: #1c1c1c; border: 1px solid #2a2a2a; border-radius: 6px;
    overflow: hidden; text-align: center;
  }
  .head-card img { width: 100%; display: block; image-rendering: pixelated; }
  .head-label { padding: 4px 0; font-size: 0.72em; color: #666; }
</style>
</head>
<body>

<div id="topbar">
  <span id="counter">— / —</span>
  <span id="object-name"></span>
  <span id="label-badge"></span>
  <div id="filter-btns">
    <button class="fbtn active-all" onclick="setFilter('all')">All</button>
    <button class="fbtn" onclick="setFilter('1')">1</button>
    <button class="fbtn" onclick="setFilter('0')">0</button>
    <button class="fbtn" onclick="setFilter('-1')">-1</button>
  </div>
  <div id="nav">
    <button id="btn-prev10" onclick="navigate(-10)" disabled>&#8647; -10</button>
    <button id="btn-prev"   onclick="navigate(-1)"  disabled>&#8592; Prev</button>
    <button id="btn-next"   onclick="navigate(1)"   disabled>Next &#8594;</button>
    <button id="btn-next10" onclick="navigate(10)"  disabled>+10 &#8649;</button>
  </div>
  <div id="jump-wrap">
    <input id="jump-input" type="number" min="1" placeholder="Go to #"
      onkeydown="if(event.key==='Enter') jumpTo()" />
    <button onclick="jumpTo()">Go</button>
  </div>
  <div id="dir-path"></div>
  <div id="progress-bar-wrap"><div id="progress-bar"></div></div>
</div>

<div id="controls-bar">
  <div class="ctrl-group">
    <span class="ctrl-label">Denoising step</span>
    <input type="number" id="step-input" class="ctrl-input" value="5" min="0" max="999"
      onkeydown="if(event.key==='Enter') loadFeatureMap()" />
  </div>
  <div class="ctrl-group">
    <span class="ctrl-label">Layer</span>
    <input type="number" id="layer-input" class="ctrl-input" value="0" min="0" max="255"
      onkeydown="if(event.key==='Enter') loadFeatureMap()" />
  </div>
  <button id="load-btn" onclick="loadFeatureMap()">Load</button>
  <span id="steps-hint"></span>
</div>

<div id="main">
  <div id="loading-msg" style="display:none">Loading…</div>
  <div id="error-msg"   style="display:none"></div>

  <div id="prompt-card" style="display:none">
    <div style="font-size:0.75em;color:#555;margin-bottom:4px">PROMPT</div>
    <div id="prompt-text"></div>
    <div class="info-row">
      <span class="info-key">Word</span>
      <span class="info-val" id="info-word"></span>
      <span class="info-key">Label</span>
      <span id="info-label-badge"></span>
    </div>
  </div>

  <div id="ref-section" style="display:none">
    <div>
      <img id="ref-img" src="" alt="Mean attention" />
      <div class="ref-caption">Mean attention · middle frame</div>
    </div>
    <div>
      <img id="video-ref-img" src="" alt="Video frame" />
      <div class="ref-caption">Video · middle frame</div>
    </div>
  </div>

  <div id="heads-wrapper" style="display:none">
    <div id="heads-section-label"></div>
    <div id="heads-grid"></div>
  </div>
</div>

<script>
const CFG = {{ config_json | safe }};

let allEntries = [];
let allLabels  = [];
let filteredIndices = [];
let currentPos    = 0;
let currentCsvIdx = -1;
let activeFilter  = 'all';

async function init() {
  document.getElementById('loading-msg').style.display = '';
  const r = await fetch('/api/entries');
  allEntries = await r.json();
  allLabels  = allEntries.map(e => e.label);
  document.getElementById('loading-msg').style.display = 'none';

  if (allEntries.length === 0) {
    document.getElementById('error-msg').textContent = 'No rows found in the CSV.';
    document.getElementById('error-msg').style.display = '';
    return;
  }
  applyFilter('all');
}

function applyFilter(filter) {
  activeFilter = filter;
  const fNum = parseFloat(filter);
  filteredIndices = filter === 'all'
    ? allEntries.map((_, i) => i)
    : allEntries.reduce((acc, e, i) => {
        if (parseFloat(e.label) === fNum) acc.push(i);
        return acc;
      }, []);

  const cls = { 'all': 'active-all', '1': 'active-pos', '0': 'active-neg', '-1': 'active-soft' };
  document.querySelectorAll('.fbtn').forEach(btn => {
    btn.className = 'fbtn';
    const txt = btn.textContent.trim();
    if ((filter === 'all' && txt === 'All') || txt === filter) btn.classList.add(cls[filter]);
  });

  currentPos = 0;
  if (filteredIndices.length === 0) {
    document.getElementById('counter').textContent = '0 / 0';
    document.getElementById('progress-bar').style.width = '0%';
    document.getElementById('object-name').textContent = '';
    document.getElementById('label-badge').textContent = '';
    document.getElementById('dir-path').textContent = '';
    ['btn-prev','btn-prev10','btn-next','btn-next10'].forEach(id =>
      document.getElementById(id).disabled = true);
    document.getElementById('error-msg').textContent = 'No entries match this filter.';
    document.getElementById('error-msg').style.display = '';
    return;
  }
  document.getElementById('error-msg').style.display = 'none';
  loadEntry(filteredIndices[0]);
}

function setFilter(filter) { applyFilter(filter); }

function loadEntry(csvIdx) {
  currentCsvIdx = csvIdx;
  currentPos    = filteredIndices.indexOf(csvIdx);
  if (currentPos < 0) currentPos = 0;

  const entry = allEntries[csvIdx];
  const total  = filteredIndices.length;

  document.getElementById('object-name').textContent = entry.object_name || '—';
  const badge = document.getElementById('label-badge');
  badge.textContent = 'label: ' + entry.label;
  badge.className   = labelClass(entry.label);
  document.getElementById('dir-path').textContent = entry.extraction_dir || '';
  document.getElementById('counter').textContent  = (currentPos + 1) + ' / ' + total;
  document.getElementById('progress-bar').style.width = ((currentPos + 1) / total * 100) + '%';

  document.getElementById('btn-prev').disabled    = (currentPos === 0);
  document.getElementById('btn-prev10').disabled  = (currentPos === 0);
  document.getElementById('btn-next').disabled    = (currentPos === total - 1);
  document.getElementById('btn-next10').disabled  = (currentPos === total - 1);

  loadFeatureMap();
}

async function loadFeatureMap() {
  if (currentCsvIdx < 0) return;
  const step  = parseInt(document.getElementById('step-input').value,  10);
  const layer = parseInt(document.getElementById('layer-input').value, 10);

  document.getElementById('loading-msg').style.display  = '';
  document.getElementById('error-msg').style.display    = 'none';
  document.getElementById('prompt-card').style.display  = 'none';
  document.getElementById('ref-section').style.display  = 'none';
  document.getElementById('heads-wrapper').style.display = 'none';
  document.getElementById('heads-grid').innerHTML = '';

  let d;
  try {
    const r = await fetch('/api/feature_map?idx=' + currentCsvIdx + '&step=' + step + '&layer=' + layer);
    d = await r.json();
  } catch(e) {
    document.getElementById('loading-msg').style.display = 'none';
    document.getElementById('error-msg').textContent = 'Network error: ' + e;
    document.getElementById('error-msg').style.display = '';
    return;
  }
  document.getElementById('loading-msg').style.display = 'none';

  if (d.available_steps && d.available_steps.length > 0) {
    document.getElementById('steps-hint').textContent = 'Available steps: ' + d.available_steps.join(', ');
  }

  if (d.error) {
    document.getElementById('error-msg').textContent = d.error;
    document.getElementById('error-msg').style.display = '';
    return;
  }

  // Prompt card
  document.getElementById('prompt-text').textContent = d.prompt || '—';
  document.getElementById('info-word').textContent = d.word || '—';
  const lblBadge = document.getElementById('info-label-badge');
  lblBadge.textContent = d.label;
  lblBadge.className = labelClass(d.label);
  document.getElementById('prompt-card').style.display = '';

  // Reference images
  document.getElementById('ref-img').src = 'data:image/png;base64,' + d.avg_mid_b64;
  const videoRefImg = document.getElementById('video-ref-img');
  if (d.video_frame_b64) {
    videoRefImg.src = 'data:image/jpeg;base64,' + d.video_frame_b64;
    videoRefImg.style.display = '';
  } else {
    videoRefImg.style.display = 'none';
  }
  document.getElementById('ref-section').style.display = '';

  // Heads grid
  const grid = document.getElementById('heads-grid');
  document.getElementById('heads-section-label').textContent =
    d.n_heads + ' heads · step ' + step + ' · layer ' + layer +
    ' · ' + d.n_frames + ' frames · ' + d.spatial_hw[0] + '×' + d.spatial_hw[1] + ' spatial';
  d.heads.forEach(function(head, i) {
    const card = document.createElement('div');
    card.className = 'head-card';
    const img = document.createElement('img');
    img.src = 'data:image/gif;base64,' + head.gif_b64;
    img.alt = 'Head ' + i;
    const lbl = document.createElement('div');
    lbl.className = 'head-label';
    lbl.textContent = 'Head ' + i;
    card.appendChild(img);
    card.appendChild(lbl);
    grid.appendChild(card);
  });
  document.getElementById('heads-wrapper').style.display = '';
}

function navigate(delta) {
  const nxt = Math.max(0, Math.min(filteredIndices.length - 1, currentPos + delta));
  if (nxt !== currentPos) loadEntry(filteredIndices[nxt]);
}

function jumpTo() {
  const input = document.getElementById('jump-input');
  const n = parseInt(input.value, 10);
  input.value = '';
  if (isNaN(n)) return;
  const pos = Math.max(1, Math.min(filteredIndices.length, n)) - 1;
  loadEntry(filteredIndices[pos]);
}

function labelClass(val) {
  const n = parseFloat(val);
  if (isNaN(n) || n === -1) return 'label-soft';
  return n >= 0.5 ? 'label-pos' : 'label-neg';
}

document.addEventListener('keydown', function(e) {
  const active = document.activeElement;
  const inputs = ['jump-input', 'step-input', 'layer-input'];
  if (inputs.some(id => active === document.getElementById(id))) return;
  if (e.key === 'ArrowRight' || e.key === 'PageDown') navigate(e.key === 'PageDown' ? 10 : 1);
  if (e.key === 'ArrowLeft'  || e.key === 'PageUp')   navigate(e.key === 'PageUp'   ? -10 : -1);
});

init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Config / metadata helpers
# ---------------------------------------------------------------------------


def _load_probe_config(extraction_dir: str) -> dict | None:
    cfg_path = Path(extraction_dir).parent / "probe_config.yaml"
    if not cfg_path.exists():
        return None
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def _get_prompt(extraction_dir: str) -> str:
    meta = Path(extraction_dir) / "feature_metadata.json"
    if meta.exists():
        with open(meta) as f:
            d = json.load(f)
        for key in ("prompt", "text", "caption", "description"):
            if d.get(key):
                return str(d[key])
    return Path(extraction_dir).name


def _available_steps(extraction_dir: str) -> list[int]:
    steps = []
    for p in Path(extraction_dir).glob("step_*.safetensors"):
        try:
            steps.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(steps)


# ---------------------------------------------------------------------------
# Image / GIF helpers
# ---------------------------------------------------------------------------


def _to_rgb_colormap(arr: np.ndarray) -> np.ndarray:
    """float32 [H, W] in [0,1] → uint8 [H, W, 3] cold-to-hot colormap."""
    anchors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    r_vals  = np.array([  0,   0,   0, 255, 220], dtype=np.float32)
    g_vals  = np.array([  0, 150, 220, 200,   0], dtype=np.float32)
    b_vals  = np.array([128, 255, 100,   0,   0], dtype=np.float32)
    flat = arr.ravel()
    r = np.interp(flat, anchors, r_vals)
    g = np.interp(flat, anchors, g_vals)
    b = np.interp(flat, anchors, b_vals)
    return np.stack([r, g, b], axis=1).reshape(*arr.shape, 3).astype(np.uint8)


def _frames_to_gif_b64(frames_thw: np.ndarray, scale: int = 12, duration_ms: int = 500) -> str:
    """float32 [T, H, W] already normalized to [0,1] → base64 animated GIF."""
    T, H, W = frames_thw.shape
    pil_frames = []
    for t in range(T):
        rgb = _to_rgb_colormap(frames_thw[t])
        img = Image.fromarray(rgb, "RGB").resize((W * scale, H * scale), Image.NEAREST)
        pil_frames.append(img)
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True,
        append_images=pil_frames[1:], loop=0, duration=duration_ms,
    )
    return base64.b64encode(buf.getvalue()).decode()


def _frame_to_png_b64(frame_hw: np.ndarray, scale: int = 24) -> str:
    """float32 [H, W] in [0,1] → base64 PNG."""
    H, W = frame_hw.shape
    rgb = _to_rgb_colormap(frame_hw)
    img = Image.fromarray(rgb, "RGB").resize((W * scale, H * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _extract_mid_frame_b64(extraction_dir: str) -> str | None:
    """Extract the middle frame from output.mp4 → base64 JPEG, or None if unavailable."""
    d = Path(extraction_dir)
    mp4s = list(d.glob("*.mp4"))
    if not mp4s:
        return None
    mp4 = d / "output.mp4" if (d / "output.mp4") in mp4s else mp4s[0]
    cap = cv2.VideoCapture(str(mp4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode() if ok2 else None


def _normalize_minmax(arr: np.ndarray) -> np.ndarray:
    vmin, vmax = arr.min(), arr.max()
    if vmax == vmin:
        return np.zeros_like(arr)
    return (arr - vmin) / (vmax - vmin)


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------


def _load_attn(extraction_dir: str, step: int, word: str, cfg: dict) -> np.ndarray:
    """Load safetensor and reshape to [n_blocks, n_heads, T, h, w]."""
    feat_path = Path(extraction_dir) / f"step_{step:03d}.safetensors"
    if not feat_path.exists():
        raise FileNotFoundError(f"step_{step:03d}.safetensors not found in {extraction_dir}")
    with safe_open(str(feat_path), framework="pt") as f:
        word_key = f"features__{word}"
        key = word_key if word_key in f.keys() else "feature"
        flat = f.get_tensor(key).float()

    n_blocks  = cfg["n_blocks"]
    n_heads   = cfg["n_heads"]
    Lq        = flat.numel() // (n_blocks * n_heads)
    latent_hw = cfg["latent_hw"]
    assert latent_hw is not None, "probe_config.yaml is missing latent_hw — re-run extraction"
    T, h, w   = _resolve_spatial(Lq, latent_hw[0], latent_hw[1])
    return flat.view(n_blocks, n_heads, T, h, w).numpy()


# ---------------------------------------------------------------------------
# Flask app factory
# ---------------------------------------------------------------------------


def create_app(df: pd.DataFrame) -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    config_json = json.dumps({"total": len(df)})

    @app.route("/")
    def index():
        return render_template_string(_HTML, config_json=config_json)

    @app.route("/api/entries")
    def entries():
        result = []
        for seq_idx, (_, row) in enumerate(df.iterrows()):
            result.append({
                "idx": seq_idx,
                "object_name":   str(row.get("object_name", "")),
                "label":         str(row.get("label", "")),
                "extraction_dir": str(row.get("extraction_dir", "")),
            })
        return jsonify(result)

    @app.route("/api/feature_map")
    def feature_map():
        idx   = int(request.args.get("idx",   0))
        step  = int(request.args.get("step",  0))
        layer = int(request.args.get("layer", 0))

        if idx < 0 or idx >= len(df):
            return jsonify({"error": f"Index {idx} out of range"}), 404

        row           = df.iloc[idx]
        extraction_dir = str(row.get("extraction_dir", ""))
        word          = str(row.get("object_name", ""))
        label         = str(row.get("label", ""))

        cfg = _load_probe_config(extraction_dir)
        if cfg is None:
            return jsonify({"error": f"probe_config.yaml not found near {extraction_dir}"})

        prompt          = _get_prompt(extraction_dir)
        available_steps = _available_steps(extraction_dir)

        try:
            attn = _load_attn(extraction_dir, step, word, cfg)  # [n_blocks, n_heads, T, h, w]
        except FileNotFoundError as e:
            return jsonify({
                "error":           str(e),
                "prompt":          prompt,
                "word":            word,
                "label":           label,
                "available_steps": available_steps,
            })

        n_blocks, n_heads, T, h, w = attn.shape

        if layer < 0 or layer >= n_blocks:
            return jsonify({
                "error":           f"Layer {layer} out of range [0, {n_blocks - 1}]",
                "available_steps": available_steps,
            })

        layer_attn = attn[layer]                   # [n_heads, T, h, w]
        layer_norm = _normalize_minmax(layer_attn) # [0,1], shared scale across all heads

        heads = []
        for h_idx in range(n_heads):
            gif = _frames_to_gif_b64(layer_norm[h_idx])
            heads.append({"gif_b64": gif})

        # Mean across heads, middle T frame
        mid_t    = T // 2
        mean_mid = layer_norm[:, mid_t, :, :].mean(axis=0)  # [h, w]
        avg_mid  = _frame_to_png_b64(mean_mid)

        return jsonify({
            "prompt":           prompt,
            "word":             word,
            "label":            label,
            "n_heads":          n_heads,
            "n_frames":         T,
            "spatial_hw":       [h, w],
            "available_steps":  available_steps,
            "heads":            heads,
            "avg_mid_b64":      avg_mid,
            "video_frame_b64":  _extract_mid_frame_b64(extraction_dir),
        })

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _df_from_root(root: str) -> pd.DataFrame:
    """Scan an extraction root dir and build a DataFrame of (extraction_dir, object_name, label)."""
    rows = []
    for subdir in sorted(Path(root).iterdir()):
        if not subdir.is_dir():
            continue
        meta = subdir / "feature_metadata.json"
        if meta.exists():
            with open(meta) as f:
                words = json.load(f).get("words", [])
        else:
            # No metadata — skip silently
            continue
        for word in words:
            rows.append({
                "extraction_dir": str(subdir),
                "object_name":    word,
                "label":          float("nan"),
            })
    return pd.DataFrame(rows, columns=["extraction_dir", "object_name", "label"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive attention-map visualization (Flask server)"
    )
    parser.add_argument(
        "labels_csv",
        nargs="?",
        default=None,
        help=(
            "Path to a labels CSV (columns: extraction_dir, object_name, label) "
            "or a prompts CSV (columns: prompt, object_name, …). "
            "Omit when using --root alone to browse an extraction directory without labels."
        ),
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Extraction root directory. Required when the CSV is a prompts CSV "
            "(no extraction_dir column). Can be used alone without a CSV to browse "
            "all extraction subdirs directly."
        ),
    )
    parser.add_argument("--port", type=int, default=8080, help="HTTP port (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    args = parser.parse_args()

    if args.labels_csv is None and args.root is None:
        parser.error("Provide a labels CSV, a --root directory, or both.")

    if args.labels_csv is None:
        # Directory-only mode: scan root, no labels
        root_path = Path(args.root)
        if not root_path.is_dir():
            sys.exit(f"Error: --root directory not found: {root_path}")
        df = _df_from_root(args.root)
        if df.empty:
            sys.exit(f"Error: no extraction subdirs with feature_metadata.json found under {root_path}")
        source_desc = str(root_path)
    else:
        csv_path = Path(args.labels_csv)
        if not csv_path.exists():
            sys.exit(f"Error: CSV not found: {csv_path}")
        raw = pd.read_csv(csv_path)

        if "extraction_dir" in raw.columns:
            # Labels CSV — use as-is
            required = {"extraction_dir", "object_name", "label"}
            missing  = required - set(raw.columns)
            if missing:
                sys.exit(f"Error: labels CSV is missing columns: {missing}")
            df = raw
            source_desc = str(csv_path)
        else:
            # Prompts CSV (prompt, object_name, target_words, …) — scan --root
            if args.root is None:
                sys.exit(
                    "Error: CSV has no 'extraction_dir' column (prompts format). "
                    "Pass --root pointing to the extraction output directory."
                )
            root_path = Path(args.root)
            if not root_path.is_dir():
                sys.exit(f"Error: --root directory not found: {root_path}")
            df = _df_from_root(args.root)
            if df.empty:
                sys.exit(f"Error: no extraction subdirs with feature_metadata.json found under {root_path}")
            source_desc = f"{csv_path} + root {root_path}"

    n        = len(df)
    hostname = socket.gethostname()
    print(f"Loaded {n} entr{'ies' if n != 1 else 'y'} from {source_desc}")
    print(f"Serving at http://localhost:{args.port}")
    print(f"Access via SSH port forwarding:")
    print(f"  ssh -L {args.port}:localhost:{args.port} {hostname}")
    print(f"Then open http://localhost:{args.port} in your local browser.")
    print("Press Ctrl+C to stop.")

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app = create_app(df)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
