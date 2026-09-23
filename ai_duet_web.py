import random
from typing import Optional
import asyncio
import os

# 設置 JAX / XLA CPU 多執行緒加速旗標，釋放 8 核 16 緒硬體運算力
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import io
import wave
import base64
import numpy as np
import json
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import time

jax_pool = ThreadPoolExecutor(max_workers=4)

try:
    from duet_system import (
        agent_a as mrt2_agent_a,
        agent_b as mrt2_agent_b,
        agent_bass as mrt2_agent_bass,
    )
except ImportError:
    mrt2_agent_a = None
    mrt2_agent_b = None
    mrt2_agent_bass = None

import scipy.signal

def encode_audio_base64(audio_array: np.ndarray, sample_rate: int = 48000) -> str:
    import scipy.io.wavfile as wavfile
    if audio_array is None:
        return ""
    if hasattr(audio_array, "samples"):
        audio_array = audio_array.samples
    # 若為 (2, N) 立體聲，轉為 (N, 2) interleaved 格式
    if audio_array.ndim == 2 and audio_array.shape[0] == 2 and audio_array.shape[1] > 2:
        audio_array = audio_array.T
    audio_array = np.ascontiguousarray(audio_array, dtype=np.float32)
    pcm16 = (np.clip(audio_array, -1.0, 1.0) * 32767).astype(np.int16)
    with io.BytesIO() as buf:
        wavfile.write(buf, sample_rate, pcm16)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def audio_waveform_to_midi_events(
    audio_array: np.ndarray,
    sample_rate: int = 48000,
    duration_sec: float = 0.8,
    scale: Optional[list] = None,
    num_notes: int = 6,
    min_pitch: int = 48,
    max_pitch: int = 84,
) -> list:
    """
    從 MRT2 48kHz 音訊波形中以 FFT 能量峰值精準萃取 num_notes 個真實 MIDI 音符事件。
    保證 100% 產出對齊的音符方塊，絕不遺漏。
    """
    if audio_array is None:
        return []
    if audio_array.ndim == 2:
        mono = np.mean(audio_array, axis=0 if audio_array.shape[0] == 2 else 1)
    else:
        mono = audio_array

    total_samples = len(mono)
    if total_samples < 512:
        return []

    events = []
    step_duration = duration_sec / num_notes
    step_samples = max(256, total_samples // num_notes)

    for i in range(num_notes):
        chunk = mono[i * step_samples : min(total_samples, (i + 1) * step_samples)]
        if len(chunk) < 128:
            continue

        fft_res = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk))))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sample_rate)

        # 鎖定鋼琴有效音域 (65Hz ~ 1500Hz, C2 ~ G6)
        valid = np.where((freqs >= 65) & (freqs <= 1500))[0]
        if len(valid) > 0:
            peak_idx = valid[np.argmax(fft_res[valid])]
            freq = freqs[peak_idx]
            if freq > 0:
                p = int(round(69 + 12 * np.log2(freq / 440.0)))
                p = max(min_pitch, min(max_pitch, p))
                if scale:
                    p = min(scale, key=lambda s: abs(s - p))
            else:
                p = scale[i % len(scale)] if scale else 60
        else:
            p = scale[i % len(scale)] if scale else 60

        events.append({
            "note": int(p),
            "play_delay_sec": round(i * step_duration, 3),
            "duration": round(step_duration * 0.8, 3),
        })

    return events

# --- 定義多種調性字典 ---
def generate_all_scales():
    scales = {}
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    major_intervals = [0, 2, 4, 5, 7, 9, 11]
    minor_intervals = [0, 2, 3, 5, 7, 8, 10]
    
    for i in range(12):
        note_name = notes[i]
        root_midi = 48 + i
        scales[f"{note_name}_Major"] = sorted([root_midi + interval + oct*12 for oct in range(-1, 3) for interval in major_intervals])
        scales[f"{note_name}_Minor"] = sorted([root_midi + interval + oct*12 for oct in range(-1, 3) for interval in minor_intervals])
    return scales

SCALES_DICT = generate_all_scales()

# --- 嫩哈常見和弦進行 (minor 音階 degree index 0-6 = i ii III iv V VI VII) ---
# 讓每段弦路持續 2成4 小節，對應嫩哈重復 loop 感
HH_PROGRESSIONS = [
    [0, 6, 5, 6],   # i - VII - VI - VII  最常見 (Kendrick/Drake 風格)
    [0, 2, 6, 5],   # i - III - VII - VI   classic boom bap
    [0, 3, 6, 2],   # i - iv  - VII - III  沉鬱 lofi
    [0, 5, 2, 6],   # i - VI  - III - VII  R&B 色彩
    [0, 6, 0, 6],   # i - VII - i   - VII  minimal trap
    [0, 3, 5, 6],   # i - iv  - VI  - VII
    [0, 5, 6, 2],   # i - VI  - VII - III
    [0, 2, 3, 6],   # i - III - iv  - VII  魓幻 drill
    [0, 5, 3, 6],   # i - VI  - iv  - VII  neo soul
    [0, 6, 3, 5],   # i - VII - iv  - VI
]

options_html = ""
default_keys = ["G_Minor", "C_Major"]
for k in default_keys + [x for x in SCALES_DICT.keys() if x not in default_keys]:
    name = k.replace("_", " ")
    options_html += f'<option value="{k}">{name}</option>\n'

def midi_to_note_name(midi_num):
    if isinstance(midi_num, (list, tuple)):
        return "[" + " ".join([midi_to_note_name(n) for n in midi_num]) + "]"
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

class MusicAgent:
    def __init__(self, name, base_scale, play_probability=0.6):
        self.name = name
        self.base_scale = base_scale
        self.play_probability = play_probability
        self.memory = {}
        
    def listen_and_learn(self, melody_events):
        def get_root(n):
            return n[0] if isinstance(n, (list, tuple)) else n
            
        notes_only = [get_root(ev['note']) for ev in melody_events if 'note' in ev]
            
        for i in range(len(notes_only) - 1):
            current_note = notes_only[i]
            next_note = notes_only[i+1]
            if current_note not in self.memory:
                self.memory[current_note] = {}
            if next_note not in self.memory[current_note]:
                self.memory[current_note][next_note] = 0
            self.memory[current_note][next_note] += 1

    def generate_full_performance(self, start_note, global_triad_midi, steps=16, step_duration=0.125):
        events = []
        chord_pcs = [n % 12 for n in global_triad_midi]
        
        if random.random() < 0.7:  
            my_chord = [n + 12 for n in global_triad_midi]
                
            onset = random.choice([0, 2, 4])
            events.append({
                "note": my_chord,
                "onset_beat": onset,
                "duration": 1.5,
                "play_delay_sec": onset * step_duration,
                "note_name": midi_to_note_name(my_chord)
            })

        current_step = 0
        current_note = start_note if isinstance(start_note, int) else start_note[0]
        
        while current_step < steps:
            if random.random() > self.play_probability:
                current_step += 1 
                continue
                
            duration_steps = random.choice([1, 2, 3, 4])
            if current_step + duration_steps > steps:
                duration_steps = steps - current_step
            
            if random.random() < 0.8:
                valid_notes = [n for n in self.base_scale if (n % 12) in chord_pcs]
            else:
                valid_notes = self.base_scale
                
            if self.name == "Agent_B":
                valid_notes = [n for n in valid_notes if 65 <= n <= 88]
            else:
                valid_notes = [n for n in valid_notes if 60 <= n <= 84]
            if not valid_notes: valid_notes = [72]
            
            if current_note in self.memory and random.random() < 0.4:
                choices = list(self.memory[current_note].keys())
                weights = list(self.memory[current_note].values())
                candidate = random.choices(choices, weights=weights)[0]
                next_note = min(valid_notes, key=lambda n: abs(n - candidate))
            else:
                if current_note in valid_notes:
                    idx = valid_notes.index(current_note)
                else:
                    idx = min(range(len(valid_notes)), key=lambda i: abs(valid_notes[i] - current_note))
                idx += random.choice([-2, -1, 0, 1, 2])
                idx = max(0, min(len(valid_notes) - 1, idx))
                next_note = valid_notes[idx]
                
            duration_val = max(0.25, duration_steps * step_duration)
            events.append({
                "note": next_note,
                "onset_beat": current_step,
                "duration": duration_val,
                "play_delay_sec": current_step * step_duration,
                "note_name": midi_to_note_name(next_note)
            })
                
            current_note = next_note
            current_step += duration_steps
            
        return events

    def generate_bass_performance(self, global_triad_midi, steps=16, step_duration=0.125):
        events = []
        chord_pcs = [n % 12 for n in global_triad_midi]
        root_pc = global_triad_midi[0] % 12

        # 限定在鋼琴顯示範圍內的低音區 (MIDI 36-55, C2~G3)
        bass_notes = [n for n in self.base_scale if 36 <= n <= 55]
        if not bass_notes:
            bass_notes = [global_triad_midi[0]]

        root_notes = [n for n in bass_notes if n % 12 == root_pc] or bass_notes
        chord_notes = [n for n in bass_notes if n % 12 in chord_pcs] or bass_notes

        step = 0
        while step < steps:
            is_downbeat = (step % 4 == 0)
            if is_downbeat:
                if random.random() < 0.88:
                    note = random.choice(root_notes)
                    dur_steps = random.choice([2, 2, 4])
                    events.append({
                        "note": note, "onset_beat": step,
                        "duration": max(0.15, dur_steps * step_duration),
                        "play_delay_sec": step * step_duration,
                        "note_name": midi_to_note_name(note)
                    })
                    step += dur_steps
                    continue
            else:
                if random.random() < 0.38:
                    note = random.choice(chord_notes)
                    dur_steps = random.choice([1, 2])
                    events.append({
                        "note": note, "onset_beat": step,
                        "duration": max(0.1, dur_steps * step_duration),
                        "play_delay_sec": step * step_duration,
                        "note_name": midi_to_note_name(note)
                    })
                    step += dur_steps
                    continue
            step += 1
        return events

app = FastAPI()

if os.path.isdir("samples"):
    app.mount("/samples", StaticFiles(directory="samples"), name="samples")

html_content = """
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <title>AI Duet — Piano Session</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js"></script>
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --accent-a: #60a5fa;
            --accent-b: #fb923c;
            --accent-human: #34d399;
            --accent-bass: #a78bfa;
            --surface: rgba(255,255,255,0.04);
            --border: rgba(255,255,255,0.08);
            --text: #f0f0f0;
            --muted: #6b7280;
        }

        body {
            background: #080808;
            color: var(--text);
            font-family: 'Inter', sans-serif;
            overflow: hidden;
            margin: 0; padding: 0;
        }

        /* ── Splash ────────────────────────────────────── */
        #splash {
            position: fixed;
            inset: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 44px;
            background: #080808;
            z-index: 500;
        }

        .splash-title {
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.24em;
            text-transform: uppercase;
            color: var(--muted);
        }

        .splash-heading {
            font-size: clamp(36px, 5.5vw, 56px);
            font-weight: 300;
            letter-spacing: -0.025em;
            color: #fff;
            line-height: 1;
        }

        .splash-heading span {
            background: linear-gradient(120deg, var(--accent-a) 0%, var(--accent-human) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 600;
        }

        .controls-row {
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
            justify-content: center;
        }

        .ctrl-group {
            display: flex;
            align-items: center;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 10px;
            overflow: hidden;
        }

        .ctrl-group label {
            padding: 0 14px;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            white-space: nowrap;
            border-right: 1px solid var(--border);
            height: 100%;
            display: flex;
            align-items: center;
        }

        select, input[type="number"] {
            padding: 13px 14px;
            font-size: 14px;
            font-family: 'Inter', sans-serif;
            font-weight: 400;
            background: transparent;
            color: var(--text);
            border: none;
            outline: none;
            cursor: pointer;
            -webkit-appearance: none;
            appearance: none;
        }

        select { min-width: 150px; padding-right: 32px; }
        input[type="number"] { width: 68px; text-align: center; }

        .ctrl-group.sel-wrap { position: relative; }
        .ctrl-group.sel-wrap::after {
            content: '';
            position: absolute;
            right: 12px;
            top: 50%;
            transform: translateY(-50%);
            width: 0; height: 0;
            border-left: 4px solid transparent;
            border-right: 4px solid transparent;
            border-top: 5px solid var(--muted);
            pointer-events: none;
        }

        #startBtn {
            padding: 14px 36px;
            font-size: 13px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            cursor: pointer;
            background: #fff;
            color: #080808;
            border: none;
            border-radius: 10px;
            transition: background 0.15s, transform 0.12s, box-shadow 0.15s;
        }
        #startBtn:hover {
            background: #e8e8e8;
            transform: translateY(-1px);
            box-shadow: 0 8px 28px rgba(255,255,255,0.1);
        }
        #startBtn:active { transform: translateY(0); }
        #startBtn:disabled { opacity: 0.45; cursor: not-allowed; transform: none; }

        /* ── Game ──────────────────────────────────────── */
        #game-container {
            position: absolute;
            top: 0; left: 0; right: 0; bottom: 0;
            display: none;
            flex-direction: column;
        }

        #roll {
            position: relative;
            height: 85vh;
            width: 100%;
            background: #070707;
            overflow: hidden;
        }

        #roll::before {
            content: '';
            position: absolute;
            inset: 0;
            background-image: repeating-linear-gradient(
                to bottom,
                transparent 0px,
                transparent calc(85vh/16 - 1px),
                rgba(255,255,255,0.018) calc(85vh/16 - 1px),
                rgba(255,255,255,0.018) calc(85vh/16)
            );
            pointer-events: none;
            z-index: 1;
        }

        #piano {
            position: relative;
            height: 15vh;
            width: 100%;
            background: #0f0f0f;
            border-top: 1px solid rgba(255,255,255,0.06);
        }

        /* ── Keys ──────────────────────────────────────── */
        .key { position: absolute; bottom: 0; box-sizing: border-box; border-radius: 0 0 3px 3px; transition: background 0.07s; }
        .key.white { height: 100%; background: #eee; border: 1px solid #bbb; border-top: none; z-index: 1; }
        .key.black { height: 62%; background: #181818; border: 1px solid #000; z-index: 2; border-radius: 0 0 2px 2px; }

        .key.white.active-a     { background: #93c5fd !important; }
        .key.black.active-a     { background: #2563eb !important; }
        .key.white.active-b     { background: #fdba74 !important; }
        .key.black.active-b     { background: #ea580c !important; }
        .key.white.active-human { background: #6ee7b7 !important; }
        .key.black.active-human { background: #059669 !important; }
        .key.white.active-bass  { background: #ddd6fe !important; }
        .key.black.active-bass  { background: #7c3aed !important; }

        /* ── Notes ─────────────────────────────────────── */
        .falling-note { position: absolute; border-radius: 2px; z-index: 2; opacity: 0.85; }
        .falling-note.color-a {
            background: linear-gradient(180deg, #bfdbfe 0%, #3b82f6 100%);
            box-shadow: 0 0 10px rgba(96,165,250,0.45);
        }
        .falling-note.color-b {
            background: linear-gradient(180deg, #fed7aa 0%, #ea580c 100%);
            box-shadow: 0 0 10px rgba(251,146,60,0.45);
        }
        .falling-note.color-human {
            background: linear-gradient(0deg, #6ee7b7 0%, #10b981 100%);
            box-shadow: 0 0 14px rgba(52,211,153,0.75);
        }
        .falling-note.color-bass {
            background: linear-gradient(180deg, #ede9fe 0%, #7c3aed 100%);
            box-shadow: 0 0 12px rgba(167,139,250,0.55);
        }

        /* ── Labels ────────────────────────────────────── */
        .key-label {
            position: absolute;
            bottom: 5px; left: 50%;
            transform: translateX(-50%);
            font-size: 8px;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: rgba(0,0,0,0.25);
            pointer-events: none;
            z-index: 5;
            user-select: none;
        }
        .key.black .key-label { color: rgba(255,255,255,0.25); font-size: 7px; bottom: 3px; }

        /* ── HUD ────────────────────────────────────────── */
        .hud {
            position: fixed;
            z-index: 300;
            pointer-events: none;
            border-radius: 6px;
            font-family: 'Inter', sans-serif;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            padding: 5px 12px;
        }

        #human-badge {
            top: 50px; right: 16px;
            background: rgba(52,211,153,0.08);
            border: 1px solid rgba(52,211,153,0.3);
            color: var(--accent-human);
            opacity: 0;
            transition: opacity 0.2s;
        }
        #human-badge.show { opacity: 1; }

        #octave-badge {
            bottom: calc(15vh + 12px);
            right: 14px;
            background: rgba(255,255,255,0.04);
            border: 1px solid var(--border);
            color: var(--muted);
            display: none;
        }

        #kb-hint {
            bottom: calc(15vh + 12px);
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--border);
            color: #555;
            white-space: nowrap;
            display: none;
        }

        /* ── Top info bar ───────────────────────────────── */
        #info-bar {
            position: fixed;
            top: 0; left: 0; right: 0;
            height: 36px;
            display: none;
            align-items: center;
            justify-content: space-between;
            padding: 0 18px;
            background: rgba(8,8,8,0.75);
            border-bottom: 1px solid var(--border);
            backdrop-filter: blur(16px);
            z-index: 200;
            font-size: 10px;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
        }
        .i-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; margin-right: 6px; }
        .d-a { background: var(--accent-a); }
        .d-b { background: var(--accent-b); }
        .d-h { background: var(--accent-human); }
        .d-bass { background: var(--accent-bass); }
        .i-item { display: flex; align-items: center; }
        .i-sep { color: rgba(255,255,255,0.08); margin: 0 12px; }
    </style>
</head>
<body>

    <!-- Splash / Start Screen -->
    <div id="splash">
        <div style="display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center;">
            <div class="splash-title">Real-time generative music</div>
        </div>

        <div class="controls-row">
            <div class="ctrl-group sel-wrap">
                <label for="scaleSelect">Key</label>
                <select id="scaleSelect">
                    PLACEHOLDER_OPTIONS
                </select>
            </div>
            <div class="ctrl-group">
                <label for="bpmInput">BPM</label>
                <input type="number" id="bpmInput" value="120" min="40" max="240" step="5" />
            </div>
        </div>

        <button id="startBtn">Start Session</button>
    </div>

    <!-- In-game info bar -->
    <div id="info-bar">
        <div style="display:flex;align-items:center;">
            <div class="i-item"><span class="i-dot d-a"></span>Agent A</div>
            <div class="i-sep">|</div>
            <div class="i-item"><span class="i-dot d-b"></span>Agent B</div>
            <div class="i-sep">|</div>
            <div class="i-item"><span class="i-dot d-bass"></span>Bass</div>
            <div class="i-sep">|</div>
            <div class="i-item"><span class="i-dot d-h"></span>You</div>
            <div class="i-sep">|</div>
            <div class="i-item" style="cursor:pointer;opacity:0.7;" onclick="document.getElementById('mrt2-panel').classList.toggle('hidden')">
                🎛️ MRT2
            </div>
        </div>

        <div id="info-right" style="color:#333;">—</div>
    </div>

    <!-- MRT2 Features Panel -->
    <div id="mrt2-panel" class="hidden" style="
        position:fixed; top:40px; right:8px; z-index:200;
        background:rgba(20,20,30,0.95); border:1px solid rgba(255,255,255,0.1);
        border-radius:12px; padding:16px; width:300px;
        font-size:12px; color:#e0e0e0; backdrop-filter:blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    ">
        <div style="font-size:14px;font-weight:600;margin-bottom:12px;color:#60a5fa;">🎵 MRT2 Features</div>
        <div id="mrt2-status" style="font-size:11px;color:#6b7280;margin-bottom:10px;">Status: checking...</div>

        <!-- Text-to-Synth -->
        <div style="margin-bottom:14px;">
            <div style="font-weight:500;margin-bottom:6px;color:#a78bfa;">📝 Text-to-Synth</div>
            <div style="display:flex;gap:4px;margin-bottom:4px;">
                <select id="synth-agent" style="background:#1a1a2e;border:1px solid #333;color:#fff;border-radius:6px;padding:4px 6px;font-size:11px;">
                    <option value="a">Agent A</option>
                    <option value="b">Agent B</option>
                </select>
                <input id="synth-prompt" type="text" placeholder="e.g. string ensemble" style="
                    flex:1;background:#1a1a2e;border:1px solid #333;color:#fff;border-radius:6px;padding:4px 8px;font-size:11px;
                " />
                <button onclick="sendSynth()" style="
                    background:#5b4cdb;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;
                ">Apply</button>
            </div>
            <div style="display:flex;gap:4px;flex-wrap:wrap;">
                <button class="preset-btn" onclick="quickSynth('jazz piano improvisation')">🎹 Jazz Piano</button>
                <button class="preset-btn" onclick="quickSynth('warm analog synth pad')">🎛️ Synth Pad</button>
                <button class="preset-btn" onclick="quickSynth('string ensemble legato')">🎻 Strings</button>
                <button class="preset-btn" onclick="quickSynth('bright pluck lead')">✨ Pluck</button>
            </div>
        </div>

        <!-- Audio Cloning -->
        <div style="margin-bottom:14px;">
            <div style="font-weight:500;margin-bottom:6px;color:#34d399;">🎤 Audio Cloning</div>
            <div style="display:flex;gap:4px;align-items:center;">
                <select id="clone-agent" style="background:#1a1a2e;border:1px solid #333;color:#fff;border-radius:6px;padding:4px 6px;font-size:11px;">
                    <option value="a">Agent A</option>
                    <option value="b">Agent B</option>
                </select>
                <label style="
                    flex:1;background:#1a1a2e;border:1px dashed #555;border-radius:6px;padding:6px 8px;
                    text-align:center;cursor:pointer;font-size:11px;color:#888;
                ">
                    <span id="clone-label">Drop audio or click</span>
                    <input id="clone-file" type="file" accept="audio/*" style="display:none" onchange="handleCloneFile(this)" />
                </label>
            </div>
        </div>

        <!-- Prompt Mixing -->
        <div style="margin-bottom:8px;">
            <div style="font-weight:500;margin-bottom:6px;color:#fb923c;">🔀 Prompt Mixing</div>
            <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:10px;color:#60a5fa;">A</span>
                <input id="mix-slider" type="range" min="0" max="100" value="50" style="flex:1;accent-color:#fb923c;" oninput="updateMixLabel(this.value)" />
                <span style="font-size:10px;color:#fb923c;">B</span>
                <span id="mix-label" style="font-size:10px;color:#888;min-width:35px;">50%</span>
                <button onclick="sendMix()" style="
                    background:#d4600a;color:#fff;border:none;border-radius:6px;padding:4px 10px;font-size:11px;cursor:pointer;
                ">Mix</button>
            </div>
        </div>

        <div id="mrt2-log" style="max-height:60px;overflow-y:auto;font-size:10px;color:#555;margin-top:8px;border-top:1px solid #222;padding-top:6px;"></div>
    </div>

    <style>
        .hidden { display: none !important; }
        .preset-btn {
            background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
            color: #ccc; border-radius: 6px; padding: 3px 8px; font-size: 10px;
            cursor: pointer; transition: background 0.2s;
        }
        .preset-btn:hover { background: rgba(255,255,255,0.15); }
    </style>

    <div id="human-badge" class="hud">You</div>
    <div id="octave-badge" class="hud">Oct 4</div>
    <div id="kb-hint" class="hud">A S D F G H J K &nbsp;|&nbsp; W E T Y U &nbsp;|&nbsp; Z / X = oct</div>

    <div id="game-container">
        <div id="roll"></div>
        <div id="piano"></div>
    </div>
    
    <script>
        const btn = document.getElementById('startBtn');
        const scaleSelect = document.getElementById('scaleSelect');
        const bpmInput = document.getElementById('bpmInput');
        const rollContainer = document.getElementById('roll');
        const pianoContainer = document.getElementById('piano');
        const gameContainer = document.getElementById('game-container');
        const splashEl = document.getElementById('splash');
        const infoBar = document.getElementById('info-bar');
        
        // --- 建構 4 個八度虛擬鋼琴鍵盤 (MIDI 36 ~ 84) ---
        const startMidi = 36;
        const endMidi = 84;
        const numWhiteKeys = 29;
        const whiteKeyWidth = 100 / numWhiteKeys;
        const keyPos = {}; 
        
        let whiteIndex = 0;
        for (let midi = startMidi; midi <= endMidi; midi++) {
            const noteClass = midi % 12;
            const isBlack = [1, 3, 6, 8, 10].includes(noteClass);
            
            if (!isBlack) {
                keyPos[midi] = { left: whiteIndex * whiteKeyWidth, width: whiteKeyWidth, isBlack: false };
                whiteIndex++;
            } else {
                const prevWhiteLeft = (whiteIndex - 1) * whiteKeyWidth;
                keyPos[midi] = { left: prevWhiteLeft + whiteKeyWidth * 0.65, width: whiteKeyWidth * 0.7, isBlack: true };
            }
            
            const el = document.createElement('div');
            el.id = `key-${midi}`;
            el.className = `key ${keyPos[midi].isBlack ? 'black' : 'white'}`;
            el.style.left = `${keyPos[midi].left}%`;
            el.style.width = `${keyPos[midi].width}%`;
            pianoContainer.appendChild(el);
        }

        // --- 鍵盤對應音符 (相對半音, C=0) ---
        const KEY_SEMITONE = { 'a':0,'w':1,'s':2,'e':3,'d':4,'f':5,'t':6,'g':7,'y':8,'h':9,'u':10,'j':11,'k':12 };
        const KEY_LABEL_MAP = { 0:'A',1:'W',2:'S',3:'E',4:'D',5:'F',6:'T',7:'G',8:'Y',9:'H',10:'U',11:'J',12:'K' };
        let humanBaseMidi = 60; // C4
        const pressedKeys = new Set();
        let humanBadgeTimer = null;
        let audioCtx = null;

        function updateKeyLabels() {
            document.querySelectorAll('.key-label').forEach(el => el.remove());
            for (const [semi, letter] of Object.entries(KEY_LABEL_MAP)) {
                const midi = humanBaseMidi + parseInt(semi);
                const keyEl = document.getElementById('key-' + midi);
                if (!keyEl) continue;
                const lbl = document.createElement('span');
                lbl.className = 'key-label';
                lbl.textContent = letter;
                keyEl.appendChild(lbl);
            }
            document.getElementById('octave-badge').textContent = 'Oct ' + (Math.floor(humanBaseMidi / 12) - 1);
        }

        // --- 樂器取樣引擎 ---
        const localBaseUrl = "/samples/";
        const localGuitarUrls = {
            "A2": "guitar_A2.mp3",
            "C3": "guitar_C3.mp3",
            "E4": "guitar_E4.mp3",
            "G4": "guitar_G4.mp3"
        };

        // Agent A (Melody)
        const agentASynth = new Tone.Sampler({ urls: localGuitarUrls, baseUrl: localBaseUrl }).toDestination();

        // Agent B (Harmony) & Human: 使用本地 samples
        const agentBSynth = new Tone.Sampler({ urls: localGuitarUrls, baseUrl: localBaseUrl }).toDestination();
        const humanSynth = new Tone.Sampler({ urls: localGuitarUrls, baseUrl: localBaseUrl }).toDestination();

        agentASynth.volume.value = 2;
        agentBSynth.volume.value = -3;
        humanSynth.volume.value = 5;

        // Bass Agent 合成器
        const bassSynth = new Tone.Synth({
            oscillator: { type: 'triangle' },
            envelope: { attack: 0.025, decay: 0.3, sustain: 0.55, release: 1.6 }
        }).toDestination();
        bassSynth.volume.value = 1;
        
        // 從後端獲取當前音階的音符集合
        let currentScaleNotes = new Set();
        
        // --- 鼓聲音源 ---
        // 預設使用內建合成器。如果你想用自己的鼓聲音效，請將下方的 false 改為 true，
        // 並且把你的 kick.mp3 (大鼓) 和 snare.mp3 (軍鼓) 丟進 samples/ 資料夾裡！
        const useCustomDrums = true; 
        
        let drumSampler = null;
        if (useCustomDrums) {
            const _v = Date.now();
            drumSampler = new Tone.Sampler({ 
                urls: { "C1": `kick.wav?v=${_v}`, "D1": `snare.wav?v=${_v}` }, 
                baseUrl: "/samples/" 
            }).toDestination();
            drumSampler.volume.value = 5;
        }

        // (預設備用) 內建的高保真合成器大鼓與軍鼓
        const kickSynth = new Tone.MembraneSynth({
            pitchDecay: 0.05,
            octaves: 10,
            oscillator: { type: "sine" },
            envelope: { attack: 0.001, decay: 0.4, sustain: 0.01, release: 1.4, attackCurve: "exponential" }
        }).toDestination();
        kickSynth.volume.value = 2;
        
        const snareSynth = new Tone.NoiseSynth({
            noise: { type: "white" },
            envelope: { attack: 0.005, decay: 0.1, sustain: 0.02, release: 0.05 }
        }).toDestination();
        snareSynth.volume.value = -3;

        // Hi-hat: 短噪音合成
        const hiHatSynth = new Tone.NoiseSynth({
            noise: { type: "pink" },
            envelope: { attack: 0.001, decay: 0.04, sustain: 0.0, release: 0.02 }
        });
        const hiHatFilter = new Tone.Filter(8000, "highpass").toDestination();
        hiHatSynth.connect(hiHatFilter);
        hiHatSynth.volume.value = -12;

        // 預讀落下時間 (秒)
        const FALL_DURATION = 2.0; 
        let ws = null;
        let midiOutput = null;
        if (navigator.requestMIDIAccess) {
            navigator.requestMIDIAccess().then(access => {
                const outputs = Array.from(access.outputs.values());
                if (outputs.length > 0) {
                    midiOutput = outputs[0];
                    console.log("Web MIDI 埠已連線:", midiOutput.name);
                }
            }).catch(() => {});
        }
        
        // --- MRT2 Feature Functions ---
        let savedEmbA = null, savedEmbB = null;

        function mrt2Log(msg) {
            const log = document.getElementById('mrt2-log');
            if (log) { log.innerHTML = `<div>${new Date().toLocaleTimeString()} — ${msg}</div>` + log.innerHTML; }
        }

        function sendSynth() {
            const agent = document.getElementById('synth-agent').value;
            const prompt = document.getElementById('synth-prompt').value.trim();
            if (!prompt) return;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'text_to_synth', agent, prompt }));
                mrt2Log(`🎹 Text-to-Synth → Agent ${agent.toUpperCase()}: "${prompt}"`);
            }
        }

        function quickSynth(prompt) {
            document.getElementById('synth-prompt').value = prompt;
            sendSynth();
        }

        function handleCloneFile(input) {
            const file = input.files[0];
            if (!file) return;
            document.getElementById('clone-label').textContent = file.name;
            const reader = new FileReader();
            reader.onload = (e) => {
                const agent = document.getElementById('clone-agent').value;
                const base64 = e.target.result.split(',')[1];
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: 'clone_audio', agent, audio_base64: base64, filename: file.name }));
                    mrt2Log(`🎤 Audio Cloning → Agent ${agent.toUpperCase()}: ${file.name}`);
                }
            };
            reader.readAsDataURL(file);
        }

        function updateMixLabel(val) {
            document.getElementById('mix-label').textContent = val + '%';
        }

        function sendMix() {
            const alpha = parseInt(document.getElementById('mix-slider').value) / 100;
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ action: 'prompt_mix', alpha }));
                mrt2Log(`🔀 Prompt Mixing: α=${alpha.toFixed(2)}`);
            }
        }

        function tellBackend() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ 
                    action: "update_settings", 
                    scale: scaleSelect.value,
                    bpm: parseInt(bpmInput.value)
                }));
            }
        }
        
        scaleSelect.onchange = () => { tellBackend(); };
        bpmInput.onchange = () => { tellBackend(); };

        btn.onclick = async () => {
            btn.disabled = true;
            btn.innerText = 'Loading samples...';
            
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            if (audioCtx.state === 'suspended') {
                audioCtx.resume();
            }

            await Tone.start();
            await Tone.loaded();

            splashEl.style.display = 'none';
            gameContainer.style.display = 'flex';
            infoBar.style.display = 'flex';
            document.getElementById('octave-badge').style.display = 'block';
            document.getElementById('kb-hint').style.display = 'block';
            updateKeyLabels();

            // === 人類鍵盤輸入 ===
            document.addEventListener('keydown', (ev) => {
                if (ev.repeat || !ws || ws.readyState !== WebSocket.OPEN) return;
                if (['INPUT','SELECT'].includes(ev.target.tagName)) return;
                const key = ev.key.toLowerCase();
                if (key === 'z') { humanBaseMidi = Math.max(36, humanBaseMidi - 12); updateKeyLabels(); return; }
                if (key === 'x') { humanBaseMidi = Math.min(72, humanBaseMidi + 12); updateKeyLabels(); return; }
                const semi = KEY_SEMITONE[key];
                if (semi === undefined) return;
                const midi = humanBaseMidi + semi;
                if (midi < startMidi || midi > endMidi || pressedKeys.has(midi)) return;
                
                // 防呆：只允許彈奏當前音階內的音
                const noteClass = midi % 12;
                if (currentScaleNotes.size > 0 && !currentScaleNotes.has(noteClass)) {
                    // 彈錯音，可以給點視覺提示
                    const keyEl = document.getElementById('key-' + midi);
                    if (keyEl) {
                        keyEl.style.backgroundColor = '#ff4757'; // 閃紅燈
                        setTimeout(() => keyEl.style.backgroundColor = '', 200);
                    }
                    return;
                }

                pressedKeys.add(midi);
                humanSynth.triggerAttack(Tone.Frequency(midi,'midi').toFrequency(), Tone.now());
                const keyEl = document.getElementById('key-' + midi);
                if (keyEl) keyEl.classList.add('active-human');
                spawnHumanBubble(midi);
                ws.send(JSON.stringify({ action: 'human_note', midi }));
                const badge = document.getElementById('human-badge');
                badge.classList.add('show');
                clearTimeout(humanBadgeTimer);
                humanBadgeTimer = setTimeout(() => badge.classList.remove('show'), 1800);
            });
            document.addEventListener('keyup', (ev) => {
                const key = ev.key.toLowerCase();
                const semi = KEY_SEMITONE[key];
                if (semi === undefined) return;
                const midi = humanBaseMidi + semi;
                pressedKeys.delete(midi);
                humanSynth.triggerRelease(Tone.Frequency(midi,'midi').toFrequency(), Tone.now() + 0.08);
                const keyEl = document.getElementById('key-' + midi);
                if (keyEl) keyEl.classList.remove('active-human');
            });
            
            ws = new WebSocket(`ws://${location.host}/ws`);
            ws.onopen = () => { tellBackend(); };
            
            ws.onmessage = (e) => {
                const data = JSON.parse(e.data);
                
                // 接收後端傳來的音階資訊
                if (data.type === 'scale_info') {
                    currentScaleNotes = new Set(data.notes.map(n => n % 12));
                    return;
                }

                // MRT2 feature 回饋
                if (data.type === 'mrt2_status') {
                    document.getElementById('mrt2-status').textContent = 'Status: ' + data.message;
                    document.getElementById('mrt2-status').style.color = data.ok ? '#34d399' : '#f87171';
                    mrt2Log(data.ok ? `✅ ${data.message}` : `❌ ${data.message}`);
                    return;
                }

                // === 處理多軌同步合奏封包 (同時播放 Drums + MRT2 Bass + MRT2 Melody) ===
                if (data.type === 'measure_bundle') {
                    if (data.scale) {
                        currentScaleNotes = new Set(data.scale.map(n => n % 12));
                    }
                    const measureRecvToneTime = Tone.now();
                    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    if (audioCtx.state === 'suspended') audioCtx.resume();
                    const measureRecvAudioTime = audioCtx.currentTime;

                    // 1. 播放鼓聲 (對齊 measureRecvToneTime + FALL_DURATION)
                    if (data.drum && data.drum.events) {
                        data.drum.events.forEach((ev) => {
                            const playTime = measureRecvToneTime + (ev.play_delay_sec || 0) + FALL_DURATION;
                            const vel = ev.velocity !== undefined ? ev.velocity : 0.8;
                            if (ev.note_name === 'KICK') {
                                if (useCustomDrums && drumSampler) {
                                    drumSampler.triggerAttackRelease("C1", ev.duration, playTime, vel);
                                } else {
                                    kickSynth.volume.value = Tone.gainToDb(vel) + 2;
                                    kickSynth.triggerAttackRelease("C1", ev.duration, playTime);
                                }
                            } else if (ev.note_name === 'SNARE') {
                                if (useCustomDrums && drumSampler) {
                                    drumSampler.triggerAttackRelease("D1", ev.duration, playTime, vel);
                                } else {
                                    snareSynth.volume.value = Tone.gainToDb(vel) - 3;
                                    snareSynth.triggerAttackRelease(ev.duration, playTime);
                                }
                            } else if (ev.note_name === 'HH' || ev.note_name === 'HH_OPEN') {
                                const dur = ev.note_name === 'HH_OPEN' ? 0.18 : 0.03;
                                hiHatSynth.volume.value = Tone.gainToDb(vel) - 6;
                                hiHatSynth.triggerAttackRelease(dur, playTime);
                            }
                        });
                    }

                    // 2. 播放 MRT2 Bass 音訊 (立體聲神經低音波形)
                    if (data.bass && data.bass.audio_base64) {
                        const playTime = measureRecvAudioTime + FALL_DURATION;
                        const bin = window.atob(data.bass.audio_base64);
                        const bytes = new Uint8Array(bin.length);
                        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                        audioCtx.decodeAudioData(bytes.buffer, (buffer) => {
                            const src = audioCtx.createBufferSource();
                            src.buffer = buffer;
                            src.connect(audioCtx.destination);
                            src.start(playTime);
                        });
                    }

                    // 3. 渲染 Bass 音符方塊與琴鍵下沉 (與鼓/Bass音訊同步)
                    if (data.bass && data.bass.events) {
                        data.bass.events.forEach((ev) => {
                            if (typeof ev.note !== 'number') return;
                            const fallTime = measureRecvToneTime + (ev.play_delay_sec || 0);
                            const hitTime = fallTime + FALL_DURATION;
                            const noteNum = ev.note;
                            if (noteNum >= startMidi && noteNum <= endMidi && keyPos[noteNum]) {
                                const heightVh = (ev.duration / FALL_DURATION) * 85;
                                const totalDistance = 85 + heightVh;
                                const totalAnimTime = totalDistance / (85 / FALL_DURATION);
                                const block = document.createElement('div');
                                block.className = 'falling-note color-bass';
                                block.style.left = `${keyPos[noteNum].left}%`;
                                block.style.width = `${keyPos[noteNum].width}%`;
                                block.style.height = `${heightVh}vh`;
                                block.style.top = `-${heightVh}vh`;
                                rollContainer.appendChild(block);
                                setTimeout(() => {
                                    block.style.transition = `transform ${totalAnimTime}s linear`;
                                    block.style.transform = `translateY(${totalDistance}vh)`;
                                }, Math.max(0, (fallTime - Tone.now()) * 1000));
                                setTimeout(() => {
                                    const key = document.getElementById(`key-${noteNum}`);
                                    if (key) {
                                        key.classList.add('active-bass');
                                        key.style.transform = 'translateY(3px)';
                                        setTimeout(() => {
                                            key.classList.remove('active-bass');
                                            key.style.transform = '';
                                        }, ev.duration * 1000);
                                    }
                                    if (midiOutput) {
                                        try {
                                            midiOutput.send([0x90, noteNum, 110]);
                                            setTimeout(() => midiOutput.send([0x80, noteNum, 0]), ev.duration * 1000);
                                        } catch(e) {}
                                    }
                                }, Math.max(0, (hitTime - Tone.now()) * 1000));
                                setTimeout(() => block.remove(), Math.max(0, (fallTime + totalAnimTime - Tone.now()) * 1000) + 100);
                            }
                        });
                    }

                    // 4. 播放 MRT2 旋律音訊 (Agent A 或 B，與 Bass/鼓聲同一瞬間響起)
                    if (data.melody && data.melody.audio_base64) {
                        const playTime = measureRecvAudioTime + FALL_DURATION;
                        const bin = window.atob(data.melody.audio_base64);
                        const bytes = new Uint8Array(bin.length);
                        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                        audioCtx.decodeAudioData(bytes.buffer, (buffer) => {
                            const src = audioCtx.createBufferSource();
                            src.buffer = buffer;
                            src.connect(audioCtx.destination);
                            src.start(playTime);
                        });
                    }

                    // 5. 渲染 Melody 音符方塊與琴鍵下沉 (與 Bass 同步在鍵盤上交響演奏)
                    if (data.melody && data.melody.events) {
                        const isA = (data.melody.agent === 'Agent_A');
                        const synth = isA ? agentASynth : agentBSynth;
                        data.melody.events.forEach((ev) => {
                            const fallTime = measureRecvToneTime + (ev.play_delay_sec || 0);
                            const hitTime = fallTime + FALL_DURATION;
                            const notesArr = Array.isArray(ev.note) ? ev.note : [ev.note];
                            if (!data.melody.audio_base64) {
                                const freqs = notesArr.map(n => Tone.Frequency(n, "midi").toFrequency());
                                synth.triggerAttackRelease(freqs, ev.duration, hitTime);
                            }
                            notesArr.forEach(noteNum => {
                                if (noteNum < startMidi || noteNum > endMidi || !keyPos[noteNum]) return;
                                const heightVh = (ev.duration / FALL_DURATION) * 85;
                                const totalDistance = 85 + heightVh;
                                const totalAnimationTime = totalDistance / (85 / FALL_DURATION);
                                const block = document.createElement('div');
                                block.className = `falling-note ${isA ? 'color-a' : 'color-b'}`;
                                block.style.left = `${keyPos[noteNum].left}%`;
                                block.style.width = `${keyPos[noteNum].width}%`;
                                block.style.height = `${heightVh}vh`;
                                block.style.top = `-${heightVh}vh`;
                                rollContainer.appendChild(block);
                                setTimeout(() => {
                                    block.style.transition = `transform ${totalAnimationTime}s linear`;
                                    block.style.transform = `translateY(${totalDistance}vh)`;
                                }, Math.max(0, (fallTime - Tone.now()) * 1000));
                                setTimeout(() => {
                                    const key = document.getElementById(`key-${noteNum}`);
                                    if (key) {
                                        key.classList.add(`active-${isA ? 'a' : 'b'}`);
                                        key.style.transform = 'translateY(3px)';
                                        setTimeout(() => {
                                            key.classList.remove(`active-${isA ? 'a' : 'b'}`);
                                            key.style.transform = '';
                                        }, ev.duration * 1000);
                                    }
                                    if (midiOutput) {
                                        try {
                                            midiOutput.send([0x90, noteNum, 100]);
                                            setTimeout(() => midiOutput.send([0x80, noteNum, 0]), ev.duration * 1000);
                                        } catch(e) {}
                                    }
                                }, Math.max(0, (hitTime - Tone.now()) * 1000));
                                setTimeout(() => {
                                    block.remove();
                                }, Math.max(0, (fallTime + totalAnimationTime - Tone.now()) * 1000) + 100);
                            });
                        });
                    }
                    return;
                }

                const measureRecvTime = Tone.now();
                
                // 處理鼓聲軌道
                if (data.agent === 'Drum') {
                    data.events.forEach((ev) => {
                        const playTime = measureRecvTime + (ev.play_delay_sec || 0) + FALL_DURATION;
                        const vel = ev.velocity !== undefined ? ev.velocity : 0.8;

                        if (ev.note_name === 'KICK') {
                            if (useCustomDrums && drumSampler) {
                                drumSampler.triggerAttackRelease("C1", ev.duration, playTime, vel);
                            } else {
                                kickSynth.volume.value = Tone.gainToDb(vel) + 2;
                                kickSynth.triggerAttackRelease("C1", ev.duration, playTime);
                            }
                        } else if (ev.note_name === 'SNARE') {
                            if (useCustomDrums && drumSampler) {
                                drumSampler.triggerAttackRelease("D1", ev.duration, playTime, vel);
                            } else {
                                snareSynth.volume.value = Tone.gainToDb(vel) - 3;
                                snareSynth.triggerAttackRelease(ev.duration, playTime);
                            }
                        } else if (ev.note_name === 'GHOST') {
                            // Ghost snare: 低音量
                            const ghostSnare = new Tone.NoiseSynth({
                                noise: { type: "white" },
                                envelope: { attack: 0.003, decay: 0.06, sustain: 0, release: 0.03 }
                            }).toDestination();
                            ghostSnare.volume.value = Tone.gainToDb(ev.velocity || 0.15) - 6;
                            ghostSnare.triggerAttackRelease(ev.duration, playTime);
                        } else if (ev.note_name === 'HH' || ev.note_name === 'HH_OPEN') {
                            const dur = ev.note_name === 'HH_OPEN' ? 0.18 : 0.03;
                            // velocity 控制音量
                            hiHatSynth.volume.value = Tone.gainToDb(vel) - 6;
                            hiHatSynth.triggerAttackRelease(dur, playTime);
                        }
                    });
                    return;
                }

                // 處理 Bass Agent 軌道
                if (data.agent === 'Agent_Bass') {
                    if (data.audio_base64) {
                        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                        if (audioCtx.state === 'suspended') audioCtx.resume();
                        const playTime = audioCtx.currentTime + FALL_DURATION;
                        const bin = window.atob(data.audio_base64);
                        const bytes = new Uint8Array(bin.length);
                        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
                        audioCtx.decodeAudioData(bytes.buffer, (buffer) => {
                            const src = audioCtx.createBufferSource();
                            src.buffer = buffer;
                            src.connect(audioCtx.destination);
                            src.start(playTime);
                        });
                    }

                    data.events.forEach((ev) => {
                        if (typeof ev.note !== 'number') return;
                        const fallTime = measureRecvTime + ev.play_delay_sec;
                        const hitTime = fallTime + FALL_DURATION;
                        const noteNum = ev.note;
                        if (!data.audio_base64) {
                            bassSynth.triggerAttackRelease(
                                Tone.Frequency(noteNum, 'midi').toFrequency(), ev.duration, hitTime
                            );
                        }
                        if (noteNum >= startMidi && noteNum <= endMidi && keyPos[noteNum]) {
                            const heightVh = (ev.duration / FALL_DURATION) * 85;
                            const totalDistance = 85 + heightVh;
                            const totalAnimTime = totalDistance / (85 / FALL_DURATION);
                            const block = document.createElement('div');
                            block.className = 'falling-note color-bass';
                            block.style.left = `${keyPos[noteNum].left}%`;
                            block.style.width = `${keyPos[noteNum].width}%`;
                            block.style.height = `${heightVh}vh`;
                            block.style.top = `-${heightVh}vh`;
                            rollContainer.appendChild(block);
                            setTimeout(() => {
                                block.style.transition = `transform ${totalAnimTime}s linear`;
                                block.style.transform = `translateY(${totalDistance}vh)`;
                            }, Math.max(0, (fallTime - Tone.now()) * 1000));
                            setTimeout(() => {
                                const key = document.getElementById(`key-${noteNum}`);
                                if (key) {
                                    key.classList.add('active-bass');
                                    key.style.transform = 'translateY(3px)';
                                    setTimeout(() => {
                                        key.classList.remove('active-bass');
                                        key.style.transform = '';
                                    }, ev.duration * 1000);
                                }
                                if (midiOutput) {
                                    try {
                                        midiOutput.send([0x90, noteNum, 110]);
                                        setTimeout(() => midiOutput.send([0x80, noteNum, 0]), ev.duration * 1000);
                                    } catch(e) {}
                                }
                            }, Math.max(0, (hitTime - Tone.now()) * 1000));
                            setTimeout(() => block.remove(), Math.max(0, (fallTime + totalAnimTime - Tone.now()) * 1000) + 100);
                        }
                    });
                    return;
                }

                // 處理 Agent 軌道
                const isA = data.agent === 'Agent_A';
                const synth = isA ? agentASynth : agentBSynth; 
                
                if (data.audio_base64) {
                    if (!audioCtx) {
                        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    if (audioCtx.state === 'suspended') {
                        audioCtx.resume();
                    }
                    const playTime = audioCtx.currentTime + FALL_DURATION;
                    
                    const binaryString = window.atob(data.audio_base64);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {
                        bytes[i] = binaryString.charCodeAt(i);
                    }
                    audioCtx.decodeAudioData(
                        bytes.buffer,
                        (buffer) => {
                            const source = audioCtx.createBufferSource();
                            source.buffer = buffer;
                            source.connect(audioCtx.destination);
                            source.start(playTime);
                        },
                        (err) => {
                            console.error("WAV Decode Error for " + data.agent, err);
                        }
                    );
                } else if (!data.is_async_audio) {
                    if (data.agent === 'Agent_A' || data.agent === 'Agent_B') {
                        // 方塊掉下來時，因為 MRT2 音頻還在後台算，這裡會先用 Tone.js 播放對齊的 MIDI 聲音
                    }
                }
                
                data.events.forEach((ev) => {
                    const fallTime = measureRecvTime + ev.play_delay_sec;
                    const hitTime = fallTime + FALL_DURATION;
                    
                    const notesArr = Array.isArray(ev.note) ? ev.note : [ev.note];
                    
                    // -- 聲音排程 (沒有 MRT2 的時候才發聲) --
                    if (!data.audio_base64) {
                        const freqs = notesArr.map(n => Tone.Frequency(n, "midi").toFrequency());
                        synth.triggerAttackRelease(freqs, ev.duration, hitTime);
                    }
                    
                    // -- 視覺方塊排程 --
                    notesArr.forEach(noteNum => {
                        if(noteNum < startMidi || noteNum > endMidi) return;
                        
                        const heightVh = (ev.duration / FALL_DURATION) * 85; 
                        const totalDistance = 85 + heightVh;
                        const totalAnimationTime = totalDistance / (85 / FALL_DURATION);
                        
                        const block = document.createElement('div');
                        block.className = `falling-note ${isA ? 'color-a' : 'color-b'}`;
                        block.style.left = `${keyPos[noteNum].left}%`;
                        block.style.width = `${keyPos[noteNum].width}%`;
                        block.style.height = `${heightVh}vh`;
                        block.style.top = `-${heightVh}vh`; 
                        rollContainer.appendChild(block);
                        
                        setTimeout(() => {
                            block.style.transition = `transform ${totalAnimationTime}s linear`;
                            block.style.transform = `translateY(${totalDistance}vh)`;
                        }, Math.max(0, (fallTime - Tone.now()) * 1000));
                        
                        setTimeout(() => {
                            const key = document.getElementById(`key-${noteNum}`);
                            if (key) {
                                key.classList.add(`active-${isA ? 'a' : 'b'}`);
                                key.style.transform = 'translateY(3px)';
                                setTimeout(() => {
                                    key.classList.remove(`active-${isA ? 'a' : 'b'}`);
                                    key.style.transform = '';
                                }, ev.duration * 1000);
                            }
                            if (midiOutput) {
                                try {
                                    midiOutput.send([0x90, noteNum, 100]);
                                    setTimeout(() => midiOutput.send([0x80, noteNum, 0]), ev.duration * 1000);
                                } catch(e) {}
                            }
                        }, Math.max(0, (hitTime - Tone.now()) * 1000));
                        setTimeout(() => {
                            block.remove();
                        }, Math.max(0, (fallTime + totalAnimationTime - Tone.now()) * 1000) + 100);
                    });
                });
            };
        };

        function spawnHumanBubble(noteNum) {
            if (!keyPos[noteNum]) return;
            const kp = keyPos[noteNum];
            const block = document.createElement('div');
            block.className = 'falling-note color-human';
            block.style.left = `${kp.left}%`;
            block.style.width = `${kp.width}%`;
            block.style.height = '6vh';
            block.style.top = `79vh`;
            block.style.opacity = '1';
            rollContainer.appendChild(block);
            requestAnimationFrame(() => {
                block.style.transition = 'transform 1.2s ease-out, opacity 1.2s ease-out';
                block.style.transform = 'translateY(-35vh)';
                block.style.opacity = '0';
            });
            setTimeout(() => block.remove(), 1300);
        }
    </script>
</body>
</html>
""".replace('PLACEHOLDER_OPTIONS', options_html)

@app.get("/")
async def get_ui():
    return HTMLResponse(html_content)

@app.get("/favicon.ico")
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools():
    from fastapi.responses import JSONResponse
    return JSONResponse({})

@app.websocket("/ws")
async def websocket_music_loop(websocket: WebSocket):
    await websocket.accept()

    has_mrt = (mrt2_agent_a is not None and mrt2_agent_a.has_mrt2)
    session_active = asyncio.Event()
    session_active.set()

    state = {
        "scale": SCALES_DICT["G_Minor"],
        "bpm": 120,
        "human_notes": [],
        "chord_loop": random.choice(HH_PROGRESSIONS),
        "chord_pos": 0,
        "loop_count": 0,
    }

    async def safe_send_json(data: dict) -> bool:
        if not session_active.is_set():
            return False
        try:
            from starlette.websockets import WebSocketState
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
                return True
        except Exception:
            session_active.clear()
        return False

    await safe_send_json({
        "type": "mrt2_status",
        "ok": has_mrt,
        "message": "MRT2 Ready (mrt2_small JAX)" if has_mrt else "MRT2 Unavailable (Fallback Mode)"
    })

    async def settings_listener():
        try:
            while session_active.is_set():
                msg = await websocket.receive_json()
                if msg.get("action") == "update_settings":
                    if msg.get("scale") in SCALES_DICT:
                        state["scale"] = SCALES_DICT[msg["scale"]]
                    if msg.get("bpm"):
                        state["bpm"] = max(40, min(240, int(msg["bpm"])))
                elif msg.get("action") == "human_note":
                    midi = int(msg.get("midi", 60))
                    state["human_notes"].append(midi)
                    if len(state["human_notes"]) > 16:
                        state["human_notes"] = state["human_notes"][-16:]

                elif msg.get("action") == "text_to_synth":
                    agent_name = msg.get("agent", "a")
                    prompt = msg.get("prompt", "")
                    target = mrt2_agent_a if agent_name == "a" else mrt2_agent_b
                    if target and target.has_mrt2:
                        result = target.text_to_synth(prompt)
                        await safe_send_json({
                            "type": "mrt2_status", "ok": result is not None,
                            "message": f"Text-to-Synth Agent {agent_name.upper()}: '{prompt}'"
                        })

                elif msg.get("action") == "clone_audio":
                    agent_name = msg.get("agent", "a")
                    audio_b64 = msg.get("audio_base64", "")
                    filename = msg.get("filename", "unknown")
                    target = mrt2_agent_a if agent_name == "a" else mrt2_agent_b
                    if target and target.has_mrt2 and audio_b64:
                        try:
                            import base64 as b64mod
                            raw = b64mod.b64decode(audio_b64)
                            audio_np = np.frombuffer(raw, dtype=np.float32)
                            if len(audio_np) == 0:
                                audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                            result = target.clone_audio(audio_np, sample_rate=48000)
                            await safe_send_json({
                                "type": "mrt2_status", "ok": result is not None,
                                "message": f"Audio Clone {agent_name.upper()}: {filename}"
                            })
                        except Exception as e:
                            await safe_send_json({"type": "mrt2_status", "ok": False, "message": str(e)})

                elif msg.get("action") == "prompt_mix":
                    alpha = float(msg.get("alpha", 0.5))
                    if mrt2_agent_a and mrt2_agent_b:
                        emb_a = mrt2_agent_a._style_embedding
                        emb_b = mrt2_agent_b._style_embedding
                        if emb_a is not None and emb_b is not None:
                            mixed = mrt2_agent_a.mix_prompts(emb_a, emb_b, alpha=alpha)
                            await safe_send_json({
                                "type": "mrt2_status", "ok": mixed is not None,
                                "message": f"Prompt Mix α={alpha:.0%}"
                            })
        except Exception:
            session_active.clear()

    listener_task = asyncio.create_task(settings_listener())

    agent_bass: Optional[MusicAgent] = None
    round_count = 0
    loop = asyncio.get_running_loop()
    import functools

    # 建立非同步佇列：預先生成雙緩衝 (Prefetch Pipeline)，徹底消除等待延遲
    bundle_queue = asyncio.Queue(maxsize=2)

    async def measure_producer():
        nonlocal round_count, agent_bass
        while session_active.is_set():
            current_scale = list(state["scale"])
            current_bpm = state["bpm"]

            beat_sec = 60.0 / current_bpm
            step_duration = beat_sec / 4.0
            steps_per_measure = 16
            measure_sec = steps_per_measure * step_duration
            gen_duration = min(0.8, measure_sec)

            # 1. 鼓點事件
            events_drum = []
            for d_step in [0, 8]:
                events_drum.append({
                    "note": "C1", "onset_beat": d_step, "duration": 0.4,
                    "play_delay_sec": round(d_step * step_duration, 3),
                    "note_name": "KICK", "velocity": 0.9
                })
            for d_step in [4, 12]:
                events_drum.append({
                    "note": "D1", "onset_beat": d_step, "duration": 0.3,
                    "play_delay_sec": round(d_step * step_duration, 3),
                    "note_name": "SNARE", "velocity": 0.85
                })
            for d_step in [0, 2, 4, 6, 8, 10, 12, 14]:
                events_drum.append({
                    "note": "HH", "onset_beat": d_step, "duration": 0.05,
                    "play_delay_sec": round(d_step * step_duration, 3),
                    "note_name": "HH", "velocity": 0.4
                })

            # 2. 決定 Melody Agent (Agent A 與 Agent B 永續輪流)
            is_turn_a = (round_count % 2 == 0)
            current_agent = mrt2_agent_a if is_turn_a else (mrt2_agent_b or mrt2_agent_a)
            agent_name = "Agent_A" if is_turn_a else "Agent_B"
            round_count += 1

            human_notes_to_play = list(state["human_notes"])
            state["human_notes"].clear()

            # 3. 同時並行生成 Bass 與 Melody (Concurrent Inference via asyncio.gather)
            async def generate_bass_task():
                if mrt2_agent_bass and mrt2_agent_bass.has_mrt2:
                    raw = await loop.run_in_executor(
                        jax_pool,
                        functools.partial(mrt2_agent_bass.generate_audio, duration_sec=gen_duration)
                    )
                    if raw is not None:
                        b64 = encode_audio_base64(raw)
                        evs = audio_waveform_to_midi_events(
                            raw, sample_rate=48000, duration_sec=gen_duration,
                            num_notes=4, min_pitch=36, max_pitch=52
                        )
                        return {"agent": "Agent_Bass", "events": evs, "audio_base64": b64}
                # Fallback
                chord_loop = state["chord_loop"]
                degree = chord_loop[state["chord_pos"] % len(chord_loop)]
                state["chord_pos"] += 1
                idx = 7 + degree
                triad = [current_scale[idx] - 24, current_scale[idx + 2] - 24, current_scale[idx + 4] - 24]
                nonlocal agent_bass
                if agent_bass is None or agent_bass.base_scale != current_scale:
                    agent_bass = MusicAgent("Agent_Bass", current_scale, play_probability=0.8)
                evs = agent_bass.generate_bass_performance(triad, steps=steps_per_measure, step_duration=step_duration)
                return {"agent": "Agent_Bass", "events": evs, "audio_base64": ""}

            async def generate_melody_task():
                if current_agent and current_agent.has_mrt2:
                    if human_notes_to_play and hasattr(current_agent, "notes_to_continuation"):
                        fmt_notes = [
                            {"pitch": p, "start_time": i * 0.15, "end_time": (i + 1) * 0.15}
                            for i, p in enumerate(human_notes_to_play[-4:])
                        ]
                        payload = await loop.run_in_executor(
                            jax_pool,
                            functools.partial(current_agent.notes_to_continuation, fmt_notes, duration_sec=gen_duration)
                        )
                        raw = (
                            np.frombuffer(payload["audio_bytes"], dtype=np.float32).reshape(tuple(payload["metadata"]["shape"]))
                            if payload and "audio_bytes" in payload else None
                        )
                    else:
                        raw = await loop.run_in_executor(
                            jax_pool,
                            functools.partial(current_agent.generate_audio, duration_sec=gen_duration)
                        )
                    if raw is not None:
                        b64 = encode_audio_base64(raw)
                        evs = audio_waveform_to_midi_events(
                            raw, sample_rate=48000, duration_sec=gen_duration,
                            scale=current_scale, num_notes=6, min_pitch=48, max_pitch=84
                        )
                        return {"agent": agent_name, "events": evs, "audio_base64": b64}
                return {"agent": agent_name, "events": [], "audio_base64": ""}

            try:
                bass_data, melody_data = await asyncio.gather(
                    generate_bass_task(),
                    generate_melody_task()
                )
            except Exception as e:
                print(f"[MRT2 Gather Error]: {e}")
                bass_data = {"agent": "Agent_Bass", "events": [], "audio_base64": ""}
                melody_data = {"agent": agent_name, "events": [], "audio_base64": ""}

            bundle = {
                "type": "measure_bundle",
                "measure_index": round_count,
                "measure_duration": measure_sec,
                "scale": current_scale,
                "drum": {"events": events_drum},
                "bass": bass_data,
                "melody": melody_data,
            }

            if session_active.is_set():
                await bundle_queue.put(bundle)
            else:
                break

    producer_task = asyncio.create_task(measure_producer())

    try:
        while session_active.is_set():
            bundle = await bundle_queue.get()
            if not session_active.is_set():
                break

            # 100% 同步發送完整合奏封包給前端 (Drum + Bass + Melody 同一時間軸)
            await safe_send_json(bundle)
            print(f"[MRT2 Sync Duet] 第 {bundle['measure_index']} 小節已同步發送: "
                  f"Bass({len(bundle['bass']['events'])}音符) + "
                  f"{bundle['melody']['agent']}({len(bundle['melody']['events'])}音符)")

            # 精確等待本小節播放時間，維持完美音樂律動與無縫接棒
            await asyncio.sleep(bundle["measure_duration"])

    except Exception as e:
        print(f"Session closed: {e}")
    finally:
        session_active.clear()
        listener_task.cancel()
        producer_task.cancel()

if __name__ == "__main__":
    print("啟動 AI Duet Web Server: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
