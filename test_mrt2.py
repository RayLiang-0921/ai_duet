#from magenta_rt import utils
#from magenta_rt import MagentaRT2Jax
#mrt = MagentaRT2Jax(size="mrt2_small")
#embedding = mrt.embed_style("dj scratch, 140bpm, trap with piano sample, G minor")


'''
    Args:
      conditioning: Dictionary mapping TokensConfig.key strings to their values
        Values can be lists of integers (tokens) or raw embeddings (e.g. Style)
      cfg_scales: Optional dictionary to override default CFG scales for this call.
          Example: {'musiccoca': 5.0, 'notes': 3.0}
      temperature: Sampling temperature. None falls back to
        ``self.temperature``.
      top_k: Top-k sampling threshold. None falls back to ``self.top_k``.
      frames: Number of frames to generate (25 frames = 1 second at 48kHz).
      state: Streaming state from a previous call. If None, a fresh state is
          created.
    
    Returns:
      (waveform, state) — a Waveform at 48kHz stereo, and the updated state
      for continuation.
'''

#wav, state = mrt.generate(state=None, conditioning={"mulan_tokens_25hz": embedding}, frames=100) 
#wav.write("output.wav")
#沒有使用GPU
#有沒有機會測試base


import queue
import threading
import time

import numpy as np
import pasimple  # WSL 內經 WSLg PulseAudio 播放（sounddevice 在 WSL 需要 sudo 安裝 portaudio）
from magenta_rt import MagentaRT2Jax

CHUNK_FRAMES = 50          # 25 frames = 1 second, so each chunk is 2 s of audio
BUFFER_CHUNKS = 4          # how many chunks to generate ahead of playback
PREBUFFER = 2              # 先存滿幾段才開始播放。GPU 偶爾會有一段略超過 2 秒，
                           # 預存可吸收這種抖動，避免開頭就斷音

# MRT2 沒有 BPM 參數，用兩種方式控制速度：
#   1. 在風格文字裡寫 BPM（模型的「建議」，不保證準）
#   2. 用 drum_pianoroll_tokens 在每一拍的位置送出鼓點，把節拍鎖在 BPM 上
BPM = 78
USE_DRUMS = True           # False = 完全不送鼓點條件。注意：不要改成每幀送 [0]，
                           # 那等於在叫模型「不要有鼓」，CFG 越大壓得越兇
DRUM_CFG = 2.0             # 鼓點 conditioning 強度（預設 1.0），越大越緊跟節拍
DRUM_SUBDIV = 2            # 一拍送幾個鼓點。2（八分音符）實測律動最穩，1（每拍）較鬆散
FRAMES_PER_MIN = 25 * 60   # 1 frame = 40 ms

# MRT2 也沒有調性參數。只寫在文字裡幾乎無效；改用 pianoroll_with_onsets_tokens
# 送出該調的和弦進行（每小節一個和弦），實測主要音高都會落在調內。
USE_CHORDS = True          # False = 不鎖調性，讓模型自由發揮（最連貫，但調性不固定）
KEY_ROOT = "A"             # C C# D D# E F F# G G# A A# B
KEY_MODE = "minor"         # "major" 或 "minor"
PROGRESSION = [0, 5, 3, 4] # 級數（0 起算）：I–vi–IV–V
NOTES_CFG = 0.4            # 音符 conditioning 強度（預設 1.0）。實測 1.0 會把旋律綁死、
                           # 律動消失；0.4 保留調性又不僵硬
BARS_PER_CHORD = 2         # 一個和弦撐幾小節，越多越不急促
CHORD_OCTAVES = (48, 60)   # 和弦音所在音域 (C3–B3)；範圍越窄留給模型的空間越大
CHORD_ONSET = False        # True = 每次換和弦都硬起音，聽起來像整塊和弦敲下去

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE_STEPS = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}[KEY_MODE]

mrt = MagentaRT2Jax(size="mrt2_small")
style = mrt.embed_style(f"jazz instrumental, in {KEY_ROOT} {KEY_MODE}")  # or any other style prompt you like
# 先轉成 token，避免逐幀生成時每一幀都重新 tokenize
style_tokens = list(mrt.tokenize_style(style))


def beat_index(frame: int) -> int:
    return (frame * BPM) // FRAMES_PER_MIN


def _subdiv_index(frame: int) -> int:
    return (frame * BPM * DRUM_SUBDIV) // FRAMES_PER_MIN


def is_beat(frame: int) -> bool:
    """第 frame 幀是否剛好落在新的鼓點上（依 DRUM_SUBDIV 決定密度）。"""
    return frame == 0 or _subdiv_index(frame) != _subdiv_index(frame - 1)


def chord_tokens(frame: int) -> list:
    """目前和弦的 pianoroll：和弦音設為發聲，其餘交給模型（-1）。"""
    chord_idx = beat_index(frame) // 4 // BARS_PER_CHORD
    degree = PROGRESSION[chord_idx % len(PROGRESSION)]
    root = NOTE_NAMES.index(KEY_ROOT)
    chord = {(root + SCALE_STEPS[(degree + k) % 7]) % 12 for k in (0, 2, 4)}
    new_chord = frame == 0 or chord_idx != beat_index(frame - 1) // 4 // BARS_PER_CHORD
    # 2 = 起音（onset），3 = 持續發聲但由模型決定怎麼彈
    value = 2 if (new_chord and CHORD_ONSET) else 3
    lo, hi = CHORD_OCTAVES
    return [value if lo <= p < hi and p % 12 in chord else -1 for p in range(128)]


def build_conditioning(frame: int) -> tuple:
    """組出這一幀的 conditioning 與 cfg_scales；沒用到的條件就整個不送。"""
    cond = {"mulan_tokens_25hz": style_tokens}
    cfg = {}
    if USE_DRUMS:
        cond["drum_pianoroll_tokens"] = [1 if is_beat(frame) else 0]
        cfg["drums"] = DRUM_CFG
    if USE_CHORDS:
        cond["pianoroll_with_onsets_tokens"] = chord_tokens(frame)
        cfg["notes"] = NOTES_CFG
    return cond, cfg

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=BUFFER_CHUNKS)
stop = threading.Event()


def generator():
    state = None
    frame = 0
    while not stop.is_set():
        # conditioning 在一次 generate() 內固定不變，所以逐幀生成才能讓鼓點落在指定的幀上
        parts = []
        for _ in range(CHUNK_FRAMES):
            cond, cfg = build_conditioning(frame)
            wav, state = mrt.generate(                  # always pass the NEW state back in:
                conditioning=cond,                      # the old one is freed after each call
                cfg_scales=cfg,
                frames=1,
                state=state,
            )
            parts.append(wav.samples)
            frame += 1
        samples = np.concatenate(parts).astype(np.float32)
        while not stop.is_set():                        # wait while the buffer is full
            try:
                audio_q.put(samples, timeout=0.5)
                break
            except queue.Full:
                continue


def player():
    # 等預存足夠再開始，開頭才不會因為第一段生成較慢而斷掉
    while not stop.is_set() and audio_q.qsize() < PREBUFFER:
        time.sleep(0.05)
    with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                           2, 48000, app_name="ai_duet") as pa:
        while not stop.is_set():
            try:
                samples = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            pa.write(np.ascontiguousarray(samples).tobytes())  # blocks while playing


gen_thread = threading.Thread(target=generator)
play_thread = threading.Thread(target=player)
gen_thread.start()
play_thread.start()

print("Playing - press Enter to stop")
input()
stop.set()
# 等 GPU 上正在跑的 generate() 結束再離開，否則 JAX 會在程式關閉時 abort (core dump)
gen_thread.join()
play_thread.join()
