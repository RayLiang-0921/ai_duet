"""
test_mrt2.py — MRT2 即時串流播放：可設定風格、速度、調性、有沒有鼓。

在 WSL GPU 環境執行：雙擊 run_stream_gputest.bat
"""
import queue
import threading

import numpy as np
from magenta_rt import MagentaRT2Jax

# ===== 設定 =====
INSTRUMENT = "solo electric guitar"  # 這個 agent 負責的樂器；其他樂器之後交給其他 agent
STYLE = "neo soul"
BPM = 78
KEY = "D minor"             # 例如 "C major"、"F# minor"
PROGRESSION = [2, 4, 0, 3]  # 和弦級數（0 = I 級），每小節一個和弦
DRUMS = False               # True = 依 BPM 打鼓；False = 叫 AI 不要生成鼓（吉他 agent 保持 False）
# ================

FRAMES_PER_MIN = 25 * 60    # MRT2 每幀 40 ms
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def count(frame: int, per_minute: float) -> int:
    """到第 frame 幀為止經過了幾個單位（每分鐘 per_minute 個）。"""
    return int(frame * per_minute // FRAMES_PER_MIN)


def starts(frame: int, per_minute: float) -> bool:
    """第 frame 幀是否剛好是新的一個單位的開頭。"""
    return frame == 0 or count(frame, per_minute) != count(frame - 1, per_minute)


def style_prompt() -> str:
    # 樂器寫在最前面：實測 "neo soul" 單獨使用時會生成合成器 + 貝斯 + 鼓，
    # 加上 "solo electric guitar" 後四個 10 秒片段都被辨識為吉他，低頻（<80 Hz）從 59% 降到 0.1%
    return f"{INSTRUMENT}, {STYLE}, in {KEY}"


def conditioning(frame: int, style_tokens: list) -> tuple:
    """回傳這一幀的 (conditioning, cfg_scales)。

    MRT2 沒有 BPM / 調性參數，改用它的條件輸入控制（數值皆為實測最佳）：
    - 調性：每小節送一個和弦（C3–B3），調外音設為靜音
    - 速度：有鼓時每個八分音符送一次鼓點（實測鎖拍最穩）；沒鼓時只剩換和弦的起音，
      速度較不穩
    - 沒鼓：每幀都送「沒有鼓」（寫在風格文字裡幾乎無效）
    """
    root_name, mode = KEY.split()
    root, steps = NOTE_NAMES.index(root_name), SCALES[mode]
    degree = PROGRESSION[count(frame, BPM / 4) % len(PROGRESSION)]
    chord = {(root + steps[(degree + k) % 7]) % 12 for k in (0, 2, 4)}
    scale = {(root + s) % 12 for s in steps}

    # pianoroll 數值：2 = 起音、3 = 發聲（模型決定怎麼彈）、0 = 靜音、-1 = 交給模型
    on = 2 if starts(frame, BPM / 4) else 3
    roll = [on if 48 <= p < 60 and p % 12 in chord else (-1 if p % 12 in scale else 0)
            for p in range(128)]
    drum = [int(starts(frame, BPM * 2))] if DRUMS else [0]

    cond = {"mulan_tokens_25hz": style_tokens,
            "pianoroll_with_onsets_tokens": roll,
            "drum_pianoroll_tokens": drum}
    return cond, {"notes": 1.0, "drums": 2.0 if DRUMS else 5.0}


def main() -> None:
    import pasimple  # WSL 內經 WSLg PulseAudio 播放

    mrt = MagentaRT2Jax(size="mrt2_small")
    style_tokens = list(mrt.tokenize_style(mrt.embed_style(style_prompt())))

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=4)
    stop = threading.Event()

    def generator() -> None:
        state, frame = None, 0
        while not stop.is_set():
            parts = []
            for _ in range(50):  # 每段 2 秒；逐幀生成，條件才能精準落在每一幀
                cond, cfg = conditioning(frame, style_tokens)
                wav, state = mrt.generate(conditioning=cond, cfg_scales=cfg, frames=1, state=state)
                parts.append(wav.samples)
                frame += 1
            samples = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
            while not stop.is_set():
                try:
                    audio_q.put(samples, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def player() -> None:
        while not stop.is_set() and audio_q.qsize() < 2:  # 先預存 2 段，吸收 GPU 偶發的慢段
            stop.wait(0.05)
        with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                               2, 48000, app_name="ai_duet") as pa:
            while not stop.is_set():
                try:
                    pa.write(audio_q.get(timeout=0.5).tobytes())
                except queue.Empty:
                    continue

    threads = [threading.Thread(target=generator), threading.Thread(target=player)]
    for t in threads:
        t.start()
    print("Playing - press Enter to stop")
    input()
    stop.set()
    for t in threads:  # 等 GPU 上的 generate() 結束再離開，否則 JAX 會 abort
        t.join()


if __name__ == "__main__":
    main()
