"""
test_mrt2.py — MRT2 即時串流：可設定風格、速度、調性、有沒有鼓，並可用 MIDI 鍵盤即時影響旋律。

在 WSL GPU 環境執行：雙擊 run_stream_gputest.bat
WSL 讀不到 USB MIDI 鍵盤，所以程式會自動在 Windows 端開一個「MIDI 轉送」（同一個檔案，
用 Windows venv 執行 --midi-bridge），把鍵盤的音符經本機網路傳進 WSL。
"""
import os
import queue
import socket
import subprocess
import sys
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
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 使用第一個找到的
# ================

FRAMES_PER_MIN = 25 * 60    # MRT2 每幀 40 ms
FRAME_BYTES = 1920 * 2 * 4  # 一幀 = 1920 個取樣 × 立體聲 × float32
QUEUE_FRAMES = 3            # 預先生成幾幀（120 ms）。5 分鐘實測：生成端領先 3 幀以上就不會斷音；
                            # 音效緩衝裡的 2 幀也是已生成的音訊，合計約 320 ms 的餘裕
MIDI_TCP = ("127.0.0.1", 47600)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def count(frame: int, per_minute: float) -> int:
    """到第 frame 幀為止經過了幾個單位（每分鐘 per_minute 個）。"""
    return int(frame * per_minute // FRAMES_PER_MIN)


def starts(frame: int, per_minute: float) -> bool:
    """第 frame 幀是否剛好是新的一個單位的開頭。"""
    return frame == 0 or count(frame, per_minute) != count(frame - 1, per_minute)


def note_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def style_prompt() -> str:
    # 樂器寫在最前面：實測 "neo soul" 單獨使用時會生成合成器 + 貝斯 + 鼓，
    # 加上 "solo electric guitar" 後四個 10 秒片段都被辨識為吉他，低頻（<80 Hz）從 59% 降到 0.1%
    return f"{INSTRUMENT}, {STYLE}, in {KEY}"


def conditioning(frame: int, style_tokens: list, human: tuple = ((), ())) -> tuple:
    """回傳這一幀的 (conditioning, cfg_scales)。human = (正在按的音, 這一幀剛按下的音)。

    MRT2 沒有 BPM / 調性參數，改用它的條件輸入控制（數值皆為實測結果，多個隨機種子驗證）：
    - 人類有按鍵時：音符輸入只送人類的音，取代和弦。實測 AI 0.5 秒內就跟上（調外音比例 8% → 39%）；
      疊加在和弦上幾乎沒效果（13%）。放開後回到和弦，調性也恢復
    - 調性：沒人彈時每小節送一個和弦（C3–B3），調外音設為靜音
    - 速度：和弦在每個八分音符重新起音（鎖拍 0.27 → 0.55），有鼓時再加八分音符鼓點
    - 沒鼓：每幀都送「沒有鼓」（寫在風格文字裡幾乎無效）
    - 音符強度 2.0：3.0 鎖拍差不多，但較常被辨識成貝斯
    """
    # pianoroll 數值：2 = 起音、1 = 持續、3 = 發聲（模型決定怎麼彈）、0 = 靜音、-1 = 交給模型
    sounding, onsets = human
    if sounding:
        roll = [0] * 128
        for p in sounding:
            roll[p] = 2 if p in onsets else 1
    else:
        root_name, mode = KEY.split()
        root, steps = NOTE_NAMES.index(root_name), SCALES[mode]
        degree = PROGRESSION[count(frame, BPM / 4) % len(PROGRESSION)]
        chord = {(root + steps[(degree + k) % 7]) % 12 for k in (0, 2, 4)}
        scale = {(root + s) % 12 for s in steps}
        on = 2 if starts(frame, BPM * 2) else 3
        roll = [on if 48 <= p < 60 and p % 12 in chord else (-1 if p % 12 in scale else 0)
                for p in range(128)]
    drum = [int(starts(frame, BPM * 2))] if DRUMS else [0]

    cond = {"mulan_tokens_25hz": style_tokens,
            "pianoroll_with_onsets_tokens": roll,
            "drum_pianoroll_tokens": drum}
    return cond, {"notes": 2.0, "drums": 2.0 if DRUMS else 5.0}


class HumanNotes:
    """MIDI 音符的共享狀態：接收執行緒寫入，生成執行緒每幀取用。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: set = set()
        self._new: set = set()

    def press(self, note: int) -> None:
        with self._lock:
            self._held.add(note)
            self._new.add(note)

    def release(self, note: int) -> None:
        with self._lock:
            self._held.discard(note)

    def take(self) -> tuple:
        """回傳 (正在發聲的音, 剛按下的音)。比一幀還短的點按也至少會出現一幀。"""
        with self._lock:
            new, self._new = self._new, set()
            return self._held | new, new


def listen_midi(human: HumanNotes, stop: threading.Event) -> None:
    """接收 MIDI 轉送傳來的音符（每行 "1 60" = 按下、"0 60" = 放開）。"""
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(MIDI_TCP)
    server.listen()
    server.settimeout(0.5)
    while not stop.is_set():
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        conn.settimeout(0.5)
        buf = b""
        while not stop.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                down, note = map(int, line.split())
                if down:
                    human.press(note)
                    print(f"🎹 {note_name(note)}", flush=True)
                else:
                    human.release(note)
        conn.close()
    server.close()


def start_midi_bridge() -> "subprocess.Popen | None":
    """在 Windows 端執行這個檔案的 --midi-bridge 模式（WSL 讀不到 USB MIDI）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    win_python = os.path.join(here, "venv", "Scripts", "python.exe")
    if not os.path.exists(win_python):
        print("[MIDI] 找不到 Windows venv，略過 MIDI 鍵盤", flush=True)
        return None
    win_script = subprocess.run(["wslpath", "-w", os.path.abspath(__file__)],
                                capture_output=True, text=True).stdout.strip()
    return subprocess.Popen([win_python, win_script, "--midi-bridge"])


def midi_bridge() -> None:
    """（Windows 端）讀取 MIDI 鍵盤，把音符傳給 WSL 裡的生成程式。"""
    import time
    import mido

    sys.stdout.reconfigure(encoding="utf-8")  # Windows 預設 Big5，輸出到 WSL 主控台會變亂碼

    names = mido.get_input_names()
    if not names:
        print("[MIDI] 沒有偵測到 MIDI 鍵盤；音樂照常播放，接上鍵盤後請重新啟動", flush=True)
        return
    name = MIDI_PORT if MIDI_PORT in names else names[0]
    for _ in range(100):  # 等 WSL 端開始接收
        try:
            sock = socket.create_connection(MIDI_TCP, timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        print("[MIDI] 無法連到生成程式", flush=True)
        return
    print(f"[MIDI] 已連接鍵盤：{name}", flush=True)
    with mido.open_input(name) as port:
        for msg in port:
            if msg.type == "note_on" and msg.velocity > 0:
                sock.sendall(f"1 {msg.note}\n".encode())
            elif msg.type in ("note_off", "note_on"):
                sock.sendall(f"0 {msg.note}\n".encode())


def main() -> None:
    import pasimple  # WSL 內經 WSLg PulseAudio 播放

    mrt = MagentaRT2Jax(size="mrt2_small")
    style_tokens = list(mrt.tokenize_style(mrt.embed_style(style_prompt())))

    human = HumanNotes()
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=QUEUE_FRAMES)
    stop = threading.Event()
    dropouts = [0]

    def generator() -> None:
        # 逐幀生成並逐幀送出：人類的音符在下一幀就會進到條件裡，延遲只剩佇列長度
        state, frame = None, 0
        while not stop.is_set():
            cond, cfg = conditioning(frame, style_tokens, human.take())
            wav, state = mrt.generate(conditioning=cond, cfg_scales=cfg, frames=1, state=state)
            frame += 1
            samples = np.ascontiguousarray(wav.samples, dtype=np.float32)
            while not stop.is_set():
                try:
                    audio_q.put(samples, timeout=0.5)
                    break
                except queue.Full:
                    continue

    def player() -> None:
        while not stop.is_set() and not audio_q.full():  # 先存滿佇列再開始播
            stop.wait(0.01)
        # 預設的 PulseAudio 緩衝實測約 0.9 秒（最高 1.5 秒），人類按鍵要很久才聽到回應。
        # 實測 WSLg：緩衝 3 幀 → 305 ms、2 幀 → 199 ms、1 幀 → 100 ms；取 2 幀兼顧穩定
        with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                               2, 48000, app_name="ai_duet",
                               tlength=FRAME_BYTES * 2, prebuf=FRAME_BYTES * 2,
                               minreq=FRAME_BYTES) as pa:
            first = True
            while not stop.is_set():
                if audio_q.empty():  # 生成跟不上，這一幀會斷音
                    dropouts[0] += 1
                try:
                    pa.write(audio_q.get(timeout=0.5).tobytes())
                except queue.Empty:
                    continue
                if first:
                    first = False
                    threading.Timer(2.0, lambda: print(
                        f"[延遲] 按鍵到聽到約 {QUEUE_FRAMES * 40 + pa.get_latency() / 1000:.0f} ms "
                        f"（不含 Windows 音效輸出）", flush=True)).start()

    threads = [threading.Thread(target=generator), threading.Thread(target=player),
               threading.Thread(target=listen_midi, args=(human, stop))]
    for t in threads:
        t.start()
    bridge = start_midi_bridge()
    print("Playing - press Enter to stop", flush=True)
    input()
    stop.set()
    if bridge:
        bridge.terminate()
    for t in threads:  # 等 GPU 上的 generate() 結束再離開，否則 JAX 會 abort
        t.join()
    print(f"斷音次數：{dropouts[0]}", flush=True)


if __name__ == "__main__":
    if "--midi-bridge" in sys.argv:
        midi_bridge()
    else:
        main()
