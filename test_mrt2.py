"""
test_mrt2.py — MRT2 即時串流，可用 MIDI 鍵盤即時影響旋律。
執行：雙擊 run_stream_gputest.bat（WSL GPU 環境）
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
INSTRUMENT = "solo electric guitar"  # 這個 agent 的樂器
STYLE = "neo soul"
BPM = 78
KEY = "D minor"             # 例如 "C major"、"F# minor"
PROGRESSION = [2, 4, 0, 3]  # 和弦級數（0 = I 級），每小節一個
DRUMS = False               # True = 有鼓；False = 不要鼓
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
# ================

FRAMES_PER_MIN = 25 * 60    # MRT2 每幀 40 ms
FRAME_BYTES = 1920 * 2 * 4  # 一幀 = 1920 個取樣 × 立體聲 × float32
QUEUE_FRAMES = 3            # 預先生成的幀數，太少會斷音、太多會增加延遲
MIDI_TCP = ("127.0.0.1", 47600)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def count(frame: int, per_minute: float) -> int:
    """第 frame 幀時已經過幾個單位（每分鐘 per_minute 個）。"""
    return int(frame * per_minute // FRAMES_PER_MIN)


def starts(frame: int, per_minute: float) -> bool:
    """第 frame 幀是否為新單位的開頭。"""
    return frame == 0 or count(frame, per_minute) != count(frame - 1, per_minute)


def note_name(n: int) -> str:
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def style_prompt() -> str:
    # 樂器寫在最前面，否則模型會自己加入其他樂器
    return f"{INSTRUMENT}, {STYLE}, in {KEY}"


def conditioning(frame: int, style_tokens: list, human: tuple = ((), ())) -> tuple:
    """回傳這一幀的 (conditioning, cfg_scales)。human = (正在按的音, 剛按下的音)。

    MRT2 沒有 BPM / 調性參數，改用條件輸入控制：
    - 人類按鍵時：只送人類的音（取代和弦），AI 會跟著改變旋律
    - 沒人彈時：每小節送一個和弦（鎖調性），每個八分音符重新起音（鎖速度）
    - 鼓：有鼓時每個八分音符送鼓點；沒鼓時每幀都送「沒有鼓」
    """
    # 音符值：2 = 起音、1 = 持續、3 = 發聲、0 = 靜音、-1 = 交給模型
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
    """MIDI 音符狀態：接收端寫入，生成端每幀讀取。"""

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
        """回傳 (正在發聲的音, 剛按下的音)。很短的點按也至少保留一幀。"""
        with self._lock:
            new, self._new = self._new, set()
            return self._held | new, new


def listen_midi(human: HumanNotes, stop: threading.Event) -> None:
    """接收 MIDI 轉送的音符（"1 60" = 按下、"0 60" = 放開）。"""
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
    """WSL 讀不到 USB MIDI，改在 Windows 端以 --midi-bridge 模式執行本檔。"""
    here = os.path.dirname(os.path.abspath(__file__))
    win_python = os.path.join(here, "venv", "Scripts", "python.exe")
    if not os.path.exists(win_python):
        print("[MIDI] 找不到 Windows venv，略過 MIDI 鍵盤", flush=True)
        return None
    win_script = subprocess.run(["wslpath", "-w", os.path.abspath(__file__)],
                                capture_output=True, text=True).stdout.strip()
    return subprocess.Popen([win_python, win_script, "--midi-bridge"])


def midi_bridge() -> None:
    """（Windows 端）讀取 MIDI 鍵盤，把音符傳給 WSL。"""
    import time
    import mido

    sys.stdout.reconfigure(encoding="utf-8")  # 避免中文在 WSL 主控台變亂碼

    names = mido.get_input_names()
    if not names:
        print("[MIDI] 沒有偵測到 MIDI 鍵盤；音樂照常播放，接上後請重新啟動", flush=True)
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
    import pasimple  # 經 WSLg 播放到 Windows 喇叭

    mrt = MagentaRT2Jax(size="mrt2_small")
    style_tokens = list(mrt.tokenize_style(mrt.embed_style(style_prompt())))

    human = HumanNotes()
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=QUEUE_FRAMES)
    stop = threading.Event()
    dropouts = [0]

    def generator() -> None:
        # 逐幀生成，人類的音符下一幀就會生效
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
        while not stop.is_set() and not audio_q.full():  # 存滿再開始播
            stop.wait(0.01)
        # 音效緩衝設為 2 幀；預設值約 1 秒，延遲太大
        with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                               2, 48000, app_name="ai_duet",
                               tlength=FRAME_BYTES * 2, prebuf=FRAME_BYTES * 2,
                               minreq=FRAME_BYTES) as pa:
            first = True
            while not stop.is_set():
                if audio_q.empty():  # 生成跟不上，會斷音
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
    for t in threads:  # 等生成結束再離開，否則 JAX 會當掉
        t.join()
    print(f"斷音次數：{dropouts[0]}", flush=True)


if __name__ == "__main__":
    if "--midi-bridge" in sys.argv:
        midi_bridge()
    else:
        main()
