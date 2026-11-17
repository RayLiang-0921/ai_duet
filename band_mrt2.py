"""
band_mrt2.py — 五個 MRT2 agent（Rhodes、吉他、貝斯、弦樂、鐘）加一個鼓手（鼓機排 pattern、GrooVAE 加人味、one-shot 取樣發聲）
組成的即時樂團，可在 neo soul 與 memphis rap（強制小調、較暗）之間切換；每段（4 小節）樂器會隨機退出或加入。
彈一個和弦開始：和弦決定調（根音 + 大/小調），那一刻就是第一拍。之後和弦進行會自己走；
按住音時樂團跟著你，彈一個和弦則讓和弦進行從那個和弦接下去。
在視覺視窗按空白鍵切換風格，在下一段開頭生效；開始前按則是選擇起始風格。
執行：雙擊 run_band.bat（WSL GPU 環境）
"""
import bisect
import json
import math
import os
import queue
import random
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import test_mrt2 as tm  # 共用時間計算、MIDI 接收等工具

# ===== 設定 =====
BPM = 75                    # 兩種風格共用；75 BPM 的 16 分音符剛好 5 幀（40 ms）
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
PIANO_PORT = "Microsoft GS Wavetable Synth"  # Windows 端發出鋼琴聲的合成器（名稱開頭）；None = 彈的人不聽自己的琴聲
GROOVE_PYTHON = "~/groove_venv/bin/python"  # 鼓手 groove_server.py（GrooVAE）的環境
DRUM_LAG = 2                # 鼓延後幾幀：MRT2 的樂器比鼓點條件晚約 70 ms 出聲，鼓往後挪才對齊
# ================

AGENTS = ["rhodes", "guitar", "bass", "strings", "bell"]
STYLES = {  # 每種風格：hi-hat 每拍幾個、各 agent 的提示詞；prompt = 接在提示詞後的風格描述、mode = 強制的調式
    "neo soul": {"hat_div": 2,
                 "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                             "bass": "solo electric bass", "strings": "cinematic strings", "bell": "digital bell"}},
    "memphis rap": {"hat_div": 4, "prompt": "dark and moody", "mode": "minor",  # 提示詞不寫 rap（會帶出鼓），一律小調
                    "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                                "bass": "deep sub bass",  # "808 bass" 的長滑音跟不上拍子
                                "strings": "cinematic strings", "bell": "digital bell"}},
}
PHRASE_BARS = 4             # 風格與編制在每段（4 小節）的開頭變，不在半途變
DROP_P, REJOIN_P = 0.3, 0.6  # 每段開頭：在場的樂器退出的機率、退出的樂器回來的機率（至少留 3 個）
BLENDS = (0.25, 0.5, 0.75)  # 切換風格前的最後一小節，提示詞逐拍混合過去，接得比較順
NEXT_CHORD = {  # 和弦進行自己走：每小節從上一個和弦的常見去處隨機挑一個（級數，0 = 主和弦）
    "minor": {0: [3, 5, 6, 2], 2: [5, 3, 0], 3: [4, 6, 0, 5], 4: [0, 5, 3], 5: [3, 6, 2, 0], 6: [0, 2, 5]},
    "major": {0: [5, 3, 1, 4], 1: [4, 6, 3], 3: [4, 0, 1, 5], 4: [0, 5, 3], 5: [3, 1, 4, 0], 6: [0, 4]},
}
GAINS = {"rhodes": 0.8, "guitar": 0.8, "bass": 1.0, "drums": 0.7, "strings": 0.8, "bell": 0.5}
PAN = {"rhodes": -0.4, "guitar": 0.4, "bass": 0.0, "drums": 0.0, "strings": 0.0, "bell": 0.2}  # −1 = 最左、+1 = 最右
MASTER = 0.85               # 整體音量；配合下面的限制器，混音的失真約 −49 dB
TEMPERATURE = 1.1           # 取樣隨機度（MRT2 預設 1.3）：吉他更穩，貝斯不變
FRAMES_PER_CALL = 5         # 一次在 GPU 上算幾幀：5 個 agent 時 5 幀約 34–37 ms/幀（預算 40 ms）
QUEUE_FRAMES = 8            # 預先生成的幀數，太少會斷音、太多會增加延遲
CHORD_SETTLE = 0.08         # 第一個音按下後等多久再判斷和弦（和弦的音不會剛好同時按下）
CHORDS = {"major": (0, 4, 7), "minor": (0, 3, 7)}
VISUAL_TCP = ("127.0.0.1", 47601)  # 視覺程式 vj_tk.py（Windows 端）連到這裡接收資料
GROOVE_TCP = ("127.0.0.1", 47602)  # 鼓手 groove_server.py 聽這裡
SAMPLES = {36: ("kick", 1.0), 38: ("snare", 1.0), 42: ("hihat", 0.7), 46: ("openhat", 0.7),  # GrooVAE 的鼓音高 → (取樣, 音量)
           45: ("rim", 0.8), 48: ("rim", 0.8), 50: ("rim", 0.8), 49: ("openhat", 0.6), 51: ("openhat", 0.5)}  # 取樣在 samples/<風格>/
# samples/<風格>/808.wav 存在時，那種風格的貝斯改由 808 取樣跟著大鼓打（音高 = 和弦根音），取代 MRT2 的貝斯
FFT_FREQ = np.fft.rfftfreq(1920, 1 / 48000)
WINDOW = np.hanning(1920).astype(np.float32)
HERE = os.path.dirname(os.path.abspath(__file__))


def detect_chord(notes) -> "tuple | None":
    """從按住的音判斷 (根音, "major"/"minor")。需要根音和三度音；分數相同時以最低音為根音。"""
    pcs = {n % 12 for n in notes}
    if len(pcs) < 2:
        return None
    bass, best, best_score = min(notes) % 12, None, None
    for root in range(12):
        for quality, iv in CHORDS.items():
            if root not in pcs or (root + iv[1]) % 12 not in pcs:
                continue
            chord = {(root + i) % 12 for i in iv}
            score = (2 * len(pcs & chord) - len(pcs - chord), root == bass)
            if best_score is None or score > best_score:
                best, best_score = (root, quality), score
    return best


def pan_gains(position: float) -> np.ndarray:
    """等功率平移：回傳 [左, 右] 增益，放在中間時兩邊都是 1，移到一邊時總音量不變。"""
    angle = (position + 1) * np.pi / 4
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * np.sqrt(2)


def limit(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """軟拐點限制器：threshold 以下完全不動，超過才柔和壓縮，避免爆音。"""
    a = np.abs(x)
    return np.where(a <= threshold, x,
                    np.sign(x) * (threshold + (1 - threshold) * np.tanh((a - threshold) / (1 - threshold))))


def levels(stems: dict) -> str:
    """一幀各樂器的聲音 → 給視覺的 8 個數值（大約 0–1）：大鼓、小鼓、hi-hat、貝斯、鍵盤、旋律、鍵盤音色、旋律高低。
    除數是實際音量的第 90 百分位（鼓是 one-shot 取樣的重擊），所以強的音約為 1。"""
    mono = {a: s.mean(axis=1) for a, s in stems.items()}
    spectrum = lambda a: np.abs(np.fft.rfft(mono[a] * WINDOW)) ** 2
    rms = lambda a: float(np.sqrt((mono[a] ** 2).mean()))
    centroid = lambda P, lo, hi: min(1.0, max(0.0, math.log2(max(lo, (P * FFT_FREQ).sum() / (P.sum() + 1e-9)) / lo)
                                               / math.log2(hi / lo)))
    D = spectrum("drums")
    band = lambda lo, hi: float(np.sqrt(D[(FFT_FREQ >= lo) & (FFT_FREQ < hi)].sum())) / 1920
    v = [band(0, 150) / 0.2, band(150, 2500) / 0.1, band(6000, 24000) / 0.07, rms("bass") / 0.15,
         (rms("rhodes") + rms("strings")) / 0.25, (rms("guitar") + rms("bell")) / 0.15,  # 弦樂一直在，鍵盤除數放大才不會整個畫面過亮
         centroid(spectrum("rhodes"), 200, 2000), centroid(spectrum("guitar"), 150, 3000)]
    return " ".join(f"{min(x, 2.0):.2f}" for x in v)


class VisualLink:
    """把每幀的樂器音量、拍子、風格、觀眾的音送給 Windows 端的 vj_tk.py；沒有接上就直接丟掉。
    視覺視窗按空白鍵會送回一個 "s"（切換風格）。"""

    def __init__(self, stop: threading.Event, switch: threading.Event) -> None:
        self.clients, self.stop, self.switch, self.lock = [], stop, switch, threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

    def _read(self, conn: socket.socket) -> None:
        while not self.stop.is_set():
            try:
                data = conn.recv(64)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            if b"s" in data:
                self.switch.set()

    def _accept(self) -> None:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(VISUAL_TCP)
        server.listen()
        server.settimeout(0.5)
        while not self.stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            conn.settimeout(0.05)
            self.clients.append(conn)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()
        server.close()

    def send(self, line: str) -> None:
        with self.lock:  # 播放執行緒和 MIDI 執行緒都會送
            for c in list(self.clients):
                try:
                    c.sendall((line + "\n").encode())
                except socket.timeout:
                    pass  # 視覺來不及收，丟掉這一行
                except OSError:
                    self.clients.remove(c)
                    c.close()


def agent_conditioning(agent: str, frame: int, root: int, mode: str, tokens: dict,
                       human: tuple, degree: int = 0) -> tuple:
    """回傳某個 agent 這一幀的 (conditioning, cfg_scales)。human = (正在按的音, 剛按下的音)；degree = 這小節的和弦級數。

    - Rhodes：和弦（C3–B4），每拍起音；吉他：和弦（C3–B3），每個八分音符起音；鐘（bell）：和弦（C5–B6），每拍起音
    - 弦樂：和弦（G3–F#5），每小節起音一次後一直持續（鋪底）
    - 貝斯：根音，斷奏八分音符（明確給停止點，才會彈出有節奏的貝斯線）
    - 人類按鍵時：其他樂器在自己的音域彈人類按的音級（剛按下時有起音）、更聽話；貝斯跟人類最低的音
    """
    steps = tm.SCALES[mode]
    scale = {(root + s) % 12 for s in steps}
    chord = {(root + steps[(degree + k) % 7]) % 12 for k in (0, 2, 4)}
    sounding, onsets = human
    other = lambda p: -1 if p % 12 in scale else 0  # 調內音交給模型，調外音靜音

    cond = {"mulan_tokens_25hz": tokens[agent], "drum_pianoroll_tokens": [0]}  # 鼓點 0：這些 agent 不打鼓
    cfg = {"notes": 2.0, "drums": 5.0}
    if agent == "bass":
        bass_pc = min(sounding) % 12 if sounding else min(chord, key=lambda p: (p - root) % 12 != steps[degree % 7])
        eighth = tm.count(frame, BPM * 2)
        since = frame - math.ceil(eighth * tm.FRAMES_PER_MIN / (BPM * 2))  # 這個八分音符開始後第幾幀
        v = 2 if since == 0 else (1 if since <= 3 else 0)                  # 起音、持續 120 ms、停
        roll = [v if 28 <= p < 48 and p % 12 == bass_pc else other(p) for p in range(128)]
    else:
        lo, hi, per_minute = {"rhodes": (48, 72, BPM), "guitar": (48, 60, BPM * 2), "bell": (72, 96, BPM),
                              "strings": (55, 79, BPM / 4)}[agent]
        if sounding:
            pcs, on, cfg["notes"] = {p % 12 for p in sounding}, 2 if onsets else 1, 3.5
        else:
            pcs, on = chord, 2 if tm.starts(frame, per_minute) else (1 if agent == "strings" else 3)
        roll = [on if lo <= p < hi and p % 12 in pcs else other(p) for p in range(128)]
    if agent == "strings":
        cfg["musiccoca"] = 4.0  # 加強風格提示，弦樂聽起來才像弦樂
    cond["pianoroll_with_onsets_tokens"] = roll
    return cond, cfg


class BatchedMRT2:
    """把多個 agent 合成一批，一次在 GPU 上生成（4 個依序生成要 139 ms/幀，合成一批約 38 ms）。"""

    def __init__(self, mrt, batch: int) -> None:
        import jax
        import jax.numpy as jnp
        from magenta_rt.jax import system  # noqa: F401  載入 MRT2 內附的 sequence_layers
        import sequence_layers.jax as sl

        self.jax, self.jnp, self.mrt, self.batch = jax, jnp, mrt, batch
        self._rows = {}  # frame_row 的快取
        sampler = mrt._sampler
        spec = jax.ShapeDtypeStruct([mrt._num_channels], jnp.int32)
        rngs = {"params": jax.random.PRNGKey(42), "random": jax.random.PRNGKey(0)}
        _, const = mrt._build_conditioning({})
        const = {k: jnp.repeat(v, batch, axis=0) for k, v in const.items()}

        self._init = jax.jit(lambda p: sampler.apply(p, batch, spec, constants={}, training=False,
                                                      rngs=rngs, method=sampler.get_initial_state))

        def run(p, xs, s):  # xs: (幀數, batch, 1, C)，在 GPU 上連續算完這幾幀
            def body(s, x):
                out, s, _ = sampler.apply(p, x=sl.Sequence.from_values(x), state=s, constants=const,
                                          training=False, rngs=rngs, method=sampler.step_with_emits)
                return s, out.values
            s, outs = jax.lax.scan(body, s, xs)
            return outs, s

        self._run = jax.jit(run, donate_argnums=(2,))

    def new_state(self, seed: int):
        state = self._init(self.mrt._params)
        leaves, tree = self.jax.tree_util.tree_flatten(state)
        # 狀態裡唯一的亂數金鑰（uint32，形狀 [batch, 2]）：每個 agent 各給一個種子
        i = next(i for i, l in enumerate(leaves) if l.dtype == self.jnp.uint32 and l.shape == (self.batch, 2))
        leaves[i] = self.jnp.stack([self.jax.random.PRNGKey(seed + k) for k in range(self.batch)])
        return self.jax.tree_util.tree_unflatten(tree, leaves)

    def frame_row(self, conds: list) -> np.ndarray:
        """一幀所有 agent 的條件 → (batch, 1, C)。同樣的條件只算一次（每小節大多重複），省下約 5 ms/幀。"""
        rows = []
        for c, cfg in conds:
            key = tuple((k, tuple(v)) for k, v in sorted(c.items())) + tuple(sorted(cfg.items()))
            if key not in self._rows:
                if len(self._rows) > 20000:
                    self._rows.clear()
                self._rows[key] = np.asarray(self.mrt._build_conditioning(c, cfg)[0].values)
            rows.append(self._rows[key])
        return np.concatenate(rows)

    def generate(self, rows: list, state) -> tuple:
        """rows：每幀一個 frame_row。回傳 (幀數, batch, 1920, 2) 的 float32 音訊與新的狀態。"""
        out, state = self._run(self.mrt._params, self.jnp.asarray(np.stack(rows)), state)
        return np.asarray(self.jax.device_get(out)).astype(np.float32) / 32768.0, state


class Human(tm.HumanNotes):
    """人類的音符；每個按鍵也立刻送給視覺。"""

    def __init__(self, link: VisualLink) -> None:
        super().__init__()
        self.link = link

    def press(self, note: int) -> None:
        super().press(note)
        self.link.send(f"n 1 {note}")

    def release(self, note: int) -> None:
        super().release(note)
        self.link.send(f"n 0 {note}")

    def held(self) -> set:
        with self._lock:
            return set(self._held)


def load_sample(path: str, max_s: float = 1.5) -> np.ndarray:
    """one-shot 取樣 → 48 kHz 立體聲、最長 max_s 秒、頭尾淡入淡出（避免喀聲）、峰值 1。"""
    y, sr = sf.read(path, dtype="float32")
    y = y if y.ndim == 2 else np.stack([y, y], 1)
    if sr != 48000:
        y = resample_poly(y, 48000, sr, axis=0).astype(np.float32)
    y = y[: int(max_s * 48000)]
    y[:48] *= np.linspace(0, 1, 48)[:, None]
    y[-480:] *= np.linspace(1, 0, min(480, len(y)))[:, None]
    return y / (np.abs(y).max() + 1e-9)


class Drummer:
    """鼓手。鼓機照規則排每 2 小節的大鼓 / 小鼓 / open hat：小鼓在 2、4 拍；大鼓比小鼓多一點、通常不跟小鼓同時；
    open hat 每 4 小節一次或偶爾跟大鼓一起；每小節變化一點；每段最後一拍多半有過門，每 8 小節偶爾整小節只剩小鼓滾奏。
    排好的 pattern 提前一段送給 groove_server.py（GrooVAE）補 hi-hat、力度和微小的提前延後，再用 one-shot 取樣發聲；
    沒拿到回覆就直接打鼓機的 pattern。"""
    SNARE, KICK_POOL = (4, 12), (6, 7, 10, 11, 14, 3, 9, 13)
    FILLS = ([(12, 38), (13, 38), (14, 38), (15, 38)], [(12, 38), (14, 45), (15, 48)],  # 最後一拍的過門：小鼓滾奏、小鼓接 tom
             [(13, 38), (14, 38), (15, 46)], [(12, 45), (13, 45), (14, 48), (15, 50)])
    BREAK = [(8, 38), (10, 38), (12, 38), (13, 38), (14, 38), (15, 38)]  # 整小節的斷：後半小節小鼓漸密

    def __init__(self, seed: int) -> None:
        self.samples = {st: {n: load_sample(os.path.join(HERE, "samples", st, f"{n}.wav")) for n, _ in SAMPLES.values()}
                        for st in STYLES}
        self.s808 = {}  # 風格 → {根音音級: 調好音高的 808 取樣}
        for st in STYLES:
            path = os.path.join(HERE, "samples", st, "808.wav")
            if os.path.exists(path):
                y = load_sample(path, 3.0)
                spec = np.abs(np.fft.rfft(y[4800:28800].mean(1) * np.hanning(24000), n=1 << 18))  # 取樣的基頻（25–200 Hz 最強處）
                fr = np.fft.rfftfreq(1 << 18, 1 / 48000)
                f0 = fr[np.argmax(np.where((fr > 25) & (fr < 200), spec, 0))]
                self.s808[st] = {}
                for pc in range(12):  # 根音放在 G#1–G2
                    ratio = 440 * 2 ** (((36 + pc if pc <= 7 else 24 + pc) - 69) / 12) / f0
                    idx = np.arange(0, len(y) - 1, ratio)
                    self.s808[st][pc] = np.stack([np.interp(idx, np.arange(len(y)), y[:, c]) for c in (0, 1)], 1).astype(np.float32)
        self.rng, self.kicks = random.Random(seed), {0, 6, 10}
        self.played, self.net, self.io = {}, threading.Lock(), None  # played[段] = [(秒, 音高, 力度)]；一次只送一個請求
        self.pending, self.buf = [], np.zeros((2 * 48000, 2), np.float32)  # pending：等著發聲的 (取樣位置, 音高, 力度)
        self.buf808 = np.zeros((4 * 48000, 2), np.float32)

    def pattern(self, seg: int) -> list:
        """鼓機：第 seg 段（2 小節）的 [(16 分音符位置 0–31, 音高)]。"""
        r, notes = self.rng, []
        for bar in (2 * seg, 2 * seg + 1):
            if r.random() < 0.5:  # 變化：拿掉或加一個大鼓（第一拍不動，共 2–4 個）
                extra = sorted(self.kicks - {0})
                if extra and (len(extra) >= 3 or r.random() < 0.5):
                    self.kicks.discard(r.choice(extra))
                else:
                    self.kicks.add(r.choice(self.KICK_POOL))
            if r.random() > 0.1:
                self.kicks -= set(self.SNARE)
            oh = {14} if bar % 4 == 3 else ({r.choice(sorted(self.kicks))} if r.random() < 0.15 else set())
            hits = [(s, 36) for s in self.kicks] + [(s, 38) for s in self.SNARE] + [(s, 46) for s in oh]
            if bar % 8 == 7 and r.random() < 0.3:
                hits = [(0, 36), (4, 38)] + self.BREAK
            elif bar % 4 == 3 and r.random() < 0.7:
                hits = [(s, p) for s, p in hits if s < 12] + r.choice(self.FILLS)
            notes += [(16 * (bar % 2) + s, p) for s, p in hits]
        return notes

    def prepare(self, seg: int) -> None:
        """排第 seg 段的 pattern，先存鼓機版本，再在背景送去 GrooVAE（約 0.3 秒）換成演奏版本。"""
        notes = self.pattern(seg)
        self.played[seg] = [(s * 60 / BPM / 4, p, 100) for s, p in notes]
        threading.Thread(target=self._ask, args=(seg, notes), daemon=True).start()

    def _ask(self, seg: int, notes: list) -> None:
        with self.net:
            try:
                for attempt in range(20):  # groove_server 還在載入時等它
                    try:
                        self.io = self.io or socket.create_connection(GROOVE_TCP, timeout=10).makefile("rw")
                        break
                    except ConnectionRefusedError:
                        if attempt == 19:
                            raise
                        time.sleep(1)
                self.io.write(json.dumps({"bpm": BPM, "notes": notes}) + "\n")
                self.io.flush()
                self.played[seg] = [tuple(n) for n in json.loads(self.io.readline())]
            except (OSError, ValueError) as e:
                print(f"[鼓手] GrooVAE 沒回應（{e}），第 {seg} 段用鼓機的 pattern", flush=True)
                self.io = None

    def frame(self, frame: int, style: str, root_pc: int) -> tuple:
        """這一幀的 (鼓聲, 808 貝斯或 None)，各 (1920, 2)。段落開頭把這段的演奏排進 pending、並準備下一段；
        風格決定鼓組和 hi-hat 密度；有 808 的風格，每個大鼓同時打一個 root_pc 音高的 808（單音，新的會切掉舊的）。"""
        start, hat_div = frame * 1920, STYLES[style]["hat_div"]
        if tm.starts(frame, BPM / 8):  # 每 2 小節一段
            seg, step = tm.count(frame, BPM / 8), 60 / BPM / 4
            self.prepare(seg + 1)
            for t, p, v in self.played.pop(seg, []):
                if p in SAMPLES and (p != 42 or round(t / step) % (4 // hat_div) == 0):  # neo soul 只留八分音符的 hi-hat
                    bisect.insort(self.pending, (max(start, start + int(t * 48000)) + DRUM_LAG * 1920, p, v))
        while self.pending and self.pending[0][0] < start + 1920:
            pos, p, v = self.pending.pop(0)
            name, gain = SAMPLES[p]
            y = self.samples[style][name]
            self.buf[pos - start: pos - start + len(y)] += gain * (v / 127) ** 0.5 * y  # 力度曲線：輕的不會太小聲
            if p == 36 and style in self.s808:
                i, y = pos - start, self.s808[style][root_pc]
                self.buf808[i: i + 240] *= np.linspace(1, 0, 240)[:, None]  # 5 ms 內切掉上一個 808
                self.buf808[i + 240:] = 0
                self.buf808[i: i + len(y)] += 0.35 * (v / 127) ** 0.5 * y  # 取樣峰值 1，壓到跟 MRT2 貝斯差不多的音量
        out = self.buf[:1920].copy()
        self.buf[:-1920] = self.buf[1920:]
        self.buf[-1920:] = 0
        bass = None
        if style in self.s808:
            bass = self.buf808[:1920].copy()
            self.buf808[:-1920] = self.buf808[1920:]
            self.buf808[-1920:] = 0
        return out, bass


def style_tokens(mrt) -> dict:
    """每種風格各 agent 的提示詞 → 風格 token；另外兩種風格之間 BLENDS 比例的混合（embedding 加權平均），切換前用。
    調性靠和弦音符維持，提示詞寫不寫調名測起來沒差，所以不寫。回傳 {風格: {agent: tokens}, (從, 到, 比例): {agent: tokens}}。"""
    emb = {s: {a: np.asarray(mrt.embed_style(f"{p}, {STYLES[s].get('prompt', s)}")) for a, p in STYLES[s]["prompts"].items()}
           for s in STYLES}
    tok = lambda e: list(mrt.tokenize_style(e))
    tokens = {s: {a: tok(e) for a, e in es.items()} for s, es in emb.items()}
    for s1 in STYLES:
        for s2 in STYLES:
            for w in BLENDS if s1 != s2 else ():
                tokens[(s1, s2, w)] = {a: tok((1 - w) * emb[s1][a] + w * emb[s2][a]) for a in AGENTS}
    return tokens


def midi_bridge() -> None:
    """（Windows 端）讀 MIDI 鍵盤：音符送給 WSL 的樂團，同時用 Windows 的合成器發鋼琴聲，彈的人立刻聽得到。"""
    import mido

    sys.stdout.reconfigure(encoding="utf-8")  # 避免中文在 WSL 主控台變亂碼
    names = mido.get_input_names()
    if not names:
        print("[MIDI] 沒有偵測到 MIDI 鍵盤；音樂照常播放，接上後請重新啟動", flush=True)
        return
    name = MIDI_PORT if MIDI_PORT in names else names[0]
    for _ in range(100):  # 等 WSL 端開始接收
        try:
            sock = socket.create_connection(tm.MIDI_TCP, timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        print("[MIDI] 無法連到生成程式", flush=True)
        return
    piano = next((mido.open_output(n) for n in mido.get_output_names() if PIANO_PORT and n.startswith(PIANO_PORT)), None)
    if piano:
        piano.send(mido.Message("program_change", program=0))  # 平台鋼琴
    print(f"[MIDI] 已連接鍵盤：{name}" + (f"，鋼琴聲：{PIANO_PORT}" if piano else ""), flush=True)
    with mido.open_input(name) as port:
        for msg in port:
            if msg.type not in ("note_on", "note_off"):
                continue
            down = int(msg.type == "note_on" and msg.velocity > 0)
            sock.sendall(f"{down} {msg.note}\n".encode())
            if piano:
                piano.send(msg)


def start_windows(script: str, *args: str) -> "subprocess.Popen | None":
    """在 Windows 端用專案的 venv 執行 script：WSL 讀不到 USB MIDI（--midi-bridge），視窗也在 Windows 開（vj_tk.py）。"""
    win_python = os.path.join(HERE, "venv", "Scripts", "python.exe")
    if not os.path.exists(win_python):
        print(f"[{script}] 找不到 Windows venv，略過", flush=True)
        return None
    win_script = subprocess.run(["wslpath", "-w", os.path.join(HERE, script)],
                                capture_output=True, text=True).stdout.strip()
    return subprocess.Popen([win_python, win_script, *args], stdin=subprocess.DEVNULL)  # 不讓它吃掉這邊的 Enter


def main() -> None:
    import jax
    import pasimple  # 經 WSLg 播放到 Windows 喇叭

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))  # 編譯結果存起來，第二次啟動快很多

    styles = list(STYLES)
    mix_gains = {a: GAINS[a] * pan_gains(PAN[a]) for a in GAINS}  # 每個樂器的 [左, 右] 音量
    stop, switch = threading.Event(), threading.Event()  # switch：視覺視窗要求切換風格
    link = VisualLink(stop, switch)
    visual = None if "--no-visual" in sys.argv else start_windows("vj_tk.py")  # 視覺先開，載入模型時就看得到
    groove_python, groove = os.path.expanduser(GROOVE_PYTHON), None
    if os.path.exists(groove_python):
        groove = subprocess.Popen([groove_python, os.path.join(HERE, "groove_server.py")], stdin=subprocess.DEVNULL)
    else:
        print(f"[鼓手] 找不到 {GROOVE_PYTHON}，鼓只用鼓機的 pattern", flush=True)
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, len(AGENTS))
    tokens = style_tokens(mrt)  # 兩種風格和中間的混合都先準備好，切換時不用等

    # 先編譯，人類開始彈時就不用等。要算兩次：第二次的狀態來自第一次的輸出，型別略有不同，JAX 會再編譯一次（約 5 秒）
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", tokens[styles[0]], (set(), set()))
                            for a in AGENTS]) for f in range(FRAMES_PER_CALL)]
    _, state = band.generate(rows, band.new_state(0))
    band.generate(rows, state)
    drummer = Drummer(int(time.time()))
    drummer.prepare(0)  # 第一段先排好，開始時不用等

    human = Human(link)
    audio_q: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_FRAMES)  # (混音, 給視覺的那一幀資料)
    dropouts, gen_ms = [0], []

    def generator() -> None:
        style = styles[0]
        print(f"🎹 請彈一個和弦開始（需要根音和三度音）；在視覺視窗按空白鍵選風格，目前：{style}", flush=True)
        link.send(f"w|{style}")
        key = None
        while not stop.is_set() and key is None:
            if switch.is_set():
                switch.clear()
                style = styles[(styles.index(style) + 1) % len(styles)]
                print(f"🔀 起始風格：{style}", flush=True)
                link.send(f"w|{style}")
            if human.held():
                stop.wait(CHORD_SETTLE)
                key = detect_chord(human.held())
            stop.wait(0.01)
        if key is None:
            return
        root, played_mode = key
        mode = STYLES[style].get("mode", played_mode)  # memphis 一律小調
        key_name = f"{tm.NOTE_NAMES[root]} {mode}"
        print(f"🎵 {key_name}，{style}，開始！", flush=True)
        link.send(f"k|{key_name}|{style}")
        state, frame, pending = band.new_state(int(time.time()) % 10000 * 10), 0, None
        rng, degree, human_degree = random.Random(), 0, None
        steps = tm.SCALES[mode]
        target = {a: 1.0 for a in GAINS}  # 編制：每個樂器 1 = 在場、0 = 退出；active 是淡入淡出中的實際音量
        active = dict(target)
        rec = None  # --record 檔名：各樂器分軌存成多聲道 wav（測試用）
        if "--record" in sys.argv:
            rec = sf.SoundFile(sys.argv[sys.argv.index("--record") + 1], "w", 48000, 2 * (len(AGENTS) + 1), "FLOAT")
        while not stop.is_set():
            if switch.is_set():
                switch.clear()
                pending = styles[(styles.index(pending or style) + 1) % len(styles)]
                print(f"🔀 下一段開頭切換到 {pending}", flush=True)
                link.send(f"p|{pending}")
            sounding, onsets = human.take()
            played = detect_chord(sounding) if len(sounding) >= 2 else None  # 彈和弦 → 和弦進行從那個和弦接下去
            if played and (played[0] - root) % 12 in steps:
                human_degree = steps.index((played[0] - root) % 12)
            rows, frame_styles, drums = [], [], []
            for k in range(FRAMES_PER_CALL):
                if tm.starts(frame + k, BPM / 4):  # 小節開頭：決定這小節的和弦；段落開頭才換風格、改編制，拍子不中斷
                    bar = tm.count(frame + k, BPM / 4)
                    if pending and bar % PHRASE_BARS == 0:
                        style, pending, degree = pending, None, 0
                        mode = STYLES[style].get("mode", played_mode)
                        steps = tm.SCALES[mode]
                    elif human_degree is not None:
                        degree, human_degree = human_degree, None
                    elif bar > 0:
                        degree = rng.choice(NEXT_CHORD[mode].get(degree, [0]))
                    if bar % PHRASE_BARS == PHRASE_BARS - 1:  # 編制：下一段誰在場，這一小節慢慢淡入淡出過去；至少留 3 個
                        target = {a: float(rng.random() >= (DROP_P if on else 1 - REJOIN_P)) for a, on in target.items()}
                        for a in rng.sample(list(target), 3 - int(sum(target.values()))) if sum(target.values()) < 3 else ():
                            target[a] = 1.0
                        print("🎚 下一段：" + "、".join(a for a, on in target.items() if on), flush=True)
                frame_styles.append(styles.index(style))
                bar_beat = tm.count(frame + k, BPM) % 4
                blend = pending and bar % PHRASE_BARS == PHRASE_BARS - 1 and bar_beat > 0  # 切換前最後一小節逐拍混合
                tok = tokens[(style, pending, BLENDS[bar_beat - 1])] if blend else tokens[style]
                root_pc = min(sounding) % 12 if sounding else (root + steps[degree % 7]) % 12
                drums.append(drummer.frame(frame + k, style, root_pc))
                rows.append(band.frame_row([agent_conditioning(a, frame + k, root, mode, tok,
                                                               (sounding, onsets if k == 0 else set()), degree)
                                            for a in AGENTS]))
            t0 = time.perf_counter()
            audio, state = band.generate(rows, state)
            gen_ms.append((time.perf_counter() - t0) * 1000 / FRAMES_PER_CALL)
            for k in range(FRAMES_PER_CALL):
                active = {a: g + ((1.0 if sounding else target[a]) - g) * 0.04 for a, g in active.items()}  # 約一小節淡入淡出；人類彈琴時全員回來
                stems = {a: audio[k, i] * active[a] for i, a in enumerate(AGENTS)} | {"drums": drums[k][0] * active["drums"]}
                if drums[k][1] is not None:
                    stems["bass"] = drums[k][1] * active["bass"]  # 這種風格的貝斯是 808
                if rec:
                    rec.write(np.concatenate(list(stems.values()), axis=1))
                mix = MASTER * sum(mix_gains[a] * s for a, s in stems.items())
                mix = np.ascontiguousarray(limit(mix), dtype=np.float32)
                beat = (frame + k) * BPM / tm.FRAMES_PER_MIN
                info = f"f {levels(stems)} {beat:.2f} {frame_styles[k]}"
                while not stop.is_set():
                    try:
                        audio_q.put((mix, info), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            frame += FRAMES_PER_CALL
        if rec:
            rec.close()

    def player() -> None:
        while not stop.is_set() and not audio_q.full():  # 存滿再開始播
            stop.wait(0.01)
        # 音效緩衝設為 2 幀；預設值約 1 秒，延遲太大
        with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                               2, 48000, app_name="ai_duet_band",
                               tlength=tm.FRAME_BYTES * 2, prebuf=tm.FRAME_BYTES * 2,
                               minreq=tm.FRAME_BYTES) as pa:
            first = True
            while not stop.is_set():
                if audio_q.empty():  # 生成跟不上，會斷音
                    dropouts[0] += 1
                try:
                    mix, info = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                pa.write(mix.tobytes())
                link.send(info)  # 在真正播出時才送，畫面自然跟聲音對齊
                if first:
                    first = False
                    threading.Timer(2.0, report_latency, args=(pa,)).start()

    def report_latency(pa) -> None:  # 播放 2 秒、緩衝穩定後再量
        if not stop.is_set():
            print(f"[延遲] 按鍵到聽到約 {QUEUE_FRAMES * 40 + pa.get_latency() / 1000:.0f} ms "
                  f"（不含 Windows 音效輸出）", flush=True)

    threads = [threading.Thread(target=generator), threading.Thread(target=player),
               threading.Thread(target=tm.listen_midi, args=(human, stop))]
    for t in threads:
        t.start()
    bridge = start_windows("band_mrt2.py", "--midi-bridge")
    print("按 Enter 結束", flush=True)
    input()
    stop.set()
    for p in (bridge, visual, groove):
        if p:
            p.terminate()
    for t in threads:  # 等生成結束再離開，否則 JAX 會當掉
        t.join()
    if gen_ms:
        print(f"生成每幀中位數 {np.median(gen_ms):.1f} ms、95% 在 {np.percentile(gen_ms, 95):.1f} ms 內、最慢 {max(gen_ms):.0f} ms"
              f"（預算 40 ms）", flush=True)
    print(f"斷音次數：{dropouts[0]}", flush=True)


if __name__ == "__main__":
    if "--midi-bridge" in sys.argv:
        midi_bridge()
    else:
        main()
