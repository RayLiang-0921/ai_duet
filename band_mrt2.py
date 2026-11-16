"""
band_mrt2.py — 五個 MRT2 agent（Rhodes、吉他、貝斯、鼓、弦樂）組成的即時樂團，可在 neo soul 與 memphis rap 之間切換。
彈一個和弦開始：和弦決定調（根音 + 大/小調），那一刻就是第一拍。之後和弦進行會自己走；
按住音時樂團跟著你，彈一個和弦則讓和弦進行從那個和弦接下去。
按 SWITCH_KEY（鍵盤最左邊的鍵）切換風格，在下一段（4 小節）開頭生效；開始前按則是選擇起始風格。
執行：雙擊 run_band.bat（WSL GPU 環境）
"""
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
from scipy.signal import lfilter

import test_mrt2 as tm  # 共用時間計算、MIDI 接收等工具

# ===== 設定 =====
BPM = 75                    # 兩種風格共用；75 BPM 的 16 分音符剛好 5 幀（40 ms），鼓最穩（78 時 memphis 鼓會跑掉）
SWITCH_KEY = 36             # 這個鍵（含更低的鍵）用來切換風格，不當作音符；36 = C2，61 鍵鍵盤最左邊的鍵
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
# ================

AGENTS = ["rhodes", "guitar", "bass", "drums", "strings"]
STYLES = {  # 每種風格：鼓點（每拍幾個、強度）、各 agent 的提示詞
    "neo soul": {"drum_div": 2, "drum_cfg": 2.0,
                 "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                             "bass": "solo electric bass",
                             "drums": "dry drum kit, tight hi-hats, snappy snare",  # 要求銅鈸明顯會讓銅鈸拖長
                             "strings": "cinematic strings"}},
    "memphis rap": {"drum_div": 4, "drum_cfg": 4.0,  # 16 分音符 hi-hat；強度 2 時鼓常跑掉
                    "prompts": {"rhodes": "solo rhodes electric piano", "guitar": "solo electric guitar",
                                "bass": "deep sub bass",  # "808 bass" 的長滑音跟不上拍子
                                "drums": "memphis rap drums with cowbell and hi-hats",
                                "strings": "cinematic strings"}},
}
PHRASE_BARS = 4             # 風格在每段（4 小節）的開頭切換，不在半途切
NEXT_CHORD = {  # 和弦進行自己走：每小節從上一個和弦的常見去處隨機挑一個（級數，0 = 主和弦）
    "minor": {0: [3, 5, 6, 2], 2: [5, 3, 0], 3: [4, 6, 0, 5], 4: [0, 5, 3], 5: [3, 6, 2, 0], 6: [0, 2, 5]},
    "major": {0: [5, 3, 1, 4], 1: [4, 6, 3], 3: [4, 0, 1, 5], 4: [0, 5, 3], 5: [3, 1, 4, 0], 6: [0, 4]},
}
GAINS = {"rhodes": 0.8, "guitar": 0.8, "bass": 1.0, "drums": 1.12, "strings": 0.8}
PAN = {"rhodes": -0.4, "guitar": 0.4, "bass": 0.0, "drums": 0.0, "strings": 0.0}  # −1 = 最左、+1 = 最右
DRUM_BRIGHTEN = 6.0         # 鼓 5 kHz 以上提亮幾 dB，讓 hi-hat 聽得清楚
MASTER = 0.85               # 整體音量；配合下面的限制器，混音的失真約 −49 dB
TEMPERATURE = 1.1           # 取樣隨機度（MRT2 預設 1.3）：吉他、鼓更穩，貝斯不變
FRAMES_PER_CALL = 5         # 一次在 GPU 上算幾幀：5 個 agent 時 5 幀約 34–38 ms/幀（預算 40 ms）
QUEUE_FRAMES = 8            # 預先生成的幀數，太少會斷音、太多會增加延遲
CHORD_SETTLE = 0.08         # 第一個音按下後等多久再判斷和弦（和弦的音不會剛好同時按下）
CHORDS = {"major": (0, 4, 7), "minor": (0, 3, 7)}
VISUAL_TCP = ("127.0.0.1", 47601)  # 視覺程式 vj_tk.py（Windows 端）連到這裡接收資料
FFT_FREQ = np.fft.rfftfreq(1920, 1 / 48000)
WINDOW = np.hanning(1920).astype(np.float32)


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


def high_shelf(gain_db: float, f0: float = 5000, q: float = 0.7, sr: int = 48000) -> tuple:
    """高頻棚架 EQ（RBJ biquad）：f0 以上提高 gain_db，回傳 (b, a) 給 scipy.signal.lfilter。"""
    A, w = 10 ** (gain_db / 40), 2 * np.pi * f0 / sr
    al, c = np.sin(w) / (2 * q), np.cos(w)
    b = [A * ((A + 1) + (A - 1) * c + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * c),
         A * ((A + 1) + (A - 1) * c - 2 * np.sqrt(A) * al)]
    a = [(A + 1) - (A - 1) * c + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * c),
         (A + 1) - (A - 1) * c - 2 * np.sqrt(A) * al]
    return np.array(b) / a[0], np.array(a) / a[0]


def limit(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """軟拐點限制器：threshold 以下完全不動，超過才柔和壓縮，避免爆音。"""
    a = np.abs(x)
    return np.where(a <= threshold, x,
                    np.sign(x) * (threshold + (1 - threshold) * np.tanh((a - threshold) / (1 - threshold))))


def levels(stems: dict) -> str:
    """一幀各樂器的聲音 → 給視覺的 8 個數值（大約 0–1）：大鼓、小鼓、hi-hat、貝斯、鍵盤、旋律、鍵盤音色、旋律高低。
    除數是實際生成出來的音量的第 90 百分位，所以強的音約為 1。"""
    mono = {a: s.mean(axis=1) for a, s in stems.items()}
    spectrum = lambda a: np.abs(np.fft.rfft(mono[a] * WINDOW)) ** 2
    rms = lambda a: float(np.sqrt((mono[a] ** 2).mean()))
    centroid = lambda P, lo, hi: min(1.0, max(0.0, math.log2(max(lo, (P * FFT_FREQ).sum() / (P.sum() + 1e-9)) / lo)
                                               / math.log2(hi / lo)))
    D = spectrum("drums")
    band = lambda lo, hi: float(np.sqrt(D[(FFT_FREQ >= lo) & (FFT_FREQ < hi)].sum())) / 1920
    v = [band(0, 150) / 0.06, band(150, 2500) / 0.04, band(6000, 24000) / 0.02, rms("bass") / 0.15,
         (rms("rhodes") + rms("strings")) / 0.25, rms("guitar") / 0.15,  # 弦樂一直在，鍵盤除數放大才不會整個畫面過亮
         centroid(spectrum("rhodes"), 200, 2000), centroid(spectrum("guitar"), 150, 3000)]
    return " ".join(f"{min(x, 2.0):.2f}" for x in v)


class VisualLink:
    """把每幀的樂器音量、拍子、風格、觀眾的音送給 Windows 端的 vj_tk.py；沒有接上就直接丟掉。"""

    def __init__(self, stop: threading.Event) -> None:
        self.clients, self.stop, self.lock = [], stop, threading.Lock()
        threading.Thread(target=self._accept, daemon=True).start()

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


def agent_conditioning(agent: str, frame: int, root: int, mode: str, style: dict, tokens: dict,
                       human: tuple, degree: int = 0) -> tuple:
    """回傳某個 agent 這一幀的 (conditioning, cfg_scales)。human = (正在按的音, 剛按下的音)；degree = 這小節的和弦級數。

    - Rhodes：和弦（C3–B4），每拍起音；吉他：和弦（C3–B3），每個八分音符起音
    - 弦樂：和弦（G3–F#5），每小節起音一次後一直持續（鋪底）
    - 貝斯：根音，斷奏八分音符（明確給停止點，才會彈出有節奏的貝斯線）
    - 鼓：不送音符，依風格每個八分或十六分音符送鼓點
    - 人類按鍵時：Rhodes、吉他、弦樂改彈人類的音；貝斯跟人類最低的音；鼓不受影響
    """
    steps = tm.SCALES[mode]
    scale = {(root + s) % 12 for s in steps}
    chord = [(root + steps[(degree + k) % 7]) % 12 for k in (0, 2, 4)]
    sounding, onsets = human
    other = lambda p: -1 if p % 12 in scale else 0  # 調內音交給模型，調外音靜音

    cond = {"mulan_tokens_25hz": tokens[agent], "drum_pianoroll_tokens": [0]}
    cfg = {"notes": 2.0, "drums": 5.0}
    if agent == "drums":
        cond["drum_pianoroll_tokens"] = [int(tm.starts(frame, BPM * style["drum_div"]))]
        return cond, {"drums": style["drum_cfg"]}
    if agent == "bass":
        bass_pc = min(sounding) % 12 if sounding else chord[0]
        eighth = tm.count(frame, BPM * 2)
        since = frame - math.ceil(eighth * tm.FRAMES_PER_MIN / (BPM * 2))  # 這個八分音符開始後第幾幀
        v = 2 if since == 0 else (1 if since <= 3 else 0)                  # 起音、持續 120 ms、停
        roll = [v if 28 <= p < 48 and p % 12 == bass_pc else other(p) for p in range(128)]
    elif sounding:
        roll = [0] * 128
        for p in sounding:
            roll[p] = 2 if p in onsets else 1
    elif agent == "strings":
        on = 2 if tm.starts(frame, BPM / 4) else 1
        roll = [on if 55 <= p < 79 and p % 12 in chord else other(p) for p in range(128)]
    else:
        lo, hi, per_minute = (48, 72, BPM) if agent == "rhodes" else (48, 60, BPM * 2)
        on = 2 if tm.starts(frame, per_minute) else 3
        roll = [on if lo <= p < hi and p % 12 in chord else other(p) for p in range(128)]
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
    """人類的音符；SWITCH_KEY 以下的鍵不算音符，而是要求切換風格。每個按鍵也立刻送給視覺。"""

    def __init__(self, link: VisualLink) -> None:
        super().__init__()
        self.switch, self.link = threading.Event(), link

    def press(self, note: int) -> None:
        if note <= SWITCH_KEY:
            self.switch.set()
        else:
            super().press(note)
            self.link.send(f"n 1 {note}")

    def release(self, note: int) -> None:
        if note > SWITCH_KEY:
            super().release(note)
            self.link.send(f"n 0 {note}")

    def held(self) -> set:
        with self._lock:
            return set(self._held)


def encode(mrt, prompt: str) -> list:
    return list(mrt.tokenize_style(mrt.embed_style(prompt)))


def encode_style(mrt, style: str) -> dict:
    """一種風格所有 agent 的提示詞 → 風格 token。調性靠和弦音符維持，提示詞寫不寫調名測起來沒差，所以不寫、不用重編碼。"""
    return {a: encode(mrt, f"{p}, {style}") for a, p in STYLES[style]["prompts"].items()}


def start_windows(script: str, *args: str) -> "subprocess.Popen | None":
    """在 Windows 端用專案的 venv 執行 script：WSL 讀不到 USB MIDI（--midi-bridge），視窗也在 Windows 開（vj_tk.py）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    win_python = os.path.join(here, "venv", "Scripts", "python.exe")
    if not os.path.exists(win_python):
        print(f"[{script}] 找不到 Windows venv，略過", flush=True)
        return None
    win_script = subprocess.run(["wslpath", "-w", os.path.join(here, script)],
                                capture_output=True, text=True).stdout.strip()
    return subprocess.Popen([win_python, win_script, *args], stdin=subprocess.DEVNULL)  # 不讓它吃掉這邊的 Enter


def main() -> None:
    import pasimple  # 經 WSLg 播放到 Windows 喇叭

    styles = list(STYLES)
    mix_gains = {a: GAINS[a] * pan_gains(PAN[a]) for a in AGENTS}  # 每個 agent 的 [左, 右] 音量
    stop = threading.Event()
    link = VisualLink(stop)
    visual = None if "--no-visual" in sys.argv else start_windows("vj_tk.py")  # 視覺先開，載入模型時就看得到
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, len(AGENTS))
    tokens = {s: encode_style(mrt, s) for s in styles}  # 兩種風格都先準備好，切換時不用等

    # 先編譯，人類開始彈時就不用等。要算兩次：第二次的狀態來自第一次的輸出，型別略有不同，JAX 會再編譯一次（約 5 秒）
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", STYLES[styles[0]], tokens[styles[0]],
                                               (set(), set())) for a in AGENTS]) for f in range(FRAMES_PER_CALL)]
    _, state = band.generate(rows, band.new_state(0))
    band.generate(rows, state)

    human = Human(link)
    audio_q: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_FRAMES)  # (混音, 給視覺的那一幀資料)
    dropouts, gen_ms = [0], []

    def generator() -> None:
        style = styles[0]
        print(f"🎹 請彈一個和弦開始（需要根音和三度音）；按最左邊的鍵選風格，目前：{style}", flush=True)
        link.send(f"w|{style}")
        key = None
        while not stop.is_set() and key is None:
            if human.switch.is_set():
                human.switch.clear()
                style = styles[(styles.index(style) + 1) % len(styles)]
                print(f"🔀 起始風格：{style}", flush=True)
                link.send(f"w|{style}")
            if human.held():
                stop.wait(CHORD_SETTLE)
                key = detect_chord(human.held())
            stop.wait(0.01)
        if key is None:
            return
        root, mode = key
        key_name = f"{tm.NOTE_NAMES[root]} {mode}"
        print(f"🎵 {key_name}，{style}，開始！", flush=True)
        link.send(f"k|{key_name}|{style}")
        state, frame, pending = band.new_state(int(time.time()) % 10000 * 10), 0, None
        eq, eq_state = high_shelf(DRUM_BRIGHTEN), np.zeros((2, 2))
        rng, degree, human_degree = random.Random(), 0, None
        steps = tm.SCALES[mode]
        while not stop.is_set():
            if human.switch.is_set():
                human.switch.clear()
                pending = styles[(styles.index(pending or style) + 1) % len(styles)]
                print(f"🔀 下一段開頭切換到 {pending}", flush=True)
            sounding, onsets = human.take()
            played = detect_chord(sounding) if len(sounding) >= 2 else None  # 彈和弦 → 和弦進行從那個和弦接下去
            if played and (played[0] - root) % 12 in steps:
                human_degree = steps.index((played[0] - root) % 12)
            rows, frame_styles = [], []
            for k in range(FRAMES_PER_CALL):
                if tm.starts(frame + k, BPM / 4):  # 小節開頭：決定這小節的和弦；段落開頭才換風格，拍子不中斷
                    bar = tm.count(frame + k, BPM / 4)
                    if pending and bar % PHRASE_BARS == 0:
                        style, pending, degree = pending, None, 0
                    elif human_degree is not None:
                        degree, human_degree = human_degree, None
                    elif bar > 0:
                        degree = rng.choice(NEXT_CHORD[mode].get(degree, [0]))
                frame_styles.append(styles.index(style))
                rows.append(band.frame_row([agent_conditioning(a, frame + k, root, mode, STYLES[style], tokens[style],
                                                               (sounding, onsets if k == 0 else set()), degree)
                                            for a in AGENTS]))
            t0 = time.perf_counter()
            audio, state = band.generate(rows, state)
            gen_ms.append((time.perf_counter() - t0) * 1000 / FRAMES_PER_CALL)
            d = AGENTS.index("drums")  # 鼓提亮高頻；濾波器狀態 eq_state 接續上一批，幀與幀之間不會有接縫
            drums, eq_state = lfilter(*eq, audio[:, d].reshape(-1, 2), axis=0, zi=eq_state)
            audio[:, d] = drums.reshape(audio[:, d].shape)
            for k in range(FRAMES_PER_CALL):
                mix = MASTER * sum(mix_gains[a] * audio[k, i] for i, a in enumerate(AGENTS))
                mix = np.ascontiguousarray(limit(mix), dtype=np.float32)
                beat = (frame + k) * BPM / tm.FRAMES_PER_MIN
                info = f"f {levels({a: audio[k, i] for i, a in enumerate(AGENTS)})} {beat:.2f} {frame_styles[k]}"
                while not stop.is_set():
                    try:
                        audio_q.put((mix, info), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            frame += FRAMES_PER_CALL

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
    for p in (bridge, visual):
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
        tm.MIDI_PORT = MIDI_PORT
        tm.midi_bridge()
    else:
        main()
