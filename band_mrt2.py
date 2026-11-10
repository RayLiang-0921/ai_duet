"""
band_mrt2.py — 四個 MRT2 agent（鋼琴、吉他、貝斯、鼓）組成的即時樂團。
彈一個和弦開始：和弦決定調（根音 + 大/小調），那一刻就是第一拍；之後按住音時樂團跟著你。
執行：雙擊 run_band.bat（WSL GPU 環境）
"""
import math
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

import test_mrt2 as tm  # 共用時間計算、MIDI 接收等工具

# ===== 設定 =====
STYLE = "neo soul"
BPM = 78
PROGRESSION = [0, 3, 4, 0]  # 和弦級數（0 = 你彈的第一個和弦），每小節一個
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
# ================

AGENTS = {"piano": "solo piano", "guitar": "solo electric guitar",
          "bass": "solo electric bass",
          "drums": "drum kit with crisp hi-hats and crash cymbals"}  # 只寫 drum kit 幾乎沒有銅鈸聲
GAINS = {"piano": 0.8, "guitar": 0.8, "bass": 1.0, "drums": 1.12}  # 貝斯、鼓各比原本大 1 dB
PAN = {"piano": -0.4, "guitar": 0.4, "bass": 0.0, "drums": 0.0}  # −1 = 最左、+1 = 最右；低音和鼓放中間
MASTER = 0.85               # 整體音量；配合下面的限制器，混音的失真約 −49 dB
TEMPERATURE = 1.1           # 取樣隨機度（MRT2 預設 1.3）：吉他、鼓更穩，貝斯不變
FRAMES_PER_CALL = 5         # 一次在 GPU 上算幾幀：4 個 agent 時 5 幀約 38 ms/幀（預算 40 ms）
QUEUE_FRAMES = 8            # 預先生成的幀數，太少會斷音、太多會增加延遲
CHORD_SETTLE = 0.08         # 第一個音按下後等多久再判斷和弦（和弦的音不會剛好同時按下）
CHORDS = {"major": (0, 4, 7), "minor": (0, 3, 7)}


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


def agent_conditioning(agent: str, frame: int, root: int, mode: str, tokens: dict, human: tuple) -> tuple:
    """回傳某個 agent 這一幀的 (conditioning, cfg_scales)。human = (正在按的音, 剛按下的音)。

    - 鋼琴：和弦（C3–B4），每拍起音；吉他：和弦（C3–B3），每個八分音符起音
    - 貝斯：根音，斷奏八分音符（明確給停止點，才會彈出有節奏的貝斯線）
    - 鼓：不送音符，每個八分音符送鼓點
    - 人類按鍵時：鋼琴、吉他改彈人類的音；貝斯跟人類最低的音；鼓不受影響
    """
    steps = tm.SCALES[mode]
    scale = {(root + s) % 12 for s in steps}
    degree = PROGRESSION[tm.count(frame, BPM / 4) % len(PROGRESSION)]
    chord = [(root + steps[(degree + k) % 7]) % 12 for k in (0, 2, 4)]
    sounding, onsets = human
    other = lambda p: -1 if p % 12 in scale else 0  # 調內音交給模型，調外音靜音

    cond = {"mulan_tokens_25hz": tokens[agent], "drum_pianoroll_tokens": [0]}
    if agent == "drums":
        cond["drum_pianoroll_tokens"] = [int(tm.starts(frame, BPM * 2))]
        return cond, {"drums": 2.0}
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
    else:
        lo, hi, per_minute = (48, 72, BPM) if agent == "piano" else (48, 60, BPM * 2)
        on = 2 if tm.starts(frame, per_minute) else 3
        roll = [on if lo <= p < hi and p % 12 in chord else other(p) for p in range(128)]
    cond["pianoroll_with_onsets_tokens"] = roll
    return cond, {"notes": 2.0, "drums": 5.0}


class BatchedMRT2:
    """把多個 agent 合成一批，一次在 GPU 上生成（4 個依序生成要 139 ms/幀，合成一批約 38 ms）。"""

    def __init__(self, mrt, batch: int) -> None:
        import jax
        import jax.numpy as jnp
        from magenta_rt.jax import system  # noqa: F401  載入 MRT2 內附的 sequence_layers
        import sequence_layers.jax as sl

        self.jax, self.jnp, self.mrt, self.batch = jax, jnp, mrt, batch
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
        """一幀所有 agent 的條件 → (batch, 1, C)。"""
        return np.concatenate([np.asarray(self.mrt._build_conditioning(c, cfg)[0].values) for c, cfg in conds])

    def generate(self, rows: list, state) -> tuple:
        """rows：每幀一個 frame_row。回傳 (幀數, batch, 1920, 2) 的 float32 音訊與新的狀態。"""
        out, state = self._run(self.mrt._params, self.jnp.asarray(np.stack(rows)), state)
        return np.asarray(self.jax.device_get(out)).astype(np.float32) / 32768.0, state


class Human(tm.HumanNotes):
    def held(self) -> set:
        with self._lock:
            return set(self._held)


def encode(mrt, prompt: str) -> list:
    return list(mrt.tokenize_style(mrt.embed_style(prompt)))


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


def main() -> None:
    import pasimple  # 經 WSLg 播放到 Windows 喇叭

    names = list(AGENTS)
    mix_gains = {a: GAINS[a] * pan_gains(PAN[a]) for a in names}  # 每個 agent 的 [左, 右] 音量
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, len(names))
    live = {"tokens": {a: encode(mrt, f"{inst}, {STYLE}") for a, inst in AGENTS.items()}}

    # 先編譯一次，人類開始彈時就不用等
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", live["tokens"], (set(), set())) for a in names])
            for f in range(FRAMES_PER_CALL)]
    band.generate(rows, band.new_state(0))

    human = Human()
    audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=QUEUE_FRAMES)
    stop = threading.Event()
    dropouts = [0]

    def generator() -> None:
        print("🎹 請彈一個和弦開始（需要根音和三度音）", flush=True)
        key = None
        while not stop.is_set() and key is None:
            if human.held():
                stop.wait(CHORD_SETTLE)
                key = detect_chord(human.held())
            stop.wait(0.01)
        if key is None:
            return
        root, mode = key
        key_name = f"{tm.NOTE_NAMES[root]} {mode}"
        print(f"🎵 {key_name}，開始！", flush=True)

        def add_key_to_prompts() -> None:  # 先用沒寫調名的提示詞開始，約 1 秒後換成有調名的版本
            live["tokens"] = {a: encode(mrt, f"{inst}, {STYLE}, in {key_name}") for a, inst in AGENTS.items()}
        threading.Thread(target=add_key_to_prompts, daemon=True).start()

        state, frame = band.new_state(int(time.time()) % 10000 * 10), 0
        while not stop.is_set():
            sounding, onsets = human.take()
            rows = [band.frame_row([agent_conditioning(a, frame + k, root, mode, live["tokens"],
                                                       (sounding, onsets if k == 0 else set())) for a in names])
                    for k in range(FRAMES_PER_CALL)]
            audio, state = band.generate(rows, state)
            frame += FRAMES_PER_CALL
            for k in range(FRAMES_PER_CALL):
                mix = MASTER * sum(mix_gains[a] * audio[k, i] for i, a in enumerate(names))
                mix = np.ascontiguousarray(limit(mix), dtype=np.float32)
                while not stop.is_set():
                    try:
                        audio_q.put(mix, timeout=0.5)
                        break
                    except queue.Full:
                        continue

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
                    pa.write(audio_q.get(timeout=0.5).tobytes())
                except queue.Empty:
                    continue
                if first:
                    first = False
                    threading.Timer(2.0, lambda: print(
                        f"[延遲] 按鍵到聽到約 {QUEUE_FRAMES * 40 + pa.get_latency() / 1000:.0f} ms "
                        f"（不含 Windows 音效輸出）", flush=True)).start()

    threads = [threading.Thread(target=generator), threading.Thread(target=player),
               threading.Thread(target=tm.listen_midi, args=(human, stop))]
    for t in threads:
        t.start()
    bridge = start_midi_bridge()
    print("按 Enter 結束", flush=True)
    input()
    stop.set()
    if bridge:
        bridge.terminate()
    for t in threads:  # 等生成結束再離開，否則 JAX 會當掉
        t.join()
    print(f"斷音次數：{dropouts[0]}", flush=True)


if __name__ == "__main__":
    if "--midi-bridge" in sys.argv:
        tm.MIDI_PORT = MIDI_PORT
        tm.midi_bridge()
    else:
        main()
