"""鼓手：鼓機排 pattern → groove_server.py（GrooVAE）加人味 → 從 samples/ 的鼓組取樣發聲（會輪換鼓組）；memphis 另外有跟著大鼓的 808 貝斯。"""
import bisect
import functools
import json
import os
import random
import socket
import threading
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

import test_mrt2 as tm
from band.settings import BPM, DRUM_LAG, GROOVE_TCP, KIT_SEGMENTS, ROOT, SAMPLES, STYLES


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


def tuned_808(path: str) -> dict:
    """808 取樣 → {根音音級: 調到 G#1–G2 的取樣}。先量取樣的基頻（25–200 Hz 最強處），再用線性內插變速。"""
    y = load_sample(path, 3.0)
    spec = np.abs(np.fft.rfft(y[4800:28800].mean(1) * np.hanning(24000), n=1 << 18))
    fr = np.fft.rfftfreq(1 << 18, 1 / 48000)
    f0 = fr[np.argmax(np.where((fr > 25) & (fr < 200), spec, 0))]
    out = {}
    for pc in range(12):
        ratio = 440 * 2 ** (((36 + pc if pc <= 7 else 24 + pc) - 69) / 12) / f0
        idx = np.arange(0, len(y) - 1, ratio)
        out[pc] = np.stack([np.interp(idx, np.arange(len(y)), y[:, c]) for c in (0, 1)], 1).astype(np.float32)
    return out


KIT_ROLES = {  # 鼓組資料夾裡，路徑（不分大小寫）含這些字的 wav 當作這個角色；依序試，前面的找不到才用後面的
    "kick": [("kick",)], "snare": [("snare",), ("clap",)], "hihat": [("hi hat", "hihat", "hi-hat", "closed hat")],
    "openhat": [("open hat",)], "rim": [("rim",), ("perc", "stick", "snap", "clap")], "808": [("808",)],
}
KIT_SKIP = ("loop", "fill", "transition", "fx", "vox", "sample", "scratch", "vinyl", "extra", "texture", "reverse")
KIT_MAX_S = {"kick": 1.5, "snare": 1.0, "hihat": 0.4, "openhat": 2.0, "rim": 0.8, "808": 10.0}  # 比這長的多半是 roll 或 loop，不挑


@functools.lru_cache(maxsize=None)
def kit_candidates(name: str) -> dict:
    """samples/<name>/ 掃描一次：{角色: [可以用的 wav 路徑]}。頂層有 kick.wav 這種固定檔名時就只用它；
    否則照 KIT_ROLES 比對子資料夾名稱，跳過 KIT_SKIP 的資料夾和超過 KIT_MAX_S 長度的檔案。"""
    root = os.path.join(ROOT, "samples", name)
    files = []
    for d, _, fs in os.walk(root):
        rel = os.path.relpath(d, root).lower().replace("\\", "/")
        if not any(s in rel for s in KIT_SKIP):
            files += [(rel, os.path.join(d, f)) for f in fs if f.lower().endswith(".wav")]
    out = {}
    for role, choices in KIT_ROLES.items():
        fixed = os.path.join(root, f"{role}.wav")
        if os.path.exists(fixed):
            out[role] = [fixed]
            continue
        for words in choices:
            hits = [p for r, p in files if any(w in r for w in words) and (role != "hihat" or "open" not in r)]
            if hits:
                out[role] = [p for p in hits if sf.info(p).duration <= KIT_MAX_S[role]] or hits
                break
        else:
            out[role] = []
    return out


def load_kit(name: str, rng: random.Random) -> dict:
    """鼓組 → {角色: 取樣, "808": 調好音高的 808 或 None, "name": name}；每個角色從候選裡隨機挑一個（每次載入挑的不一樣）。"""
    kit = {"name": name, "808": None}
    for role, paths in kit_candidates(name).items():
        path = rng.choice(paths) if paths else None
        if role == "808":
            kit["808"] = tuned_808(path) if path else None
        else:
            kit[role] = load_sample(path) if path else np.zeros((480, 2), np.float32)
    return kit


class Drummer:
    """鼓機照規則排每 2 小節的大鼓 / 小鼓 / open hat：小鼓在 2、4 拍；大鼓比小鼓多一點、通常不跟小鼓同時；
    open hat 每 4 小節一次或偶爾跟大鼓一起；每小節變化一點；每段最後一拍多半有過門，每 8 小節偶爾整小節只剩小鼓滾奏。
    排好的 pattern 提前一段送給 groove_server.py（GrooVAE）補 hi-hat、力度和微小的提前延後，再用 one-shot 取樣發聲；
    沒拿到回覆就直接打鼓機的 pattern。"""
    SNARE, KICK_POOL = (4, 12), (6, 7, 10, 11, 14, 3, 9, 13)
    FILLS = ([(12, 38), (13, 38), (14, 38), (15, 38)], [(12, 38), (14, 45), (15, 48)],  # 最後一拍的過門：小鼓滾奏、小鼓接 tom
             [(13, 38), (14, 38), (15, 46)], [(12, 45), (13, 45), (14, 48), (15, 50)])
    BREAK = [(8, 38), (10, 38), (12, 38), (13, 38), (14, 38), (15, 38)]  # 整小節的斷：後半小節小鼓漸密

    def __init__(self, seed: int) -> None:
        self.rng, self.kicks = random.Random(seed), {0, 6, 10}
        names = {k for st in STYLES.values() for k in st["kits"] if os.path.isdir(os.path.join(ROOT, "samples", k))}  # 沒有的資料夾跳過
        self.kits = {k: load_kit(k, self.rng) for k in sorted(names)}  # 每個鼓組先載一套，換組時不用等
        self.kit, self.kit_style, self.last808 = None, None, None  # 正在用的鼓組；last808 = 鼓組沒有 808 時沿用上一個
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

    def pick_kit(self, style: str) -> None:
        """換鼓組：從這種風格的鼓組裡隨機挑一個（盡量不跟現在的一樣），並在背景重新載入它的另一套取樣，下次輪到時音色又不同。"""
        names = [n for n in STYLES[style]["kits"] if n in self.kits]
        if not names:  # samples/ 裡沒有這種風格的鼓組（見 samples/README.md）：鼓安靜，其他樂器照常
            if self.kit_style != style:
                print(f"🥁 samples/ 裡沒有 {style} 的鼓組，鼓先安靜", flush=True)
            self.kit, self.kit_style = {r: np.zeros((480, 2), np.float32) for r in KIT_ROLES} | {"name": "", "808": None}, style
            return
        name = self.rng.choice([n for n in names if not self.kit or n != self.kit["name"]] or names)
        self.kit, self.kit_style = self.kits[name], style
        self.last808 = self.kit["808"] or self.last808
        print(f"🥁 鼓組：{name}", flush=True)
        seed = self.rng.random()
        threading.Thread(target=lambda: self.kits.__setitem__(name, load_kit(name, random.Random(seed))), daemon=True).start()

    def frame(self, frame: int, style: str, root_pc: int) -> tuple:
        """這一幀的 (鼓聲, 808 貝斯或 None, 打擊)。鼓聲、808 各 (1920, 2)；打擊 = [大鼓, 小鼓類, hi-hat 類] 這一幀打下去的力度
        （0–1，給視覺用，比從聲音猜準）。段落開頭把這段的演奏排進 pending、並準備下一段；風格決定鼓組和 hi-hat 密度；
        bass808 風格，每個大鼓同時打一個 root_pc 音高的 808（單音，新的會切掉舊的）。"""
        start, hat_div, hits = frame * 1920, STYLES[style]["hat_div"], [0.0, 0.0, 0.0]
        use808 = STYLES[style].get("bass808", False) and self.last808 is not None
        if tm.starts(frame, BPM / 8):  # 每 2 小節一段
            seg, step = tm.count(frame, BPM / 8), 60 / BPM / 4
            if style != self.kit_style or seg % KIT_SEGMENTS == 0:  # 換風格時、或每隔一段時間換鼓組
                self.pick_kit(style)
                use808 = STYLES[style].get("bass808", False) and self.last808 is not None
            self.prepare(seg + 1)
            for t, p, v in self.played.pop(seg, []):
                if p in SAMPLES and (p != 42 or round(t / step) % (4 // hat_div) == 0):  # neo soul 只留八分音符的 hi-hat
                    bisect.insort(self.pending, (max(start, start + int(t * 48000)) + DRUM_LAG * 1920, p, v))
        while self.pending and self.pending[0][0] < start + 1920:
            pos, p, v = self.pending.pop(0)
            name, gain = SAMPLES[p]
            y = self.kit[name]
            self.buf[pos - start: pos - start + len(y)] += gain * (v / 127) ** 0.5 * y  # 力度曲線：輕的不會太小聲
            kind = 0 if p == 36 else (2 if name in ("hihat", "openhat") else 1)  # 小鼓類含 tom、rim
            hits[kind] = max(hits[kind], (v / 127) ** 0.5)
            if p == 36 and use808:
                i, y = pos - start, self.last808[root_pc]
                self.buf808[i: i + 240] *= np.linspace(1, 0, 240)[:, None]  # 5 ms 內切掉上一個 808
                self.buf808[i + 240:] = 0
                y = y[: len(self.buf808) - i]  # 調低音高後可能比緩衝還長，超過的切掉
                self.buf808[i: i + len(y)] += 0.35 * (v / 127) ** 0.5 * y  # 取樣峰值 1，壓到跟 MRT2 貝斯差不多的音量
        out = self.buf[:1920].copy()
        self.buf[:-1920] = self.buf[1920:]
        self.buf[-1920:] = 0
        bass = None
        if use808:
            bass = self.buf808[:1920].copy()
            self.buf808[:-1920] = self.buf808[1920:]
            self.buf808[-1920:] = 0
        return out, bass, hits
