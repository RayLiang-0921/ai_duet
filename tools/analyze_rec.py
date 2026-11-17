"""
分析 `band_mrt2.py --record` 錄下的分軌（12 聲道：AGENTS 的順序再加鼓，各立體聲）：
  1. 調性：每個樂器的 chroma 能量有多少在音階上（隨機約 0.58）、最強的音級；貝斯另外用音高追蹤算在調內的比例與音準
  2. 跟拍：每個樂器的起音相對於鼓提前或延後幾 ms（互相關峰值）、鎖在八分音符格上的程度（0–1）
  3. 鼓味：MRT2 樂器有沒有偷偷打鼓（6 kHz 以上的起音鎖在 16 分音符格上的程度，> 0.4 就可疑）
用法（在 WSL 的 GPU 環境裡）：python tools/analyze_rec.py /tmp/rec.wav C major [切換到 memphis 的秒數]
"""
import sys

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0].rsplit("\\tools\\", 1)[0])
from band.settings import AGENTS, BPM  # noqa: E402

NN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALE = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
names = AGENTS + ["drums"]


def onset_lock(x: np.ndarray, sr: int, period: float, lo: float = 0, hi: float = 0) -> float:
    """起音包絡（可先帶通）鎖在 period 秒格子上的程度，0–1。"""
    if hi:
        x = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x).astype(np.float32)
    env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=240)
    t = np.arange(len(env)) * 240 / sr
    w = np.maximum(env - np.median(env), 0)
    return float(abs((w * np.exp(2j * np.pi * t / period)).sum() / (w.sum() + 1e-9)))


def main() -> None:
    path, root, mode = sys.argv[1], NN.index(sys.argv[2]), sys.argv[3]
    switch = float(sys.argv[4]) if len(sys.argv) > 4 else None
    y, sr = sf.read(path, dtype="float32")
    dur = len(y) / sr
    stems = {n: y[:, 2 * i: 2 * i + 2].mean(1) for i, n in enumerate(names)}
    segs = [(f"{NN[root]} {mode}", 0, switch or dur, mode)]
    if switch:
        segs.append((f"memphis：{NN[root]} minor", switch, dur, "minor"))

    for label, a, b, m in segs:
        print(f"\n===== {label}（{a:.0f}–{b:.0f} s）=====")
        mask = np.zeros(12)
        mask[[(root + s) % 12 for s in SCALE[m]]] = 1
        cut = lambda n: stems[n][int(a * sr): int(b * sr)]
        print("調性：chroma 能量在音階上的比例、最強音級")
        for n in AGENTS:
            x = cut(n)
            if np.sqrt((x ** 2).mean()) < 1e-4:
                print(f"   {n:8s} 靜音")
                continue
            c = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=1024, bins_per_octave=36).mean(1)
            c /= c.sum()
            print(f"   {n:8s} {float((c * mask).sum()):.2f}   " + " ".join(f"{NN[i]}:{c[i]:.2f}" for i in np.argsort(c)[::-1][:4]))
        f0, voiced, _ = librosa.pyin(cut("bass"), fmin=30, fmax=400, sr=sr, frame_length=4096, hop_length=1024)
        if voiced.any():
            midi = 12 * np.log2(f0[voiced] / 440) + 69
            pcs = np.round(midi).astype(int) % 12
            print(f"   貝斯音高追蹤：{np.isin(pcs, np.flatnonzero(mask)).mean():.0%} 在調內，音準誤差中位數 "
                  f"{np.median(np.abs(midi - np.round(midi)) * 100):.0f} cents，有聲 {voiced.mean():.0%}")

        print("跟拍：相對於鼓的延後（+ = 比鼓晚）、鎖在八分音符格上的程度")
        env = {n: librosa.onset.onset_strength(y=cut(n), sr=sr, hop_length=240) for n in names}
        d = env["drums"] - env["drums"].mean()
        for n in AGENTS:
            e, L = env[n] - env[n].mean(), 60
            cc = np.array([np.dot(d[max(0, -k): len(d) - max(0, k)], e[max(0, k): len(d) - max(0, -k)]) for k in range(-L, L + 1)])
            cc /= np.linalg.norm(d) * np.linalg.norm(e) + 1e-9
            print(f"   {n:8s} {(int(np.argmax(cc)) - L) * 5:+4d} ms（相關 {cc.max():.2f}）  鎖拍 {onset_lock(cut(n), sr, 30 / BPM):.2f}")
        print(f"   {'drums':8s} 鎖拍 {onset_lock(cut('drums'), sr, 30 / BPM):.2f}")

        print("鼓味：6 kHz 以上的起音鎖在 16 分音符格上的程度（> 0.4 可疑）、6 kHz 以上佔的能量")
        for n in AGENTS:
            x = cut(n)
            hf = sosfiltfilt(butter(4, [6000, 16000], btype="band", fs=sr, output="sos"), x)
            print(f"   {n:8s} {onset_lock(x, sr, 15 / BPM, 6000, 16000):.2f}（{(hf ** 2).mean() / ((x ** 2).mean() + 1e-12):.1%}）")


if __name__ == "__main__":
    main()
