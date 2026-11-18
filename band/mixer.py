"""混音：平移、限制器，以及算給視覺用的音量數值。"""
import math

import numpy as np

FFT_FREQ = np.fft.rfftfreq(1920, 1 / 48000)
WINDOW = np.hanning(1920).astype(np.float32)


def pan_gains(position: float) -> np.ndarray:
    """等功率平移：回傳 [左, 右] 增益，放在中間時兩邊都是 1，移到一邊時總音量不變。"""
    angle = (position + 1) * np.pi / 4
    return np.array([np.cos(angle), np.sin(angle)], dtype=np.float32) * np.sqrt(2)


def limit(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
    """軟拐點限制器：threshold 以下完全不動，超過才柔和壓縮，避免爆音。"""
    a = np.abs(x)
    return np.where(a <= threshold, x,
                    np.sign(x) * (threshold + (1 - threshold) * np.tanh((a - threshold) / (1 - threshold))))


def levels(stems: dict, hits: list) -> str:
    """一幀 → 給視覺的 9 個數值（大約 0–1），順序對應 vj_tk.Show.handle：
    大鼓、小鼓、hi-hat（hits：鼓手這一幀真的打下去的力度，不從聲音猜）、貝斯、Rhodes、旋律（吉他＋鐘）、
    Rhodes 音色、旋律高低、鍵盤與弦樂合起來（背景亮度）。除數是實際音量的第 90 百分位，所以強的音約為 1。"""
    mono = {a: s.mean(axis=1) for a, s in stems.items()}
    spectrum = lambda a: np.abs(np.fft.rfft(mono[a] * WINDOW)) ** 2
    rms = lambda a: float(np.sqrt((mono[a] ** 2).mean()))
    centroid = lambda P, lo, hi: min(1.0, max(0.0, math.log2(max(lo, (P * FFT_FREQ).sum() / (P.sum() + 1e-9)) / lo)
                                               / math.log2(hi / lo)))
    v = [*hits, rms("bass") / 0.15, rms("rhodes") / 0.1, (rms("guitar") + rms("bell")) / 0.15,
         centroid(spectrum("rhodes"), 200, 2000), centroid(spectrum("guitar"), 150, 3000),
         (rms("rhodes") + rms("strings")) / 0.25]  # 弦樂一直在，除數放大才不會整個畫面過亮
    return " ".join(f"{min(x, 2.0):.2f}" for x in v)
