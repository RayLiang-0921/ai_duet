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
