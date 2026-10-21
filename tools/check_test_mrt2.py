"""
檢查 test_mrt2.py 的速度、調性、鼓量是否符合設定（WSL GPU 環境執行）。

    python tools/check_test_mrt2.py

每個組合連續生成 60 秒（不播放），切成三個 20 秒視窗分別量測，列出平均與範圍：
  - 速度：拍點起音包絡在目標 BPM 週期上的自相關，扣掉鄰近速度（越高越鎖拍）
  - 調性：諧波成分中落在調內的能量比例，以及 12 個調裡哪個最吻合（應該是設定的調）
  - 鼓量：大鼓（40–120 Hz）與鈸（6 kHz 以上）頻段的打擊能量佔總能量比例。
          不用全頻段打擊比例，因為和弦重新起音也會被算成打擊聲
"""
import os
import sys

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_mrt2 as tm  # noqa: E402

SR, SECONDS, WINDOW = 48000, 63, 20  # 前 3 秒暖機不計


def measure(samples: np.ndarray, bpm: int) -> tuple:
    y = samples.mean(axis=1)
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=512)
    ac = librosa.autocorrelate(env - env.mean())
    ac /= ac[0]
    lag = lambda b: int(round(SR / 512 * 60 / b))
    peak = lambda b: ac[lag(b) - 1:lag(b) + 2].max()
    lock = peak(bpm) - np.mean([peak(bpm * r) for r in (0.8, 0.9, 1.1, 1.25)])

    S = np.abs(librosa.stft(y))
    H, P = librosa.decompose.hpss(S)
    freqs = librosa.fft_frequencies(sr=SR)
    band = ((freqs > 40) & (freqs < 120)) | (freqs > 6000)
    drums = (P[band] ** 2).sum() / ((S ** 2).sum() + 1e-9)

    chroma = librosa.feature.chroma_cqt(y=librosa.effects.harmonic(y), sr=SR).sum(axis=1)
    root_name, mode = tm.KEY.split()
    share = [chroma[[(r + s) % 12 for s in tm.SCALES[mode]]].sum() / chroma.sum() for r in range(12)]
    in_key = share[tm.NOTE_NAMES.index(root_name)]
    best = int(np.argmax(share))
    return lock, in_key, drums, best


mrt = tm.MagentaRT2Jax(size="mrt2_small")
mrt.generate(conditioning={}, frames=2)  # JIT 暖機
print(f"風格={tm.STYLE!r} 調性={tm.KEY} 和弦={tm.PROGRESSION}")
for drums in (False, True):
    for bpm in (78, 120):
        tm.DRUMS, tm.BPM = drums, bpm
        tokens = list(mrt.tokenize_style(mrt.embed_style(f"{tm.STYLE}, in {tm.KEY}")))
        state, parts = None, []
        for f in range(SECONDS * 25):
            cond, cfg = tm.conditioning(f, tokens)
            wav, state = mrt.generate(conditioning=cond, cfg_scales=cfg, frames=1, state=state)
            parts.append(wav.samples)
        audio = np.concatenate(parts)[SR * 3:]
        rows = np.array([measure(audio[i * SR * WINDOW:(i + 1) * SR * WINDOW], bpm) for i in range(3)])
        best = "/".join(tm.NOTE_NAMES[int(b)] for b in rows[:, 3])
        fmt = lambda c, k=1, u="": f"{rows[:, c].mean() * k:5.2f}{u} ({rows[:, c].min() * k:.2f}–{rows[:, c].max() * k:.2f})"
        print(f"DRUMS={str(drums):5s} BPM={bpm:3d}  鎖拍={fmt(0)}  調內音={fmt(1, 100, '%')}  "
              f"鼓頻段={fmt(2, 100, '%')}  最吻合的調={best}", flush=True)
