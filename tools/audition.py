"""
試音：把候選樂器放進某個位置（slot）在 MRT2 上生成，量它能不能用；順便量文字轉風格的時間。
  1. 文字 → 風格 token 要多久；演奏中做這件事會不會拖慢 GPU 生成（每幀 ms）
  2. 每個位置一次試 5 個候選（同一批在 GPU 上），兩種風格各 SECONDS 秒，量：
     調內比例（chroma 能量在音階上的比例，隨機約 0.58）、鎖拍（起音鎖在八分音符格上，0–1）、
     鼓味（6 kHz 以上的起音鎖在 16 分音符格上 > 0.4 可疑、100 Hz 以下的能量比例 = 悶響）、音量（有聲時的第 90 百分位）、
     鼓聲 / 人聲（MusicCoCa 聽生成的聲音，判斷像「鼓」「人聲」的機率；> 0.3 就不要用）
用法（WSL GPU 環境，和 run_band.sh 一樣的環境變數）：python tools/audition.py [輸出資料夾]
"""
import os
import random
import sys
import threading
import time

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_mrt2 as tm  # noqa: E402
from band.harmony import agent_conditioning  # noqa: E402
from band.mrt2 import BatchedMRT2  # noqa: E402
from band.settings import BPM, FRAMES_PER_CALL, NEXT_CHORD, STYLES, TEMPERATURE  # noqa: E402

SECONDS = 20
CANDIDATES = {  # 位置 → 5 個候選提示詞；量完把可以用的加進 band/settings.py 的 INSTRUMENTS
    "keys": ["solo rhodes electric piano", "wurlitzer electric piano", "hammond organ", "acoustic grand piano", "clavinet"],
    "rhythm": ["solo electric guitar", "nylon string guitar", "muted funk guitar", "harp", "marimba"],
    "bass": ["solo electric bass", "upright bass", "synth bass", "fretless bass", "deep sub bass"],
    "pad": ["cinematic strings", "dark strings", "legato string ensemble", "warm synth pad", "cello"],  # choir：人聲 0.95，不用
    "lead": ["digital bell", "vibraphone", "flute", "music box", "synth lead"],
}
SCALE = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
LABELS = ["drums and percussion", "drum kit", "hi-hat and snare", "singing voice", "choir vocals", "human voice humming",  # 前 3 個 = 鼓、再 3 個 = 人聲
          "piano", "electric piano", "electric guitar", "acoustic guitar", "harp", "bass guitar", "strings", "cello",
          "bells", "music box", "vibraphone", "flute", "synthesizer pad", "organ"]


def leak(mrt, x: np.ndarray, labels: np.ndarray) -> tuple:
    """(鼓聲機率, 人聲機率)：MusicCoCa 的聲音 embedding 和 LABELS 文字 embedding 比相似度（零樣本分類）。"""
    from magenta_rt import audio as au
    e = np.asarray(mrt._style_model.embed(au.Waveform(x.astype(np.float32), 48000)), np.float32).ravel()
    p = np.exp(30 * labels @ (e / np.linalg.norm(e)))
    p /= p.sum()
    return float(p[:3].sum()), float(p[3:6].sum())


def lock(x: np.ndarray, sr: int, period: float, lo: float = 0, hi: float = 0) -> float:
    if hi:
        x = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x).astype(np.float32)
    env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=240)
    t = np.arange(len(env)) * 240 / sr
    w = np.maximum(env - np.median(env), 0)
    return float(abs((w * np.exp(2j * np.pi * t / period)).sum() / (w.sum() + 1e-9)))


def metrics(x: np.ndarray, sr: int, mode: str) -> dict:
    mask = np.zeros(12)
    mask[SCALE[mode]] = 1  # 根音 C
    c = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=1024, bins_per_octave=36).mean(1)
    frames = np.sqrt((x[: len(x) // 1920 * 1920].reshape(-1, 1920) ** 2).mean(1))
    on = frames[frames > 1e-3]
    hf = sosfiltfilt(butter(4, [6000, 16000], btype="band", fs=sr, output="sos"), x)
    lf = sosfiltfilt(butter(4, 100, btype="low", fs=sr, output="sos"), x)
    p = (x ** 2).mean() + 1e-12
    return {"key": float((c / c.sum() * mask).sum()), "beat": lock(x, sr, 30 / BPM), "hat": lock(x, sr, 15 / BPM, 6000, 16000),
            "hf%": 100 * (hf ** 2).mean() / p, "lf%": 100 * (lf ** 2).mean() / p,
            "loud": float(np.percentile(on, 90)) if len(on) else 0.0, "sound%": 100 * len(on) / len(frames)}


def main() -> None:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audition"
    os.makedirs(out_dir, exist_ok=True)
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, 5)

    # ---- 1. 文字轉風格的時間 ----
    texts = ["rainy night lo-fi", "sunny bossa nova", "dark jazz club", "dreamy synthwave", "warm gospel"]
    t = time.perf_counter()
    e = mrt.embed_style(texts[0])
    t_first = time.perf_counter() - t
    times = []
    for s in texts[1:]:
        t = time.perf_counter()
        mrt.tokenize_style(mrt.embed_style(s))
        times.append(time.perf_counter() - t)
    t = time.perf_counter()
    for _ in range(20):
        mrt.tokenize_style(e)
    print(f"文字轉風格：第一次 {t_first * 1000:.0f} ms、之後每句 {np.median(times) * 1000:.0f} ms（embed + tokenize），"
          f"只 tokenize {(time.perf_counter() - t) / 20 * 1000:.1f} ms", flush=True)

    # ---- 演奏中轉風格會不會拖慢生成 ----
    tok = {a: list(mrt.tokenize_style(mrt.embed_style(f"{c[0]}, neo soul"))) for a, c in CANDIDATES.items()}
    agents = list(tok)
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", tok, (set(), set())) for a in agents])
            for f in range(FRAMES_PER_CALL)]
    _, state = band.generate(rows, band.new_state(0))
    _, state = band.generate(rows, state)

    def gen_ms(n: int = 40) -> list:
        nonlocal state
        ms = []
        for _ in range(n):
            t = time.perf_counter()
            _, state = band.generate(rows, state)
            ms.append((time.perf_counter() - t) * 1000 / FRAMES_PER_CALL)
        return ms

    base = gen_ms()
    busy, done = threading.Event(), []
    def embed_loop():
        i = 0
        while not busy.is_set():
            mrt.tokenize_style(mrt.embed_style(f"{texts[i % 5]} {i}"))
            done.append(i)
            i += 1
    th = threading.Thread(target=embed_loop)
    th.start()
    loaded = gen_ms()
    busy.set()
    th.join()
    print(f"生成每幀：平常 中位數 {np.median(base):.1f} / 最慢 {max(base):.1f} ms；"
          f"同時不停轉風格（轉了 {len(done)} 句）中位數 {np.median(loaded):.1f} / 最慢 {max(loaded):.1f} ms", flush=True)

    # ---- 2. 候選樂器 ----
    labels = np.stack([np.asarray(mrt.embed_style(t), np.float32).ravel() for t in LABELS])
    labels /= np.linalg.norm(labels, axis=1, keepdims=True)
    rng = random.Random(3)
    n = int(SECONDS * 25) // FRAMES_PER_CALL * FRAMES_PER_CALL
    results = {}
    for style, mode in (("neo soul", "major"), ("memphis rap", "minor")):
        suffix = STYLES[style].get("prompt", style)
        degrees, d = [], 0  # 每小節的和弦級數（固定種子，每個候選聽到的一樣）
        for bar in range(n // 80 + 1):
            degrees.append(d)
            d = rng.choice(NEXT_CHORD[mode].get(d, [0]))
        for slot, cands in CANDIDATES.items():
            toks = [{slot: list(mrt.tokenize_style(mrt.embed_style(f"{c}, {suffix}, instrumental")))} for c in cands]  # 和 band.py 的 prompt() 一樣
            state, audio = band.new_state(7), []
            for f0 in range(0, n, FRAMES_PER_CALL):
                rows = [band.frame_row([agent_conditioning(slot, f, 0, mode, tk, (set(), set()), degrees[f // 80]) for tk in toks])
                        for f in range(f0, f0 + FRAMES_PER_CALL)]
                a, state = band.generate(rows, state)
                audio.append(a)
            audio = np.concatenate(audio)  # (幀, 5, 1920, 2)
            for i, c in enumerate(cands):
                y = audio[:, i].reshape(-1, 2)
                sf.write(f"{out_dir}/{style.split()[0]}_{slot}_{c.replace(' ', '_')}.wav", y, 48000)
                results[(style, slot, c)] = metrics(y.mean(1), 48000, mode)
                results[(style, slot, c)]["drum"], results[(style, slot, c)]["voice"] = leak(mrt, y[50 * 1920:], labels)
            print(f"{style} / {slot} 完成", flush=True)

    print("\n位置     候選                         風格      調內  鎖拍  hat鎖  hf%   lf%   音量   有聲%  鼓聲  人聲")
    for (style, slot, c), m in results.items():
        print(f"{slot:8s} {c:28s} {style.split()[0]:8s} {m['key']:.2f}  {m['beat']:.2f}  {m['hat']:.2f}  "
              f"{m['hf%']:4.1f}  {m['lf%']:4.1f}  {m['loud']:.3f}  {m['sound%']:3.0f}  {m['drum']:.2f}  {m['voice']:.2f}")


if __name__ == "__main__":
    main()
