"""和聲：判斷觀眾彈的和弦、決定每個 agent 每一幀要彈什麼（MRT2 的 conditioning）。"""
import math

import test_mrt2 as tm
from band.settings import BPM, CHORDS


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
        bass_pc = min(sounding) % 12 if sounding else (root + steps[degree % 7]) % 12
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
