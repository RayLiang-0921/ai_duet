"""
找新樂器（整晚跑用）：每個候選樂器在每種風格（兩種原本的 + custom_styles.txt 的自訂風格）各生成 SECONDS 秒、2 個種子，
量調內比例、跟拍、鼓聲、人聲、低頻、音量（tools/audition.py 的量法）。每段結果馬上寫進 reports/explore-日期.csv（斷電也留得住），
最後寫 reports/explore-日期.md：每個位置可以加進 band/settings.py 的 INSTRUMENTS 的樂器。
用法（WSL GPU 環境）：python tools/explore.py        只重寫報告：python tools/explore.py --summary reports/explore-日期.csv
"""
import csv
import datetime
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

SECONDS, SEEDS = 20, (11, 12)
CANDIDATES = {  # 位置 → [(顯示名稱, 提示詞)]
    "keys": [("鋼琴", "acoustic grand piano"), ("直立鋼琴", "upright piano"), ("Clavinet", "clavinet"), ("鋼片琴", "celesta"),
             ("大鍵琴", "harpsichord"), ("手風琴", "accordion"), ("Mellotron", "mellotron"), ("合成器和弦", "analog synth chords"),
             ("玩具鋼琴", "toy piano"), ("柔和電鋼琴", "soft electric piano with chorus")],
    "rhythm": [("爵士吉他", "jazz guitar"), ("木吉他", "acoustic guitar"), ("班鳩琴", "banjo"), ("曼陀林", "mandolin"),
               ("烏克麗麗", "ukulele"), ("古箏", "guzheng"), ("馬林巴", "marimba"), ("拇指琴", "kalimba"), ("鋼鼓", "steel pan"),
               ("西塔琴", "sitar")],
    "bass": [("合成貝斯", "synth bass"), ("Moog 貝斯", "moog bass"), ("撥弦低音提琴", "pizzicato double bass"),
             ("低音單簧管", "bass clarinet"), ("低音號", "tuba"), ("Pick 貝斯", "picked electric bass"), ("Slap 貝斯", "slap bass"),
             ("大提琴撥奏", "pizzicato cello")],
    "pad": [("合成鋪底", "warm analog synth pad"), ("簧風琴", "harmonium"), ("Mellotron 弦樂", "mellotron strings"),
            ("法國號", "french horn section"), ("銅管", "soft brass section"), ("木管", "woodwind ensemble"),
            ("玻璃鋪底", "glass pad"), ("環境音", "ambient drone"), ("中提琴", "viola section"), ("教堂管風琴", "church organ")],
    "lead": [("薩克斯風", "saxophone"), ("小號", "trumpet"), ("弱音小號", "muted trumpet"), ("單簧管", "clarinet"),
             ("口琴", "harmonica"), ("小提琴", "solo violin"), ("雙簧管", "oboe"), ("鐘琴", "glockenspiel"), ("排笛", "pan flute"),
             ("陶笛", "ocarina"), ("合成主音", "synth lead"), ("Theremin", "theremin")],
}
FIELDS = ["slot", "name", "prompt", "style", "base", "seed", "key", "beat", "hat", "hf%", "lf%", "loud", "sound%", "drum", "voice"]


def summary(path: str) -> str:
    """CSV → markdown：每個 (位置, 樂器) 在每種風格的最差數值，以及能不能用（每種原本的風格分開判斷）。"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    by = {}
    for r in rows:
        by.setdefault((r["slot"], r["name"], r["prompt"]), []).append(r)
    out = [f"# 新樂器試音（{os.path.basename(path)}，{len(rows)} 段）", "",
           "能用 = 在這種風格（含以它為基礎的自訂風格）每一段都：鼓聲、人聲 ≤ 0.3，調內 ≥ 0.6，有聲 ≥ 50%，"
           "非貝斯位置低頻 ≤ 30%。音量 = 平均的有聲第 90 百分位，直接填進 INSTRUMENTS。", ""]
    for slot in CANDIDATES:
        out += [f"## {slot}", "", "| 樂器 | 提示詞 | neo soul | memphis rap | 最大鼓聲 | 最大人聲 | 最低調內 | 最大低頻% | 音量 neo / memphis |",
                "|---|---|---|---|---|---|---|---|---|"]
        for (s, name, prompt), rs in by.items():
            if s != slot:
                continue
            verdict, loud = {}, {}
            for base in ("neo soul", "memphis rap"):
                b = [r for r in rs if r["base"] == base]
                if not b:
                    verdict[base], loud[base] = "—", 0.0
                    continue
                f = lambda k: [float(r[k]) for r in b]
                ok = (max(f("drum")) <= .3 and max(f("voice")) <= .3 and min(f("key")) >= .6 and min(f("sound%")) >= 50
                      and (slot == "bass" or max(f("lf%")) <= 30))
                verdict[base], loud[base] = ("能用" if ok else "不行"), float(np.mean(f("loud")))
            g = lambda k, fn: fn(float(r[k]) for r in rs)
            out.append(f"| {name} | {prompt} | {verdict['neo soul']} | {verdict['memphis rap']} | {g('drum', max):.2f} | "
                       f"{g('voice', max):.2f} | {g('key', min):.2f} | {g('lf%', max):.0f} | {loud['neo soul']:.3f} / {loud['memphis rap']:.3f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    if "--summary" in sys.argv:
        path = sys.argv[sys.argv.index("--summary") + 1]
        open(path.replace(".csv", ".md"), "w", encoding="utf-8").write(summary(path))
        return
    import test_mrt2 as tm
    from audition import LABELS, leak, metrics
    from band.band import clean_style, prompt
    from band.harmony import agent_conditioning
    from band.mrt2 import BatchedMRT2
    from band.settings import FRAMES_PER_CALL, NEXT_CHORD, STYLES, TEMPERATURE

    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    path = os.path.join(ROOT, "reports", f"explore-{datetime.date.today()}.csv")
    done = set()
    if os.path.exists(path):  # 接著上次（例如斷電）沒跑完的繼續
        done = {(r["slot"], r["prompt"], r["style"], r["seed"]) for r in csv.DictReader(open(path, encoding="utf-8"))}
    new_file = not os.path.exists(path)
    out = open(path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(out, FIELDS)
    if new_file:
        writer.writeheader()

    styles = {"neo soul": "neo soul", "memphis rap": "memphis rap"}  # 風格 → 根據的風格
    for line in open(os.path.join(ROOT, "custom_styles.txt"), encoding="utf-8"):
        text, _, base = line.split("#")[0].partition("|")
        name = clean_style(text.strip())[0]
        if name and base.strip() in styles:
            styles[name[:40]] = base.strip()
            STYLES[name[:40]] = {**STYLES[base.strip()], "prompt": name, "base": base.strip()}
    jobs = [(slot, inst, st, seed) for slot, cands in CANDIDATES.items() for inst in cands for st in styles for seed in SEEDS
            if (slot, inst[1], st, str(seed)) not in done]
    print(f"{len(jobs)} 段要跑（{len(done)} 段之前跑過）", flush=True)

    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, 5)
    labels = np.stack([np.asarray(mrt.embed_style(t), np.float32).ravel() for t in LABELS])
    labels /= np.linalg.norm(labels, axis=1, keepdims=True)
    n = SECONDS * 25 // FRAMES_PER_CALL * FRAMES_PER_CALL
    rng, degrees, d = random.Random(3), [], 0
    for _ in range(n // 80 + 2):  # 和弦進行固定，每個候選聽到的一樣
        degrees.append(d)
        d = rng.choice(NEXT_CHORD["major"].get(d, [0]))
    t0 = time.time()
    for b in range(0, len(jobs), 5):
        batch = jobs[b:b + 5]
        batch += [batch[-1]] * (5 - len(batch))
        toks = [list(mrt.tokenize_style(mrt.embed_style(prompt(slot, st, (inst[0], inst[1], .1))))) for slot, inst, st, _ in batch]
        modes = [STYLES[st].get("mode", "major") for _, _, st, _ in batch]
        state, audio = band.new_state(batch[0][3] * 1000 + b), []
        for f0 in range(0, n, FRAMES_PER_CALL):
            rows = [band.frame_row([agent_conditioning(slot, f, 0, modes[i], {slot: toks[i]}, (set(), set()), degrees[f // 80])
                                    for i, (slot, _, _, _) in enumerate(batch)]) for f in range(f0, f0 + FRAMES_PER_CALL)]
            a, state = band.generate(rows, state)
            audio.append(a)
        audio = np.concatenate(audio)
        for i, (slot, inst, st, seed) in enumerate(jobs[b:b + 5]):
            y = audio[50:, i].reshape(-1, 2)
            m = metrics(y.mean(1), 48000, modes[i])
            m["drum"], m["voice"] = leak(mrt, y, labels)
            writer.writerow({"slot": slot, "name": inst[0], "prompt": inst[1], "style": st, "base": styles[st], "seed": seed,
                             **{k: f"{v:.3f}" for k, v in m.items()}})
        out.flush()
        left = (time.time() - t0) / (b + 5) * (len(jobs) - b - 5)
        print(f"{min(b + 5, len(jobs))}/{len(jobs)}，還要約 {left / 60:.0f} 分鐘", flush=True)
    out.close()
    open(path.replace(".csv", ".md"), "w", encoding="utf-8").write(summary(path))
    print(f"報告：{path.replace('.csv', '.md')}", flush=True)


if __name__ == "__main__":
    main()
