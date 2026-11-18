"""主流程：載入模型 → 等觀眾彈和弦 → 生成執行緒（和弦進行、風格、編制、樂器、鼓）＋ 播放執行緒 → 結束統計。"""
import os
import queue
import random
import re
import socket
import subprocess
import sys
import threading
import time

import numpy as np
import soundfile as sf

import test_mrt2 as tm
from band.drummer import Drummer
from band.harmony import agent_conditioning, detect_chord
from band.links import Human, VisualLink, start_windows
from band.mixer import levels, limit, pan_gains
from band.mrt2 import BatchedMRT2, StyleBank
from band.settings import (AGENTS, BLENDS, BPM, CHORD_SETTLE, DROP_P, DRUM_WORDS, FRAMES_PER_CALL, GAINS, GROOVE_PYTHON,
                           INSTRUMENTS, LEVEL, MASTER, NEXT_CHORD, PAN, PHRASE_BARS, QUEUE_FRAMES, REJOIN_P, ROOT, STYLES,
                           TEMPERATURE, VISUAL_TCP)

CUSTOM_STYLES = os.path.join(ROOT, "custom_styles.txt")  # 自訂風格清單：一行一個「風格描述 | 根據的風格」


def pool(slot: str, style: str) -> list:
    """這個位置在這種風格可以用的樂器 [(名稱, 提示詞, 音量)]；打字新增的風格用它所根據的風格的樂器。"""
    return INSTRUMENTS[slot][STYLES[style].get("base", style)]


def prompt(slot: str, style: str, inst: tuple) -> str:
    """送給 MRT2 的提示詞 = 樂器 + 風格描述。"""
    return f"{inst[1]}, {STYLES[style].get('prompt', style)}"


def clean_style(text: str) -> tuple:
    """觀眾打的風格 → (拿掉會帶出鼓聲的字之後的文字, 拿掉的字)。"""
    removed = []
    for w in sorted(DRUM_WORDS, key=len, reverse=True):
        pattern = rf"(?<![\w-]){re.escape(w)}(?![\w-])"
        if re.search(pattern, text, flags=re.I):
            removed.append(w)
            text = re.sub(pattern, " ", text, flags=re.I)
    text = re.sub(r"\s*,[\s,]*", ", ", re.sub(r"\s+", " ", text)).strip(" ,")
    text = re.sub(r"(\s+(with|and|of|plus|featuring))+$", "", text, flags=re.I)  # 拿掉字之後尾巴剩下的連接詞
    return text, removed


def main() -> None:
    import jax
    import pasimple  # 經 WSLg 播放到 Windows 喇叭

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))  # 編譯結果存起來，第二次啟動快很多
    try:  # 已經有一個樂團在跑時，連接埠都被占用，新的會收不到鍵盤和視覺的指令
        socket.create_connection(VISUAL_TCP, timeout=0.5).close()
        print("⚠ 已經有一個樂團在執行（連接埠被占用）；請先結束它（視覺視窗按 Q 或終端機按 Enter）", flush=True)
        return
    except OSError:
        pass

    # ---- 啟動其他程式、載入模型 ----
    styles = list(STYLES)  # 選單裡的風格；自訂風格（custom_styles.txt、觀眾打的）會加在後面
    pan = {a: pan_gains(PAN[a]) for a in GAINS}
    stop, paused = threading.Event(), threading.Event()
    link = VisualLink(stop)
    visual = None if "--no-visual" in sys.argv else start_windows("vj_tk.py")  # 視覺先開，載入模型時就看得到

    def progress(step: str) -> None:  # 載入進度：終端機和視覺畫面都顯示
        print(step, flush=True)
        link.send(f"t|樂團準備中…|{step}|600")

    try:  # 先確認聲音送得出去；WSLg 的音效伺服器有時會不見，不先檢查的話會整場沒聲音卻沒有任何提示
        pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE, 2, 48000, app_name="ai_duet_check").close()
    except Exception:
        print("⚠ 沒有聲音輸出：WSL 的音效伺服器（WSLg）沒有回應。請關掉這個視窗，"
              "在 Windows 命令列執行 wsl --shutdown，再重開 run_band.bat", flush=True)
        for _ in range(150):  # 等視覺連上（最多 15 秒）再顯示
            if link.clients:
                break
            time.sleep(0.1)
        link.send("t|沒有聲音輸出|WSL 的音效伺服器沒有回應：請執行 wsl --shutdown 後重開 run_band.bat|600")
        time.sleep(15)
        stop.set()
        if visual:
            visual.terminate()
        return

    groove_python, groove = os.path.expanduser(GROOVE_PYTHON), None
    if os.path.exists(groove_python):  # TensorFlow 的大量警告寫進紀錄檔，終端機只留「就緒」那一行
        log = os.path.expanduser("~/.cache/ai_duet/groove.log")
        os.makedirs(os.path.dirname(log), exist_ok=True)
        groove = subprocess.Popen([groove_python, "-W", "ignore", os.path.join(ROOT, "groove_server.py")],
                                  stdin=subprocess.DEVNULL, stderr=open(log, "w"))
    else:
        print(f"[鼓手] 找不到 {GROOVE_PYTHON}，鼓只用鼓機的 pattern", flush=True)
    ready = {}  # 鼓手（載入所有鼓組約 9 秒）在背景準備，和載入模型同時進行
    loader = threading.Thread(target=lambda: ready.update(drummer=Drummer(int(time.time()))))
    loader.start()
    progress("1/4 載入 MRT2 模型")
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, len(AGENTS))
    bank = StyleBank(mrt)
    encoder = threading.Thread(target=bank.load_encoder)  # 文字模型在背景載入（約 18 秒），和下面的準備、編譯同時進行
    encoder.start()
    progress("2/4 準備每種風格的樂器（第一次約 30 秒，之後很快）")
    for st in STYLES:  # 每種風格每個樂器的風格 token 先準備好（之後從硬碟讀）
        for a in AGENTS:
            for inst in pool(a, st):
                bank.warm(prompt(a, st, inst))

    # 先編譯，人類開始彈時就不用等。要算兩次：第二次的狀態來自第一次的輸出，型別略有不同，JAX 會再編譯一次（約 5 秒）
    progress("3/4 編譯（第一次約 4 分鐘，之後約 1 分鐘）")
    tok0 = {a: bank.get(prompt(a, styles[0], pool(a, styles[0])[0])) for a in AGENTS}
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", tok0, (set(), set())) for a in AGENTS])
            for f in range(FRAMES_PER_CALL)]
    _, state = band.generate(rows, band.new_state(0))
    band.generate(rows, state)
    progress("4/4 鼓手與自訂風格")
    loader.join()
    encoder.join()
    drummer = ready["drummer"]
    drummer.prepare(0)  # 第一段先排好，開始時不用等

    human = Human(link)
    audio_q: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_FRAMES)  # (混音, 給視覺的那一幀資料)
    dropouts, gen_ms = [0], []
    muted = set()  # 觀眾手動靜音的樂器（m1–m6 = GAINS 的順序）
    new_styles: "queue.Queue[str]" = queue.Queue()  # 打字新增、已經準備好的風格
    playing = threading.Event()  # 樂器開始演奏了：之後不能再打字新增風格，只能從選單選現有的

    def pick_base(text: str) -> str:
        """觀眾打的風格（還沒拿掉鼓的字，trap、drill 這類字正好說明該用哪種鼓）文字上最像哪種原本的風格；
        自訂風格用那種風格的鼓組、調式和樂器，鼓才會符合曲風。"""
        roots = [s for s in STYLES if "base" not in STYLES[s]]
        return max(roots, key=lambda b: bank.similarity(text, b) + STYLES[b].get("lean", 0.0))

    def make_style(cleaned: str, base: str) -> str:
        """新增一個自訂風格：風格描述 = cleaned，沿用 base 風格的鼓、調式和樂器，把每個樂器的 token 準備好。回傳風格名稱。"""
        name = cleaned[:40]
        if name not in STYLES:
            STYLES[name] = {**STYLES[base], "prompt": cleaned, "base": base}
            STYLES[name].pop("lean", None)
            for a in AGENTS:
                for inst in pool(a, name):
                    bank.warm(prompt(a, name, inst))
            styles.append(name)
            link.send("y|" + "|".join(styles))
        return name

    def add_style(text: str) -> None:
        """（背景執行緒）開始演奏前觀眾打的風格：拿掉會帶出鼓聲的字、判斷用哪種風格的鼓、準備好、
        記進 custom_styles.txt（下次開機還在），當作起始風格。"""
        cleaned, removed = clean_style(text)
        if removed:
            print(f"✂ 拿掉會帶出鼓聲的字：{'、'.join(removed)}", flush=True)
        if not cleaned:
            link.send(f"t|沒有可以用的風格描述|{'（拿掉了：' + '、'.join(removed) + '）' if removed else ''}|4")
            return
        link.send(f"t|{cleaned[:40]}|風格準備中…{'（拿掉了：' + '、'.join(removed) + '）' if removed else ''}|30")
        known, t = cleaned[:40] in STYLES, time.perf_counter()
        print(f"✍ 開始準備「{cleaned[:40]}」", flush=True)
        name = make_style(cleaned, STYLES[cleaned[:40]]["base"] if known else pick_base(text))
        if not known:
            print(f"✍ 新風格「{name}」準備好了（{time.perf_counter() - t:.1f} 秒，最像 {STYLES[name]['base']}，用它的鼓和樂器）", flush=True)
            with open(CUSTOM_STYLES, "a", encoding="utf-8") as f:
                print(f"{cleaned} | {STYLES[name]['base']}", file=f)
        new_styles.put(name)

    # custom_styles.txt：事先寫好、或之前打過的自訂風格，開機就準備好，選單裡選得到（演出中不用打字）
    for line in open(CUSTOM_STYLES, encoding="utf-8") if os.path.exists(CUSTOM_STYLES) else ():
        text, _, base = line.split("#")[0].partition("|")
        cleaned, _ = clean_style(text.strip())
        if cleaned:
            make_style(cleaned, base.strip() if base.strip() in INSTRUMENTS["keys"] else pick_base(text.strip()))
    if len(styles) > 2:
        print(f"自訂風格：{'、'.join(styles[2:])}", flush=True)

    def commands() -> list:
        """視覺視窗送來的指令：q 結束、p 暫停/繼續、m<n> 切靜音、t <文字> 新風格（只有開始演奏前）在這裡處理；
        g <風格>（選單選的風格）回傳那個風格名稱的清單。"""
        out = []
        while not link.commands.empty():
            c = link.commands.get()
            if c == "q":
                stop.set()
            elif c == "p":
                paused.clear() if paused.is_set() else paused.set()
                print("⏸ 暫停" if paused.is_set() else "▶ 繼續", flush=True)
            elif c.startswith("m") and c[1:].isdigit() and 1 <= int(c[1:]) <= len(GAINS):
                a = list(GAINS)[int(c[1:]) - 1]
                muted.symmetric_difference_update({a})
                print(f"🔇 {a} 靜音" if a in muted else f"🔊 {a} 回來", flush=True)
            elif c.startswith("t ") and c[2:].strip() and playing.is_set():
                print(f"✍ 演出中不能新增風格，略過：{c[2:].strip()}", flush=True)
            elif c.startswith("t ") and c[2:].strip():
                print(f"✍ 觀眾輸入風格：{c[2:].strip()}", flush=True)
                threading.Thread(target=add_style, args=(c[2:].strip(),), daemon=True).start()
            elif c.startswith("g ") and c[2:] in STYLES:
                out.append(c[2:])
        return out

    def wait_for_chord(style: str) -> tuple:
        """等觀眾彈第一個和弦；期間可以用選單（或打字）換起始風格。回傳 (起始風格, 和弦) 或 (風格, None) = 被結束。"""
        print(f"🎹 請彈一個和弦開始（需要根音和三度音）；點一下視覺畫面可以選風格，目前：{style}", flush=True)
        link.send("y|" + "|".join(styles))
        link.send(f"w|{style}|{STYLES[style]['scene']}")
        key = None
        while not stop.is_set() and key is None:
            new = next(iter(commands()[-1:]), None)
            while not new_styles.empty():
                new = new_styles.get()
            if new:
                style = new
                print(f"🔀 起始風格：{style}", flush=True)
                link.send(f"w|{style}|{STYLES[style]['scene']}")
            if human.held():
                stop.wait(CHORD_SETTLE)
                key = detect_chord(human.held())
            stop.wait(0.01)
        return style, key

    def generator() -> None:
        style, key = wait_for_chord(styles[0])
        if key is None:
            return
        root, played_mode = key
        playing.set()
        mode = STYLES[style].get("mode", played_mode)  # memphis 一律小調
        key_name = f"{tm.NOTE_NAMES[root]} {mode}"
        print(f"🎵 {key_name}，{style}，開始！", flush=True)
        link.send(f"k|{key_name}|{style}")

        state, frame, pending = band.new_state(int(time.time()) % 10000 * 10), 0, None
        rng, degree, human_degree = random.Random(), 0, None
        steps = tm.SCALES[mode]
        lineup = {a: pool(a, style)[0] for a in AGENTS}  # 每個位置現在彈的樂器 (名稱, 提示詞, 音量)
        next_lineup, swap = {}, {}  # 切換風格後的樂器；休息中、下一段要換的樂器
        last_tok = {a: bank.get(prompt(a, style, lineup[a])) for a in AGENTS}
        sent_names = None  # 上次送給視覺的樂器名稱
        target = {a: 1.0 for a in GAINS}  # 編制：每個樂器 1 = 在場、0 = 退出；active 是淡入淡出中的實際音量
        active = dict(target)
        rec = None  # --record 檔名：各樂器分軌存成多聲道 wav（測試用）
        if "--record" in sys.argv:
            rec = sf.SoundFile(sys.argv[sys.argv.index("--record") + 1], "w", 48000, 2 * (len(AGENTS) + 1), "FLOAT")

        def plan_switch(new: str) -> dict:
            """要切到 new：每個位置留著現在的樂器（新風格也有的話），否則用新風格的第一個；在背景先把逐拍混合的 token 算好。"""
            nxt = {a: next((i for i in pool(a, new) if i[0] == lineup[a][0]), pool(a, new)[0]) for a in AGENTS}
            pairs = [(prompt(a, style, lineup[a]), prompt(a, new, nxt[a])) for a in AGENTS]
            threading.Thread(target=lambda: [bank.warm(p, q, w) for w in BLENDS for p, q in pairs], daemon=True).start()
            return nxt

        while not stop.is_set():
            # ---- 觀眾的輸入 ----
            new = next(iter(commands()[-1:]), None)  # 選單選的風格
            while not new_styles.empty():
                new = new_styles.get()
            if new == style and pending:  # 選回現在的風格 = 取消切換
                pending = None
                link.send(f"c|{style}")
            elif new and new != style:
                pending, swap = new, {}
                next_lineup = plan_switch(pending)
                print(f"🔀 下一段開頭切換到 {pending}", flush=True)
                link.send(f"p|{pending}")
            sounding, onsets = human.take()
            played = detect_chord(sounding) if len(sounding) >= 2 else None  # 彈和弦 → 和弦進行從那個和弦接下去
            if played and (played[0] - root) % 12 in steps:
                human_degree = steps.index((played[0] - root) % 12)

            # ---- 這一批幀的條件：和弦進行、風格、編制、樂器、鼓 ----
            rows, scenes, drums = [], [], []
            for k in range(FRAMES_PER_CALL):
                if tm.starts(frame + k, BPM / 4):  # 小節開頭：決定這小節的和弦；段落開頭才換風格、改編制，拍子不中斷
                    bar = tm.count(frame + k, BPM / 4)
                    if pending and bar % PHRASE_BARS == 0:
                        style, lineup, pending, degree = pending, next_lineup, None, 0
                        mode = STYLES[style].get("mode", played_mode)
                        steps = tm.SCALES[mode]
                        link.send(f"c|{style}")
                    elif human_degree is not None:
                        degree, human_degree = human_degree, None
                    elif bar > 0:
                        degree = rng.choice(NEXT_CHORD[mode].get(degree, [0]))
                    if bar % PHRASE_BARS == 0 and swap:  # 休息過的位置換上新樂器
                        for a, inst in swap.items():
                            print(f"🎷 {a}：{lineup[a][0]} → {inst[0]}", flush=True)
                            lineup[a] = inst
                        swap = {}
                    if bar % PHRASE_BARS == PHRASE_BARS - 1:  # 編制：下一段誰在場，這一小節慢慢淡入淡出過去；至少留 3 個
                        old = target
                        target = {a: float(rng.random() >= (DROP_P if on else 1 - REJOIN_P)) for a, on in target.items()}
                        for a in rng.sample(list(target), 3 - int(sum(target.values()))) if sum(target.values()) < 3 else ():
                            target[a] = 1.0
                        if not pending:  # 這一段要休息的位置，下一段開頭換樂器（休息時換，聽不到切換）
                            swap = {a: rng.choice([i for i in pool(a, style) if i != lineup[a]]) for a in AGENTS
                                    if old[a] and not target[a] and len(pool(a, style)) > 1}
                        print("🎚 下一段：" + "、".join(a for a, on in target.items() if on), flush=True)
                scenes.append(STYLES[style]["scene"])
                bar_beat = tm.count(frame + k, BPM) % 4
                w = BLENDS[bar_beat - 1] if pending and bar % PHRASE_BARS == PHRASE_BARS - 1 and bar_beat > 0 else 0
                tok = {}  # 這一幀每個位置的風格 token；切換前最後一小節逐拍混合（還沒算好就先用原本的）
                for a in AGENTS:
                    cur = prompt(a, style, lineup[a])
                    mix_tok = bank.get(cur, prompt(a, pending, next_lineup[a]), w) if w else None
                    tok[a] = last_tok[a] = mix_tok or bank.get(cur) or last_tok[a]
                root_pc = min(sounding) % 12 if sounding else (root + steps[degree % 7]) % 12
                drums.append(drummer.frame(frame + k, style, root_pc))
                use808 = drums[-1][1] is not None
                names = [{"drums": "鼓", "bass": "808" if use808 else lineup["bass"][0]}.get(a) or lineup[a][0] for a in GAINS]
                if names != sent_names:  # 樂器名稱給視覺：靜音提示（i|，GAINS 的順序）和光柱名稱（l|，按 S 顯示）
                    sent_names = names
                    link.send("i|" + "|".join(names))
                    link.send(f"l|鼓 · {drummer.kit['name']}|{names[2]}|你|{lineup['keys'][0]}|"
                              f"{lineup['rhythm'][0]}＋{lineup['lead'][0]}")
                rows.append(band.frame_row([agent_conditioning(a, frame + k, root, mode, tok,
                                                               (sounding, onsets if k == 0 else set()), degree)
                                            for a in AGENTS]))

            # ---- 生成、混音 ----
            t0 = time.perf_counter()
            audio, state = band.generate(rows, state)
            gen_ms.append((time.perf_counter() - t0) * 1000 / FRAMES_PER_CALL)
            if gen_ms[-1] > 60:  # 一批特別慢（會斷音）：印出來，方便對照當時發生什麼事
                print(f"⚠ 第 {frame * 0.04:.1f} 秒這一批生成很慢：每幀 {gen_ms[-1]:.0f} ms", flush=True)
            level = {a: min(2.0, max(0.5, LEVEL / lineup[a][2])) for a in AGENTS}  # 每種樂器拉到差不多大聲
            for k in range(FRAMES_PER_CALL):
                goal = lambda a: 0.0 if a in muted else (1.0 if sounding else target[a])  # 手動靜音優先；人類彈琴時全員回來
                active = {a: g + (goal(a) - g) * 0.04 for a, g in active.items()}  # 約一小節淡入淡出
                stems = {a: audio[k, i] * active[a] * level[a] for i, a in enumerate(AGENTS)}
                stems["drums"] = drums[k][0] * active["drums"]
                if drums[k][1] is not None:
                    stems["bass"] = drums[k][1] * active["bass"]  # 這種風格的貝斯是 808
                if rec:
                    rec.write(np.concatenate(list(stems.values()), axis=1))
                mix = MASTER * sum(GAINS[a] * pan[a] * s for a, s in stems.items())
                mix = np.ascontiguousarray(limit(mix), dtype=np.float32)
                beat = (frame + k) * BPM / tm.FRAMES_PER_MIN
                info = f"f {levels(stems, [h * active['drums'] for h in drums[k][2]])} {beat:.2f} {scenes[k]}"
                while not stop.is_set():
                    try:
                        audio_q.put((mix, info), timeout=0.5)
                        break
                    except queue.Full:
                        continue
            frame += FRAMES_PER_CALL
        if rec:
            rec.close()

    def player() -> None:
        try:
            play()
        except Exception as e:  # 音效伺服器斷了：說清楚再結束，而不是無聲地卡著
            print(f"⚠ 聲音送不出去（{e}）：請在 Windows 命令列執行 wsl --shutdown，再重開 run_band.bat", flush=True)
            link.send("t|聲音送不出去|請執行 wsl --shutdown 後重開 run_band.bat|600")
            stop.set()

    def play() -> None:
        while not stop.is_set() and not audio_q.full():  # 存滿再開始播
            stop.wait(0.01)
        # 音效緩衝設為 2 幀；預設值約 1 秒，延遲太大
        with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                               2, 48000, app_name="ai_duet_band",
                               tlength=tm.FRAME_BYTES * 2, prebuf=tm.FRAME_BYTES * 2,
                               minreq=tm.FRAME_BYTES) as pa:
            first, silence = True, np.zeros((1920, 2), np.float32)
            while not stop.is_set():
                if audio_q.empty():  # 生成跟不上，會斷音
                    dropouts[0] += 1
                try:
                    mix, info = audio_q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if paused.is_set():  # 暫停：樂團在背後照常生成、跟著拍子走，只是不出聲；GPU 不會降頻，繼續時馬上接得上
                    pa.write(silence.tobytes())
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

    # ---- 開始；終端機按 Enter 或視覺視窗按 Q 結束 ----
    threads = [threading.Thread(target=generator), threading.Thread(target=player),
               threading.Thread(target=tm.listen_midi, args=(human, stop))]
    for t in threads:
        t.start()
    bridge = start_windows("band_mrt2.py", "--midi-bridge")
    print("按 Enter（或在視覺視窗按 Q）結束", flush=True)

    def wait_enter() -> None:
        try:
            input()
        except EOFError:  # 沒有終端機（測試時），就只能從視覺視窗結束
            return
        stop.set()

    threading.Thread(target=wait_enter, daemon=True).start()
    stop.wait()
    for p in (bridge, visual, groove):
        if p:
            p.terminate()
    for t in threads:  # 等生成結束再離開，否則 JAX 會當掉
        t.join()
    if gen_ms:
        print(f"生成每幀中位數 {np.median(gen_ms):.1f} ms、95% 在 {np.percentile(gen_ms, 95):.1f} ms 內、最慢 {max(gen_ms):.0f} ms"
              f"（預算 40 ms）", flush=True)
    print(f"斷音次數：{dropouts[0]}", flush=True)
