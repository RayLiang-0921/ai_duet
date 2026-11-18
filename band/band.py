"""主流程：載入模型 → 等觀眾彈和弦 → 生成執行緒（和弦進行、風格、編制、鼓）＋ 播放執行緒 → 結束統計。"""
import os
import queue
import random
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
from band.mrt2 import BatchedMRT2, style_tokens
from band.settings import (AGENTS, BLENDS, BPM, CHORD_SETTLE, DROP_P, FRAMES_PER_CALL, GAINS, GROOVE_PYTHON, MASTER,
                           NEXT_CHORD, PAN, PHRASE_BARS, QUEUE_FRAMES, REJOIN_P, ROOT, STYLES, TEMPERATURE, VISUAL_TCP)


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
    styles = list(STYLES)
    mix_gains = {a: GAINS[a] * pan_gains(PAN[a]) for a in GAINS}  # 每個樂器的 [左, 右] 音量
    stop, paused = threading.Event(), threading.Event()
    link = VisualLink(stop)
    visual = None if "--no-visual" in sys.argv else start_windows("vj_tk.py")  # 視覺先開，載入模型時就看得到
    groove_python, groove = os.path.expanduser(GROOVE_PYTHON), None
    if os.path.exists(groove_python):
        groove = subprocess.Popen([groove_python, os.path.join(ROOT, "groove_server.py")], stdin=subprocess.DEVNULL)
    else:
        print(f"[鼓手] 找不到 {GROOVE_PYTHON}，鼓只用鼓機的 pattern", flush=True)
    mrt = tm.MagentaRT2Jax(size="mrt2_small", temperature=TEMPERATURE)
    band = BatchedMRT2(mrt, len(AGENTS))
    tokens = style_tokens(mrt)  # 兩種風格和中間的混合都先準備好，切換時不用等

    # 先編譯，人類開始彈時就不用等。要算兩次：第二次的狀態來自第一次的輸出，型別略有不同，JAX 會再編譯一次（約 5 秒）
    rows = [band.frame_row([agent_conditioning(a, f, 0, "major", tokens[styles[0]], (set(), set()))
                            for a in AGENTS]) for f in range(FRAMES_PER_CALL)]
    _, state = band.generate(rows, band.new_state(0))
    band.generate(rows, state)
    drummer = Drummer(int(time.time()))
    drummer.prepare(0)  # 第一段先排好，開始時不用等

    human = Human(link)
    audio_q: "queue.Queue[tuple]" = queue.Queue(maxsize=QUEUE_FRAMES)  # (混音, 給視覺的那一幀資料)
    dropouts, gen_ms = [0], []
    instruments = list(GAINS)  # m1–m6 對應的樂器
    muted = set()  # 觀眾手動靜音的樂器

    def commands() -> list:
        """視覺視窗送來的指令：q 結束、p 暫停/繼續、m<n> 切靜音，這三個在這裡處理；其他（s = 切換風格）回傳給呼叫的人。"""
        out = []
        while not link.commands.empty():
            c = link.commands.get()
            if c == "q":
                stop.set()
            elif c == "p":
                paused.clear() if paused.is_set() else paused.set()
                print("⏸ 暫停" if paused.is_set() else "▶ 繼續", flush=True)
            elif c.startswith("m") and c[1:].isdigit() and 1 <= int(c[1:]) <= len(instruments):
                a = instruments[int(c[1:]) - 1]
                muted.symmetric_difference_update({a})
                print(f"🔇 {a} 靜音" if a in muted else f"🔊 {a} 回來", flush=True)
            else:
                out.append(c)
        return out

    def wait_for_chord(style: str) -> tuple:
        """等觀眾彈第一個和弦；期間按空白鍵可以換起始風格。回傳 (起始風格, 和弦) 或 (風格, None) = 被結束。"""
        print(f"🎹 請彈一個和弦開始（需要根音和三度音）；在視覺視窗按空白鍵選風格，目前：{style}", flush=True)
        link.send(f"w|{style}")
        key = None
        while not stop.is_set() and key is None:
            if "s" in commands():
                style = styles[(styles.index(style) + 1) % len(styles)]
                print(f"🔀 起始風格：{style}", flush=True)
                link.send(f"w|{style}")
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
        mode = STYLES[style].get("mode", played_mode)  # memphis 一律小調
        key_name = f"{tm.NOTE_NAMES[root]} {mode}"
        print(f"🎵 {key_name}，{style}，開始！", flush=True)
        link.send(f"k|{key_name}|{style}")

        state, frame, pending = band.new_state(int(time.time()) % 10000 * 10), 0, None
        rng, degree, human_degree = random.Random(), 0, None
        steps = tm.SCALES[mode]
        target = {a: 1.0 for a in GAINS}  # 編制：每個樂器 1 = 在場、0 = 退出；active 是淡入淡出中的實際音量
        active = dict(target)
        rec = None  # --record 檔名：各樂器分軌存成多聲道 wav（測試用）
        if "--record" in sys.argv:
            rec = sf.SoundFile(sys.argv[sys.argv.index("--record") + 1], "w", 48000, 2 * (len(AGENTS) + 1), "FLOAT")

        while not stop.is_set():
            # ---- 觀眾的輸入 ----
            if "s" in commands():
                pending = styles[(styles.index(pending or style) + 1) % len(styles)]
                print(f"🔀 下一段開頭切換到 {pending}", flush=True)
                link.send(f"p|{pending}")
            sounding, onsets = human.take()
            played = detect_chord(sounding) if len(sounding) >= 2 else None  # 彈和弦 → 和弦進行從那個和弦接下去
            if played and (played[0] - root) % 12 in steps:
                human_degree = steps.index((played[0] - root) % 12)

            # ---- 這一批幀的條件：和弦進行、風格、編制、鼓 ----
            rows, frame_styles, drums = [], [], []
            for k in range(FRAMES_PER_CALL):
                if tm.starts(frame + k, BPM / 4):  # 小節開頭：決定這小節的和弦；段落開頭才換風格、改編制，拍子不中斷
                    bar = tm.count(frame + k, BPM / 4)
                    if pending and bar % PHRASE_BARS == 0:
                        style, pending, degree = pending, None, 0
                        mode = STYLES[style].get("mode", played_mode)
                        steps = tm.SCALES[mode]
                    elif human_degree is not None:
                        degree, human_degree = human_degree, None
                    elif bar > 0:
                        degree = rng.choice(NEXT_CHORD[mode].get(degree, [0]))
                    if bar % PHRASE_BARS == PHRASE_BARS - 1:  # 編制：下一段誰在場，這一小節慢慢淡入淡出過去；至少留 3 個
                        target = {a: float(rng.random() >= (DROP_P if on else 1 - REJOIN_P)) for a, on in target.items()}
                        for a in rng.sample(list(target), 3 - int(sum(target.values()))) if sum(target.values()) < 3 else ():
                            target[a] = 1.0
                        print("🎚 下一段：" + "、".join(a for a, on in target.items() if on), flush=True)
                frame_styles.append(styles.index(style))
                bar_beat = tm.count(frame + k, BPM) % 4
                blend = pending and bar % PHRASE_BARS == PHRASE_BARS - 1 and bar_beat > 0  # 切換前最後一小節逐拍混合
                tok = tokens[(style, pending, BLENDS[bar_beat - 1])] if blend else tokens[style]
                root_pc = min(sounding) % 12 if sounding else (root + steps[degree % 7]) % 12
                drums.append(drummer.frame(frame + k, style, root_pc))
                rows.append(band.frame_row([agent_conditioning(a, frame + k, root, mode, tok,
                                                               (sounding, onsets if k == 0 else set()), degree)
                                            for a in AGENTS]))

            # ---- 生成、混音 ----
            t0 = time.perf_counter()
            audio, state = band.generate(rows, state)
            gen_ms.append((time.perf_counter() - t0) * 1000 / FRAMES_PER_CALL)
            for k in range(FRAMES_PER_CALL):
                goal = lambda a: 0.0 if a in muted else (1.0 if sounding else target[a])  # 手動靜音優先；人類彈琴時全員回來
                active = {a: g + (goal(a) - g) * 0.04 for a, g in active.items()}  # 約一小節淡入淡出
                stems = {a: audio[k, i] * active[a] for i, a in enumerate(AGENTS)} | {"drums": drums[k][0] * active["drums"]}
                if drums[k][1] is not None:
                    stems["bass"] = drums[k][1] * active["bass"]  # 這種風格的貝斯是 808
                if rec:
                    rec.write(np.concatenate(list(stems.values()), axis=1))
                mix = MASTER * sum(mix_gains[a] * s for a, s in stems.items())
                mix = np.ascontiguousarray(limit(mix), dtype=np.float32)
                beat = (frame + k) * BPM / tm.FRAMES_PER_MIN
                info = f"f {levels(stems)} {beat:.2f} {frame_styles[k]}"
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
