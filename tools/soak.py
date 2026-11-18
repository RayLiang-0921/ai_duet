"""
長時間測試（整晚跑用）：假觀眾一直彈，樂團一直演，看幾個小時後會不會出問題。
假觀眾：隨機彈和弦或單音、每 1–3 分鐘從選單換風格、偶爾靜音某個樂器、偶爾暫停。
記錄：生成很慢的批次、監聽抓到的鼓聲 / 人聲（哪個風格的哪個樂器）、錯誤、記憶體，寫進 reports/soak-日期.md。
樂團當掉就記下錯誤、重開，繼續測。
用法（WSL GPU 環境、沒有樂團在跑時）：python tools/soak.py 結束時間（例如 07:00）
"""
import collections
import datetime
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}


def connect(port: int, tries: int = 600) -> socket.socket:
    for _ in range(tries):
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=5)
        except OSError:
            time.sleep(1)
    raise RuntimeError(f"連不上 {port}")


class Run:
    """一次樂團執行：開樂團、等它準備好、讀它的輸出。"""

    def __init__(self, log) -> None:
        self.lines, self.log = [], log
        self.proc = subprocess.Popen([sys.executable, "band_mrt2.py", "--no-visual"], cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.ready = threading.Event()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            line = line.rstrip()
            self.lines.append((time.time(), line))
            self.log.write(f"{datetime.datetime.now():%H:%M:%S} {line}\n")
            self.log.flush()
            if "請彈一個和弦" in line:
                self.ready.set()

    def rss_mb(self) -> float:
        out = subprocess.run(["ps", "-o", "rss=", "--ppid", str(self.proc.pid), "-p", str(self.proc.pid)],
                             capture_output=True, text=True).stdout.split()
        return sum(int(x) for x in out) / 1024


def play(run: Run, end: float, rng: random.Random, stats: collections.Counter) -> None:
    """假觀眾，直到 end 或樂團結束。"""
    midi, vis = connect(47600), connect(47601)
    vis.settimeout(None)
    styles = []

    def read_visual() -> None:  # 一定要一直讀，否則樂團送給視覺的資料會塞住
        buf = b""
        while True:
            try:
                data = vis.recv(65536)
            except OSError:
                return
            if not data:
                return
            buf += data
            for line in buf.split(b"\n")[:-1]:
                if line.startswith(b"y|"):
                    styles[:] = line[2:].decode().split("|")
            buf = buf.split(b"\n")[-1]
    threading.Thread(target=read_visual, daemon=True).start()
    send = lambda s, msg: s.sendall(msg.encode())
    send(midi, "0 0\n")
    time.sleep(1)
    root = rng.randrange(12)
    chord = lambda deg, mode: [48 + root + SCALES[mode][(deg + k) % 7] + 12 * ((deg + k) // 7) for k in (0, 2, 4)]
    notes = chord(0, "major")
    send(midi, "".join(f"1 {n}\n" for n in notes))
    time.sleep(1.5)
    send(midi, "".join(f"0 {n}\n" for n in notes))
    next_style, unmute = time.time() + rng.uniform(60, 180), {}
    while time.time() < end and run.proc.poll() is None:
        time.sleep(rng.uniform(4, 20))
        r = rng.random()
        if r < 0.45:  # 和弦
            notes = chord(rng.randrange(7), "major")
            send(midi, "".join(f"1 {n}\n" for n in notes))
            time.sleep(rng.uniform(0.5, 3))
            send(midi, "".join(f"0 {n}\n" for n in notes))
            stats["和弦"] += 1
        elif r < 0.65:  # 單音旋律
            for _ in range(rng.randint(3, 8)):
                n = 60 + root + rng.choice(SCALES["major"])
                send(midi, f"1 {n}\n")
                time.sleep(rng.uniform(0.15, 0.6))
                send(midi, f"0 {n}\n")
            stats["旋律"] += 1
        elif r < 0.7:  # 靜音一個樂器，20–60 秒後恢復
            k = rng.randint(1, 6)
            if k not in unmute:
                send(vis, f"m{k}\n")
                unmute[k] = time.time() + rng.uniform(20, 60)
                stats["靜音"] += 1
        elif r < 0.72:  # 暫停 5–15 秒
            send(vis, "p\n")
            time.sleep(rng.uniform(5, 15))
            send(vis, "p\n")
            stats["暫停"] += 1
        for k, t in list(unmute.items()):
            if time.time() > t:
                send(vis, f"m{k}\n")
                del unmute[k]
        if time.time() > next_style and styles:
            send(vis, f"g {rng.choice(styles)}\n")
            next_style = time.time() + rng.uniform(60, 180)
            stats["換風格"] += 1
    for k in unmute:
        send(vis, f"m{k}\n")
    if run.proc.poll() is None:
        send(vis, "q\n")


def main() -> None:
    hh, mm = map(int, (sys.argv[1] if len(sys.argv) > 1 else "07:00").split(":"))
    now = datetime.datetime.now()
    end_dt = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    end_dt += datetime.timedelta(days=1) if end_dt <= now else datetime.timedelta()
    end = end_dt.timestamp()
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    name = os.path.join(ROOT, "reports", f"soak-{datetime.date.today()}")
    log, rng, stats = open(name + ".log", "a", encoding="utf-8"), random.Random(), collections.Counter()
    runs, rss, t_start = [], [], time.time()
    print(f"長時間測試到 {end_dt:%H:%M}", flush=True)
    while time.time() < end - 300:
        run = Run(log)
        runs.append(run)
        if not run.ready.wait(900):
            run.proc.kill()
            stats["開不起來"] += 1
            continue
        player = threading.Thread(target=play, args=(run, end, rng, stats), daemon=True)
        player.start()
        while player.is_alive() and run.proc.poll() is None:
            player.join(600)
            if run.proc.poll() is None:
                rss.append((time.time(), run.rss_mb()))
            if any("Traceback" in l for _, l in run.lines[-200:]):  # 生成或播放當掉：記下來、重開
                stats["當掉重開"] += 1
                run.proc.kill()
                break
        try:
            run.proc.wait(60)
        except subprocess.TimeoutExpired:
            run.proc.kill()
        print(f"{datetime.datetime.now():%H:%M} 樂團結束（第 {len(runs)} 次）", flush=True)

    lines = [l for r in runs for _, l in r.lines]
    slow = [l for l in lines if "生成很慢" in l]
    guard = collections.Counter(re.sub(r"（[\d.]+）.*", "", l.split("🚫 ")[1]) for l in lines if "🚫" in l)
    errors = [l for l in lines if "Error" in l or "Traceback" in l or "⚠ 聲音送不出去" in l]
    drops = [l for l in lines if l.startswith("斷音次數")]
    speed = [l for l in lines if l.startswith("生成每幀")]
    hours = (time.time() - t_start) / 3600
    md = [f"# 長時間測試 {datetime.date.today()}（{hours:.1f} 小時，樂團開了 {len(runs)} 次）", "",
          "## 假觀眾做了什麼", "", "、".join(f"{k} {v} 次" for k, v in stats.items()) or "（沒有）", "",
          "## 穩定性", "", *[f"- {l}" for l in speed + drops],
          f"- 生成很慢的批次：{len(slow)} 次" + (f"（第一次：{slow[0]}）" if slow else ""),
          f"- 錯誤：{len(errors)} 行" + "".join(f"\n  - `{e}`" for e in errors[:20]),
          f"- 記憶體（MB）：" + ("、".join(f"{datetime.datetime.fromtimestamp(t):%H:%M} {m:.0f}" for t, m in rss) or "沒量到"), "",
          "## 監聽抓到的鼓聲 / 人聲（同一個樂器被抓越多次越該從樂器池拿掉）", "",
          *([f"- {k}：{v} 次" for k, v in guard.most_common()] or ["（沒有）"]), "",
          f"完整紀錄：{os.path.basename(name)}.log"]
    open(name + ".md", "w", encoding="utf-8").write("\n".join(md))
    print(f"報告：{name}.md", flush=True)


if __name__ == "__main__":
    main()
