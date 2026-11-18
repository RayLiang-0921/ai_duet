"""對外連線：送資料給視覺（並接收選單和按鍵的指令）、觀眾的音符、Windows 端的 MIDI 橋接與啟動 Windows 程式。"""
import os
import queue
import socket
import subprocess
import sys
import threading
import time

import test_mrt2 as tm
from band.settings import MIDI_PORT, PIANO_PORT, ROOT, VISUAL_TCP


class VisualLink:
    """把每幀的樂器音量、拍子、風格、觀眾的音送給 Windows 端的 vj_tk.py；沒有接上就直接丟掉。
    視覺視窗的按鍵會送回一行指令，放進 commands：s = 切換風格、m1–m6 = 切某個樂器靜音、p = 暫停/繼續、q = 結束。"""

    STICKY = ("y|", "w|", "k|", "c|", "i|", "l|")  # 狀態訊息：新連上的視覺（例如視窗重開）會先收到每種最新的一行

    def __init__(self, stop: threading.Event) -> None:
        self.clients, self.stop, self.lock, self.state = [], stop, threading.Lock(), {}
        self.commands: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(VISUAL_TCP)
        server.listen()
        server.settimeout(0.5)
        while not self.stop.is_set():
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            conn.settimeout(0.05)
            with self.lock:
                try:
                    conn.sendall("".join(line + "\n" for line in self.state.values()).encode())
                except OSError:
                    continue
                self.clients.append(conn)
            threading.Thread(target=self._read, args=(conn,), daemon=True).start()
        server.close()

    def _read(self, conn: socket.socket) -> None:
        buf = b""
        while not self.stop.is_set():
            try:
                data = conn.recv(64)
            except socket.timeout:
                continue
            except OSError:
                return
            if not data:
                return
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.commands.put(line.decode().strip())

    def send(self, line: str) -> None:
        with self.lock:  # 播放執行緒和 MIDI 執行緒都會送
            if line[:2] in self.STICKY:
                self.state[line[:2]] = line
            for c in list(self.clients):
                try:
                    c.sendall((line + "\n").encode())
                except socket.timeout:
                    pass  # 視覺來不及收，丟掉這一行
                except OSError:
                    self.clients.remove(c)
                    c.close()


class Human(tm.HumanNotes):
    """觀眾的音符；每個按鍵也立刻送給視覺。"""

    def __init__(self, link: VisualLink) -> None:
        super().__init__()
        self.link = link

    def press(self, note: int) -> None:
        super().press(note)
        self.link.send(f"n 1 {note}")

    def release(self, note: int) -> None:
        super().release(note)
        self.link.send(f"n 0 {note}")

    def held(self) -> set:
        with self._lock:
            return set(self._held)


def midi_bridge() -> None:
    """（Windows 端）讀 MIDI 鍵盤：音符送給 WSL 的樂團，同時用 Windows 的合成器發鋼琴聲，彈的人立刻聽得到。"""
    import mido

    sys.stdout.reconfigure(encoding="utf-8")  # 避免中文在 WSL 主控台變亂碼
    names = mido.get_input_names()
    if not names:
        print("[MIDI] 沒有偵測到 MIDI 鍵盤；音樂照常播放，接上後請重新啟動", flush=True)
        return
    name = MIDI_PORT if MIDI_PORT in names else names[0]
    for _ in range(100):  # 等 WSL 端開始接收
        try:
            sock = socket.create_connection(tm.MIDI_TCP, timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    else:
        print("[MIDI] 無法連到生成程式", flush=True)
        return
    piano = next((mido.open_output(n) for n in mido.get_output_names() if PIANO_PORT and n.startswith(PIANO_PORT)), None)
    if piano:
        piano.send(mido.Message("program_change", program=0))  # 平台鋼琴
    print(f"[MIDI] 已連接鍵盤：{name}" + (f"，鋼琴聲：{PIANO_PORT}" if piano else ""), flush=True)
    with mido.open_input(name) as port:
        for msg in port:
            if msg.type not in ("note_on", "note_off"):
                continue
            down = int(msg.type == "note_on" and msg.velocity > 0)
            sock.sendall(f"{down} {msg.note}\n".encode())
            if piano:
                piano.send(msg)


def start_windows(script: str, *args: str) -> "subprocess.Popen | None":
    """在 Windows 端用專案的 venv 執行 script：WSL 讀不到 USB MIDI（--midi-bridge），視窗也在 Windows 開（vj_tk.py）。"""
    win_python = os.path.join(ROOT, "venv", "Scripts", "python.exe")
    if not os.path.exists(win_python):
        print(f"[{script}] 找不到 Windows venv，略過", flush=True)
        return None
    win_script = subprocess.run(["wslpath", "-w", os.path.join(ROOT, script)],
                                capture_output=True, text=True).stdout.strip()
    return subprocess.Popen([win_python, win_script, *args], stdin=subprocess.DEVNULL)  # 不讓它吃掉這邊的 Enter
