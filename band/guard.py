"""監聽：另一個低優先權的行程用 MusicCoCa 聽某個 AI 樂器最近 10 秒的聲音，判斷像不像鼓聲或人聲。
提示詞加了 instrumental、「不要鼓」也調強了，MRT2 偶爾還是會讓樂器像在唱歌或打鼓；聽到了，樂團就把那個樂器關掉、下一段換別的。
聽一段約 2 秒 CPU、不用 GPU；用 nice 19 跑，不搶生成的 CPU。主程式用 Guard，這個檔案直接執行時是子行程（serve）。"""
import os
import queue
import subprocess
import sys
import threading

import numpy as np

LABELS = ["drums and percussion", "drum kit", "hi-hat and snare",            # 前 3 個 = 鼓聲
          "singing voice", "choir vocals", "human voice humming",            # 再 3 個 = 人聲
          "piano", "electric piano", "electric guitar", "acoustic guitar", "harp", "bass guitar", "strings", "cello",
          "bells", "music box", "vibraphone", "flute", "synthesizer pad", "organ"]


def serve() -> None:
    """子行程：讀「編號 樣本數」一行 + 16 kHz float32 單聲道 → 回「編號 鼓聲機率 人聲機率」。"""
    from magenta_rt import audio as au
    from magenta_rt import musiccoca

    model = musiccoca.MusicCoCa()
    text = np.stack([np.asarray(model.embed(t), np.float32).ravel() for t in LABELS])
    text /= np.linalg.norm(text, axis=1, keepdims=True)
    model.embed(au.Waveform(np.zeros((160000, 1), np.float32), 16000))  # 聲音那半邊第一次用要載入約 20 秒，先做掉
    print("ready", flush=True)
    inp = sys.stdin.buffer
    while head := inp.readline():
        tag, n = head.split()
        x = np.frombuffer(inp.read(4 * int(n)), np.float32)
        e = np.asarray(model.embed(au.Waveform(x[:, None].copy(), 16000)), np.float32).ravel()
        p = np.exp(30 * text @ (e / np.linalg.norm(e)))  # 零樣本分類：聲音和每個標籤文字的相似度 → 機率
        p /= p.sum()
        print(f"{tag.decode()} {p[:3].sum():.3f} {p[3:6].sum():.3f}", flush=True)


class Guard:
    """主程式這邊：check() 送一段聲音去聽（一次只聽一段，忙的時候回傳 False），結果放進 results：(編號, 鼓聲, 人聲)。"""

    def __init__(self, root: str) -> None:
        self.results: "queue.Queue[tuple]" = queue.Queue()
        self.ready, self.busy = threading.Event(), threading.Event()
        self.proc = subprocess.Popen(["nice", "-n", "19", sys.executable, "-W", "ignore", "-m", "band.guard"], cwd=root,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            parts = line.decode().split()
            if parts == ["ready"]:
                self.ready.set()
            elif len(parts) == 3:
                self.results.put((int(parts[0]), float(parts[1]), float(parts[2])))
                self.busy.clear()

    def check(self, tag: int, audio: np.ndarray) -> bool:
        """audio：48 kHz 單聲道。送出去就回傳 True；在背景寫入，不卡住生成。"""
        if not self.ready.is_set() or self.busy.is_set() or self.proc.poll() is not None:
            return False
        self.busy.set()

        def send() -> None:
            from scipy.signal import resample_poly
            x = resample_poly(audio, 1, 3).astype(np.float32)  # 48 kHz → 16 kHz
            try:
                self.proc.stdin.write(f"{tag} {len(x)}\n".encode() + x.tobytes())
                self.proc.stdin.flush()
            except OSError:
                pass
        threading.Thread(target=send, daemon=True).start()
        return True

    def close(self) -> None:
        self.proc.terminate()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    serve()
