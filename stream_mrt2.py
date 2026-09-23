"""
stream_mrt2.py — MRT2 無限串流播放（邊生成邊播放）

一個執行緒持續呼叫 mrt.generate() 並把 streaming state 接續下去，另一個執行緒把
生成好的音訊依序送到喇叭。播放中可直接輸入新的風格文字，下一段就會切換過去。

在 WSL2 GPU 環境執行（需要 GPU 才跟得上即時播放）：
    雙擊 run_stream_gpu.bat，或在 WSL 內 bash run_stream_gpu.sh
"""
import queue
import threading

import numpy as np
import pasimple
from magenta_rt import MagentaRT2Jax

SAMPLE_RATE = 48000
CHUNK_FRAMES = 25          # 25 frames = 1 秒；越小切換風格反應越快
BUFFER_CHUNKS = 2          # 預先生成幾段（越大越不容易斷音，但風格切換延遲越久）

mrt = MagentaRT2Jax(size="mrt2_small")
style = mrt.embed_style("dj scratch, 140bpm, trap with piano sample, G minor")

audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=BUFFER_CHUNKS)
stop = threading.Event()


def generator() -> None:
    state = None
    while not stop.is_set():
        # 每次都要傳入「新的」state：舊的 state 在呼叫後會被 JAX 釋放
        wav, state = mrt.generate(
            conditioning={"mulan_tokens_25hz": style},
            frames=CHUNK_FRAMES,
            state=state,
        )
        audio_q.put(wav.samples)  # float32 [N, 2]；佇列滿時會等待


def player() -> None:
    with pasimple.PaSimple(pasimple.PA_STREAM_PLAYBACK, pasimple.PA_SAMPLE_FLOAT32LE,
                           2, SAMPLE_RATE, app_name="ai_duet") as pa:
        while not stop.is_set():
            try:
                samples = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            pa.write(np.ascontiguousarray(samples, dtype=np.float32).tobytes())
        pa.drain()


threading.Thread(target=generator, daemon=True).start()
threading.Thread(target=player, daemon=True).start()

print("播放中。輸入新的風格文字後按 Enter 即可切換（例如：funky slap bass, 100bpm），輸入 q 結束。")
while True:
    text = input("> ").strip()
    if text.lower() in ("q", "quit", "exit"):
        break
    if text:
        style = mrt.embed_style(text)
        print(f"   下一段切換為：{text}")
stop.set()
