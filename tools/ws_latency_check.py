"""
連線到正在執行的 ai_duet_web.py，量測每小節封包是否能即時送達（驗證 GPU 是否跟得上播放）。

用法：  python tools/ws_latency_check.py [秒數=60] [url=ws://127.0.0.1:8000/ws]

判讀：
  - 每小節的「到達間隔」≈ measure_duration  → 生成速度跟得上播放（即時）
  - 到達間隔明顯大於 measure_duration     → 生成太慢，前端會出現空檔
  - melody/bass 的 audio 欄位非空          → 真的是 MRT2 生成的音訊，不是 fallback
"""
import asyncio
import base64
import io
import json
import wave
import sys
import time

import websockets


def audio_seconds(b64: str) -> float:
    if not b64:
        return 0.0
    with wave.open(io.BytesIO(base64.b64decode(b64))) as w:
        return w.getnframes() / w.getframerate()


async def main(seconds: float, url: str) -> None:
    async with websockets.connect(url, max_size=None) as ws:
        t0 = time.time()
        last = None
        gaps, rows, coverage = [], [], []
        sent_human = False
        while time.time() - t0 < seconds:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, seconds - (time.time() - t0)))
            except asyncio.TimeoutError:
                break
            msg = json.loads(raw)
            if msg.get("type") != "measure_bundle":
                continue
            now = time.time()
            dur = msg["measure_duration"]
            gap = None if last is None else now - last
            last = now
            mel, bass = msg["melody"], msg["bass"]
            mel_sec = audio_seconds(mel["audio_base64"])
            coverage.append(mel_sec / dur)
            rows.append((msg["measure_index"], now - t0, gap, dur, mel["agent"],
                         bool(mel["audio_base64"]), bool(bass["audio_base64"]),
                         len(mel["events"])))
            g = "  -  " if gap is None else f"{gap:5.2f}"
            print(f"小節 {msg['measure_index']:3d}  t={now - t0:6.2f}s  間隔={g}s  小節長={dur:.2f}s  "
                  f"{mel['agent']:8s} MRT2音訊: melody={'Y' if mel['audio_base64'] else 'N'} "
                  f"bass={'Y' if bass['audio_base64'] else 'N'}  旋律音訊={mel_sec:.2f}s  音符={len(mel['events'])}", flush=True)
            if gap is not None:
                gaps.append((gap, dur))
            # 過了一半時間模擬人類彈奏 C-E-G，驗證人類插入的路徑也能即時
            if not sent_human and now - t0 > seconds / 2:
                for midi in (60, 64, 67):
                    await ws.send(json.dumps({"action": "human_note", "midi": midi}))
                sent_human = True
                print("   >>> 送出人類音符 C4 E4 G4", flush=True)

    if not gaps:
        print("沒有收到足夠的小節可供計算")
        return
    # 前兩個間隔包含 JIT 暖機，另外統計穩態
    steady = gaps[2:] or gaps
    avg_gap = sum(g for g, _ in steady) / len(steady)
    avg_dur = sum(d for _, d in steady) / len(steady)
    late = sum(1 for g, d in steady if g > d * 1.1)
    first = rows[0][1]
    print("\n==== 結果 ====")
    print(f"第一小節到達時間（含暖機）: {first:.2f}s")
    print(f"穩態平均間隔: {avg_gap:.2f}s   小節長: {avg_dur:.2f}s   比值: {avg_gap / avg_dur:.2f}x (≤1.0 = 即時)")
    print(f"延遲超過 10% 的小節: {late}/{len(steady)}")
    print(f"旋律音訊覆蓋小節比例: 平均 {100 * sum(coverage) / len(coverage):.0f}%（100% = 整小節都有聲音）")
    print(f"MRT2 真實音訊小節: {sum(1 for r in rows if r[5])}/{len(rows)}")


if __name__ == "__main__":
    secs = float(sys.argv[1]) if len(sys.argv) > 1 else 60
    url = sys.argv[2] if len(sys.argv) > 2 else "ws://127.0.0.1:8000/ws"
    asyncio.run(main(secs, url))
