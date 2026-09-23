"""
量測「每小節生成完整長度音訊」在 GPU 上是否跟得上播放。

模擬網頁版每小節的工作量：Bass + Melody 兩個 MRT2 串流並行，各生成 measure_sec 秒音訊。
比值 = 生成耗時 / 小節長度，≤ 1.0 代表能即時。

用法（WSL GPU venv）： python tools/gpu_measure_bench.py [measure_sec=2.0] [rounds=4]
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.disable(logging.INFO)

import jax  # noqa: E402

from duet_system import agent_a, agent_bass  # noqa: E402

measure_sec = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
rounds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
alloc = os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "bfc(default)")
prealloc = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "true(default)")
frac = os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75(default)")
print(f"backend={jax.default_backend()} allocator={alloc} preallocate={prealloc} mem_fraction={frac}", flush=True)

# 暖機（JIT）
agent_a.generate_audio(duration_sec=0.2)
agent_bass.generate_audio(duration_sec=0.2)

single = []
for _ in range(rounds):
    t = time.time()
    agent_a.generate_audio(duration_sec=measure_sec)
    single.append(time.time() - t)

pool = ThreadPoolExecutor(max_workers=2)
both = []
for _ in range(rounds):
    t = time.time()
    fa = pool.submit(agent_a.generate_audio, duration_sec=measure_sec)
    fb = pool.submit(agent_bass.generate_audio, duration_sec=measure_sec)
    fa.result(), fb.result()
    both.append(time.time() - t)

avg = lambda xs: sum(xs[1:]) / max(1, len(xs) - 1)  # 去掉第一輪
print(f"單一串流 {measure_sec:.1f}s 音訊: {avg(single):.2f}s  (比值 {avg(single) / measure_sec:.2f}x)")
print(f"Bass+Melody 並行各 {measure_sec:.1f}s: {avg(both):.2f}s  (比值 {avg(both) / measure_sec:.2f}x)")
