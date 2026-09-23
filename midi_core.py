"""
midi_core.py — 多軌 MIDI 路由系統 + 人類亂入攔截機制

架構說明：
  - AI_Melody_Out / AI_Harmony_Out / Preset_Bass_Out  → 三個虛擬輸出軌
  - Human_In                                          → 模擬人類輸入
  - Preset Bass 以獨立 asyncio Task 不斷循環播放
  - Human_In 一有音符，立刻觸發 on_human_interrupt()，
    清空 AI 待發音符 Buffer，並將人類音符寫入 Human Buffer

Windows 注意：mido 虛擬埠在 Windows rtmidi 後端不支援，
此腳本以「軟路由＋佇列模擬」取代實體虛擬埠，
確保邏輯在無 DAW/loopMIDI 環境下也能完整跑通。
"""

import asyncio
import threading
import time
import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

# ──────────────────────────────────────────
# Logging 設定
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("midi_core")


# ──────────────────────────────────────────
# 資料型別
# ──────────────────────────────────────────
@dataclass
class MidiNote:
    channel: int
    pitch: int
    velocity: int
    source: str  # 'AI_Melody' | 'AI_Harmony' | 'Preset_Bass' | 'Human'

    def note_name(self) -> str:
        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (self.pitch // 12) - 1
        return f"{names[self.pitch % 12]}{octave}"

    def __str__(self):
        return (
            f"[{self.source}] ch={self.channel} "
            f"note={self.note_name()}({self.pitch}) vel={self.velocity}"
        )


# ──────────────────────────────────────────
# 全域緩衝區 (Global Buffer)
# ──────────────────────────────────────────
class MidiBuffer:
    """執行緒安全的 MIDI 音符佇列"""

    def __init__(self, name: str, maxlen: int = 64):
        self.name = name
        self._lock = threading.Lock()
        self._buf: deque[MidiNote] = deque(maxlen=maxlen)

    def push(self, note: MidiNote) -> None:
        with self._lock:
            self._buf.append(note)
        log.debug("Buffer[%s] push → %s", self.name, note)

    def pop(self) -> Optional[MidiNote]:
        with self._lock:
            return self._buf.popleft() if self._buf else None

    def clear(self) -> int:
        with self._lock:
            count = len(self._buf)
            self._buf.clear()
        if count:
            log.info("Buffer[%s] 已清空 %d 個待發音符", self.name, count)
        return count

    def __len__(self):
        with self._lock:
            return len(self._buf)


# 全域實例
ai_pending_buffer = MidiBuffer("AI_Pending")    # AI 預計要發送的下一個音符
human_buffer = MidiBuffer("Human_Input")        # 人類彈奏的音符


# ──────────────────────────────────────────
# 虛擬埠口 (軟路由模擬)
# ──────────────────────────────────────────
class VirtualMidiPort:
    """以 callback 模擬 MIDI Output 埠口（無需 DAW/loopMIDI）"""

    def __init__(self, name: str):
        self.name = name
        self._callbacks: list[Callable[[MidiNote], None]] = []
        log.info("VirtualMidiPort '%s' 已開啟", self.name)

    def send(self, note: MidiNote) -> None:
        log.info("  ▶ %s → SEND %s", self.name, note)
        for cb in self._callbacks:
            cb(note)

    def connect(self, callback: Callable[[MidiNote], None]) -> None:
        self._callbacks.append(callback)

    def close(self) -> None:
        log.info("VirtualMidiPort '%s' 已關閉", self.name)


# ──────────────────────────────────────────
# 系統狀態
# ──────────────────────────────────────────
class SystemState:
    AI_MODE = "AI_互彈模式"
    HUMAN_MODE = "人類亂入模式"

    def __init__(self):
        self._mode = self.AI_MODE
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        with self._lock:
            return self._mode

    def set_mode(self, mode: str) -> None:
        with self._lock:
            if self._mode != mode:
                self._mode = mode
                log.warning("★ 系統狀態切換 → %s", mode)


state = SystemState()


# ──────────────────────────────────────────
# 人類亂入攔截器
# ──────────────────────────────────────────
def on_human_interrupt(note: MidiNote) -> None:
    """
    人類亂入事件處理器。
    1. 切換系統狀態
    2. 強制清空 AI 待發 Buffer
    3. 將音符寫入 Human Buffer
    """
    log.warning("╔══════════════════════════════════╗")
    log.warning("║  ⚡ 人類亂入攔截！Human_In 偵測  ║")
    log.warning("╚══════════════════════════════════╝")
    log.warning("  觸發音符: %s", note)

    state.set_mode(SystemState.HUMAN_MODE)
    cleared = ai_pending_buffer.clear()
    log.warning("  已清空 AI 待發音符 %d 個", cleared)

    human_buffer.push(note)
    log.info("  人類音符已寫入 Human Buffer，共 %d 個", len(human_buffer))


# ──────────────────────────────────────────
# 埠口初始化
# ──────────────────────────────────────────
melody_out = VirtualMidiPort("AI_Melody_Out")
harmony_out = VirtualMidiPort("AI_Harmony_Out")
bass_out = VirtualMidiPort("Preset_Bass_Out")
human_in = VirtualMidiPort("Human_In")   # 以軟路由模擬實體輸入

# Human_In 送進來的音符自動路由至攔截器
def _human_in_router(note: MidiNote) -> None:
    on_human_interrupt(note)

human_in.connect(_human_in_router)


# ──────────────────────────────────────────
# Preset Bass 背景軌 (asyncio Task)
# ──────────────────────────────────────────
# C-G-F-G Bass line (MIDI pitch: C2=36, G2=43, F2=41)
BASS_LINE = [
    MidiNote(channel=2, pitch=36, velocity=80, source="Preset_Bass"),  # C2
    MidiNote(channel=2, pitch=43, velocity=80, source="Preset_Bass"),  # G2
    MidiNote(channel=2, pitch=41, velocity=75, source="Preset_Bass"),  # F2
    MidiNote(channel=2, pitch=43, velocity=80, source="Preset_Bass"),  # G2
]
BASS_BPM = 120
BASS_BEAT_SEC = 60.0 / BASS_BPM  # 0.5s per beat


async def preset_bass_task(stop_event: asyncio.Event) -> None:
    """不斷循環發送 C-G-F-G Bass line"""
    log.info("[Preset_Bass] 背景軌啟動，BPM=%d", BASS_BPM)
    idx = 0
    while not stop_event.is_set():
        note = BASS_LINE[idx % len(BASS_LINE)]
        bass_out.send(note)
        idx += 1
        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=BASS_BEAT_SEC,
            )
        except asyncio.TimeoutError:
            pass  # 正常：timeout 表示還沒停止，繼續循環
    log.info("[Preset_Bass] 背景軌已停止")


# ──────────────────────────────────────────
# AI 互彈任務 (asyncio Task)
# ──────────────────────────────────────────
C_MAJOR = [60, 62, 64, 65, 67, 69, 71, 72]
G_MIXO = [55, 57, 59, 60, 62, 64, 65, 67]


async def ai_duet_task(stop_event: asyncio.Event) -> None:
    """AI 互彈：Melody 與 Harmony 輪流往 Buffer 推音符，再從 Buffer 取出發送"""
    log.info("[AI_Duet] AI 互彈任務啟動")
    last_melody_pitch = random.choice(C_MAJOR)
    last_harmony_pitch = random.choice(G_MIXO)

    while not stop_event.is_set():
        if state.mode == SystemState.AI_MODE:
            # 產生 AI Melody 音符並推入 Buffer
            melody_pitch = random.choice(C_MAJOR)
            melody_note = MidiNote(channel=0, pitch=melody_pitch, velocity=85, source="AI_Melody")
            ai_pending_buffer.push(melody_note)

            # 取出並發送（若沒被清空）
            pending = ai_pending_buffer.pop()
            if pending:
                melody_out.send(pending)
                last_melody_pitch = pending.pitch

            # 稍後 Harmony 跟進
            await asyncio.sleep(0.25)

            if state.mode == SystemState.AI_MODE:
                harmony_pitch = random.choice(G_MIXO)
                harmony_note = MidiNote(channel=1, pitch=harmony_pitch, velocity=70, source="AI_Harmony")
                ai_pending_buffer.push(harmony_note)

                pending = ai_pending_buffer.pop()
                if pending:
                    harmony_out.send(pending)
                    last_harmony_pitch = pending.pitch

        elif state.mode == SystemState.HUMAN_MODE:
            # 人類模式：消化 Human Buffer，AI 暫停發音
            hn = human_buffer.pop()
            if hn:
                log.info("  [AI 聆聽] 消化人類音符: %s", hn)
            else:
                # Buffer 清空 → 切回 AI 模式
                log.warning("  Human Buffer 已清空，系統回到 AI 互彈模式")
                state.set_mode(SystemState.AI_MODE)

        try:
            await asyncio.wait_for(
                asyncio.shield(stop_event.wait()),
                timeout=0.5,
            )
        except asyncio.TimeoutError:
            pass

    log.info("[AI_Duet] AI 互彈任務已停止")


# ──────────────────────────────────────────
# 人類亂入模擬器 (在獨立 thread 中執行)
# ──────────────────────────────────────────
def simulate_human_input(
    loop: asyncio.AbstractEventLoop,
    stop_event: threading.Event,
    delay_sec: float = 3.0,
    note_count: int = 4,
) -> None:
    """
    模擬人類在 delay_sec 秒後亂入，連彈 note_count 個音符。
    真實場景中，這裡替換為 mido.open_input() 的 callback。
    """
    log.info("[Simulator] 模擬人類將在 %.1f 秒後亂入...", delay_sec)
    time.sleep(delay_sec)

    if stop_event.is_set():
        return

    human_notes = [
        MidiNote(channel=3, pitch=p, velocity=100, source="Human")
        for p in [64, 67, 69, 72][:note_count]
    ]
    log.warning(">>> 模擬人類彈下第一個音符！<<<")
    for i, note in enumerate(human_notes):
        if stop_event.is_set():
            break
        # 透過 asyncio loop 安全地呼叫（因為 human_in.send 是同步的）
        loop.call_soon_threadsafe(human_in.send, note)
        time.sleep(0.4)


# ──────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────
async def main() -> None:
    log.info("=" * 50)
    log.info("  AI Duet MIDI Core 啟動")
    log.info("  埠口: AI_Melody_Out / AI_Harmony_Out / Preset_Bass_Out / Human_In")
    log.info("=" * 50)

    asyncio_stop = asyncio.Event()
    thread_stop = threading.Event()
    loop = asyncio.get_running_loop()

    # 啟動模擬人類亂入的 Thread
    human_thread = threading.Thread(
        target=simulate_human_input,
        args=(loop, thread_stop),
        kwargs={"delay_sec": 3.0, "note_count": 4},
        daemon=True,
        name="HumanSimulator",
    )

    bass_task: Optional[asyncio.Task[None]] = None
    duet_task: Optional[asyncio.Task[None]] = None

    try:
        human_thread.start()

        # 同時啟動 Preset Bass 與 AI 互彈兩個非同步任務
        bass_task = asyncio.create_task(preset_bass_task(asyncio_stop), name="preset_bass")
        duet_task = asyncio.create_task(ai_duet_task(asyncio_stop), name="ai_duet")

        # 跑 8 秒後自動停止（示範用）
        log.info("[Main] 系統運行 8 秒，接著自動停止...")
        await asyncio.sleep(8.0)

    except KeyboardInterrupt:
        log.warning("[Main] 收到 KeyboardInterrupt，正在關閉...")

    finally:
        log.info("[Main] 發送停止信號...")
        asyncio_stop.set()
        thread_stop.set()

        tasks_to_await = [t for t in (bass_task, duet_task) if t is not None]
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)

        melody_out.close()
        harmony_out.close()
        bass_out.close()
        human_in.close()

        log.info("=" * 50)
        log.info("  Human Buffer 最終內容 (%d 個音符):", len(human_buffer))
        while (n := human_buffer.pop()):
            log.info("    %s", n)
        log.info("  AI Duet MIDI Core 已完全關閉")
        log.info("=" * 50)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception("未預期錯誤: %s", e)
        raise
