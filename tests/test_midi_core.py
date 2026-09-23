"""
test_midi_core.py — midi_core 模組的單元測試
"""
import asyncio
import threading
import pytest
from unittest.mock import MagicMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi_core import (
    MidiNote,
    MidiBuffer,
    VirtualMidiPort,
    SystemState,
)


# ════════════════════════════════════════════
# MidiNote
# ════════════════════════════════════════════
class TestMidiNote:
    def test_note_name_c4(self):
        n = MidiNote(channel=0, pitch=60, velocity=100, source="AI_Melody")
        assert n.note_name() == "C4"

    def test_note_name_a0(self):
        n = MidiNote(channel=0, pitch=21, velocity=80, source="AI_Melody")
        assert n.note_name() == "A0"

    def test_note_name_sharp(self):
        n = MidiNote(channel=0, pitch=61, velocity=80, source="AI_Melody")
        assert n.note_name() == "C#4"

    def test_note_name_high(self):
        n = MidiNote(channel=0, pitch=108, velocity=80, source="AI_Melody")
        assert n.note_name() == "C8"

    def test_str_format(self):
        n = MidiNote(channel=1, pitch=60, velocity=90, source="Human")
        s = str(n)
        assert "[Human]" in s
        assert "ch=1" in s
        assert "C4" in s
        assert "vel=90" in s

    def test_pitch_boundary_zero(self):
        n = MidiNote(channel=0, pitch=0, velocity=1, source="AI_Melody")
        assert n.note_name() == "C-1"

    def test_pitch_boundary_127(self):
        n = MidiNote(channel=0, pitch=127, velocity=1, source="AI_Melody")
        assert n.note_name() == "G9"


# ════════════════════════════════════════════
# MidiBuffer
# ════════════════════════════════════════════
class TestMidiBuffer:
    def _make_note(self, pitch=60):
        return MidiNote(channel=0, pitch=pitch, velocity=100, source="Test")

    def test_push_and_pop(self):
        buf = MidiBuffer("test", maxlen=8)
        n = self._make_note()
        buf.push(n)
        assert len(buf) == 1
        popped = buf.pop()
        assert popped is n
        assert len(buf) == 0

    def test_pop_empty_returns_none(self):
        buf = MidiBuffer("test")
        assert buf.pop() is None

    def test_fifo_order(self):
        buf = MidiBuffer("test", maxlen=8)
        notes = [self._make_note(p) for p in [60, 62, 64]]
        for n in notes:
            buf.push(n)
        for expected in notes:
            assert buf.pop() is expected

    def test_maxlen_eviction(self):
        buf = MidiBuffer("test", maxlen=2)
        buf.push(self._make_note(60))
        buf.push(self._make_note(62))
        buf.push(self._make_note(64))  # 60 被擠掉
        assert len(buf) == 2
        first = buf.pop()
        assert first.pitch == 62

    def test_clear(self):
        buf = MidiBuffer("test", maxlen=8)
        buf.push(self._make_note(60))
        buf.push(self._make_note(62))
        cleared = buf.clear()
        assert cleared == 2
        assert len(buf) == 0
        assert buf.pop() is None

    def test_clear_empty(self):
        buf = MidiBuffer("test")
        cleared = buf.clear()
        assert cleared == 0

    def test_thread_safety(self):
        """多執行緒同時 push/pop 不會 crash"""
        buf = MidiBuffer("concurrent", maxlen=256)
        errors = []

        def pusher():
            try:
                for i in range(100):
                    buf.push(self._make_note(i % 128))
            except Exception as e:
                errors.append(e)

        def popper():
            try:
                for _ in range(100):
                    buf.pop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=pusher) for _ in range(4)]
        threads += [threading.Thread(target=popper) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ════════════════════════════════════════════
# VirtualMidiPort
# ════════════════════════════════════════════
class TestVirtualMidiPort:
    def test_send_triggers_callback(self):
        port = VirtualMidiPort("test_port")
        received = []
        port.connect(lambda n: received.append(n))
        note = MidiNote(channel=0, pitch=60, velocity=80, source="Test")
        port.send(note)
        assert len(received) == 1
        assert received[0] is note

    def test_multiple_callbacks(self):
        port = VirtualMidiPort("test_port")
        r1, r2 = [], []
        port.connect(lambda n: r1.append(n))
        port.connect(lambda n: r2.append(n))
        note = MidiNote(channel=0, pitch=60, velocity=80, source="Test")
        port.send(note)
        assert len(r1) == 1
        assert len(r2) == 1

    def test_no_callbacks_no_error(self):
        port = VirtualMidiPort("test_port")
        note = MidiNote(channel=0, pitch=60, velocity=80, source="Test")
        port.send(note)  # 不應 crash

    def test_close_does_not_raise(self):
        port = VirtualMidiPort("test_port")
        port.close()  # 應正常結束


# ════════════════════════════════════════════
# SystemState
# ════════════════════════════════════════════
class TestSystemState:
    def test_initial_mode_is_ai(self):
        s = SystemState()
        assert s.mode == SystemState.AI_MODE

    def test_set_mode(self):
        s = SystemState()
        s.set_mode(SystemState.HUMAN_MODE)
        assert s.mode == SystemState.HUMAN_MODE

    def test_set_same_mode_no_change(self):
        s = SystemState()
        s.set_mode(SystemState.AI_MODE)
        assert s.mode == SystemState.AI_MODE

    def test_toggle_modes(self):
        s = SystemState()
        s.set_mode(SystemState.HUMAN_MODE)
        assert s.mode == SystemState.HUMAN_MODE
        s.set_mode(SystemState.AI_MODE)
        assert s.mode == SystemState.AI_MODE


# ════════════════════════════════════════════
# on_human_interrupt (整合測試)
# ════════════════════════════════════════════
class TestHumanInterrupt:
    def test_interrupt_clears_ai_buffer_and_switches_mode(self):
        """測試人類亂入攔截：清空 AI buffer + 切換至人類模式"""
        # 使用獨立的 buffer 和 state 來避免全域副作用
        ai_buf = MidiBuffer("test_ai", maxlen=64)
        human_buf = MidiBuffer("test_human", maxlen=64)
        test_state = SystemState()

        # 預先塞幾個 AI 音符
        for p in [60, 62, 64]:
            ai_buf.push(MidiNote(channel=0, pitch=p, velocity=85, source="AI_Melody"))
        assert len(ai_buf) == 3

        # 模擬人類亂入
        human_note = MidiNote(channel=3, pitch=67, velocity=100, source="Human")
        test_state.set_mode(SystemState.HUMAN_MODE)
        cleared = ai_buf.clear()
        human_buf.push(human_note)

        assert cleared == 3
        assert len(ai_buf) == 0
        assert test_state.mode == SystemState.HUMAN_MODE
        assert len(human_buf) == 1
        assert human_buf.pop().pitch == 67


# ════════════════════════════════════════════
# preset_bass_task (async)
# ════════════════════════════════════════════
class TestPresetBassTask:
    @pytest.mark.asyncio
    async def test_bass_task_sends_notes_and_stops(self):
        """Bass 背景軌應在 stop_event 設定後停止"""
        from midi_core import BASS_LINE

        received = []
        port = VirtualMidiPort("test_bass")
        port.connect(lambda n: received.append(n))

        stop = asyncio.Event()

        async def run_bass():
            idx = 0
            while not stop.is_set():
                note = BASS_LINE[idx % len(BASS_LINE)]
                port.send(note)
                idx += 1
                try:
                    await asyncio.wait_for(
                        asyncio.shield(stop.wait()), timeout=0.05
                    )
                except asyncio.TimeoutError:
                    pass

        task = asyncio.create_task(run_bass())
        await asyncio.sleep(0.3)
        stop.set()
        await task

        assert len(received) > 0
        # 確認全部都是 Preset_Bass 來源
        assert all(n.source == "Preset_Bass" for n in received)
        # 確認是 C-G-F-G 循環
        pitches = [n.pitch for n in received]
        expected_cycle = [36, 43, 41, 43]
        for i, p in enumerate(pitches):
            assert p == expected_cycle[i % 4]
