"""
test_mrt2_agent_streaming.py — MRT2 三大核心功能驗證測試套件
=============================================================

測試涵蓋:
  1. Direct Waveform Generation — 48kHz 立體聲 PCM 波形生成 + 數值健康度
  2. Notes-to-Continuation Agent Protocol — 音符 → 接續音訊 → Agent Payload
  3. Streaming State Continuity — 跨 chunk 狀態延續性與波形拼接連續性

支援模式:
  - Mock 模式 (CI/CD，無 magenta_rt): 用 unittest.mock 模擬 MRT2 底層
  - Real 模式 (本機有模型權重): 直接載入 mrt2_small 執行端到端驗證

用法:
  pytest tests/test_mrt2_agent_streaming.py -v
  python tests/test_mrt2_agent_streaming.py          # 也可直接執行
"""

from __future__ import annotations

import os
import struct
import sys
import time
import unittest
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

# 確保能 import 專案模組
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duet_system import (
    MRT2Generator,
    C_MAJOR_PITCHES,
    MRT2_FRAME_RATE,
    MRT2_NUM_RESERVED,
    MRT2_PIANOROLL_SIZE,
    MRT2_SAMPLE_RATE,
    _midi_to_pianoroll_tokens,
)


# ══════════════════════════════════════════════════════════════════
# 偵測是否有真實 MRT2 可用
# ══════════════════════════════════════════════════════════════════
_REAL_MRT2_AVAILABLE = False
_real_mrt2_instance = None

try:
    from magenta_rt import MagentaRT2Jax  # type: ignore[import-untyped]

    _real_mrt2_instance = MagentaRT2Jax(
        size="mrt2_small",
        temperature=1.0,
        top_k=40,
        cfg_scales={"musiccoca": 3.0, "notes": 0.1, "drums": 1.0},
    )
    _REAL_MRT2_AVAILABLE = True
    print("[INFO] ✓ 偵測到 magenta_rt，將使用 Real Model 模式測試")
except Exception as exc:
    print(f"[INFO] magenta_rt 不可用 ({exc})，將使用 Mock 模式測試")


# ══════════════════════════════════════════════════════════════════
# Mock 輔助工具
# ══════════════════════════════════════════════════════════════════
def _make_mock_waveform(frames: int = 25, sr: int = MRT2_SAMPLE_RATE) -> MagicMock:
    """模擬 MRT2 輸出的 Waveform 物件 (48kHz stereo)"""
    samples_per_frame = sr // MRT2_FRAME_RATE  # 1920
    total_samples = frames * samples_per_frame
    t = np.linspace(0, frames / MRT2_FRAME_RATE, total_samples, dtype=np.float32)
    # 用多頻率合成讓 RMS > 0
    raw = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.15 * np.sin(2 * np.pi * 880 * t)
    stereo = np.stack([raw, raw * 0.95], axis=0)  # [2, N]
    wf = MagicMock()
    wf.samples = stereo
    return wf


def _make_mock_mrt2(
    stateful: bool = True,
) -> MagicMock:
    """建立一個具備完整 API 簽名的 mock MRT2"""
    mock = MagicMock()
    mock.embed_style.return_value = np.random.randn(64).astype(np.float32)

    _call_counter = {"n": 0}

    def _fake_generate(
        conditioning: dict,
        temperature: float = 1.0,
        frames: int = 25,
        state: Any = None,
    ) -> Tuple[Any, Any]:
        wf = _make_mock_waveform(frames=frames)
        _call_counter["n"] += 1
        if stateful:
            new_state = {
                "call_index": _call_counter["n"],
                "prev_state_was_none": state is None,
                "frames_generated": frames,
                # 保留上一步的 tail sample 供拼接驗證
                "tail_sample": float(wf.samples[0, -1]),
            }
        else:
            new_state = None
        return wf, new_state

    mock.generate.side_effect = _fake_generate
    return mock


def _make_gen(mrt2: Any = None, name: str = "TestGen") -> MRT2Generator:
    return MRT2Generator(
        name=name,
        scale=C_MAJOR_PITCHES,
        style_prompt="test piano",
        mrt2_system=mrt2,
    )


# ── 模擬下游接收 Agent ──────────────────────────────────────────
class AgentPayloadReceiver:
    """模擬接收 MRT2 產出的下游 Agent"""

    def __init__(self, agent_id: str = "ArrangerAgent-01") -> None:
        self.agent_id = agent_id
        self.received: List[dict] = []

    def receive(self, payload: dict) -> dict:
        """驗證並接收 payload"""
        assert isinstance(payload, dict), "Payload 必須是 dict"
        assert payload.get("message_type") == "AUDIO_CONTINUATION"
        assert payload.get("sample_rate") == MRT2_SAMPLE_RATE
        assert payload.get("channels") == 2
        assert isinstance(payload.get("audio_bytes"), bytes)
        assert len(payload["audio_bytes"]) > 0
        assert isinstance(payload.get("input_notes"), list)
        assert len(payload["input_notes"]) > 0
        assert isinstance(payload.get("metadata"), dict)

        self.received.append(payload)
        return {
            "status": "ACK",
            "agent": self.agent_id,
            "total_samples": payload["total_samples"],
        }


# ── WAV 寫入工具 ────────────────────────────────────────────────
def _write_wav(path: str, audio: np.ndarray, sr: int = MRT2_SAMPLE_RATE) -> None:
    """將 float32 numpy array 寫入 WAV（不依賴 scipy/soundfile）"""
    audio = np.clip(audio, -1.0, 1.0)
    if audio.ndim == 2:
        # [channels, samples] → interleaved
        channels = audio.shape[0]
        samples = audio.shape[1]
        interleaved = np.empty(channels * samples, dtype=np.float32)
        for ch in range(channels):
            interleaved[ch::channels] = audio[ch]
    else:
        channels = 1
        samples = len(audio)
        interleaved = audio.astype(np.float32)

    # PCM 16-bit for broad compatibility
    pcm = (interleaved * 32767).astype(np.int16)
    data_bytes = pcm.tobytes()

    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + len(data_bytes)))
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # chunk size
        f.write(struct.pack("<H", 1))  # PCM
        f.write(struct.pack("<H", channels))
        f.write(struct.pack("<I", sr))
        f.write(struct.pack("<I", sr * channels * 2))  # byte rate
        f.write(struct.pack("<H", channels * 2))  # block align
        f.write(struct.pack("<H", 16))  # bits per sample
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", len(data_bytes)))
        f.write(data_bytes)


# ══════════════════════════════════════════════════════════════════
# Test 1: Direct Waveform Generation
# ══════════════════════════════════════════════════════════════════
class TestDirectWaveformGeneration(unittest.TestCase):
    """驗證 MRT2 直接生成 48kHz 立體聲 PCM 波形"""

    def _get_gen(self) -> MRT2Generator:
        if _REAL_MRT2_AVAILABLE:
            return _make_gen(mrt2=_real_mrt2_instance, name="WaveformTest")
        return _make_gen(mrt2=_make_mock_mrt2(), name="WaveformTest_Mock")

    def test_output_is_numpy_float32(self):
        """輸出必須是 numpy float32 array"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=0.5)
        self.assertIsNotNone(audio, "generate_audio 回傳 None")
        self.assertIsInstance(audio, np.ndarray)
        self.assertEqual(audio.dtype, np.float32)

    def test_stereo_shape(self):
        """輸出必須是 [2, N] 立體聲格式"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=0.5)
        self.assertIsNotNone(audio)
        self.assertEqual(audio.ndim, 2, f"期望 2D，實際 {audio.ndim}D")
        self.assertEqual(audio.shape[0], 2, f"期望 2 聲道，實際 {audio.shape[0]}")

    def test_sample_count_matches_duration(self):
        """樣本數應接近 duration * 48000（容許 ±1 frame = 1920 samples）"""
        gen = self._get_gen()
        duration = 1.0
        audio = gen.generate_audio(duration_sec=duration)
        self.assertIsNotNone(audio)
        expected = int(MRT2_SAMPLE_RATE * duration)
        tolerance = MRT2_SAMPLE_RATE // MRT2_FRAME_RATE  # 1920
        self.assertAlmostEqual(
            audio.shape[1], expected, delta=tolerance,
            msg=f"樣本數 {audio.shape[1]} 偏離預期 {expected} 超過 {tolerance}",
        )

    def test_no_nan_inf(self):
        """波形不得包含 NaN 或 Inf"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=0.5)
        self.assertIsNotNone(audio)
        self.assertFalse(np.isnan(audio).any(), "波形含有 NaN")
        self.assertFalse(np.isinf(audio).any(), "波形含有 Inf")

    def test_rms_nonzero(self):
        """RMS 能量必須非零（非純靜音）"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=0.5)
        self.assertIsNotNone(audio)
        rms = np.sqrt(np.mean(audio ** 2))
        self.assertGreater(rms, 1e-5, f"RMS 能量 {rms:.6f} 過低，接近靜音")

    def test_amplitude_range(self):
        """所有樣本值必須在 [-1.0, 1.0] 範圍內（或非常接近）"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=0.5)
        self.assertIsNotNone(audio)
        self.assertLessEqual(
            np.max(np.abs(audio)), 1.01,
            f"波形最大振幅 {np.max(np.abs(audio)):.4f} 超出 [-1, 1] 範圍",
        )

    def test_none_without_mrt2(self):
        """無 MRT2 時 generate_audio 應回傳 None"""
        gen = _make_gen(mrt2=None)
        audio = gen.generate_audio(duration_sec=1.0)
        self.assertIsNone(audio)

    def test_write_wav_artifact(self):
        """（附帶）生成實體 WAV 驗證檔"""
        gen = self._get_gen()
        audio = gen.generate_audio(duration_sec=2.0)
        self.assertIsNotNone(audio)

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_waveform_output.wav",
        )
        _write_wav(out_path, audio)
        self.assertTrue(os.path.isfile(out_path))
        self.assertGreater(os.path.getsize(out_path), 1000)
        print(f"  ✓ WAV 已寫入: {out_path} ({os.path.getsize(out_path)} bytes)")


# ══════════════════════════════════════════════════════════════════
# Test 2: Notes-to-Continuation Agent Protocol
# ══════════════════════════════════════════════════════════════════
class TestNotesToContinuation(unittest.TestCase):
    """驗證音符 → pianoroll conditioning → 接續音訊 → Agent Payload"""

    # 測試用音符：C 大調琶音 C4-E4-G4-C5
    ARPEGGIO = [
        {"pitch": 60, "start_time": 0.0, "end_time": 0.25, "velocity": 90},
        {"pitch": 64, "start_time": 0.25, "end_time": 0.50, "velocity": 85},
        {"pitch": 67, "start_time": 0.50, "end_time": 0.75, "velocity": 95},
        {"pitch": 72, "start_time": 0.75, "end_time": 1.00, "velocity": 100},
    ]

    # 和弦：C 大三和弦同時按下
    CHORD = [
        {"pitch": 60, "start_time": 0.0, "end_time": 1.0},
        {"pitch": 64, "start_time": 0.0, "end_time": 1.0},
        {"pitch": 67, "start_time": 0.0, "end_time": 1.0},
    ]

    def _get_gen(self) -> MRT2Generator:
        if _REAL_MRT2_AVAILABLE:
            return _make_gen(mrt2=_real_mrt2_instance, name="ContTest")
        return _make_gen(mrt2=_make_mock_mrt2(), name="ContTest_Mock")

    # ── pianoroll token 轉換驗證 ──────────────────────────────────

    def test_pianoroll_tokens_length(self):
        """每幀必須有 128 個 token"""
        tokens = _midi_to_pianoroll_tokens([60, 64, 67])
        self.assertEqual(len(tokens), MRT2_PIANOROLL_SIZE)

    def test_pianoroll_onset_values(self):
        """onset pitch 對應 token 值 = NUM_RESERVED + 2"""
        tokens = _midi_to_pianoroll_tokens([60, 64], onset_pitches=[60])
        self.assertEqual(tokens[60], MRT2_NUM_RESERVED + 2)  # onset
        self.assertEqual(tokens[64], MRT2_NUM_RESERVED + 1)  # sustain
        self.assertEqual(tokens[0], MRT2_NUM_RESERVED + 0)   # off

    def test_pianoroll_all_off(self):
        """無活躍音符時所有 token = NUM_RESERVED + 0"""
        tokens = _midi_to_pianoroll_tokens([])
        self.assertTrue(all(t == MRT2_NUM_RESERVED for t in tokens))

    # ── notes_to_continuation 基本行為 ────────────────────────────

    def test_returns_none_without_mrt2(self):
        gen = _make_gen(mrt2=None)
        result = gen.notes_to_continuation(self.ARPEGGIO)
        self.assertIsNone(result)

    def test_returns_none_with_empty_notes(self):
        gen = self._get_gen()
        result = gen.notes_to_continuation([])
        self.assertIsNone(result)

    def test_payload_structure(self):
        """Payload 必須包含所有必要欄位"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.0)
        self.assertIsNotNone(payload)

        required_keys = {
            "source_agent", "message_type", "sample_rate", "channels",
            "duration_seconds", "total_samples", "input_notes",
            "audio_bytes", "metadata",
        }
        self.assertTrue(
            required_keys.issubset(payload.keys()),
            f"缺少欄位: {required_keys - payload.keys()}",
        )

    def test_payload_values(self):
        """驗證 Payload 各欄位值正確"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.0)
        self.assertIsNotNone(payload)

        self.assertEqual(payload["message_type"], "AUDIO_CONTINUATION")
        self.assertEqual(payload["sample_rate"], 48000)
        self.assertEqual(payload["channels"], 2)
        self.assertGreater(payload["total_samples"], 0)
        self.assertGreater(payload["duration_seconds"], 0)
        self.assertEqual(len(payload["input_notes"]), len(self.ARPEGGIO))

    def test_audio_bytes_integrity(self):
        """audio_bytes 可以重建為原始 numpy array"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.0)
        self.assertIsNotNone(payload)

        meta = payload["metadata"]
        shape = tuple(meta["shape"])
        dtype = np.dtype(meta["dtype"])

        reconstructed = np.frombuffer(payload["audio_bytes"], dtype=dtype).reshape(shape)
        self.assertEqual(reconstructed.dtype, np.float32)
        self.assertFalse(np.isnan(reconstructed).any())
        self.assertTrue(np.all(np.abs(reconstructed) <= 1.0))

    def test_metadata_conditioning_info(self):
        """metadata 必須包含 conditioning 資訊"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.0)
        self.assertIsNotNone(payload)

        meta = payload["metadata"]
        self.assertEqual(meta["condition_type"], "pianoroll_onset_25hz")
        self.assertGreater(meta["conditioning_frames"], 0)
        self.assertGreater(meta["generation_frames"], 0)
        self.assertIn("style_prompt", meta)

    # ── Agent 通訊模擬 ────────────────────────────────────────────

    def test_agent_receive_arpeggio(self):
        """下游 Agent 正確接收琶音 Payload"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.5)
        self.assertIsNotNone(payload)

        receiver = AgentPayloadReceiver("ArrangerAgent-01")
        ack = receiver.receive(payload)

        self.assertEqual(ack["status"], "ACK")
        self.assertEqual(ack["total_samples"], payload["total_samples"])
        self.assertEqual(len(receiver.received), 1)

    def test_agent_receive_chord(self):
        """下游 Agent 正確接收和弦 Payload"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.CHORD, duration_sec=2.0)
        self.assertIsNotNone(payload)

        receiver = AgentPayloadReceiver("MixerAgent-02")
        ack = receiver.receive(payload)
        self.assertEqual(ack["status"], "ACK")

    def test_multiple_payloads_to_different_agents(self):
        """同一段音符的 Payload 可以派發給多個 Agent"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=1.0)
        self.assertIsNotNone(payload)

        agents = [
            AgentPayloadReceiver("Arranger"),
            AgentPayloadReceiver("Mixer"),
            AgentPayloadReceiver("Effects"),
        ]
        for agent in agents:
            ack = agent.receive(payload)
            self.assertEqual(ack["status"], "ACK")

    def test_style_prompt_override(self):
        """style_prompt 參數應能覆蓋當前風格"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(
            self.ARPEGGIO,
            duration_sec=1.0,
            style_prompt="orchestral strings legato",
        )
        self.assertIsNotNone(payload)
        self.assertEqual(
            payload["metadata"]["style_prompt"],
            "orchestral strings legato",
        )

    def test_write_continuation_wav(self):
        """生成接續音訊並寫入 WAV 驗證檔"""
        gen = self._get_gen()
        payload = gen.notes_to_continuation(self.ARPEGGIO, duration_sec=2.0)
        self.assertIsNotNone(payload)

        meta = payload["metadata"]
        audio = np.frombuffer(
            payload["audio_bytes"], dtype=np.dtype(meta["dtype"]),
        ).reshape(tuple(meta["shape"]))

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_continuation_output.wav",
        )
        _write_wav(out_path, audio)
        self.assertTrue(os.path.isfile(out_path))
        print(f"  ✓ Continuation WAV: {out_path} ({os.path.getsize(out_path)} bytes)")


# ══════════════════════════════════════════════════════════════════
# Test 3: Streaming State Continuity
# ══════════════════════════════════════════════════════════════════
class TestStreamingStateContinuity(unittest.TestCase):
    """驗證 MRT2 streaming state 跨 chunk 延續性"""

    def _get_gen(self) -> MRT2Generator:
        if _REAL_MRT2_AVAILABLE:
            return _make_gen(mrt2=_real_mrt2_instance, name="StateTest")
        return _make_gen(mrt2=_make_mock_mrt2(stateful=True), name="StateTest_Mock")

    # ── 基本 state 流動 ──────────────────────────────────────────

    def test_cold_start_returns_state(self):
        """冷啟動（initial_state=None）必須回傳非 None 的 state"""
        gen = self._get_gen()
        audio, state = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(audio)
        self.assertIsNotNone(state, "冷啟動後 state 不應為 None")

    def test_state_is_updated_across_chunks(self):
        """連續兩步的 state 應該不同"""
        gen = self._get_gen()

        audio1, state1 = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(state1)

        audio2, state2 = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=state1,
        )
        self.assertIsNotNone(state2)
        self.assertIsNot(state1, state2, "state1 與 state2 不應是同一物件")

    def test_warm_start_differs_from_cold(self):
        """帶 state 的暖啟動 vs 全新冷啟動，mock 中 state 應反映歷史"""
        gen = self._get_gen()

        # 冷啟動
        _, state_cold = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )

        # 暖啟動
        _, state_warm = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=state_cold,
        )

        if isinstance(state_cold, dict) and isinstance(state_warm, dict):
            # Mock 模式：驗證 step 遞增
            self.assertGreater(
                state_warm.get("step", 0),
                state_cold.get("step", 0),
                "暖啟動的 step 計數應 > 冷啟動",
            )

    # ── 波形拼接連續性 ────────────────────────────────────────────

    def test_boundary_smoothness(self):
        """相鄰 chunk 邊界的數值跳差不應超過閾值"""
        gen = self._get_gen()

        chunk1, state1 = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(chunk1)

        chunk2, state2 = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=state1,
        )
        self.assertIsNotNone(chunk2)

        # 取得邊界值
        tail = chunk1[:, -1]   # [channels]
        head = chunk2[:, 0]    # [channels]
        diff = np.abs(head - tail)

        # 音訊波形的瞬時跳差不應超過 1.5（正常音訊極少超過 0.3）
        max_diff = np.max(diff)
        self.assertLess(
            max_diff, 1.5,
            f"邊界波形跳差 {max_diff:.4f} 過大，可能存在相位截斷",
        )
        print(f"  ✓ Chunk 邊界最大差距: {max_diff:.4f}")

    def test_three_chunk_chain(self):
        """三段連續 chunk 的 state chain 完整流動"""
        gen = self._get_gen()
        states = [None]
        chunks = []

        for i in range(3):
            chunk, new_state = gen._generate_chunk_with_state(
                duration_sec=0.5, initial_state=states[-1],
            )
            self.assertIsNotNone(chunk, f"chunk {i} 回傳 None")
            self.assertIsNotNone(new_state, f"chunk {i} 的 state 為 None")
            chunks.append(chunk)
            states.append(new_state)

        # 驗證所有 state 都不同
        for i in range(1, len(states)):
            for j in range(i + 1, len(states)):
                if states[i] is not None and states[j] is not None:
                    self.assertIsNot(states[i], states[j])

        # 拼接完整波形
        full_audio = np.concatenate(chunks, axis=1)
        self.assertEqual(full_audio.shape[0], 2)
        self.assertGreater(full_audio.shape[1], 0)
        print(f"  ✓ 三段拼接總樣本數: {full_audio.shape[1]}")

    def test_concatenated_wav_output(self):
        """拼接多段 chunk 並輸出 WAV 驗證音訊連續性"""
        gen = self._get_gen()
        chunks = []
        state = None

        for i in range(4):
            chunk, state = gen._generate_chunk_with_state(
                duration_sec=0.5, initial_state=state,
            )
            self.assertIsNotNone(chunk)
            chunks.append(chunk)

        full = np.concatenate(chunks, axis=1)

        out_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "test_streaming_concat.wav",
        )
        _write_wav(out_path, full)
        self.assertTrue(os.path.isfile(out_path))
        print(f"  ✓ Streaming WAV: {out_path} ({os.path.getsize(out_path)} bytes)")

    # ── state reset 行為 ──────────────────────────────────────────

    def test_reset_primer_clears_state(self):
        """reset_primer 必須清除 streaming state"""
        gen = self._get_gen()
        _, state1 = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(gen._state)

        gen.reset_primer([60, 64, 67])
        self.assertIsNone(gen._state, "reset_primer 後 _state 應為 None")

    def test_update_style_clears_state(self):
        """update_style 必須重設 streaming state"""
        gen = self._get_gen()
        _, _ = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(gen._state)

        gen.update_style("new ambient pad")
        self.assertIsNone(gen._state)

    # ── Mock fallback 行為 ────────────────────────────────────────

    def test_mock_fallback_produces_audio(self):
        """無 MRT2 的 Mock fallback 也能產出音訊與 state"""
        gen = _make_gen(mrt2=None)
        audio, state = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        self.assertIsNotNone(audio)
        self.assertEqual(audio.ndim, 2)
        self.assertEqual(audio.shape[0], 2)
        self.assertIsNotNone(state)
        self.assertIsInstance(state, dict)

    def test_mock_fallback_state_progression(self):
        """Mock fallback 的 state 應該遞增"""
        gen = _make_gen(mrt2=None)
        _, s1 = gen._generate_chunk_with_state(duration_sec=0.3, initial_state=None)
        _, s2 = gen._generate_chunk_with_state(duration_sec=0.3, initial_state=s1)
        _, s3 = gen._generate_chunk_with_state(duration_sec=0.3, initial_state=s2)

        self.assertEqual(s1["step"], 1)
        self.assertEqual(s2["step"], 2)
        self.assertEqual(s3["step"], 3)

    def test_mock_frequency_changes_with_state(self):
        """Mock 模式中，有 state vs 無 state 應產出不同頻率的 sine wave"""
        gen = _make_gen(mrt2=None)
        audio_cold, state = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=None,
        )
        audio_warm, _ = gen._generate_chunk_with_state(
            duration_sec=0.5, initial_state=state,
        )
        # 兩段音訊不應完全相同（不同頻率）
        self.assertFalse(
            np.allclose(audio_cold, audio_warm),
            "冷啟動與暖啟動的音訊不應完全一樣",
        )


# ══════════════════════════════════════════════════════════════════
# Test 4: 整合端到端串接
# ══════════════════════════════════════════════════════════════════
class TestEndToEndIntegration(unittest.TestCase):
    """完整流程：音符 → continuation → 拼接 → Agent 分發"""

    def _get_gen(self) -> MRT2Generator:
        if _REAL_MRT2_AVAILABLE:
            return _make_gen(mrt2=_real_mrt2_instance, name="E2ETest")
        return _make_gen(mrt2=_make_mock_mrt2(stateful=True), name="E2ETest_Mock")

    def test_notes_then_streaming_continuation(self):
        """先用 notes_to_continuation 取得初始段，再用 streaming state 續寫"""
        gen = self._get_gen()

        # Phase 1: 音符引導
        notes = [
            {"pitch": 60, "start_time": 0.0, "end_time": 0.5},
            {"pitch": 67, "start_time": 0.5, "end_time": 1.0},
        ]
        payload = gen.notes_to_continuation(notes, duration_sec=1.0)
        self.assertIsNotNone(payload)

        # 此時 gen._state 已被更新
        current_state = gen._state

        # Phase 2: 基於 state 續寫兩段 chunk
        chunks = []
        state = current_state
        for _ in range(2):
            chunk, state = gen._generate_chunk_with_state(
                duration_sec=0.5, initial_state=state,
            )
            self.assertIsNotNone(chunk)
            chunks.append(chunk)

        # Phase 3: 將續寫段落封裝派發
        full_continuation = np.concatenate(chunks, axis=1)
        receiver = AgentPayloadReceiver("PostProcessor")

        follow_up_payload = {
            "source_agent": gen.name,
            "message_type": "AUDIO_CONTINUATION",
            "sample_rate": MRT2_SAMPLE_RATE,
            "channels": 2,
            "duration_seconds": full_continuation.shape[1] / MRT2_SAMPLE_RATE,
            "total_samples": full_continuation.shape[1],
            "input_notes": notes,
            "audio_bytes": full_continuation.astype(np.float32).tobytes(),
            "metadata": {
                "format": "pcm_f32le",
                "condition_type": "streaming_continuation",
                "style_prompt": gen.style_prompt,
            },
        }
        ack = receiver.receive(follow_up_payload)
        self.assertEqual(ack["status"], "ACK")
        print(f"  ✓ E2E 完成: 初始 {payload['total_samples']} samples + 續寫 {full_continuation.shape[1]} samples")

    def test_dual_agent_collaboration(self):
        """兩個 Agent 各自生成並交叉派發"""
        gen_a = self._get_gen()
        gen_b = self._get_gen()
        gen_b.name = "AgentB"

        notes_a = [{"pitch": 60, "start_time": 0.0, "end_time": 1.0}]
        notes_b = [{"pitch": 67, "start_time": 0.0, "end_time": 1.0}]

        payload_a = gen_a.notes_to_continuation(notes_a, duration_sec=1.0)
        payload_b = gen_b.notes_to_continuation(notes_b, duration_sec=1.0)

        self.assertIsNotNone(payload_a)
        self.assertIsNotNone(payload_b)

        # 各自派發給對方的接收器
        rx_a = AgentPayloadReceiver("Receiver_A")
        rx_b = AgentPayloadReceiver("Receiver_B")

        ack_a = rx_a.receive(payload_b)  # A 接收 B 的輸出
        ack_b = rx_b.receive(payload_a)  # B 接收 A 的輸出

        self.assertEqual(ack_a["status"], "ACK")
        self.assertEqual(ack_b["status"], "ACK")
        print("  ✓ 雙 Agent 交叉派發驗證通過")


# ══════════════════════════════════════════════════════════════════
# 直接執行入口
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  MRT2 三大功能驗證測試套件")
    print(f"  模式: {'Real Model' if _REAL_MRT2_AVAILABLE else 'Mock'}")
    print("=" * 60)

    t0 = time.time()
    unittest.main(verbosity=2, exit=False)
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"  測試完成，耗時 {elapsed:.1f}s")
    print("=" * 60)
