"""
duet_system.py — Magenta RealTime 2 (MRT2) AI Duet Integration
================================================================
使用 Google Magenta RealTime 2 進行即時音樂生成。

MRT2 是音頻串流模型，接收 MIDI pianoroll + style embedding
作為 conditioning，直接輸出 48 kHz 立體聲波形。

Agent A (Primo)    → MRT2 生成 (style: "jazz piano melody")
Agent B (Secondo)  → MRT2 生成 (style: "ambient chords pad")
Preset Bass        → C-G-F-G loop → Preset_Bass_Out (MIDI)
Human_In           → on_human_interrupt() → 更新 MRT2 conditioning
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
import os
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Tuple

import numpy as np

from midi_core import (
    MidiNote,
    MidiBuffer,
    VirtualMidiPort,
    SystemState,
    ai_pending_buffer,
    human_buffer,
    state,
    melody_out,
    harmony_out,
    bass_out,
    human_in,
    BASS_LINE,
    BASS_BPM,
    BASS_BEAT_SEC,
    preset_bass_task,
    simulate_human_input,
)
import midi_core as _midi_core_module

# ──────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)-7s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("duet_system")

# ──────────────────────────────────────────────────────────────────
# 音樂常數
# ──────────────────────────────────────────────────────────────────
VOCAB_SIZE = 128
SEQUENCE_LEN = 16
DEFAULT_TEMP = 1.3
TEMP_BOOST = 0.2
REPEAT_THRESHOLD = 8

# 音階跨多個八度，供 scale mask 使用 (C3=48 ~ C7=84)
C_MAJOR_PITCHES: List[int] = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74, 76]
G_MIXO_PITCHES: List[int]  = [48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72, 74]

BPM = 120
TICKS_PER_BEAT = 480
NOTE_DURATION = 60.0 / BPM / 2  # 八分音符 = 0.25s

# MRT2 常數
MRT2_FRAME_RATE = 25  # 25 frames per second
MRT2_SAMPLE_RATE = 48000
# pianoroll_with_onsets_tokens: codebook_size=4, 128 MIDI keys
# 每一幀需要 128 個 token (每個 key 對應一個)
# token 值: 0=off, 1=sustain, 2=onset, 3=offset (加上 NUM_RESERVED_TOKENS 偏移)
MRT2_NUM_RESERVED = 6
MRT2_PIANOROLL_SIZE = 128
# drum pianoroll: 1 token per frame
MRT2_DRUM_SIZE = 1


def _pitch_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{(midi // 12) - 1}"


def _midi_to_pianoroll_tokens(
    active_pitches: List[int],
    onset_pitches: Optional[List[int]] = None,
) -> List[int]:
    """
    將當前活躍的 MIDI 音符轉換為 MRT2 pianoroll_with_onsets_tokens 格式。

    每一幀產生 128 個 token (對應 128 個 MIDI key)。
    值域: 0=off, 1=sustain, 2=onset, 3=offset
    加上 NUM_RESERVED_TOKENS (6) 偏移。
    """
    tokens = [MRT2_NUM_RESERVED + 0] * MRT2_PIANOROLL_SIZE

    onset_set = set(onset_pitches) if onset_pitches else set()

    for pitch in active_pitches:
        if 0 <= pitch < MRT2_PIANOROLL_SIZE:
            if pitch in onset_set:
                tokens[pitch] = MRT2_NUM_RESERVED + 2  # onset
            else:
                tokens[pitch] = MRT2_NUM_RESERVED + 1  # sustain

    return tokens


def _drum_tokens_from_pattern(has_kick: bool = False, has_snare: bool = False) -> List[int]:
    """將簡易鼓點轉為 drum_pianoroll_tokens。"""
    val = 1 if (has_kick or has_snare) else 0
    return [MRT2_NUM_RESERVED + val]


# ──────────────────────────────────────────────────────────────────
# MRT2 封裝層
# ──────────────────────────────────────────────────────────────────
def _try_load_mrt2(
    size: str = "mrt2_small",
    temperature: float = DEFAULT_TEMP,
) -> Optional[Any]:
    """嘗試載入 MRT2 系統；失敗回傳 None。"""
    try:
        from magenta_rt import MagentaRT2Jax  # type: ignore[import-untyped]
        mrt = MagentaRT2Jax(
            size=size,
            temperature=temperature,
            top_k=40,
            cfg_scales={
                "musiccoca": 3.0,
                "notes": 0.1,
                "drums": 1.0,
            },
        )
        log.info("[MRT2] 成功載入 %s 模型", size)
        return mrt
    except Exception as e:
        log.warning("[MRT2] 無法載入 %s: %s", size, e)
        return None


class MRT2Generator:
    """
    封裝 MRT2 系統，提供即時生成與風格控制 API。

    核心功能：
    - MRT2 直接生成音頻波形 (48kHz 立體聲)
    - 用 style embedding 控制風格
    - Text-to-Synth / Audio Cloning / Prompt Mixing
    - streaming state 跨呼叫延續
    - 無 MRT2 時 fallback 隨機音階游走
    """

    _PITCH_LOW = 36   # C2 (涵蓋低音 Bass)
    _PITCH_HIGH = 84  # C7

    @staticmethod
    def _normalize_audio(audio: np.ndarray) -> np.ndarray:
        """
        正規化 MRT2 輸出的音訊 array 為 (2, N) stereo float32。

        MRT2 JAX 原生輸出格式為 (N, 2)，本函式統一轉為 (2, N)。
        支援 1D (mono)、(N, 2)、(2, N) 等各種 shape。
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            # mono → stereo
            audio = np.stack([audio, audio], axis=0)
        elif audio.ndim == 2:
            if audio.shape[1] == 2 and audio.shape[0] != 2:
                # (N, 2) → (2, N)
                audio = audio.T
            elif audio.shape[0] != 2:
                # 未知格式，嘗試 transpose
                audio = audio.T if audio.shape[1] > audio.shape[0] else audio
        return audio

    def __init__(
        self,
        name: str,
        scale: List[int],
        style_prompt: str = "piano melody",
        mrt2_system: Optional[Any] = None,
    ) -> None:
        self.name = name
        self.scale = scale
        self.style_prompt = style_prompt
        self._lock = threading.RLock()

        self._mrt2 = mrt2_system
        self._style_embedding: Optional[np.ndarray] = None
        self._style_tokens: Optional[List[int]] = None
        self._state: Optional[Any] = None  # MRT2 streaming state

        self._temperature: float = DEFAULT_TEMP
        self._last_pitch: Optional[int] = None
        self._repeat_count: int = 0

        # 初始化 style embedding 與 pre-tokenization
        if self._mrt2 is not None:
            try:
                self._style_embedding = self._mrt2.embed_style(style_prompt)
                if hasattr(self._mrt2, "tokenize_style"):
                    self._style_tokens = self._mrt2.tokenize_style(self._style_embedding).tolist()
                log.info("[%s] Style embedding & tokens 生成完成 ('%s')", self.name, style_prompt)
            except Exception as e:
                log.warning("[%s] Style embedding 失敗: %s", self.name, e)

        log.info(
            "[%s] 初始化完成，backend=%s",
            self.name,
            "MRT2-JAX" if self._mrt2 else "Fallback-Random",
        )

    # ── 溫度自動調節 ──────────────────────────────────────────────
    def _update_temperature(self, pitch: int) -> None:
        self._last_pitch = pitch

    # ── MRT2 音頻生成 ─────────────────────────────────────────────
    def generate_audio(
        self,
        active_pitches: Optional[List[int]] = None,
        onset_pitches: Optional[List[int]] = None,
        duration_sec: float = 1.0,
    ) -> Optional[np.ndarray]:
        """
        使用 MRT2 生成音頻波形。

        Args:
            active_pitches: 可選，保留相容介面
            onset_pitches: 可選，保留相容介面
            duration_sec: 要生成的秒數 (25 frames = 1 秒)

        Returns:
            numpy array of audio samples (48kHz stereo float32), 或 None 表示失敗
        """
        if self._mrt2 is None:
            return None

        with self._lock:
            try:
                frames = max(1, int(duration_sec * MRT2_FRAME_RATE))

                conditioning: dict[str, Any] = {}
                if self._style_tokens is not None:
                    conditioning["mulan_tokens_25hz"] = self._style_tokens
                elif self._style_embedding is not None:
                    conditioning["mulan_tokens_25hz"] = self._style_embedding

                waveform, self._state = self._mrt2.generate(
                    conditioning=conditioning,
                    temperature=self._temperature,
                    frames=frames,
                    state=self._state,
                )

                if hasattr(waveform, 'samples'):
                    audio = waveform.samples
                else:
                    audio = np.array(waveform)
                return self._normalize_audio(audio)

            except Exception as e:
                log.error("[%s] MRT2 生成失敗: %s", self.name, e)
                return None

    @staticmethod
    def audio_to_pitches(
        audio: np.ndarray,
        sample_rate: int = MRT2_SAMPLE_RATE,
        length: int = 4,
        scale: Optional[List[int]] = None,
    ) -> List[int]:
        """
        從 48kHz 音訊波形中以短時頻譜峰值提取真實演奏音高。
        """
        import scipy.signal
        if audio.ndim == 2:
            mono = np.mean(audio, axis=0 if audio.shape[0] == 2 else 1)
        else:
            mono = audio

        n_fft = 4096
        hop = max(512, len(mono) // (length + 1))
        f, _, Zxx = scipy.signal.stft(mono, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(Zxx)

        pitches: List[int] = []
        step_indices = np.linspace(1, max(1, mag.shape[1] - 2), length, dtype=int)
        for idx in step_indices:
            col = mag[:, min(idx, mag.shape[1] - 1)]
            valid = np.where((f >= 65) & (f <= 1500))[0]
            if len(valid) > 0:
                p_bin = valid[np.argmax(col[valid])]
                freq = f[p_bin]
                if freq > 0:
                    p = int(round(69 + 12 * np.log2(freq / 440.0)))
                    p = max(MRT2Generator._PITCH_LOW, min(MRT2Generator._PITCH_HIGH, p))
                    if scale:
                        p = min(scale, key=lambda s: abs(s - p))
                    pitches.append(p)
                else:
                    pitches.append(60)
            else:
                pitches.append(60)
        return pitches

    # ── MRT2 音符序列生成 ─────────────────────────────────────────
    def generate(
        self,
        primer: List[int],
        length: int = 8,
        temperature: Optional[float] = None,
    ) -> List[int]:
        """
        生成音符序列。
        若 MRT2 引擎可用，100% 由 MRT2 神經音訊模型合成波形並解碼為 MIDI 音符；
        僅在無 MRT2 模型環境下使用音階隨機游走作為 fallback。
        """
        if temperature is not None:
            with self._lock:
                self._temperature = temperature

        # 1. 優先使用 MRT2 神經音訊生成並解碼真實音符
        if self._mrt2 is not None:
            try:
                gen_duration = max(0.5, length * 0.15)
                if primer:
                    notes = [
                        {"pitch": p, "start_time": i * 0.1, "end_time": (i + 1) * 0.1}
                        for i, p in enumerate(primer[-4:])
                    ]
                    payload = self.notes_to_continuation(notes, duration_sec=gen_duration)
                    if payload and "audio_bytes" in payload:
                        meta = payload["metadata"]
                        raw_audio = np.frombuffer(
                            payload["audio_bytes"], dtype=np.float32
                        ).reshape(tuple(meta["shape"]))
                        pitches = self.audio_to_pitches(
                            raw_audio, length=length, scale=self.scale
                        )
                        if pitches:
                            return pitches
                else:
                    raw_audio = self.generate_audio(duration_sec=gen_duration)
                    if raw_audio is not None:
                        pitches = self.audio_to_pitches(
                            raw_audio, length=length, scale=self.scale
                        )
                        if pitches:
                            return pitches
            except Exception as e:
                log.warning("[%s] MRT2 音符生成失敗，降級至隨機: %s", self.name, e)

        # 2. Fallback: 隨機音階游走 (僅在無 MRT2 或推論失敗時)
        if not primer:
            primer = [random.choice(self.scale)]

        result: List[int] = []
        current = primer[-1]

        for _ in range(length):
            if current in self.scale:
                idx = self.scale.index(current)
            else:
                idx = min(range(len(self.scale)), key=lambda i: abs(self.scale[i] - current))

            step = random.choice([-2, -1, 0, 1, 2])
            idx = max(0, min(len(self.scale) - 1, idx + step))
            pitch = self.scale[idx]
            pitch = max(self._PITCH_LOW, min(self._PITCH_HIGH, pitch))

            result.append(pitch)
            self._update_temperature(pitch)
            current = pitch

        return result

    # ── 公開 API ──────────────────────────────────────────────────
    def reset_primer(self, primer: List[int]) -> None:
        """人類亂入後，強制重設狀態。"""
        with self._lock:
            self._state = None  # 重設 MRT2 streaming state
            self._last_pitch = primer[-1] if primer else None
            self._repeat_count = 0
            self._temperature = DEFAULT_TEMP
        log.info("[%s] primer 重設 → %s", self.name, [_pitch_name(x) for x in primer])

    def update_style(self, prompt: str) -> None:
        """動態更新風格 embedding。"""
        if self._mrt2 is None:
            return
        with self._lock:
            try:
                self._style_embedding = self._mrt2.embed_style(prompt)
                if hasattr(self._mrt2, "tokenize_style"):
                    self._style_tokens = self._mrt2.tokenize_style(self._style_embedding).tolist()
                self.style_prompt = prompt
                self._state = None  # 風格改變，重設 state
                log.info("[%s] 風格已更新 → '%s'", self.name, prompt)
            except Exception as e:
                log.warning("[%s] 風格更新失敗: %s", self.name, e)

    # ── Text-to-Synth ─────────────────────────────────────────────
    def text_to_synth(self, prompt: str) -> Optional[np.ndarray]:
        """
        Text-to-Synth: 從文字描述生成風格 embedding。
        """
        if self._mrt2 is None or not prompt or not prompt.strip():
            log.warning("[%s] Text-to-Synth 失敗 (無 MRT2 或空 prompt)", self.name)
            return None

        with self._lock:
            try:
                if hasattr(self._mrt2, 'synthesize_from_text'):
                    synth_embedding = self._mrt2.synthesize_from_text(prompt)
                else:
                    synth_embedding = self._mrt2.embed_style(prompt)

                self._style_embedding = synth_embedding
                if hasattr(self._mrt2, "tokenize_style"):
                    self._style_tokens = self._mrt2.tokenize_style(self._style_embedding).tolist()
                self.style_prompt = prompt
                self._state = None
                log.info("[%s] Text-to-Synth 完成 → '%s'", self.name, prompt)
                return synth_embedding
            except Exception as e:
                log.error("[%s] Text-to-Synth 失敗: %s", self.name, e)
                return None

    # ── Audio Cloning ─────────────────────────────────────────────
    def clone_audio(
        self,
        audio_samples: np.ndarray,
        sample_rate: int = MRT2_SAMPLE_RATE,
    ) -> Optional[np.ndarray]:
        """
        Audio Cloning: 從一段音頻片段克隆音色，生成對應的 style embedding。
        """
        if self._mrt2 is None or audio_samples is None or len(audio_samples) == 0:
            log.warning("[%s] Audio Cloning 失敗 (無 MRT2 或空音頻)", self.name)
            return None

        # 驗證音頻長度 (至少 0.1 秒)
        min_samples = int(sample_rate * 0.1)
        if len(audio_samples) < min_samples:
            log.warning(
                "[%s] Audio Cloning 音頻太短 (%d samples < %d minimum)",
                self.name, len(audio_samples), min_samples,
            )
            return None

        with self._lock:
            try:
                if hasattr(self._mrt2, 'clone_audio'):
                    cloned_embedding = self._mrt2.clone_audio(
                        audio_samples, sample_rate=sample_rate,
                    )
                elif hasattr(self._mrt2, 'embed_audio'):
                    cloned_embedding = self._mrt2.embed_audio(
                        audio_samples, sample_rate=sample_rate,
                    )
                elif hasattr(self._mrt2, 'embed_style') and type(self._mrt2).__name__ in ('MagentaRT2System', 'MagentaRT2Jax'):
                    from magenta_rt.audio import Waveform
                    samples_float = audio_samples.astype(np.float32)
                    max_val = np.abs(samples_float).max()
                    if max_val > 1.0:
                        samples_float = samples_float / 32768.0

                    wf = Waveform(samples_float, sample_rate=sample_rate)
                    cloned_embedding = self._mrt2.embed_style(wf)
                else:
                    log.warning("[%s] MRT2 不支援 Audio Cloning", self.name)
                    return None

                self._style_embedding = cloned_embedding
                self.style_prompt = f"[cloned_audio: {len(audio_samples)} samples @ {sample_rate}Hz]"
                self._state = None
                log.info(
                    "[%s] Audio Cloning 完成 (%.2f 秒音頻)",
                    self.name, len(audio_samples) / sample_rate,
                )
                return cloned_embedding

            except Exception as e:
                log.error("[%s] Audio Cloning 失敗: %s", self.name, e)
                return None

    # ── Prompt Mixing ─────────────────────────────────────────────
    def mix_prompts(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
        alpha: float = 0.5,
    ) -> Optional[np.ndarray]:
        """
        Prompt Mixing: 在兩個 style embedding 之間平滑插值。
        """
        if embedding_a is None or embedding_b is None:
            log.warning("[%s] Prompt Mixing 收到 None embedding", self.name)
            return None

        alpha = max(0.0, min(1.0, alpha))
        if embedding_a.shape != embedding_b.shape:
            log.warning(
                "[%s] Prompt Mixing embedding 形狀不一致: %s vs %s",
                self.name, embedding_a.shape, embedding_b.shape,
            )
            return None

        with self._lock:
            try:
                if hasattr(self._mrt2, 'mix_embeddings'):
                    mixed = self._mrt2.mix_embeddings(embedding_a, embedding_b, alpha=alpha)
                else:
                    mixed = (1.0 - alpha) * embedding_a + alpha * embedding_b

                self._style_embedding = mixed
                self._state = None
                log.info("[%s] Prompt Mixing 完成 (alpha=%.2f)", self.name, alpha)
                return mixed

            except Exception as e:
                log.error("[%s] Prompt Mixing 失敗: %s", self.name, e)
                return None

    # ── Notes-to-Continuation (Agent Protocol) ──────────────────
    def notes_to_continuation(
        self,
        notes: List[dict],
        duration_sec: float = 2.0,
        style_prompt: Optional[str] = None,
    ) -> Optional[dict]:
        """
        傳入前導 MIDI 音符列表，生成接續樂曲音訊段落，
        並封裝成 Multi-Agent 通訊標準 Payload 回傳。

        Args:
            notes: 音符列表，每個 dict 包含:
                   - pitch (int): MIDI 音高 0-127
                   - start_time (float): 開始時間（秒）
                   - end_time (float): 結束時間（秒）
                   - velocity (int, optional): 力度 1-127，預設 80
            duration_sec: 要生成的接續音訊秒數
            style_prompt: 可選，覆蓋當前 style

        Returns:
            Multi-Agent payload dict，或 None 表示失敗。
            Payload 結構:
            {
                "source_agent": str,
                "message_type": "AUDIO_CONTINUATION",
                "sample_rate": 48000,
                "channels": 2,
                "duration_seconds": float,
                "total_samples": int,
                "input_notes": list,
                "audio_bytes": bytes,  # PCM float32 LE raw
                "metadata": {
                    "format": "pcm_f32le",
                    "condition_type": "pianoroll_onset_25hz",
                    "style_prompt": str,
                },
            }
        """
        if self._mrt2 is None:
            log.warning("[%s] notes_to_continuation 失敗 (無 MRT2)", self.name)
            return None

        if not notes:
            log.warning("[%s] notes_to_continuation 收到空音符列表", self.name)
            return None

        with self._lock:
            try:
                frames = max(1, int(duration_sec * MRT2_FRAME_RATE))

                # 如果指定了新 style_prompt，先更新 embedding
                effective_prompt = style_prompt or self.style_prompt
                if style_prompt and style_prompt != self.style_prompt:
                    try:
                        self._style_embedding = self._mrt2.embed_style(style_prompt)
                        self.style_prompt = style_prompt
                    except Exception:
                        pass  # 沿用既有 embedding

                # 將音符列表展開成 per-frame pianoroll tokens
                # 取得音符的最大結束時間來決定 conditioning 幀數
                max_end = max(n.get("end_time", 0) for n in notes)
                cond_frames = max(1, int(max_end * MRT2_FRAME_RATE))

                pianoroll_per_frame: List[List[int]] = []
                for f_idx in range(cond_frames):
                    frame_time = f_idx / MRT2_FRAME_RATE
                    frame_end = (f_idx + 1) / MRT2_FRAME_RATE

                    active = []
                    onsets = []
                    for n in notes:
                        p = n["pitch"]
                        st = float(n["start_time"]) if n.get("start_time") is not None else 0.0
                        et_val = n.get("end_time")
                        et = float(et_val) if et_val is not None else st + 0.25
                        if st < frame_end and et > frame_time:
                            active.append(p)
                            if st >= frame_time and st < frame_end:
                                onsets.append(p)

                    pianoroll_per_frame.append(
                        _midi_to_pianoroll_tokens(active, onsets)
                    )

                # MRT2 API: pianoroll_with_onsets_tokens 是逐步 per-frame
                # 先用 conditioning frames 餵入 pianoroll，再生成剩餘 frames
                # 將所有幀合併為一次 generate 呼叫:
                # conditioning 帶首幀的 pianoroll tokens，其餘幀會由模型自動推理
                first_frame_tokens = pianoroll_per_frame[0] if pianoroll_per_frame else _midi_to_pianoroll_tokens([])

                conditioning: dict[str, Any] = {}
                if self._style_tokens is not None:
                    conditioning["musiccoca"] = self._style_tokens
                elif self._style_embedding is not None:
                    conditioning["mulan_tokens_25hz"] = self._style_embedding

                # 每步送單幀的 128 tokens
                conditioning["pianoroll_with_onsets_tokens"] = first_frame_tokens

                # 逐步生成：先用有 pianoroll 的幀引導，再生成後續幀
                all_waveform_chunks = []
                current_state = self._state

                for f_idx in range(frames):
                    cond_step: dict[str, Any] = {}
                    if self._style_tokens is not None:
                        cond_step["mulan_tokens_25hz"] = self._style_tokens
                    elif self._style_embedding is not None:
                        cond_step["mulan_tokens_25hz"] = self._style_embedding
                    # 若在 conditioning 範圍內，帶上 pianoroll tokens
                    if f_idx < len(pianoroll_per_frame):
                        cond_step["pianoroll_with_onsets_tokens"] = pianoroll_per_frame[f_idx]
                    else:
                        # 超出音符範圍，送全 off
                        cond_step["pianoroll_with_onsets_tokens"] = _midi_to_pianoroll_tokens([])

                    wf, current_state = self._mrt2.generate(
                        conditioning=cond_step,
                        temperature=self._temperature,
                        frames=1,
                        state=current_state,
                    )
                    if hasattr(wf, "samples"):
                        all_waveform_chunks.append(np.asarray(wf.samples, dtype=np.float32))
                    else:
                        all_waveform_chunks.append(np.asarray(wf, dtype=np.float32))

                self._state = current_state

                # 拼接所有 chunk
                waveform_concat = np.concatenate(all_waveform_chunks, axis=0)

                # 正規化為 (2, N) 並 clip
                audio = self._normalize_audio(waveform_concat)
                audio = np.clip(audio, -1.0, 1.0)

                # 封裝 Multi-Agent Payload
                payload: dict[str, Any] = {
                    "source_agent": self.name,
                    "message_type": "AUDIO_CONTINUATION",
                    "sample_rate": MRT2_SAMPLE_RATE,
                    "channels": 2,
                    "duration_seconds": audio.shape[-1] / MRT2_SAMPLE_RATE if audio.ndim >= 1 else duration_sec,
                    "total_samples": audio.shape[-1] if audio.ndim >= 1 else 0,
                    "input_notes": notes,
                    "audio_bytes": audio.tobytes(),
                    "metadata": {
                        "format": "pcm_f32le",
                        "dtype": str(audio.dtype),
                        "shape": list(audio.shape),
                        "condition_type": "pianoroll_onset_25hz",
                        "conditioning_frames": cond_frames,
                        "generation_frames": frames,
                        "style_prompt": effective_prompt,
                    },
                }
                log.info(
                    "[%s] notes_to_continuation 完成: %d 個音符 → %d samples (%.2fs)",
                    self.name, len(notes), payload["total_samples"],
                    payload["duration_seconds"],
                )
                return payload

            except Exception as e:
                log.error("[%s] notes_to_continuation 失敗: %s", self.name, e)
                return None

    # ── Streaming State: 帶狀態的單步生成 ─────────────────────────
    def _generate_chunk_with_state(
        self,
        duration_sec: float = 0.5,
        style_prompt: Optional[str] = None,
        initial_state: Optional[Any] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Any]]:
        """
        供 Streaming 與 Multi-Agent 使用的單步狀態保持生成器。

        Args:
            duration_sec: 本次 chunk 的秒數
            style_prompt: 可選，覆蓋當前 style
            initial_state: 傳入的 MRT2 streaming state，None 表示冷啟動

        Returns:
            (audio_chunk, updated_state) tuple
            audio_chunk: np.ndarray float32 [2, N] 或 None
            updated_state: 更新後的 JAX state 物件
        """
        if self._mrt2 is None:
            # Mock 模式：回傳合成的 sine wave 與虛擬 state
            samples = int(MRT2_SAMPLE_RATE * duration_sec)
            t = np.linspace(0, duration_sec, samples, dtype=np.float32)
            freq = 440.0 if initial_state is None else 523.25
            sine = 0.3 * np.sin(2 * np.pi * freq * t)
            audio = np.stack([sine, sine], axis=0)  # stereo [2, N]
            step = 1 if initial_state is None else (initial_state.get("step", 0) + 1)
            mock_state = {"step": step, "last_sample": float(sine[-1])}
            return audio, mock_state

        with self._lock:
            try:
                frames = max(1, int(duration_sec * MRT2_FRAME_RATE))

                # 如果指定了新 style prompt
                if style_prompt and style_prompt != self.style_prompt:
                    try:
                        self._style_embedding = self._mrt2.embed_style(style_prompt)
                        if hasattr(self._mrt2, "tokenize_style"):
                            self._style_tokens = self._mrt2.tokenize_style(self._style_embedding).tolist()
                        self.style_prompt = style_prompt
                    except Exception:
                        pass

                conditioning: dict[str, Any] = {}
                if self._style_tokens is not None:
                    conditioning["mulan_tokens_25hz"] = self._style_tokens
                elif self._style_embedding is not None:
                    conditioning["mulan_tokens_25hz"] = self._style_embedding

                # 使用傳入的 state 或自身的 _state
                use_state = initial_state if initial_state is not None else self._state

                waveform, new_state = self._mrt2.generate(
                    conditioning=conditioning,
                    temperature=self._temperature,
                    frames=frames,
                    state=use_state,
                )

                # 同步更新自身 state
                self._state = new_state

                if hasattr(waveform, "samples"):
                    audio = waveform.samples
                else:
                    audio = np.array(waveform)

                audio = self._normalize_audio(audio)
                return audio, new_state

            except Exception as e:
                log.error("[%s] _generate_chunk_with_state 失敗: %s", self.name, e)
                return None, initial_state

    @property
    def has_mrt2(self) -> bool:
        return self._mrt2 is not None


# ──────────────────────────────────────────────────────────────────
# MRT2 系統初始化
# ──────────────────────────────────────────────────────────────────
# 嘗試載入 MRT2 (共用一個實例，因為模型很大)
_mrt2_system = _try_load_mrt2(size="mrt2_small")

agent_a = MRT2Generator(
    name="MRT2_Primo",
    scale=C_MAJOR_PITCHES,
    style_prompt="jazz piano melody improvisation",
    mrt2_system=_mrt2_system,
)
agent_b = MRT2Generator(
    name="MRT2_Secondo",
    scale=G_MIXO_PITCHES,
    style_prompt="ambient pad chords smooth",
    mrt2_system=_mrt2_system,
)
BASS_PITCHES = [36, 38, 40, 41, 43, 45, 47, 48, 50, 52, 53, 55]  # C2 ~ G3
agent_bass = MRT2Generator(
    name="MRT2_Bass",
    scale=BASS_PITCHES,
    style_prompt="deep 808 sub bass warm analog synth bassline low frequency groove",
    mrt2_system=_mrt2_system,
)

# Sliding window
_sliding_lock = threading.Lock()
_melody_window: deque[int] = deque(maxlen=SEQUENCE_LEN)
_harmony_window: deque[int] = deque(maxlen=SEQUENCE_LEN)
_bass_pitches: deque[int] = deque(maxlen=4)

for _seed in C_MAJOR_PITCHES[:4]:
    _melody_window.append(_seed)
for _seed in G_MIXO_PITCHES[:4]:
    _harmony_window.append(_seed)


# ──────────────────────────────────────────────────────────────────
# Bass 監聽
# ──────────────────────────────────────────────────────────────────
def _on_bass_received(note: MidiNote) -> None:
    with _sliding_lock:
        _bass_pitches.append(note.pitch)


bass_out.connect(_on_bass_received)


# ──────────────────────────────────────────────────────────────────
# 人類亂入擴充：覆寫 on_human_interrupt
# ──────────────────────────────────────────────────────────────────
_original_interrupt = _midi_core_module.on_human_interrupt


def _extended_human_interrupt(note: MidiNote) -> None:
    """執行原始攔截 + 更新兩個 Agent 的 primer。"""
    _original_interrupt(note)

    human_pitches: List[int] = [note.pitch]
    with _sliding_lock:
        # 把 human_buffer 裡已有的額外音符也納入
        if len(human_buffer) > 0:
            extra: List[MidiNote] = []
            while True:
                n = human_buffer.pop()
                if n is None:
                    break
                extra.append(n)
            human_pitches.extend(n.pitch for n in extra)

    primer = human_pitches[-4:]
    log.warning("[DuetSystem] 強制更新 Agent primer → %s", [_pitch_name(x) for x in primer])
    agent_a.reset_primer(primer)
    agent_b.reset_primer(primer)

    with _sliding_lock:
        _melody_window.clear()
        _harmony_window.clear()
        for p in primer:
            _melody_window.append(p)
            _harmony_window.append(p)


_midi_core_module.on_human_interrupt = _extended_human_interrupt  # type: ignore[assignment]

# 重新連接 human_in（原始 _human_in_router 已捕獲舊 reference）
human_in._callbacks.clear()
human_in.connect(_extended_human_interrupt)


# ──────────────────────────────────────────────────────────────────
# AI 互彈主迴圈
# ──────────────────────────────────────────────────────────────────
async def _play_agent_notes(
    agent: "MRT2Generator",
    pitches: List[int],
    port: Any,
    channel: int,
    velocity: int,
    source: str,
    window: Any,
    stop_event: asyncio.Event,
) -> None:
    """依序播放一個 Agent 生成的音符；無 MRT2 時經 pending buffer 以便人類打斷時清除。"""
    for pitch in pitches:
        if stop_event.is_set() or state.mode != SystemState.AI_MODE:
            break
        note = MidiNote(channel=channel, pitch=pitch, velocity=velocity, source=source)
        if agent.has_mrt2:
            port.send(note)
            with _sliding_lock:
                window.append(pitch)
        else:
            ai_pending_buffer.push(note)
            pending = ai_pending_buffer.pop()
            if pending:
                port.send(pending)
                with _sliding_lock:
                    window.append(pitch)
        await asyncio.sleep(NOTE_DURATION)


async def magenta_duet_task(stop_event: asyncio.Event) -> None:
    log.info("[DuetSystem] MRT2 互彈任務啟動")

    while not stop_event.is_set():
        if state.mode == SystemState.AI_MODE:
            with _sliding_lock:
                a_primer = list(_melody_window)
                b_primer = list(_melody_window)
                if _bass_pitches:
                    bass_hint = [p + 12 for p in _bass_pitches if p + 12 < 80]
                    a_primer = (bass_hint + a_primer)[-SEQUENCE_LEN:]

            # Agent A / B 各只呼叫一次 MRT2，並在執行緒中並行推論（GPU 上可重疊運算，
            # 也避免阻塞 event loop）
            a_pitches, b_pitches = await asyncio.gather(
                asyncio.to_thread(agent_a.generate, primer=a_primer, length=4),
                asyncio.to_thread(agent_b.generate, primer=b_primer, length=4),
            )

            await _play_agent_notes(
                agent_a, a_pitches, melody_out, channel=0, velocity=85,
                source="AI_Melody", window=_melody_window, stop_event=stop_event,
            )
            if state.mode != SystemState.AI_MODE:
                continue
            await _play_agent_notes(
                agent_b, b_pitches, harmony_out, channel=1, velocity=70,
                source="AI_Harmony", window=_harmony_window, stop_event=stop_event,
            )

        elif state.mode == SystemState.HUMAN_MODE:
            if len(human_buffer) == 0:
                log.warning("[DuetSystem] Human Buffer 清空，回到 AI 互彈模式")
                state.set_mode(SystemState.AI_MODE)
            try:
                await asyncio.wait_for(asyncio.shield(stop_event.wait()), timeout=0.1)
            except asyncio.TimeoutError:
                pass

    log.info("[DuetSystem] MRT2 互彈任務已停止")


# ──────────────────────────────────────────────────────────────────
# MIDI 錄音機
# ──────────────────────────────────────────────────────────────────
class MidiRecorder:
    """記錄所有 VirtualMidiPort 的輸出，最後寫成 .mid 檔。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: List[Tuple[float, str, int, int, int]] = []
        self._start = time.time()

    def record(self, note: MidiNote) -> None:
        t = time.time() - self._start
        with self._lock:
            self._events.append((t, note.source, note.channel, note.pitch, note.velocity))

    def save(self, path: str = "duet_output.mid") -> None:
        try:
            from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo  # type: ignore[import-untyped]

            mid = MidiFile(ticks_per_beat=TICKS_PER_BEAT)
            tempo = bpm2tempo(BPM)

            source_map: dict[str, Tuple[int, int]] = {
                "AI_Melody":   (0, 0),
                "AI_Harmony":  (1, 1),
                "Preset_Bass": (2, 2),
                "Human":       (0, 3),
            }

            tracks = [MidiTrack(), MidiTrack(), MidiTrack()]
            for trk in tracks:
                trk.append(MetaMessage("set_tempo", tempo=tempo, time=0))
            mid.tracks.extend(tracks)

            with self._lock:
                events = sorted(self._events, key=lambda e: e[0])

            def sec_to_ticks(sec: float) -> int:
                return int(sec * TICKS_PER_BEAT * BPM / 60)

            # Collect (on_tick, off_tick, vel) per (tidx, ch, pitch)
            note_on_map: dict[Tuple[int, int, int], List[Tuple[int, int, int]]] = {}
            for t_sec, source, ch, pitch, vel in events:
                tidx, _ = source_map.get(source, (0, ch))
                on_t = sec_to_ticks(t_sec)
                off_t = sec_to_ticks(t_sec + NOTE_DURATION)
                key = (tidx, ch, pitch)
                note_on_map.setdefault(key, []).append((on_t, off_t, vel))

            # Build per-track message lists
            track_msgs: dict[int, List[Tuple[int, str, int, int, int]]] = {0: [], 1: [], 2: []}
            for (tidx, ch, pitch), ons in note_on_map.items():
                for on_t, off_t, vel in ons:
                    track_msgs[tidx].append((on_t,  "note_on",  ch, pitch, vel))
                    track_msgs[tidx].append((off_t, "note_off", ch, pitch, 0))

            for tidx, msgs in track_msgs.items():
                msgs.sort(key=lambda m: m[0])
                prev = 0
                for abs_t, mtype, ch, pitch, vel in msgs:
                    delta = abs_t - prev
                    if mtype == "note_on":
                        tracks[tidx].append(
                            Message("note_on", channel=ch, note=pitch, velocity=vel, time=delta)
                        )
                    else:
                        tracks[tidx].append(
                            Message("note_off", channel=ch, note=pitch, velocity=0, time=delta)
                        )
                    prev = abs_t

            mid.save(path)
            log.info("[Recorder] MIDI 儲存完成: %s (%d 個事件)", path, len(events))
        except Exception as e:
            log.exception("[Recorder] 儲存失敗: %s", e)


recorder = MidiRecorder()
melody_out.connect(recorder.record)
harmony_out.connect(recorder.record)
bass_out.connect(recorder.record)
human_in.connect(recorder.record)


# ──────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────
async def main() -> None:
    backend = "MRT2-JAX" if _mrt2_system else "Fallback-Random"
    log.info("=" * 56)
    log.info("  AI Duet System (Magenta RealTime 2) 啟動")
    log.info("  Backend: %s", backend)
    log.info("  Agent A: MRT2_Primo    (C Major)")
    log.info("  Agent B: MRT2_Secondo  (G Mixolydian)")
    log.info("  總時長: 60 秒，人類亂入模擬: T+15s")
    log.info("=" * 56)

    asyncio_stop = asyncio.Event()
    thread_stop = threading.Event()
    loop = asyncio.get_running_loop()

    human_thread = threading.Thread(
        target=simulate_human_input,
        args=(loop, thread_stop),
        kwargs={"delay_sec": 15.0, "note_count": 6},
        daemon=True,
        name="HumanSimulator",
    )

    bass_task: Optional[asyncio.Task[None]] = None
    duet_task: Optional[asyncio.Task[None]] = None

    try:
        human_thread.start()
        bass_task = asyncio.create_task(preset_bass_task(asyncio_stop), name="preset_bass")
        duet_task = asyncio.create_task(magenta_duet_task(asyncio_stop), name="magenta_duet")

        log.info("[Main] 系統開始運行 60 秒...")
        await asyncio.sleep(60.0)

    except KeyboardInterrupt:
        log.warning("[Main] Ctrl-C，強制停止")

    finally:
        log.info("[Main] 送出停止信號...")
        asyncio_stop.set()
        thread_stop.set()

        tasks_to_await = [t for t in (bass_task, duet_task) if t is not None]
        if tasks_to_await:
            await asyncio.gather(*tasks_to_await, return_exceptions=True)

        melody_out.close()
        harmony_out.close()
        bass_out.close()
        human_in.close()

        recorder.save("duet_output.mid")

        log.info("=" * 56)
        log.info("  AI Duet System 已完全關閉")
        log.info("  輸出 MIDI: duet_output.mid")
        log.info("=" * 56)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.exception("未預期錯誤: %s", e)
        raise
