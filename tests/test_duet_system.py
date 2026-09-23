"""
test_duet_system.py — duet_system 模組的單元測試
"""
import asyncio
import threading
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duet_system import (
    MRT2Generator,
    _midi_to_pianoroll_tokens,
    _drum_tokens_from_pattern,
    _pitch_name,
    C_MAJOR_PITCHES,
    G_MIXO_PITCHES,
    VOCAB_SIZE,
    SEQUENCE_LEN,
    DEFAULT_TEMP,
    MRT2_NUM_RESERVED,
    MRT2_PIANOROLL_SIZE,
    MidiRecorder,
)
from midi_core import MidiNote, MidiBuffer, SystemState


# ════════════════════════════════════════════
# _pitch_name
# ════════════════════════════════════════════
class TestPitchName:
    def test_middle_c(self):
        assert _pitch_name(60) == "C4"

    def test_a4(self):
        assert _pitch_name(69) == "A4"

    def test_sharp(self):
        assert _pitch_name(61) == "C#4"

    def test_low_c(self):
        assert _pitch_name(36) == "C2"

    def test_high_c(self):
        assert _pitch_name(84) == "C6"


# ════════════════════════════════════════════
# _midi_to_pianoroll_tokens
# ════════════════════════════════════════════
class TestMidiToPianorollTokens:
    def test_empty_pitches(self):
        tokens = _midi_to_pianoroll_tokens([])
        assert len(tokens) == MRT2_PIANOROLL_SIZE
        assert all(t == MRT2_NUM_RESERVED for t in tokens)  # 全部 off

    def test_active_pitch_sustain(self):
        tokens = _midi_to_pianoroll_tokens([60], onset_pitches=[])
        assert tokens[60] == MRT2_NUM_RESERVED + 1  # sustain

    def test_onset_pitch(self):
        tokens = _midi_to_pianoroll_tokens([60], onset_pitches=[60])
        assert tokens[60] == MRT2_NUM_RESERVED + 2  # onset

    def test_mixed_active_and_onset(self):
        tokens = _midi_to_pianoroll_tokens([60, 64, 67], onset_pitches=[64])
        assert tokens[60] == MRT2_NUM_RESERVED + 1  # sustain
        assert tokens[64] == MRT2_NUM_RESERVED + 2  # onset
        assert tokens[67] == MRT2_NUM_RESERVED + 1  # sustain
        assert tokens[62] == MRT2_NUM_RESERVED + 0  # off (not active)

    def test_out_of_range_pitch_ignored(self):
        tokens = _midi_to_pianoroll_tokens([128, -1, 200])
        assert all(t == MRT2_NUM_RESERVED for t in tokens)

    def test_boundary_pitches(self):
        tokens = _midi_to_pianoroll_tokens([0, 127], onset_pitches=[0])
        assert tokens[0] == MRT2_NUM_RESERVED + 2   # onset
        assert tokens[127] == MRT2_NUM_RESERVED + 1  # sustain


# ════════════════════════════════════════════
# _drum_tokens_from_pattern
# ════════════════════════════════════════════
class TestDrumTokens:
    def test_no_hit(self):
        tokens = _drum_tokens_from_pattern(has_kick=False, has_snare=False)
        assert tokens == [MRT2_NUM_RESERVED + 0]

    def test_kick(self):
        tokens = _drum_tokens_from_pattern(has_kick=True)
        assert tokens == [MRT2_NUM_RESERVED + 1]

    def test_snare(self):
        tokens = _drum_tokens_from_pattern(has_snare=True)
        assert tokens == [MRT2_NUM_RESERVED + 1]

    def test_both(self):
        tokens = _drum_tokens_from_pattern(has_kick=True, has_snare=True)
        assert tokens == [MRT2_NUM_RESERVED + 1]


# ════════════════════════════════════════════
# MRT2Generator (Fallback 模式，無 MRT2)
# ════════════════════════════════════════════
class TestMRT2GeneratorFallback:
    def _make_gen(self, scale=None):
        return MRT2Generator(
            name="TestGen",
            scale=scale or C_MAJOR_PITCHES,
            style_prompt="test",
            mrt2_system=None,
        )

    def test_has_mrt2_false(self):
        gen = self._make_gen()
        assert gen.has_mrt2 is False

    def test_generate_returns_correct_length(self):
        gen = self._make_gen()
        result = gen.generate(primer=[60], length=8)
        assert len(result) == 8

    def test_generate_pitches_in_range(self):
        gen = self._make_gen()
        for _ in range(10):
            result = gen.generate(primer=[60], length=16)
            for p in result:
                assert 48 <= p <= 84, f"Pitch {p} out of range [48, 84]"

    def test_generate_with_empty_primer(self):
        gen = self._make_gen()
        result = gen.generate(primer=[], length=4)
        assert len(result) == 4

    def test_generate_with_out_of_scale_primer(self):
        gen = self._make_gen()
        result = gen.generate(primer=[61], length=4)  # C#4 不在 C major 裡
        assert len(result) == 4

    def test_generate_audio_returns_none_without_mrt2(self):
        gen = self._make_gen()
        result = gen.generate_audio(active_pitches=[60], duration_sec=1.0)
        assert result is None

    def test_reset_primer(self):
        gen = self._make_gen()
        gen.generate(primer=[60], length=4)
        gen.reset_primer([72, 74, 76])
        assert gen._last_pitch == 76
        assert gen._repeat_count == 0
        assert gen._temperature == DEFAULT_TEMP

    def test_update_style_no_crash_without_mrt2(self):
        gen = self._make_gen()
        gen.update_style("new style")  # 不應 crash

    def test_temperature_custom(self):
        gen = self._make_gen()
        gen.generate(primer=[60], length=4, temperature=2.0)
        assert gen._temperature == 2.0

    def test_generate_different_scales(self):
        gen_c = self._make_gen(scale=C_MAJOR_PITCHES)
        gen_g = self._make_gen(scale=G_MIXO_PITCHES)
        # 產生足夠多的音符來確認音階約束
        result_c = gen_c.generate(primer=[60], length=100)
        result_g = gen_g.generate(primer=[60], length=100)
        # 至少應該有不同的音符分佈
        assert len(set(result_c)) > 1
        assert len(set(result_g)) > 1


# ════════════════════════════════════════════
# MRT2Generator (Mock MRT2)
# ════════════════════════════════════════════
class TestMRT2GeneratorWithMock:
    def test_has_mrt2_true(self):
        mock_mrt = MagicMock()
        mock_mrt.embed_style.return_value = np.zeros(64)
        gen = MRT2Generator(
            name="MockGen", scale=C_MAJOR_PITCHES,
            style_prompt="test", mrt2_system=mock_mrt,
        )
        assert gen.has_mrt2 is True
        mock_mrt.embed_style.assert_called_once_with("test")

    def test_generate_audio_calls_mrt2(self):
        mock_mrt = MagicMock()
        mock_mrt.embed_style.return_value = np.zeros(64)
        mock_waveform = MagicMock()
        mock_waveform.samples = np.zeros(48000)
        mock_mrt.generate.return_value = (mock_waveform, "state")

        gen = MRT2Generator(
            name="MockGen", scale=C_MAJOR_PITCHES,
            style_prompt="test", mrt2_system=mock_mrt,
        )
        result = gen.generate_audio(active_pitches=[60, 64], onset_pitches=[60], duration_sec=1.0)
        assert result is not None
        assert result.shape == (2, 48000)  # mono is normalized to stereo
        mock_mrt.generate.assert_called_once()

    def test_generate_audio_failure_returns_none(self):
        mock_mrt = MagicMock()
        mock_mrt.embed_style.return_value = np.zeros(64)
        mock_mrt.generate.side_effect = RuntimeError("GPU OOM")

        gen = MRT2Generator(
            name="MockGen", scale=C_MAJOR_PITCHES,
            style_prompt="test", mrt2_system=mock_mrt,
        )
        result = gen.generate_audio(active_pitches=[60], duration_sec=1.0)
        assert result is None

    def test_update_style_resets_state(self):
        mock_mrt = MagicMock()
        mock_mrt.embed_style.return_value = np.zeros(64)

        gen = MRT2Generator(
            name="MockGen", scale=C_MAJOR_PITCHES,
            style_prompt="test", mrt2_system=mock_mrt,
        )
        gen._state = "old_state"
        gen.update_style("jazz")
        assert gen._state is None  # state 應被重設
        assert mock_mrt.embed_style.call_count == 2  # init + update


# ════════════════════════════════════════════
# MidiRecorder
# ════════════════════════════════════════════
class TestMidiRecorder:
    def test_record_events(self):
        rec = MidiRecorder()
        n1 = MidiNote(channel=0, pitch=60, velocity=85, source="AI_Melody")
        n2 = MidiNote(channel=1, pitch=64, velocity=70, source="AI_Harmony")
        rec.record(n1)
        rec.record(n2)
        assert len(rec._events) == 2

    def test_record_preserves_source(self):
        rec = MidiRecorder()
        n = MidiNote(channel=2, pitch=36, velocity=80, source="Preset_Bass")
        rec.record(n)
        assert rec._events[0][1] == "Preset_Bass"

    def test_record_thread_safety(self):
        rec = MidiRecorder()
        errors = []

        def worker(source):
            try:
                for i in range(50):
                    rec.record(MidiNote(channel=0, pitch=60+i%12, velocity=80, source=source))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"src_{i}",)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(rec._events) == 200


# ════════════════════════════════════════════
# 音階常數驗證
# ════════════════════════════════════════════
class TestScaleConstants:
    def test_c_major_pitches_sorted(self):
        assert C_MAJOR_PITCHES == sorted(C_MAJOR_PITCHES)

    def test_g_mixo_pitches_sorted(self):
        assert G_MIXO_PITCHES == sorted(G_MIXO_PITCHES)

    def test_c_major_in_range(self):
        for p in C_MAJOR_PITCHES:
            assert 0 <= p <= 127

    def test_g_mixo_in_range(self):
        for p in G_MIXO_PITCHES:
            assert 0 <= p <= 127

    def test_c_major_contains_middle_c(self):
        assert 60 in C_MAJOR_PITCHES

    def test_g_mixo_contains_g(self):
        assert 55 in G_MIXO_PITCHES  # G3
