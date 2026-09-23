"""
test_train_melody.py — train_melody 模組的單元測試
"""
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_melody import (
    LSTMMusicModel,
    _quantize_to_steps,
    _detect_chord_root,
    MelodyDataset,
    VOCAB_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    PITCH_LOW,
    PITCH_HIGH,
    REST_TOKEN,
)
import torch


# ════════════════════════════════════════════
# _quantize_to_steps
# ════════════════════════════════════════════
class TestQuantizeToSteps:
    def test_empty_notes(self):
        assert _quantize_to_steps([], step_sec=0.125) == []

    def test_single_note(self):
        notes = [(0.0, 60, 0.5)]  # (start, pitch, duration)
        result = _quantize_to_steps(notes, step_sec=0.125)
        assert len(result) > 0
        assert 60 in result

    def test_pitch_clamped_to_range(self):
        notes = [(0.0, 20, 0.5)]  # pitch 20 < PITCH_LOW=36
        result = _quantize_to_steps(notes, step_sec=0.125)
        for p in result:
            if p != REST_TOKEN:
                assert PITCH_LOW <= p <= PITCH_HIGH

    def test_high_pitch_clamped(self):
        notes = [(0.0, 100, 0.5)]  # pitch 100 > PITCH_HIGH=84
        result = _quantize_to_steps(notes, step_sec=0.125)
        for p in result:
            if p != REST_TOKEN:
                assert PITCH_LOW <= p <= PITCH_HIGH

    def test_rest_token_in_gap(self):
        """兩個音符之間有空白，應該填 REST_TOKEN"""
        notes = [(0.0, 60, 0.125), (1.0, 64, 0.125)]
        result = _quantize_to_steps(notes, step_sec=0.125)
        assert REST_TOKEN in result

    def test_overlapping_notes_use_highest(self):
        """同時出現多個音時，用最高音"""
        notes = [(0.0, 48, 0.5), (0.0, 60, 0.5), (0.0, 72, 0.5)]
        result = _quantize_to_steps(notes, step_sec=0.125)
        # 最高音 72 應出現
        non_rest = [p for p in result if p != REST_TOKEN]
        assert 72 in non_rest


# ════════════════════════════════════════════
# _detect_chord_root
# ════════════════════════════════════════════
class TestDetectChordRoot:
    def test_empty_window(self):
        result = _detect_chord_root([], scale_root=0)
        assert result == 0  # fallback to scale_root % 12

    def test_all_rest_tokens(self):
        result = _detect_chord_root([REST_TOKEN, REST_TOKEN], scale_root=7)
        assert result == 7  # fallback

    def test_single_pitch_class(self):
        # 全部是 C (pitch class 0)
        result = _detect_chord_root([60, 48, 72], scale_root=0)
        assert result == 0

    def test_majority_vote(self):
        # G (7) 出現最多次
        pitches = [55, 67, 55, 60]  # G, G, G, C → G wins
        result = _detect_chord_root(pitches, scale_root=0)
        assert result == 7  # G


# ════════════════════════════════════════════
# LSTMMusicModel
# ════════════════════════════════════════════
class TestLSTMMusicModel:
    def test_output_shape_pitch_only(self):
        model = LSTMMusicModel(chord_conditioned=False)
        x = torch.randint(0, VOCAB_SIZE, (2, 16))  # batch=2, seq=16
        logits, hx = model(x)
        assert logits.shape == (2, 16, VOCAB_SIZE)

    def test_output_shape_chord_conditioned(self):
        model = LSTMMusicModel(chord_conditioned=True)
        x = torch.randint(0, VOCAB_SIZE, (2, 16))
        c = torch.randint(0, 12, (2, 16))
        logits, hx = model(x, chord=c)
        assert logits.shape == (2, 16, VOCAB_SIZE)

    def test_hidden_state_returned(self):
        model = LSTMMusicModel()
        x = torch.randint(0, VOCAB_SIZE, (1, 8))
        logits, hx = model(x)
        assert hx is not None
        h, c = hx
        assert h.shape == (NUM_LAYERS, 1, HIDDEN_SIZE)
        assert c.shape == (NUM_LAYERS, 1, HIDDEN_SIZE)

    def test_hidden_state_continuity(self):
        """隱藏狀態可以跨呼叫傳遞"""
        model = LSTMMusicModel()
        x1 = torch.randint(0, VOCAB_SIZE, (1, 8))
        _, hx1 = model(x1)
        x2 = torch.randint(0, VOCAB_SIZE, (1, 8))
        logits2, hx2 = model(x2, hx=hx1)
        assert logits2.shape == (1, 8, VOCAB_SIZE)

    def test_chord_conditioned_uses_chord_embed(self):
        model = LSTMMusicModel(chord_conditioned=True)
        assert hasattr(model, 'chord_embed')
        assert model.chord_conditioned is True

    def test_non_chord_has_no_chord_embed(self):
        model = LSTMMusicModel(chord_conditioned=False)
        assert not hasattr(model, 'chord_embed')

    def test_gradient_flows(self):
        """確認梯度可以正常反向傳播"""
        model = LSTMMusicModel()
        x = torch.randint(0, VOCAB_SIZE, (2, 8))
        y = torch.randint(0, VOCAB_SIZE, (2, 8))
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), y.reshape(-1)
        )
        loss.backward()
        # 確認至少一個參數有梯度
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ════════════════════════════════════════════
# MelodyDataset
# ════════════════════════════════════════════
class TestMelodyDataset:
    def test_basic_dataset(self):
        pitches = [list(range(60, 80))]  # 20 個音符
        chords = [[0] * 20]
        ds = MelodyDataset(pitches, chords, seq_len=8, use_chord=False)
        assert len(ds) == 20 - 8  # 12 個 sample

    def test_sample_shape(self):
        pitches = [list(range(60, 80))]
        chords = [[0] * 20]
        ds = MelodyDataset(pitches, chords, seq_len=8)
        x, y, c = ds[0]
        assert x.shape == (8,)
        assert y.shape == (8,)
        assert c.shape == (8,)

    def test_xy_offset_by_one(self):
        """x 和 y 應該差一個位移"""
        pitches = [list(range(60, 80))]
        chords = [[0] * 20]
        ds = MelodyDataset(pitches, chords, seq_len=4)
        x, y, c = ds[0]
        # x = [60,61,62,63], y = [61,62,63,64]
        assert x.tolist() == [60, 61, 62, 63]
        assert y.tolist() == [61, 62, 63, 64]

    def test_empty_dataset(self):
        ds = MelodyDataset([], [], seq_len=8)
        assert len(ds) == 0

    def test_multiple_sequences(self):
        pitches = [list(range(60, 70)), list(range(70, 80))]
        chords = [[0] * 10, [0] * 10]
        ds = MelodyDataset(pitches, chords, seq_len=4)
        # 每個 seq 產生 10-4=6 個，共 12
        assert len(ds) == 12

    def test_chord_conditioned_dataset(self):
        pitches = [list(range(60, 80))]
        chords = [list(range(12)) + list(range(8))]
        ds = MelodyDataset(pitches, chords, seq_len=4, use_chord=True)
        x, y, c = ds[0]
        assert c.tolist() == [0, 1, 2, 3]
