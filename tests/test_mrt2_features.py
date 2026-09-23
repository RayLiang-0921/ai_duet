"""
test_mrt2_features.py — Text-to-Synth / Audio Cloning / Prompt Mixing 單元測試
"""
import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from duet_system import (
    MRT2Generator,
    C_MAJOR_PITCHES,
    DEFAULT_TEMP,
    MRT2_SAMPLE_RATE,
)


# ──────────────────────────────────────────────────────────────────
# Helper: 建立帶 Mock MRT2 的 Generator
# ──────────────────────────────────────────────────────────────────
def _make_mock_mrt2(**extra_attrs):
    """建立具備基本 API 的 mock MRT2 物件"""
    mock = MagicMock()
    mock.embed_style.return_value = np.random.randn(64).astype(np.float32)
    for k, v in extra_attrs.items():
        setattr(mock, k, v)
    return mock


def _make_gen(mrt2=None, **kwargs):
    return MRT2Generator(
        name="TestGen",
        scale=C_MAJOR_PITCHES,
        style_prompt="test",
        mrt2_system=mrt2,
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════
# Text-to-Synth
# ════════════════════════════════════════════════════════════════
class TestTextToSynth:
    def test_returns_none_without_mrt2(self):
        gen = _make_gen(mrt2=None)
        result = gen.text_to_synth("string ensemble")
        assert result is None

    def test_synthesize_from_text_called(self):
        """有 synthesize_from_text 時優先用它"""
        mock = _make_mock_mrt2()
        expected = np.ones(64, dtype=np.float32)
        mock.synthesize_from_text = MagicMock(return_value=expected)
        gen = _make_gen(mrt2=mock)

        result = gen.text_to_synth("warm analog bass")
        assert result is not None
        np.testing.assert_array_equal(result, expected)
        mock.synthesize_from_text.assert_called_once_with("warm analog bass")

    def test_fallback_to_embed_style(self):
        """沒有 synthesize_from_text 時 fallback 到 embed_style"""
        mock = _make_mock_mrt2()
        # 移除 synthesize_from_text 讓 hasattr 回傳 False
        del mock.synthesize_from_text
        expected = np.ones(64, dtype=np.float32)
        mock.embed_style.return_value = expected
        gen = _make_gen(mrt2=mock)

        result = gen.text_to_synth("bright pluck lead")
        assert result is not None
        # embed_style 會被呼叫兩次：__init__ + text_to_synth
        assert mock.embed_style.call_count == 2

    def test_updates_internal_state(self):
        mock = _make_mock_mrt2()
        mock.synthesize_from_text = MagicMock(return_value=np.zeros(64))
        gen = _make_gen(mrt2=mock)
        gen._state = "old_state"

        gen.text_to_synth("new sound")
        assert gen.style_prompt == "new sound"
        assert gen._state is None  # streaming state 重設
        assert gen._style_embedding is not None

    def test_empty_prompt_returns_none(self):
        mock = _make_mock_mrt2()
        gen = _make_gen(mrt2=mock)
        assert gen.text_to_synth("") is None
        assert gen.text_to_synth("   ") is None

    def test_none_prompt_returns_none(self):
        mock = _make_mock_mrt2()
        gen = _make_gen(mrt2=mock)
        # noinspection PyTypeChecker
        assert gen.text_to_synth(None) is None

    def test_exception_returns_none(self):
        mock = _make_mock_mrt2()
        mock.synthesize_from_text = MagicMock(side_effect=RuntimeError("GPU OOM"))
        gen = _make_gen(mrt2=mock)
        result = gen.text_to_synth("crash me")
        assert result is None

    def test_preserves_old_embedding_on_failure(self):
        mock = _make_mock_mrt2()
        init_embedding = mock.embed_style.return_value.copy()
        mock.synthesize_from_text = MagicMock(side_effect=RuntimeError("fail"))
        gen = _make_gen(mrt2=mock)

        old_embedding = gen._style_embedding.copy()
        gen.text_to_synth("will fail")
        # 失敗時不應修改 embedding (exception 在賦值前)
        np.testing.assert_array_equal(gen._style_embedding, old_embedding)


# ════════════════════════════════════════════════════════════════
# Audio Cloning
# ════════════════════════════════════════════════════════════════
class TestAudioCloning:
    def _make_audio(self, duration_sec=1.0, sr=MRT2_SAMPLE_RATE):
        """生成假的音頻 samples"""
        n = int(sr * duration_sec)
        return np.random.randn(n).astype(np.float32)

    def test_returns_none_without_mrt2(self):
        gen = _make_gen(mrt2=None)
        audio = self._make_audio()
        assert gen.clone_audio(audio) is None

    def test_clone_audio_called(self):
        mock = _make_mock_mrt2()
        expected = np.ones(64, dtype=np.float32)
        mock.clone_audio = MagicMock(return_value=expected)
        gen = _make_gen(mrt2=mock)

        audio = self._make_audio(duration_sec=2.0)
        result = gen.clone_audio(audio)
        assert result is not None
        np.testing.assert_array_equal(result, expected)
        mock.clone_audio.assert_called_once()

    def test_fallback_to_embed_audio(self):
        """沒有 clone_audio 時 fallback 到 embed_audio"""
        mock = _make_mock_mrt2()
        del mock.clone_audio
        expected = np.ones(64, dtype=np.float32)
        mock.embed_audio = MagicMock(return_value=expected)
        gen = _make_gen(mrt2=mock)

        audio = self._make_audio()
        result = gen.clone_audio(audio)
        assert result is not None
        mock.embed_audio.assert_called_once()

    def test_no_clone_or_embed_returns_none(self):
        """既沒有 clone_audio 也沒有 embed_audio"""
        mock = _make_mock_mrt2()
        del mock.clone_audio
        del mock.embed_audio
        gen = _make_gen(mrt2=mock)

        audio = self._make_audio()
        result = gen.clone_audio(audio)
        assert result is None

    def test_empty_audio_returns_none(self):
        mock = _make_mock_mrt2()
        mock.clone_audio = MagicMock()
        gen = _make_gen(mrt2=mock)
        assert gen.clone_audio(np.array([])) is None

    def test_none_audio_returns_none(self):
        mock = _make_mock_mrt2()
        gen = _make_gen(mrt2=mock)
        assert gen.clone_audio(None) is None

    def test_too_short_audio_returns_none(self):
        """音頻 < 0.1 秒應回傳 None"""
        mock = _make_mock_mrt2()
        mock.clone_audio = MagicMock()
        gen = _make_gen(mrt2=mock)
        short_audio = np.zeros(100, dtype=np.float32)  # 100 samples @ 48kHz ≈ 0.002s
        result = gen.clone_audio(short_audio)
        assert result is None
        mock.clone_audio.assert_not_called()

    def test_custom_sample_rate(self):
        mock = _make_mock_mrt2()
        expected = np.ones(64)
        mock.clone_audio = MagicMock(return_value=expected)
        gen = _make_gen(mrt2=mock)

        audio = self._make_audio(duration_sec=1.0, sr=44100)
        result = gen.clone_audio(audio, sample_rate=44100)
        assert result is not None
        _, kwargs = mock.clone_audio.call_args
        assert kwargs['sample_rate'] == 44100

    def test_updates_style_prompt(self):
        mock = _make_mock_mrt2()
        mock.clone_audio = MagicMock(return_value=np.zeros(64))
        gen = _make_gen(mrt2=mock)

        audio = self._make_audio(duration_sec=2.0)
        gen.clone_audio(audio)
        assert "cloned_audio" in gen.style_prompt
        assert "48000Hz" in gen.style_prompt

    def test_resets_streaming_state(self):
        mock = _make_mock_mrt2()
        mock.clone_audio = MagicMock(return_value=np.zeros(64))
        gen = _make_gen(mrt2=mock)
        gen._state = "old_state"

        gen.clone_audio(self._make_audio())
        assert gen._state is None

    def test_exception_returns_none(self):
        mock = _make_mock_mrt2()
        mock.clone_audio = MagicMock(side_effect=RuntimeError("bad audio"))
        gen = _make_gen(mrt2=mock)
        result = gen.clone_audio(self._make_audio())
        assert result is None


# ════════════════════════════════════════════════════════════════
# Prompt Mixing
# ════════════════════════════════════════════════════════════════
class TestPromptMixing:
    def _emb(self, val=1.0, size=64):
        return np.full(size, val, dtype=np.float32)

    def test_linear_interpolation_alpha_0(self):
        """alpha=0 → 完全使用 embedding_a"""
        gen = _make_gen(mrt2=None)
        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=0.0)
        assert result is not None
        np.testing.assert_allclose(result, a)

    def test_linear_interpolation_alpha_1(self):
        """alpha=1 → 完全使用 embedding_b"""
        gen = _make_gen(mrt2=None)
        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=1.0)
        assert result is not None
        np.testing.assert_allclose(result, b)

    def test_linear_interpolation_alpha_05(self):
        """alpha=0.5 → 均勻混合"""
        gen = _make_gen(mrt2=None)
        a, b = self._emb(0.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=0.5)
        assert result is not None
        np.testing.assert_allclose(result, self._emb(1.0))

    def test_arbitrary_alpha(self):
        gen = _make_gen(mrt2=None)
        a, b = self._emb(0.0), self._emb(10.0)
        result = gen.mix_prompts(a, b, alpha=0.3)
        np.testing.assert_allclose(result, self._emb(3.0))

    def test_uses_mrt2_mix_embeddings_when_available(self):
        """有 mix_embeddings 時優先用 MRT2 原生混合"""
        mock = _make_mock_mrt2()
        expected = np.ones(64, dtype=np.float32) * 99
        mock.mix_embeddings = MagicMock(return_value=expected)
        gen = _make_gen(mrt2=mock)

        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=0.7)
        np.testing.assert_array_equal(result, expected)
        mock.mix_embeddings.assert_called_once()

    def test_none_embedding_a_returns_none(self):
        gen = _make_gen(mrt2=None)
        result = gen.mix_prompts(None, self._emb())
        assert result is None

    def test_none_embedding_b_returns_none(self):
        gen = _make_gen(mrt2=None)
        result = gen.mix_prompts(self._emb(), None)
        assert result is None

    def test_shape_mismatch_returns_none(self):
        gen = _make_gen(mrt2=None)
        a = self._emb(size=64)
        b = self._emb(size=128)
        result = gen.mix_prompts(a, b)
        assert result is None

    def test_alpha_clamp_below_zero(self):
        """alpha < 0 應被 clamp 到 0"""
        gen = _make_gen(mrt2=None)
        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=-0.5)
        assert result is not None
        np.testing.assert_allclose(result, a)  # clamp 到 0 → 完全使用 a

    def test_alpha_clamp_above_one(self):
        """alpha > 1 應被 clamp 到 1"""
        gen = _make_gen(mrt2=None)
        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=1.5)
        assert result is not None
        np.testing.assert_allclose(result, b)  # clamp 到 1 → 完全使用 b

    def test_updates_internal_state(self):
        gen = _make_gen(mrt2=None)
        gen._state = "old"
        a, b = self._emb(1.0), self._emb(2.0)
        gen.mix_prompts(a, b, alpha=0.5)
        assert gen._state is None
        assert gen._style_embedding is not None

    def test_exception_returns_none(self):
        mock = _make_mock_mrt2()
        mock.mix_embeddings = MagicMock(side_effect=RuntimeError("math error"))
        gen = _make_gen(mrt2=mock)
        a, b = self._emb(1.0), self._emb(2.0)
        result = gen.mix_prompts(a, b, alpha=0.5)
        assert result is None

    def test_multidimensional_embeddings(self):
        """高維 embedding 也能正常混合"""
        gen = _make_gen(mrt2=None)
        a = np.zeros((4, 16), dtype=np.float32)
        b = np.ones((4, 16), dtype=np.float32)
        result = gen.mix_prompts(a, b, alpha=0.25)
        assert result is not None
        np.testing.assert_allclose(result, np.full((4, 16), 0.25))


# ════════════════════════════════════════════════════════════════
# 整合測試：三個功能串接
# ════════════════════════════════════════════════════════════════
class TestFeatureIntegration:
    def test_text_to_synth_then_generate_audio(self):
        """Text-to-Synth 設定音色後，能正常 generate_audio"""
        mock = _make_mock_mrt2()
        synth_emb = np.random.randn(64).astype(np.float32)
        mock.synthesize_from_text = MagicMock(return_value=synth_emb)
        mock_wav = MagicMock()
        mock_wav.samples = np.zeros(48000)
        mock.generate.return_value = (mock_wav, "state")

        gen = _make_gen(mrt2=mock)
        gen.text_to_synth("string ensemble")
        audio = gen.generate_audio(active_pitches=[60, 64, 67], duration_sec=1.0)

        assert audio is not None
        np.testing.assert_array_equal(gen._style_embedding, synth_emb)

    def test_clone_then_mix_with_text(self):
        """Audio Cloning + Text embedding → Prompt Mixing"""
        mock = _make_mock_mrt2()
        text_emb = np.ones(64, dtype=np.float32)
        audio_emb = np.ones(64, dtype=np.float32) * 2
        mock.embed_style.return_value = text_emb
        mock.clone_audio = MagicMock(return_value=audio_emb)
        # 刪除自動生成的 mix_embeddings，確保走 fallback 線性插值
        del mock.mix_embeddings

        gen = _make_gen(mrt2=mock)

        # 先用文字取得 embedding
        text_result = gen._style_embedding.copy()

        # 再 clone audio
        audio_samples = np.random.randn(48000).astype(np.float32)
        clone_result = gen.clone_audio(audio_samples)

        # 混合兩者
        mixed = gen.mix_prompts(text_result, clone_result, alpha=0.5)
        assert mixed is not None
        np.testing.assert_allclose(mixed, np.full(64, 1.5))

    def test_sequential_text_to_synth_updates(self):
        """連續多次 Text-to-Synth 切換音色"""
        mock = _make_mock_mrt2()
        gen = _make_gen(mrt2=mock)

        prompts = ["piano", "guitar", "strings", "synth pad"]
        for p in prompts:
            emb = np.random.randn(64).astype(np.float32)
            mock.synthesize_from_text = MagicMock(return_value=emb)
            result = gen.text_to_synth(p)
            assert result is not None
            assert gen.style_prompt == p
            np.testing.assert_array_equal(gen._style_embedding, emb)

    def test_mix_chain(self):
        """連續 Prompt Mixing：A+B → C, C+D → E"""
        gen = _make_gen(mrt2=None)
        a = np.full(64, 0.0, dtype=np.float32)
        b = np.full(64, 4.0, dtype=np.float32)
        c = gen.mix_prompts(a, b, alpha=0.5)  # → 2.0
        np.testing.assert_allclose(c, np.full(64, 2.0))

        d = np.full(64, 6.0, dtype=np.float32)
        e = gen.mix_prompts(c, d, alpha=0.5)  # → 4.0
        np.testing.assert_allclose(e, np.full(64, 4.0))
