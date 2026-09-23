from magenta_rt import utils
from magenta_rt import MagentaRT2Jax


mrt = MagentaRT2Jax(size="mrt2_small")
embedding = mrt.embed_style("dj scratch, 140bpm, trap with piano sample, G minor")


'''
    Args:
      conditioning: Dictionary mapping TokensConfig.key strings to their values
        Values can be lists of integers (tokens) or raw embeddings (e.g. Style)
      cfg_scales: Optional dictionary to override default CFG scales for this call.
          Example: {'musiccoca': 5.0, 'notes': 3.0}
      temperature: Sampling temperature. None falls back to
        ``self.temperature``.
      top_k: Top-k sampling threshold. None falls back to ``self.top_k``.
      frames: Number of frames to generate (25 frames = 1 second at 48kHz).
      state: Streaming state from a previous call. If None, a fresh state is
          created.
    
    Returns:
      (waveform, state) — a Waveform at 48kHz stereo, and the updated state
      for continuation.
'''

wav, state = mrt.generate(state=None, conditioning={"mulan_tokens_25hz": embedding}, frames=100) 
wav.write("output.wav")
#沒有使用GPU
#有沒有機會測試base