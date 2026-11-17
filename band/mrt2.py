"""MRT2：把五個 agent 合成一批在 GPU 上生成，以及風格提示詞的編碼。"""
import numpy as np

from band.settings import AGENTS, BLENDS, STYLES


class BatchedMRT2:
    """把多個 agent 合成一批，一次在 GPU 上生成（4 個依序生成要 139 ms/幀，合成一批約 38 ms）。"""

    def __init__(self, mrt, batch: int) -> None:
        import jax
        import jax.numpy as jnp
        from magenta_rt.jax import system  # noqa: F401  載入 MRT2 內附的 sequence_layers
        import sequence_layers.jax as sl

        self.jax, self.jnp, self.mrt, self.batch = jax, jnp, mrt, batch
        self._rows = {}  # frame_row 的快取
        sampler = mrt._sampler
        spec = jax.ShapeDtypeStruct([mrt._num_channels], jnp.int32)
        rngs = {"params": jax.random.PRNGKey(42), "random": jax.random.PRNGKey(0)}
        _, const = mrt._build_conditioning({})
        const = {k: jnp.repeat(v, batch, axis=0) for k, v in const.items()}

        self._init = jax.jit(lambda p: sampler.apply(p, batch, spec, constants={}, training=False,
                                                      rngs=rngs, method=sampler.get_initial_state))

        def run(p, xs, s):  # xs: (幀數, batch, 1, C)，在 GPU 上連續算完這幾幀
            def body(s, x):
                out, s, _ = sampler.apply(p, x=sl.Sequence.from_values(x), state=s, constants=const,
                                          training=False, rngs=rngs, method=sampler.step_with_emits)
                return s, out.values
            s, outs = jax.lax.scan(body, s, xs)
            return outs, s

        self._run = jax.jit(run, donate_argnums=(2,))

    def new_state(self, seed: int):
        state = self._init(self.mrt._params)
        leaves, tree = self.jax.tree_util.tree_flatten(state)
        # 狀態裡唯一的亂數金鑰（uint32，形狀 [batch, 2]）：每個 agent 各給一個種子
        i = next(i for i, l in enumerate(leaves) if l.dtype == self.jnp.uint32 and l.shape == (self.batch, 2))
        leaves[i] = self.jnp.stack([self.jax.random.PRNGKey(seed + k) for k in range(self.batch)])
        return self.jax.tree_util.tree_unflatten(tree, leaves)

    def frame_row(self, conds: list) -> np.ndarray:
        """一幀所有 agent 的條件 → (batch, 1, C)。同樣的條件只算一次（每小節大多重複），省下約 5 ms/幀。"""
        rows = []
        for c, cfg in conds:
            key = tuple((k, tuple(v)) for k, v in sorted(c.items())) + tuple(sorted(cfg.items()))
            if key not in self._rows:
                if len(self._rows) > 20000:
                    self._rows.clear()
                self._rows[key] = np.asarray(self.mrt._build_conditioning(c, cfg)[0].values)
            rows.append(self._rows[key])
        return np.concatenate(rows)

    def generate(self, rows: list, state) -> tuple:
        """rows：每幀一個 frame_row。回傳 (幀數, batch, 1920, 2) 的 float32 音訊與新的狀態。"""
        out, state = self._run(self.mrt._params, self.jnp.asarray(np.stack(rows)), state)
        return np.asarray(self.jax.device_get(out)).astype(np.float32) / 32768.0, state


def style_tokens(mrt) -> dict:
    """每種風格各 agent 的提示詞 → 風格 token；另外兩種風格之間 BLENDS 比例的混合（embedding 加權平均），切換前用。
    調性靠和弦音符維持，提示詞寫不寫調名測起來沒差，所以不寫。回傳 {風格: {agent: tokens}, (從, 到, 比例): {agent: tokens}}。"""
    emb = {s: {a: np.asarray(mrt.embed_style(f"{p}, {STYLES[s].get('prompt', s)}")) for a, p in STYLES[s]["prompts"].items()}
           for s in STYLES}
    tok = lambda e: list(mrt.tokenize_style(e))
    tokens = {s: {a: tok(e) for a, e in es.items()} for s, es in emb.items()}
    for s1 in STYLES:
        for s2 in STYLES:
            for w in BLENDS if s1 != s2 else ():
                tokens[(s1, s2, w)] = {a: tok((1 - w) * emb[s1][a] + w * emb[s2][a]) for a in AGENTS}
    return tokens
