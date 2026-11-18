"""MRT2：把五個 agent 合成一批在 GPU 上生成，以及風格提示詞（文字 → token）。"""
import hashlib
import os
import threading

import numpy as np


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


class StyleBank:
    """文字 → MRT2 風格 token（MusicCoCa 在 CPU 上跑，不搶 GPU；一句約 0.5 秒，開機後第一句要多 18 秒載入）。
    embedding 存在硬碟 ~/.cache/ai_duet/style，第二次啟動不用重算。兩句之間的混合 = embedding 加權平均後再轉 token。
    生成執行緒只用 get() 拿已經算好的；計算（warm）都在背景或開機時做，文字模型一次只給一個執行緒用。"""

    def __init__(self, mrt) -> None:
        self.mrt, self.emb, self.tok, self.lock = mrt, {}, {}, threading.Lock()
        self.dir = os.path.expanduser("~/.cache/ai_duet/style")
        os.makedirs(self.dir, exist_ok=True)

    def _embedding(self, text: str) -> np.ndarray:
        if text not in self.emb:
            path = os.path.join(self.dir, hashlib.sha1(text.encode()).hexdigest() + ".npy")
            if not os.path.exists(path):
                np.save(path, np.asarray(self.mrt.embed_style(text)))
            self.emb[text] = np.load(path)
        return self.emb[text]

    def warm(self, text: str, other: "str | None" = None, w: float = 0.0) -> list:
        """算好並記住 text（或 text 往 other 混合 w）的 token。"""
        key = (text, other, w)
        if key not in self.tok:
            with self.lock:
                e = self._embedding(text)
                if other is not None:
                    e = (1 - w) * e + w * self._embedding(other)
                self.tok[key] = list(self.mrt.tokenize_style(e))
        return self.tok[key]

    def load_encoder(self) -> None:
        """載入文字模型（第一次轉換要 18 秒）。風格都在硬碟快取時，開機不會用到它；要是拖到演出中第一次打字才載入，
        會卡住生成好幾秒、造成斷音，所以開機時在背景先載好。"""
        with self.lock:
            self.mrt.embed_style("warm up")

    def get(self, text: str, other: "str | None" = None, w: float = 0.0) -> "list | None":
        """已經算好的 token；還沒算好就回傳 None（不等）。"""
        return self.tok.get((text, other, w))
