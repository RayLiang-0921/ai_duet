"""
groove_server.py — 鼓手的「人味」：用 GrooVAE（Magenta 的 groovae_2bar_add_closed_hh，真人鼓手演奏資料訓練）
把鼓機排好的 2 小節 kick / snare / open hat 變成真人演奏：補上 hi-hat、每一下的力度、微小的提前或延後。
由 band_mrt2.py 自動啟動，跑在自己的環境（~/groove_venv，TensorFlow，只用 CPU，顯卡留給 MRT2）。
協定：每行一個 JSON 請求 {"bpm": 75, "notes": [[16 分音符位置 0–31, 音高], ...]} → 回一行 [[秒, 音高, 力度], ...]
"""
import glob
import json
import os
import socket

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from magenta.models.music_vae import TrainedModel, configs  # noqa: E402
from note_seq.protobuf import music_pb2  # noqa: E402

MODEL = "groovae_2bar_add_closed_hh"
CKPT_DIR = os.path.expanduser("~/groovae")  # 解壓後的 checkpoint 放這裡（見 README）
TCP = ("127.0.0.1", 47602)


def play(model, cfg, bpm: float, notes: list) -> list:
    step = 60 / bpm / 4
    ns = music_pb2.NoteSequence(ticks_per_quarter=220)
    ns.tempos.add(qpm=bpm)
    for s, pitch in notes:
        ns.notes.add(pitch=pitch, velocity=80, start_time=s * step, end_time=s * step + 0.1, is_drum=True)
    ns.total_time = 32 * step
    _, mu, _ = model.encode([ns])
    out = model.decode(mu, length=cfg.hparams.max_seq_len)[0]
    k = (out.tempos[0].qpm if out.tempos else 120) / bpm  # decode() 回傳的是 120 qpm 的時間
    return [[round(n.start_time * k, 4), n.pitch, n.velocity] for n in out.notes]


def main() -> None:
    cfg = configs.CONFIG_MAP[MODEL]
    index = glob.glob(f"{CKPT_DIR}/{MODEL}/**/*.index", recursive=True)[0]
    model = TrainedModel(cfg, batch_size=1, checkpoint_dir_or_path=index[:-len(".index")])
    play(model, cfg, 120, [[0, 36], [4, 38], [16, 36], [28, 38]])  # 先跑一次暖機，第一次請求才不會等好幾秒（輸入要有第 2 小節的音）
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(TCP)
    server.listen()
    print("[鼓手] GrooVAE 就緒", flush=True)
    while True:
        conn, _ = server.accept()
        with conn, conn.makefile("rw") as io:
            for line in io:
                req = json.loads(line)
                io.write(json.dumps(play(model, cfg, req["bpm"], req["notes"])) + "\n")
                io.flush()


if __name__ == "__main__":
    main()
