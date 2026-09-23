"""
train_melody.py — 用 MIDI 資料訓練 LSTMMusicModel (Legacy PyTorch Fallback)
=============================================================================

注意：此腳本為 PyTorch LSTM fallback 模式的訓練工具。
      若使用 Magenta RealTime 2 (MRT2)，不需要此腳本 —
      MRT2 使用 Google 提供的 pretrained weights (mrt2_small / mrt2_base)。

      此訓練器僅在以下情況使用：
      1. MRT2 無法運行（缺少 JAX / 硬體不支援）時的 fallback
      2. 需要自訂微調 LSTM 模型時

用法:
    python train_melody.py <midi_dir> [--output checkpoints/melody_rnn.pt] [--epochs 30] [--chord]

參數:
    midi_dir   : 存放 .mid / .midi 的資料夾 (會遞迴搜尋)
    --output   : checkpoint 輸出路徑 (預設 checkpoints/melody_rnn.pt)
    --epochs   : 訓練 epoch 數 (預設 30)
    --batch    : batch size (預設 64)
    --lr       : 學習率 (預設 0.001)
    --seq_len  : 輸入序列長度 (預設 16)
    --chord    : 啟用 chord-conditioned 模式 (輸入額外帶 chord root 資訊)

MIDI 資料推薦來源 (嘻哈):
    - https://www.midiworld.com/hip-hop.htm
    - https://freemidi.org/genre-hip-hop
    - Lakh MIDI Dataset (lmd_full): https://colinraffel.com/projects/lmd/
      下載後用 genre tag 過濾: grep 'hip hop\\|rap\\|trap' 之類的 metadata
    - 自己從 DAW 匯出人工製作的 MIDI 也 OK，數量不需太多，50+ 首就有效果

執行完後，把產生的 .pt 路徑設到 duet_system.py 的 fallback generator。
"""

import os
import sys
import glob
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    import pretty_midi
except ImportError:
    print("[錯誤] 請先安裝: pip install pretty-midi")
    sys.exit(1)

# ─────────────────────────────────────────────
# 超參數
# ─────────────────────────────────────────────
VOCAB_SIZE  = 129   # 0-127 pitch + 128 = REST token
HIDDEN_SIZE = 256
NUM_LAYERS  = 2
PITCH_LOW   = 36    # C2
PITCH_HIGH  = 84    # C6
REST_TOKEN  = 128

# ─────────────────────────────────────────────
# 模型 (與 duet_system.py 相同骨幹，多支援 chord conditioning)
# ─────────────────────────────────────────────
class LSTMMusicModel(nn.Module):
    """
    chord_conditioned=False: 輸入 pitch sequence，輸出 next-pitch logits
    chord_conditioned=True:  輸入 [pitch, chord_root(0-11)] concat embedding，
                             讓模型學「這個和弦下應該接什麼音」
    """
    def __init__(self, vocab=VOCAB_SIZE, hidden=HIDDEN_SIZE, layers=NUM_LAYERS,
                 chord_conditioned=False):
        super().__init__()
        self.chord_conditioned = chord_conditioned
        self.embed = nn.Embedding(vocab, hidden)

        if chord_conditioned:
            self.chord_embed = nn.Embedding(12, hidden // 4)  # 12 chord roots
            lstm_input = hidden + hidden // 4
        else:
            lstm_input = hidden

        self.lstm = nn.LSTM(lstm_input, hidden, layers, batch_first=True)
        self.fc   = nn.Linear(hidden, vocab)

    def forward(self, x, chord=None, hx=None):
        emb = self.embed(x)                          # (B, T, H)
        if self.chord_conditioned and chord is not None:
            c = self.chord_embed(chord)              # (B, T, H/4)
            emb = torch.cat([emb, c], dim=-1)        # (B, T, H + H/4)
        out, hx_out = self.lstm(emb, hx)
        return self.fc(out), hx_out                  # logits: (B, T, vocab)


# ─────────────────────────────────────────────
# MIDI 解析
# ─────────────────────────────────────────────
def _quantize_to_steps(notes, step_sec):
    """把 [(start, pitch, duration), ...] 量化成 step 序列（最高音 = 旋律）"""
    if not notes:
        return []
    max_time = max(n[0] + n[2] for n in notes)
    n_steps  = int(max_time / step_sec) + 2
    grid = [[] for _ in range(n_steps)]

    for start, pitch, dur in notes:
        s = int(round(start / step_sec))
        e = max(s + 1, int(round((start + dur) / step_sec)))
        for step in range(min(s, n_steps - 1), min(e, n_steps)):
            grid[step].append(pitch)

    seq = []
    prev = None
    for slot in grid:
        if slot:
            p = max(slot)             # 最高音 = 旋律
            p = max(PITCH_LOW, min(PITCH_HIGH, p))
            seq.append(p)
            prev = p
        else:
            seq.append(REST_TOKEN)   # 休止符
    return seq


def _detect_chord_root(notes_in_window, scale_root):
    """
    簡易和弦根音偵測：在一個視窗內統計出現最多的音級，
    以音程與 scale_root 的關係推算最可能的根音。
    回傳 0-11 的 chord root (相對於 C)。
    """
    if not notes_in_window:
        return scale_root % 12
    from collections import Counter
    pcs = Counter(p % 12 for p in notes_in_window if p != REST_TOKEN)
    if not pcs:
        return scale_root % 12
    return pcs.most_common(1)[0][0]


def extract_sequences(filepath, bpm=120, steps_per_beat=4, use_chord=False):
    """
    從單一 MIDI 檔提取:
      - pitches: List[int]  (pitch 或 REST_TOKEN)
      - chords:  List[int]  (0-11 chord root，僅 use_chord=True 時有效)
    """
    try:
        pm = pretty_midi.PrettyMIDI(filepath)
    except Exception:
        return [], []

    # 用 MIDI 自帶 tempo 或 fallback
    try:
        tempo = float(pm.estimate_tempo())
    except Exception:
        tempo = float(bpm)
    step_sec = 60.0 / tempo / steps_per_beat

    # 合併所有非鼓軌的音符
    all_notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            all_notes.append((n.start, n.pitch, n.end - n.start))

    if not all_notes:
        return [], []

    pitches = _quantize_to_steps(all_notes, step_sec)

    if not use_chord:
        return pitches, []

    # Chord detection: 每 4 steps (1 beat) 偵測一次根音
    window = 4
    chords = []
    for i, p in enumerate(pitches):
        wstart = max(0, i - window // 2)
        wend   = min(len(pitches), i + window // 2)
        chords.append(_detect_chord_root(pitches[wstart:wend], 0))

    return pitches, chords


def load_dataset(midi_dir, use_chord=False, min_len=64, max_files=None):
    files = glob.glob(os.path.join(midi_dir, "**/*.mid"),  recursive=True)
    files += glob.glob(os.path.join(midi_dir, "**/*.midi"), recursive=True)
    random.shuffle(files)
    if max_files:
        files = files[:max_files]

    print(f"[資料] 找到 {len(files)} 個 MIDI 檔")
    all_pitch, all_chord = [], []
    for i, fp in enumerate(files):
        p, c = extract_sequences(fp, use_chord=use_chord)
        if len(p) >= min_len:
            all_pitch.append(p)
            all_chord.append(c)
        if (i + 1) % 50 == 0:
            print(f"  已處理 {i+1}/{len(files)}, 有效序列: {len(all_pitch)}")

    print(f"[資料] 有效序列: {len(all_pitch)} 首，共 {sum(len(s) for s in all_pitch)} 個 step")
    return all_pitch, all_chord


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class MelodyDataset(Dataset):
    def __init__(self, pitch_seqs, chord_seqs, seq_len=16, use_chord=False):
        self.samples = []
        self.use_chord = use_chord
        for pitches, chords in zip(pitch_seqs, chord_seqs):
            for i in range(len(pitches) - seq_len):
                x = pitches[i : i + seq_len]
                y = pitches[i + 1 : i + seq_len + 1]
                c = chords[i : i + seq_len] if use_chord and chords else [0] * seq_len
                self.samples.append((x, y, c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, c = self.samples[index]
        return (torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(c, dtype=torch.long))


# ─────────────────────────────────────────────
# 訓練
# ─────────────────────────────────────────────
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[訓練] 使用裝置: {device}")

    # 資料
    pitch_seqs, chord_seqs = load_dataset(
        args.midi_dir, use_chord=args.chord, max_files=args.max_files
    )
    if not pitch_seqs:
        print("[錯誤] 沒有找到足夠的 MIDI 資料，請確認路徑與檔案格式。")
        return

    dataset = MelodyDataset(pitch_seqs, chord_seqs, seq_len=args.seq_len, use_chord=args.chord)
    loader  = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    print(f"[訓練] 樣本數: {len(dataset)}, batch 數: {len(loader)}")

    # 模型
    model = LSTMMusicModel(chord_conditioned=args.chord).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 參數量: {total_params:,} ({('chord-conditioned' if args.chord else 'pitch-only')})")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=REST_TOKEN)  # 不計算休止符的 loss

    best_loss = float("inf")
    out_path  = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y, c in loader:
            x, y, c = x.to(device), y.to(device), c.to(device)
            optimizer.zero_grad()
            chord_in = c if args.chord else None
            logits, _ = model(x, chord=chord_in)           # (B, T, vocab)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg = total_loss / len(loader)
        lr  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:3d}/{args.epochs} | loss={avg:.4f} | lr={lr:.6f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), out_path)
            print(f"  ✓ 儲存最佳模型 → {out_path}")

    print(f"\n[完成] 最佳 loss={best_loss:.4f}，模型存在: {out_path}")
    print(f"\n[下一步] 把 checkpoint 路徑設入 duet_system.py:")
    print(f'  agent_a = MagentaCompatGenerator("MelodyRNN_Primo", ..., checkpoint_path="{out_path}")')


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="訓練 MelodyRNN LSTM on MIDI 資料集")
    parser.add_argument("midi_dir", help="MIDI 資料夾路徑（遞迴搜尋 .mid/.midi）")
    parser.add_argument("--output",    default="checkpoints/melody_rnn.pt", help="輸出 checkpoint 路徑")
    parser.add_argument("--epochs",    type=int,   default=30,    help="訓練 epoch 數")
    parser.add_argument("--batch",     type=int,   default=64,    help="Batch size")
    parser.add_argument("--lr",        type=float, default=1e-3,  help="學習率")
    parser.add_argument("--seq_len",   type=int,   default=16,    help="輸入序列長度")
    parser.add_argument("--max_files", type=int,   default=None,  help="最多讀幾個 MIDI（測試用）")
    parser.add_argument("--chord",     action="store_true",       help="啟用 chord-conditioned 模式")
    args = parser.parse_args()
    train(args)
