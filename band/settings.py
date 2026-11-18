"""所有可調的設定都在這裡；其他模組只讀不寫。"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 專案資料夾

# ===== 常改的設定 =====
BPM = 75                    # 兩種風格共用；75 BPM 的 16 分音符剛好 5 幀（40 ms）
MIDI_PORT = None            # MIDI 鍵盤名稱；None = 第一個找到的
PIANO_PORT = "Microsoft GS Wavetable Synth"  # Windows 端發出鋼琴聲的合成器（名稱開頭）；None = 彈的人不聽自己的琴聲
GROOVE_PYTHON = "~/groove_venv/bin/python"  # 鼓手 groove_server.py（GrooVAE）的環境
DRUM_LAG = 2                # 鼓延後幾幀：MRT2 的樂器比鼓點條件晚約 70 ms 出聲，鼓往後挪才對齊

AGENTS = ["keys", "rhythm", "bass", "pad", "lead"]  # MRT2 的五個位置：和弦、節奏、貝斯、鋪底、主旋律；鼓另外由 Drummer 負責
STYLES = {  # 每種風格：prompt = 接在樂器提示詞後的風格描述（預設是風格名稱）、mode = 強制的調式、hat_div = hi-hat 每拍幾個、
            # kits = 可用的鼓組（samples/ 底下的資料夾）、bass808 = 貝斯改由跟著大鼓的 808 取樣打、scene = 視覺的配色（0 或 1）
            # lean = 自動判斷觀眾打的風格比較像哪一種時，加給這種的分數（文字相似度，見 band/band.py 的 pick_base）
            # 觀眾打字新增的風格會在執行時加進來（band/band.py 的 add_style），沿用最像的那種風格的設定
    "neo soul": {"hat_div": 2, "scene": 0, "kits": ["neo soul", "rnb_2000s_soul_jazz", "band sound"]},
    "memphis rap": {"hat_div": 4, "scene": 1, "prompt": "dark and moody", "mode": "minor",  # 不寫 rap（會帶出鼓），一律小調
                    "kits": ["memphis rap", "trap", "modern trap"], "bass808": True,
                    "lean": -0.04},  # 中性的描述（bossa nova、民謠）本來就稍微比較像 memphis rap，要明顯比較像才選它
}
INSTRUMENTS = {  # 位置 → 風格 → [(顯示名稱, 提示詞, 音量)]；第一個是開場用的。某個位置被自動編制安排休息時，會換成池裡的另一個
    # 音量 = 實測有聲時的第 90 百分位（提示詞加 instrumental、「不要鼓」強度 8 之後重量），混音時拉到同一個大小。
    # 只收試音時沒走音、沒偷帶鼓聲、人聲或低頻的樂器（tools/audition.py 的鼓聲、人聲欄）：
    # choir 人聲 0.95、cinematic strings 和 upright bass 常帶鼓、harp 常像人聲；memphis 的 "dark and moody" 會讓長笛冒出低頻
    "keys": {"neo soul": [("Rhodes", "solo rhodes electric piano", .106), ("Wurlitzer", "wurlitzer electric piano", .144),
                          ("管風琴", "hammond organ", .145)],
             "memphis rap": [("Rhodes", "solo rhodes electric piano", .132), ("Wurlitzer", "wurlitzer electric piano", .138),
                             ("管風琴", "hammond organ", .208)]},
    "rhythm": {"neo soul": [("電吉他", "solo electric guitar", .107), ("尼龍吉他", "nylon string guitar", .197)],
               "memphis rap": [("電吉他", "solo electric guitar", .187), ("悶音吉他", "muted funk guitar", .130)]},
    "bass": {"neo soul": [("電貝斯", "solo electric bass", .143), ("無琴格貝斯", "fretless bass", .131)],
             "memphis rap": [("貝斯", "deep sub bass", .129)]},  # memphis 由 808 取代
    "pad": {"neo soul": [("大提琴", "cello", .124), ("弦樂", "legato string ensemble", .096), ("風琴鋪底", "soft electric organ pad", .161)],
            "memphis rap": [("弦樂", "legato string ensemble", .108), ("暗色弦樂", "dark strings", .060), ("小提琴", "slow violins", .052),
                            ("風琴鋪底", "soft electric organ pad", .074)]},
    "lead": {"neo soul": [("鐘", "digital bell", .079), ("音樂盒", "music box", .129), ("鐵琴", "vibraphone", .152), ("長笛", "flute", .084)],
             "memphis rap": [("音樂盒", "music box", .062), ("鐵琴", "vibraphone", .155)]},
}
LEAK_LIMIT = 0.35           # 監聽（band/guard.py）聽到某個樂器像鼓聲或人聲的機率超過這個，就關掉它、下一段換別的樂器
LEVEL = 0.1                 # 每個樂器先拉到這個音量，再乘 GAINS
GAINS = {"keys": 0.8, "rhythm": 0.8, "bass": 1.0, "drums": 0.7, "pad": 0.8, "lead": 0.7}  # 順序 = 視覺 1–6 靜音鍵的順序
PAN = {"keys": -0.4, "rhythm": 0.4, "bass": 0.0, "drums": 0.0, "pad": 0.0, "lead": 0.2}  # −1 = 最左、+1 = 最右
MASTER = 0.85               # 整體音量；配合限制器，混音的失真約 −49 dB
DRUM_WORDS = ("hip hop", "hip-hop", "hiphop", "boom bap", "drum and bass", "drum n bass", "rap", "trap", "drill", "drum",
              "drums", "drumming", "beat", "beats", "percussion", "percussive", "cinematic", "epic", "breakbeat", "dnb",
              "edm", "house", "techno", "dubstep", "reggaeton", "808", "kick", "snare", "hi-hat", "hihat")  # 打字的風格裡會帶出鼓聲的字，自動拿掉
VOICE_WORDS = ("song", "songs", "vocal", "vocals", "vocalist", "voice", "voices", "sing", "singing", "singer", "sung", "choir",
               "choral", "chorus", "a cappella", "acapella", "humming", "lyrics", "rapper", "opera", "karaoke", "gospel")  # 會帶出人聲的字，也拿掉

# ===== 曲式 =====
PHRASE_BARS = 4             # 風格與編制在每段（4 小節）的開頭變，不在半途變
DROP_P, REJOIN_P = 0.3, 0.6  # 每段開頭：在場的樂器退出的機率、退出的樂器回來的機率（至少留 3 個）
BLENDS = (0.25, 0.5, 0.75)  # 切換風格前的最後一小節，提示詞逐拍混合過去，接得比較順
NEXT_CHORD = {  # 和弦進行自己走：每小節從上一個和弦的常見去處隨機挑一個（級數，0 = 主和弦）
    "minor": {0: [3, 5, 6, 2], 2: [5, 3, 0], 3: [4, 6, 0, 5], 4: [0, 5, 3], 5: [3, 6, 2, 0], 6: [0, 2, 5]},
    "major": {0: [5, 3, 1, 4], 1: [4, 6, 3], 3: [4, 0, 1, 5], 4: [0, 5, 3], 5: [3, 1, 4, 0], 6: [0, 4]},
}
CHORDS = {"major": (0, 4, 7), "minor": (0, 3, 7)}  # 判斷觀眾彈的和弦用

# ===== 鼓的取樣 =====
SAMPLES = {36: ("kick", 1.0), 38: ("snare", 1.0), 42: ("hihat", 0.7), 46: ("openhat", 0.7),  # GrooVAE 的鼓音高 → (取樣, 音量)
           45: ("rim", 0.8), 48: ("rim", 0.8), 50: ("rim", 0.8), 49: ("openhat", 0.6), 51: ("openhat", 0.5)}
# 鼓組 = samples/ 底下的一個資料夾：頂層有 kick.wav、snare.wav… 就直接用；否則照子資料夾名稱（Kicks、Snare、Hi hats…）
# 每次隨機挑一個取樣（規則在 band/drummer.py 的 KIT_ROLES）。808 一樣照資料夾找，自動量基頻、調到和弦根音
KIT_SEGMENTS = 8            # 同一風格每幾段（2 小節一段）換一次鼓組；換風格時一定換

# ===== 效能與連線 =====
TEMPERATURE = 1.1           # 取樣隨機度（MRT2 預設 1.3）：吉他更穩，貝斯不變
FRAMES_PER_CALL = 5         # 一次在 GPU 上算幾幀：5 個 agent 時 5 幀約 37 ms/幀（預算 40 ms）
QUEUE_FRAMES = 8            # 預先生成的幀數，太少會斷音、太多會增加延遲
CHORD_SETTLE = 0.08         # 第一個音按下後等多久再判斷和弦（和弦的音不會剛好同時按下）
VISUAL_TCP = ("127.0.0.1", 47601)  # 視覺程式 vj_tk.py（Windows 端）連到這裡接收資料
GROOVE_TCP = ("127.0.0.1", 47602)  # 鼓手 groove_server.py 聽這裡
