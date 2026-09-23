from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Cm
import copy

# ── 顏色定義 ──────────────────────────────
C_DARK       = RGBColor(0x1a, 0x1a, 0x2e)
C_WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
C_PURPLE     = RGBColor(0x5b, 0x4c, 0xdb)
C_PURPLE_BG  = RGBColor(0xed, 0xea, 0xff)
C_ORANGE     = RGBColor(0xd4, 0x60, 0x0a)
C_ORANGE_BG  = RGBColor(0xff, 0xf3, 0xe0)
C_BLUE       = RGBColor(0x15, 0x65, 0xc0)
C_BLUE_BG    = RGBColor(0xe3, 0xf2, 0xfd)
C_GREEN      = RGBColor(0x27, 0x62, 0x21)
C_GREEN_BG   = RGBColor(0xe8, 0xf5, 0xe9)
C_HEADER_BG  = RGBColor(0xb0, 0xc4, 0xde)
C_NOTE_A     = RGBColor(0x34, 0xac, 0xe0)
C_NOTE_B     = RGBColor(0xf3, 0x9c, 0x12)
C_NOTE_H     = RGBColor(0x2e, 0xd5, 0x73)
C_SLIDE_BG   = RGBColor(0xf4, 0xf4, 0xf8)

FONT = "微軟正黑體"
FONT_CODE = "Consolas"

prs = Presentation()
prs.slide_width  = Inches(13.33)
prs.slide_height = Inches(7.5)

blank_layout = prs.slide_layouts[6]  # 完全空白

# ── 輔助函式 ─────────────────────────────────────────────

def set_bg(slide, color: RGBColor):
    from pptx.oxml.ns import qn
    from lxml import etree
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_rect(slide, left, top, width, height, fill_color=None, line_color=None, line_width_pt=0):
    shape = slide.shapes.add_shape(
        1,  # MSO_SHAPE_TYPE.RECTANGLE
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    if fill_color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color and line_width_pt > 0:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(line_width_pt)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text, left, top, width, height,
                font_name=FONT, font_size=18, bold=False, color=C_DARK,
                align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox


def add_label_box(slide, label, left, top, width, height,
                  bg_color, text_color=C_WHITE, font_size=20):
    """帶顏色底色的標題條"""
    rect = add_rect(slide, left, top, width, height, fill_color=bg_color)
    tf = rect.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    run = p.add_run()
    run.text = label
    run.font.name = FONT
    run.font.size = Pt(font_size)
    run.font.bold = True
    run.font.color.rgb = text_color
    # 垂直置中
    from pptx.enum.text import MSO_ANCHOR
    tf.auto_size = None
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    return rect


def add_content_box(slide, paragraphs, left, top, width, height,
                    bg_color=C_WHITE, font_size=13, bullet=False):
    """內容文字框，paragraphs = list of (text, is_bullet, bold, font_size_override)"""
    rect = add_rect(slide, left, top, width, height, fill_color=bg_color)
    tf = rect.text_frame
    tf.word_wrap = True
    from pptx.enum.text import MSO_ANCHOR
    tf.vertical_anchor = MSO_ANCHOR.TOP
    # 內縮
    from pptx.oxml.ns import qn
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT

    first = True
    for item in paragraphs:
        if isinstance(item, str):
            text, is_bullet, is_bold, fs = item, False, False, font_size
        elif len(item) == 2:
            text, is_bullet = item; is_bold = False; fs = font_size
        elif len(item) == 3:
            text, is_bullet, is_bold = item; fs = font_size
        else:
            text, is_bullet, is_bold, fs = item

        if first:
            p = tf.paragraphs[0]
            first = False
        else:
            p = tf.add_paragraph()

        p.alignment = PP_ALIGN.LEFT
        if is_bullet:
            p.level = 1
        run = p.add_run()
        run.text = ("• " if is_bullet else "") + text
        run.font.name = FONT
        run.font.size = Pt(fs)
        run.font.bold = is_bold
        run.font.color.rgb = C_DARK
    return rect


def add_code_box(slide, code_lines, left, top, width, height, font_size=10.5):
    rect = add_rect(slide, left, top, width, height, fill_color=RGBColor(0x1e, 0x1e, 0x1e))
    tf = rect.text_frame
    tf.word_wrap = False
    from pptx.enum.text import MSO_ANCHOR
    tf.vertical_anchor = MSO_ANCHOR.TOP

    first = True
    for line in code_lines:
        if first:
            p = tf.paragraphs[0]; first = False
        else:
            p = tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line
        run.font.name = FONT_CODE
        run.font.size = Pt(font_size)
        run.font.color.rgb = RGBColor(0xd4, 0xd4, 0xd4)
    return rect


def slide_header(slide, title, subtitle=None):
    """每張投影片頂部帶色條"""
    add_rect(slide, 0, 0, 13.33, 0.6, fill_color=C_DARK)
    add_textbox(slide, title, 0.2, 0.07, 10, 0.5,
                font_size=22, bold=True, color=C_WHITE)
    if subtitle:
        add_textbox(slide, subtitle, 0.2, 0.58, 12, 0.35,
                    font_size=13, color=RGBColor(0xaa, 0xaa, 0xcc))


# ═══════════════════════════════════════════════════
# Slide 1：封面
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_DARK)

# 漸層色塊裝飾
add_rect(slide, 0, 0, 13.33, 2.2, fill_color=RGBColor(0x1e, 0x28, 0x4a))
add_rect(slide, 0, 5.8, 13.33, 1.7, fill_color=RGBColor(0x1e, 0x28, 0x4a))

# 標題
add_textbox(slide, "AI Grand Piano Duet", 0.5, 1.2, 12.3, 1.2,
            font_size=48, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
add_textbox(slide, "以馬可夫鏈實現人機即興合奏系統", 0.5, 2.3, 12.3, 0.7,
            font_size=26, bold=False, color=C_NOTE_A, align=PP_ALIGN.CENTER)

# 彩色標籤列
tags = [("Markov Chain", C_PURPLE), ("FastAPI + WebSocket", C_ORANGE),
        ("Tone.js 音訊引擎", C_BLUE), ("即時人機互動", C_GREEN)]
x = 1.3
for label, col in tags:
    add_rect(slide, x, 3.2, 2.3, 0.42, fill_color=col)
    add_textbox(slide, label, x + 0.1, 3.23, 2.1, 0.38,
                font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    x += 2.6

# 作者資訊
add_textbox(slide, "作者一¹  作者二²  作者三¹  作者四²",
            0.5, 5.9, 12.3, 0.45,
            font_size=16, color=RGBColor(0xcc, 0xcc, 0xcc), align=PP_ALIGN.CENTER)
add_textbox(slide, "¹資訊科學系   ²資訊管理系   |   指導老師：OOO",
            0.5, 6.35, 12.3, 0.4,
            font_size=13, color=RGBColor(0x88, 0x88, 0xaa), align=PP_ALIGN.CENTER)

# 裝飾音符
for ch, x, y, sz, col in [
    ("♪", 0.3, 3.0, 60, C_NOTE_A),
    ("♫", 11.8, 1.5, 50, C_NOTE_B),
    ("♩", 0.5, 5.2, 40, C_NOTE_H),
    ("♬", 12.0, 5.0, 45, C_PURPLE),
]:
    add_textbox(slide, ch, x, y, 1, 1, font_size=sz, color=col, align=PP_ALIGN.CENTER)


# ═══════════════════════════════════════════════════
# Slide 2：目的與研究問題
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "目的與研究問題", "為什麼要讓 AI 學會「聆聽」？")

# 左欄：背景說明
add_label_box(slide, "研究背景", 0.3, 0.75, 5.9, 0.42, bg_color=C_PURPLE)
add_content_box(slide, [
    ("傳統電腦音樂生成模型訓練後固定輸出，缺乏即時聆聽與回應能力", True, False, 13),
    ("本研究以輕量的馬可夫鏈作為核心決策引擎", True, False, 13),
    ("目標：建構兩個能即時學習、相互適應的音樂代理 (Agent)", True, False, 13),
    ("創造無限的 Call and Response 鋼琴對話", True, False, 13),
], 0.3, 1.17, 5.9, 1.9, bg_color=C_PURPLE_BG)

add_label_box(slide, "研究問題", 0.3, 3.15, 5.9, 0.42, bg_color=C_PURPLE)
add_content_box(slide, [
    ("① 僅用統計轉移機率，能否產出有音樂性的旋律走向？", True, False, 13),
    ("② 兩個代理隨時間聆聽，其「對話」是否會收斂或演化？", True, False, 13),
    ("③ 人類介入後，AI 的回應能否自然融入人類的音符？", True, False, 13),
], 0.3, 3.57, 5.9, 1.5, bg_color=C_PURPLE_BG)

# 右欄：系統目標
add_label_box(slide, "系統目標", 6.5, 0.75, 6.5, 0.42, bg_color=C_PURPLE)
add_content_box(slide, [
    ("雙 Agent 音色差異", False, True, 14),
    ("Agent A → C 大調（穩定和聲）", True, False, 13),
    ("Agent B → G 混合利底亞調（帶藍調色彩）", True, False, 13),
    ("音色上形成張力與對話感", True, False, 13),
    ("Web 版升級功能", False, True, 14),
    ("支援 24 種調性（12 根音 × 大調/小調）", True, False, 13),
    ("即時切換不中斷播放", True, False, 13),
    ("人類鍵盤即時「插嘴」並讓 Agent 學習", True, False, 13),
    ("前端視覺化下落式 Piano Roll 遊戲畫面", True, False, 13),
], 6.5, 1.17, 6.5, 3.3, bg_color=C_PURPLE_BG)

# 引言底部
add_rect(slide, 0.3, 5.15, 12.5, 0.75, fill_color=RGBColor(0xe0, 0xd8, 0xff))
add_textbox(slide, "「目標是讓 AI 的演奏不像『自動播放』，而是有如一個真正在聆聽的即興樂手。」",
            0.5, 5.22, 12.1, 0.6,
            font_size=14, italic=True, color=C_PURPLE, align=PP_ALIGN.CENTER)

add_textbox(slide, "2 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 3：演算法核心 - 馬可夫鏈
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "演算法核心：馬可夫鏈 (Markov Chain)", "每次聆聽都在更新「音符轉移機率矩陣」")

# 左：轉移矩陣說明
add_label_box(slide, "① listen_and_learn() — 聆聽與學習", 0.3, 0.75, 6.0, 0.42, bg_color=C_ORANGE)
add_code_box(slide, [
    "# 更新轉移矩陣（每次聆聽對方旋律）",
    "def listen_and_learn(self, melody_events):",
    "    notes = [ev['note'] for ev in melody_events]",
    "    for i in range(len(notes) - 1):",
    "        cur, nxt = notes[i], notes[i+1]",
    "        if cur not in self.memory:",
    "            self.memory[cur] = {}",
    "        if nxt not in self.memory[cur]:",
    "            self.memory[cur][nxt] = 0",
    "        self.memory[cur][nxt] += 1  # 累積權重",
], 0.3, 1.17, 6.0, 2.3)

add_label_box(slide, "② generate_response() — 生成回應", 0.3, 3.55, 6.0, 0.42, bg_color=C_ORANGE)
add_code_box(slide, [
    "# 加權抽樣 vs. 隨機游走（70 / 30 分配）",
    "if cur in self.memory and random() < 0.7:",
    "    choices = list(self.memory[cur].keys())",
    "    weights = list(self.memory[cur].values())",
    "    next_note = random.choices(choices,",
    "                               weights=weights)[0]",
    "else:",
    "    # 在音階上隨機上下跳動 1~2 度",
    "    idx += random.choice([-2, -1, 0, 1, 2])",
    "    next_note = base_scale[idx]",
], 0.3, 3.97, 6.0, 2.3)

# 右：機率概念圖示
add_label_box(slide, "轉移矩陣概念示意", 6.6, 0.75, 6.4, 0.42, bg_color=C_ORANGE)
add_content_box(slide, [
    ("記憶體結構（巢狀字典）", False, True, 13),
    ("memory[C4][D4] = 5  →  C4 後接 D4 出現 5 次", True, False, 12),
    ("memory[C4][E4] = 2  →  C4 後接 E4 出現 2 次", True, False, 12),
    ("memory[C4][G4] = 3  →  C4 後接 G4 出現 3 次", True, False, 12),
    ("加權隨機取樣結果", False, True, 13),
    ("D4 被選中機率 = 5/(5+2+3) = 50%", True, False, 12),
    ("E4 被選中機率 = 2/10 = 20%", True, False, 12),
    ("G4 被選中機率 = 3/10 = 30%", True, False, 12),
    ("越常聽到的音符連接關係，未來出現機率越高", True, False, 12),
    ("每個小節結束後，互相聆聽對方並更新矩陣", True, False, 12),
], 6.6, 1.17, 6.4, 2.3, bg_color=C_ORANGE_BG)

add_label_box(slide, "③ 和弦骨架引導（80 / 20 策略）", 6.6, 3.55, 6.4, 0.42, bg_color=C_ORANGE)
add_content_box(slide, [
    ("每小節前，後端隨機選取 I / IV / V / VI 級三和弦作為骨架", True, False, 12),
    ("80% 機率只選屬於該和弦的音符 → 確保和諧感", True, False, 12),
    ("20% 機率選非和弦音 → 增加張力與色彩", True, False, 12),
    ("有效降低純隨機游走造成的刺耳不和諧感", True, False, 12),
    ("兩個 Agent 共享同一個和弦骨架，強制讓它們「合拍」", True, False, 12),
], 6.6, 3.97, 6.4, 2.3, bg_color=C_ORANGE_BG)

add_textbox(slide, "3 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 4：系統架構
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "系統架構", "FastAPI 後端 + WebSocket + Tone.js 前端")

# 流程方塊
flow_items = [
    ("使用者選擇\n調性 / BPM", C_PURPLE),
    ("後端\nMusicAgent\n計算事件", C_ORANGE),
    ("WebSocket\n推送 MIDI\n事件陣列", C_BLUE),
    ("Tone.js\n排程播放\n+ 視覺動畫", C_GREEN),
    ("下一小節\nlisten_and\n_learn()", C_ORANGE),
]
bw = 2.0
gap = 0.15
start_x = 0.5
for i, (label, col) in enumerate(flow_items):
    x = start_x + i * (bw + gap + 0.3)
    add_rect(slide, x, 0.82, bw, 1.1, fill_color=col)
    add_textbox(slide, label, x + 0.05, 0.85, bw - 0.1, 1.05,
                font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    if i < len(flow_items) - 1:
        add_textbox(slide, "→", x + bw + 0.02, 1.1, 0.35, 0.6,
                    font_size=24, bold=True, color=C_ORANGE, align=PP_ALIGN.CENTER)

# 三欄技術細節
col_w = 4.0
# 後端
add_label_box(slide, "後端（Python / FastAPI）", 0.3, 2.1, col_w, 0.4, bg_color=C_ORANGE)
add_content_box(slide, [
    ("MusicAgent 類別：維護轉移矩陣記憶體", True, False, 12),
    ("generate_full_performance() 每小節計算 16 steps 的完整 MIDI 事件", True, False, 12),
    ("每個事件包含 onset_beat / duration / play_delay_sec / note_name", True, False, 12),
    ("鼓聲軌道：大鼓 step 0,8 / 軍鼓 step 4,12（4/4 拍）", True, False, 12),
    ("asyncio 並發：settings_listener 與音樂主迴圈同時運行", True, False, 12),
], 0.3, 2.5, col_w, 2.5, bg_color=C_ORANGE_BG)

# WebSocket
add_label_box(slide, "WebSocket 通訊協定", 4.65, 2.1, col_w, 0.4, bg_color=C_BLUE)
add_content_box(slide, [
    ("前→後：update_settings（調性、BPM）", True, False, 12),
    ("前→後：human_note（MIDI 音符編號）", True, False, 12),
    ("後→前：scale_info（當前音階音符集，防呆用）", True, False, 12),
    ("後→前：Agent_A / Agent_B / Drum 事件陣列", True, False, 12),
    ("本機測試延遲 < 50ms，每小節推送一次", True, False, 12),
], 4.65, 2.5, col_w, 2.5, bg_color=C_BLUE_BG)

# 前端
add_label_box(slide, "前端（Tone.js + 原生 JS）", 9.0, 2.1, 4.0, 0.4, bg_color=C_GREEN)
add_content_box(slide, [
    ("Tone.Sampler 載入 WAV 樣本音色", True, False, 12),
    ("低通 Filter 模擬 Bass / 吉他音色差異", True, False, 12),
    ("MIDI 36–84（4 個八度）鍵盤 Canvas", True, False, 12),
    ("下落式方塊動畫（CSS transform）", True, False, 12),
    ("人類輸入防呆：不在音階內閃紅燈", True, False, 12),
], 9.0, 2.5, 4.0, 2.5, bg_color=C_GREEN_BG)

# 底部：音符顏色圖例
add_rect(slide, 0.3, 5.2, 12.7, 0.5, fill_color=RGBColor(0x22, 0x22, 0x22))
for label, col, x in [
    ("■ Agent A（藍色）", C_NOTE_A, 1.5),
    ("■ Agent B（橘色）", C_NOTE_B, 4.5),
    ("■ 人類輸入（綠色）", C_NOTE_H, 7.5),
    ("■ 鼓聲（灰色條）", RGBColor(0x88, 0x88, 0x88), 10.5),
]:
    add_textbox(slide, label, x, 5.28, 2.5, 0.38,
                font_size=13, bold=True, color=col, align=PP_ALIGN.LEFT)

add_textbox(slide, "4 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 5：人類互動機制
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "人類互動機制", "鍵盤防呆 + Agent 即時學習")

add_label_box(slide, "鍵盤對應（相對 C 根音）", 0.3, 0.75, 6.0, 0.42, bg_color=C_GREEN)
add_content_box(slide, [
    ("白鍵：A  S  D  F  G  H  J  K  （C D E F G A B C）", True, False, 13),
    ("黑鍵：W  E  T  Y  U  （C# D# F# G# A#）", True, False, 13),
    ("切換八度：Z（降一個八度）／X（升一個八度）", True, False, 13),
    ("支援八度範圍：Oct 2 ~ Oct 6", True, False, 13),
], 0.3, 1.17, 6.0, 1.4, bg_color=C_GREEN_BG)

add_label_box(slide, "防呆機制", 0.3, 2.65, 6.0, 0.42, bg_color=C_GREEN)
add_content_box(slide, [
    ("後端每小節推送 scale_info，前端維護當前音階的音符集合 (Set)", True, False, 13),
    ("彈奏時比對 midi % 12 是否在音階內", True, False, 13),
    ("彈錯音 → 鍵盤閃紅燈 200ms，不送出訊息給後端", True, False, 13),
    ("目的：確保人類輸入不破壞當前調性框架", True, False, 13),
], 0.3, 3.07, 6.0, 1.6, bg_color=C_GREEN_BG)

add_label_box(slide, "人類音符如何影響 AI", 0.3, 4.75, 6.0, 0.42, bg_color=C_GREEN)
add_content_box(slide, [
    ("前端按下按鍵 → 送出 human_note { midi } 給後端", True, False, 13),
    ("後端暫存至 state['human_notes']（最多保留最近 16 個）", True, False, 13),
    ("下一小節：Agent B 的 listen_and_learn() 學習人類音符序列", True, False, 13),
    ("Agent A 的起始音替換為人類最後彈的音符", True, False, 13),
    ("效果：下一段旋律自然「接」在人類彈完的地方繼續", True, False, 13),
], 0.3, 5.17, 6.0, 2.05, bg_color=C_GREEN_BG)

# 右側：互動流程圖文字版
add_label_box(slide, "三方互動時序圖", 6.6, 0.75, 6.4, 0.42, bg_color=C_GREEN)
steps = [
    ("小節 N 開始", C_DARK, False),
    ("後端送出 scale_info → 前端更新音階防呆集合", C_BLUE, False),
    ("後端送出 Drum / Agent_B / Agent_A 事件", C_ORANGE, False),
    ("Tone.js 依 play_delay_sec 排程播放", C_BLUE, False),
    ("人類按鍵 → 前端立即播音 + 送 human_note", C_NOTE_H, True),
    ("Agent B 聆聽 Agent A 事件 → 更新矩陣", C_ORANGE, True),
    ("Agent A 聆聽 Agent B 事件 → 更新矩陣", C_ORANGE, True),
    ("若有 human_notes → Agent B 學習，A 起始音更新", C_NOTE_H, True),
    ("小節 N+1 開始（重複循環）", C_DARK, False),
]
y = 1.17
for i, (text, col, is_human) in enumerate(steps):
    bg = RGBColor(0xe8, 0xf5, 0xe9) if is_human else RGBColor(0xf5, 0xf5, 0xff)
    add_rect(slide, 6.6, y, 6.4, 0.55, fill_color=bg,
             line_color=col, line_width_pt=1.5)
    add_textbox(slide, f"{'→ ' if i>0 else ''}{text}", 6.75, y + 0.08, 6.1, 0.42,
                font_size=12, color=col, bold=(i == 0 or i == len(steps)-1))
    y += 0.62

add_textbox(slide, "5 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 6：開發成果 Prototype vs Web
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "開發成果：Prototype vs. Web 版", "從命令列原型到完整互動 Web 應用")

# 比較表格（手工繪製）
headers = ["比較項目", "Prototype（命令列）", "Web 版（ai_duet_web.py）"]
rows = [
    ["音樂生成引擎", "馬可夫鏈，70/30 分配", "馬可夫鏈 + 和弦骨架，40% 記憶"],
    ["音色", "MIDI Velocity 區分大小聲", "Tone.js Sampler + 低通 Filter"],
    ["人類互動", "按任意鍵停止演奏", "即時鍵盤彈奏 + 防呆機制"],
    ["調性支援", "固定 C大調 / G混合利底亞", "24 種調性，即時切換"],
    ["鼓聲軌道", "無", "4/4 大鼓+軍鼓，支援自訂 WAV"],
    ["視覺化", "終端機文字輸出", "下落式 Piano Roll 動畫"],
    ["輸出格式", "匯出 .mid 檔（mido 函式庫）", "即時 WebSocket 串流至瀏覽器"],
    ["延遲", "N/A", "本機測試 < 50ms"],
]

col_widths = [2.8, 4.5, 5.5]
col_xs = [0.3, 3.1, 7.6]
row_h = 0.52
row_y = 0.78

# Header 列
for j, (hdr, w, x) in enumerate(zip(headers, col_widths, col_xs)):
    bg = C_BLUE if j > 0 else C_DARK
    add_rect(slide, x, row_y, w, row_h, fill_color=bg)
    add_textbox(slide, hdr, x + 0.05, row_y + 0.08, w - 0.1, row_h - 0.1,
                font_size=13, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

for i, row in enumerate(rows):
    row_y += row_h
    for j, (cell, w, x) in enumerate(zip(row, col_widths, col_xs)):
        bg = (RGBColor(0xf5, 0xf5, 0xff) if i % 2 == 0 else RGBColor(0xe8, 0xf0, 0xff)) if j > 0 \
             else (RGBColor(0xee, 0xee, 0xf5) if i % 2 == 0 else RGBColor(0xe0, 0xe0, 0xf0))
        add_rect(slide, x, row_y, w, row_h, fill_color=bg,
                 line_color=RGBColor(0xcc, 0xcc, 0xdd), line_width_pt=0.5)
        add_textbox(slide, cell, x + 0.06, row_y + 0.07, w - 0.1, row_h - 0.1,
                    font_size=12, bold=(j == 0), color=C_DARK)

# 底部備註
y_note = row_y + row_h + 0.1
add_rect(slide, 0.3, y_note, 12.7, 0.45, fill_color=C_BLUE_BG,
         line_color=C_BLUE, line_width_pt=1)
add_textbox(slide, "兩個版本共用相同的 MusicAgent 核心演算法，Web 版在此基礎上擴充了多調性支援、鼓聲軌道與即時人機互動。",
            0.5, y_note + 0.07, 12.3, 0.35, font_size=12, italic=True, color=C_BLUE)

add_textbox(slide, "6 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 7：主要發現
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "主要發現", "馬可夫鏈 + 和弦引導的組合效果")

findings = [
    ("🎵 記憶感的出現",
     "即使是一階轉移矩陣，在多次對話後也能展現旋律片段的重複與變奏，"
     "兩個 Agent 的音符序列逐漸出現「慣用樂句」的雛形。",
     C_PURPLE),
    ("🎼 和聲框架有效降噪",
     "引入和弦骨架引導（80/20 比例）後，生成旋律的和諧度明顯提升。"
     "純隨機游走造成的刺耳不和諧感大幅減少，聆聽體驗更接近真實即興演奏。",
     C_ORANGE),
    ("🤝 人機對話自然過渡",
     "人類插嘴後的下一個小節，Agent A 的起始音源自人類最後輸入，"
     "Agent B 也開始學習人類序列，三方對話自然過渡，沒有明顯斷裂感。",
     C_BLUE),
    ("⏱ 低延遲即時互動達成",
     "WebSocket 架構確保每小節的事件推送延遲 < 50ms（本機環境），"
     "配合 Tone.js 的精確時間排程，前端音符播放時序與預期完全吻合。",
     C_GREEN),
]

y = 0.78
for title, body, col in findings:
    add_rect(slide, 0.3, y, 12.7, 1.4, fill_color=RGBColor(0xff,0xff,0xff),
             line_color=col, line_width_pt=2)
    add_rect(slide, 0.3, y, 0.15, 1.4, fill_color=col)
    add_textbox(slide, title, 0.55, y + 0.1, 12.2, 0.45,
                font_size=16, bold=True, color=col)
    add_textbox(slide, body, 0.55, y + 0.55, 12.2, 0.75,
                font_size=13, color=C_DARK)
    y += 1.55

add_textbox(slide, "7 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ═══════════════════════════════════════════════════
# Slide 8：結論與未來展望
# ═══════════════════════════════════════════════════
slide = prs.slides.add_slide(blank_layout)
set_bg(slide, C_SLIDE_BG)
slide_header(slide, "結論與未來展望", "這是一個起點，而非終點")

# 結論
add_label_box(slide, "結論", 0.3, 0.75, 5.9, 0.42, bg_color=C_PURPLE)
add_content_box(slide, [
    ("成功以馬可夫鏈建構即時雙 Agent 鋼琴對話系統", True, False, 13),
    ("Prototype 可匯出 .mid 檔，Web 版提供完整互動體驗", True, False, 13),
    ("人機即興互動機制運作符合預期，對話過渡自然", True, False, 13),
    ("和弦骨架引導有效提升旋律和諧度", True, False, 13),
    ("WebSocket 架構確保低延遲即時串流", True, False, 13),
], 0.3, 1.17, 5.9, 1.9, bg_color=C_PURPLE_BG)

# 技術棧
add_label_box(slide, "使用技術棧", 0.3, 3.15, 5.9, 0.42, bg_color=C_PURPLE)
add_content_box(slide, [
    ("Python 3.11 + FastAPI + uvicorn", True, False, 13),
    ("mido 1.3.x（MIDI 輸出）", True, False, 13),
    ("Tone.js 14.x（瀏覽器音訊排程引擎）", True, False, 13),
    ("原生 JavaScript + CSS 動畫（無框架）", True, False, 13),
], 0.3, 3.57, 5.9, 1.5, bg_color=C_PURPLE_BG)

# 未來展望
add_label_box(slide, "未來展望", 6.5, 0.75, 6.5, 0.42, bg_color=C_ORANGE)
roadmap = [
    ("🔬 短期", [
        "二階/三階馬可夫鏈捕捉更長樂句連貫性",
        "加入強弱力度（Velocity）動態變化",
        "改善鼓聲節奏型（加入切分音）",
    ], C_ORANGE),
    ("🚀 中期", [
        "LSTM / Transformer 取代馬可夫鏈",
        "加入情感參數調控 AI 演奏情緒曲線",
        "支援多使用者 WebSocket 多人合奏",
    ], C_BLUE),
    ("🌟 長期", [
        "整合真實鋼琴 MIDI 輸入（USB MIDI）",
        "訓練大型音樂語言模型作為生成核心",
        "商業化互動音樂教學平台",
    ], C_GREEN),
]

yr = 1.17
for stage, items, col in roadmap:
    add_rect(slide, 6.5, yr, 6.5, 1.8, fill_color=RGBColor(0xff, 0xff, 0xff),
             line_color=col, line_width_pt=1.5)
    add_textbox(slide, stage, 6.6, yr + 0.05, 6.2, 0.42,
                font_size=14, bold=True, color=col)
    for k, item in enumerate(items):
        add_textbox(slide, f"• {item}", 6.7, yr + 0.48 + k * 0.38, 6.1, 0.38,
                    font_size=12, color=C_DARK)
    yr += 1.9

# 底部結語
add_rect(slide, 0.3, 5.15, 12.7, 0.6, fill_color=C_DARK)
add_textbox(slide,
            "「這場人機之間的音樂對話，才剛剛開始——每一個小節都是一次全新的即興。」",
            0.5, 5.22, 12.3, 0.5,
            font_size=14, italic=True, color=C_NOTE_A, align=PP_ALIGN.CENTER)

add_textbox(slide, "8 / 8", 12.5, 7.1, 0.7, 0.3, font_size=11,
            color=RGBColor(0x99, 0x99, 0x99), align=PP_ALIGN.RIGHT)


# ── 儲存 ──────────────────────────────────────────
output = r"d:\Projects\work\ai_duet\AI_Duet_Presentation.pptx"
prs.save(output)
print(f"[OK] 已生成：{output}")
