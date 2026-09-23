import random
import msvcrt
import time

def midi_to_note_name(midi_num):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_num // 12) - 1
    return f"{notes[midi_num % 12]}{octave}"

# 使用馬可夫鏈 (Markov Chain) 來模擬兩個 AI 互相學習的行為
class MusicAgent:
    def __init__(self, name, base_scale):
        self.name = name
        self.base_scale = base_scale
        self.memory = {}  # 轉移矩陣：記錄 音符 A -> 音符 B 的出現次數
        
    def listen_and_learn(self, melody):
        """聆聽對方的旋律，並更新自己的轉移機率矩陣"""
        for i in range(len(melody) - 1):
            current_note = melody[i]
            next_note = melody[i+1]
            
            if current_note not in self.memory:
                self.memory[current_note] = {}
            if next_note not in self.memory[current_note]:
                self.memory[current_note][next_note] = 0
                
            # 增加權重：聽得越多，未來彈出的機率越高
            self.memory[current_note][next_note] += 1
            
    def generate_response(self, start_note, length=8):
        """根據記憶與基礎音階，生成下一段旋律"""
        response = []
        current_note = start_note
        
        for _ in range(length):
            # 70% 機率使用從對方/自己身上學到的「樂句記憶」
            if current_note in self.memory and random.random() < 0.7:
                choices = list(self.memory[current_note].keys())
                weights = list(self.memory[current_note].values())
                next_note = random.choices(choices, weights=weights)[0]
            else:
                # 30% 機率即興發揮 (在自己熟悉的音階內隨機游走)
                if current_note in self.base_scale:
                    idx = self.base_scale.index(current_note)
                else:
                    idx = len(self.base_scale) // 2
                
                # 在音階上隨機上下跳動 1~2 度
                step = random.choice([-2, -1, 0, 1, 2])
                idx = max(0, min(len(self.base_scale) - 1, idx + step))
                next_note = self.base_scale[idx]
                
            response.append(next_note)
            current_note = next_note
            
        return response

def synthesize_midi(conversation, filename):
    try:
        from mido import Message, MidiFile, MidiTrack
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 簡單的速度與樂器設定
        track.append(Message('program_change', program=0, time=0))
        ticks = int(mid.ticks_per_beat * 0.5) # 八分音符長度
        
        for name, notes in conversation:
            # 用力度 (Velocity) 區分兩個 AI：Agent A 較大聲，Agent B 較小聲
            velocity = 90 if name == "Agent_A" else 60
            for note in notes:
                track.append(Message('note_on', note=note, velocity=velocity, time=0))
                track.append(Message('note_off', note=note, velocity=velocity, time=ticks))
                
        mid.save(filename)
        print(f"\n[成功] MIDI 檔案已儲存至: {filename}")
    except ImportError:
        print("\n[提示] 若要生成真實的 MIDI 讓你能聽，請在終端機執行：pip install mido")

# C 大調 (Agent A 偏好) 與 G 混合利底亞調 (Agent B 偏好，帶有一點色彩差異)
c_major = [60, 62, 64, 65, 67, 69, 71, 72, 74, 76]  
g_mixolydian = [55, 57, 59, 60, 62, 64, 65, 67, 69, 71]

agent_a = MusicAgent("Agent_A", c_major)
agent_b = MusicAgent("Agent_B", g_mixolydian)

# 第一回合：Agent A 主動發起一個簡單的 Motif
current_melody = [60, 64, 67, 64, 60, 62, 64, 67]
conversation = [("Agent_A", current_melody)]
print(f"Agent_A 彈奏: {[midi_to_note_name(n) for n in current_melody]}")

print("\n--- 開始無限 Call and Response (按任意鍵停止) ---\n")

# 開始無限回合的 Call and Response
while True:
    if msvcrt.kbhit():
        msvcrt.getch() # 清除緩衝區
        print("\n[停止] 收到按鍵，結束對話。")
        break

    # Agent B 聽 A 的旋律並學習更新權重
    agent_b.listen_and_learn(current_melody)
    # B 接著 A 的最後一個音開始往下彈
    current_melody = agent_b.generate_response(current_melody[-1], length=8)
    conversation.append(("Agent_B", current_melody))
    print(f"Agent_B 回應: {[midi_to_note_name(n) for n in current_melody]}")
    
    # Agent A 聽 B 的旋律並學習
    agent_a.listen_and_learn(current_melody)
    # A 接著 B 的最後一個音開始往下彈
    current_melody = agent_a.generate_response(current_melody[-1], length=8)
    conversation.append(("Agent_A", current_melody))
    print(f"Agent_A 回應: {[midi_to_note_name(n) for n in current_melody]}")
    
    time.sleep(0.5) # 加上小延遲，避免印太快

synthesize_midi(conversation, "ai_duet_prototype.mid")
