"""
vj_tk.py — 樂團的視覺（Windows 端，tkinter 視窗內嵌 OpenGL，用內顯畫，不搶 RTX 3050）。
由 band_mrt2.py 自動啟動，連到 WSL 的樂團接收：每幀各樂器的音量、拍子、風格、觀眾彈的音。
run_band.bat 會把 Windows 的 python 設成用內顯（圖形設定的「省電」），RTX 3050 留給生成音樂。
畫面是你在網頁版設計的 shader；多了第五道光束（觀眾的 MIDI），音樂開始前沒有光束。
單獨測試（不用模型）：venv\\Scripts\\python.exe vj_tk.py --demo
鍵：點一下畫面（或 M）選風格、Enter 輸入風格、1–6 靜音、S 光柱名稱、P 暫停、Esc 離開全螢幕、F 全螢幕、Q 結束
"""
import math
import queue
import random
import socket
import sys
import threading
import time
import tkinter as tk
import zlib

import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pyopengltk import OpenGLFrame

VISUAL_TCP = ("127.0.0.1", 47601)  # band_mrt2.py 送資料的位置
RENDER_WIDTH = 1280                # 先畫 720p 再放大到全螢幕（內顯 1080p 只有 26 fps，720p 有 56 fps）
FONT = r"C:\Windows\Fonts\msjh.ttc"
BPM = 75

VERT = "#version 120\nattribute vec2 a;\nvoid main(){gl_Position=vec4(a,0.,1.);}"
FRAG = r"""#version 120
uniform vec2 uRes;
uniform float uTime,uBeat,uFlow,uMix,uKickAge,uStarted,uTextA;
uniform float uKick,uSnare,uHat,uBass,uKeys,uTone,uMel,uMelPos,uHum;
uniform sampler2D uObjA,uObjB,uText;
uniform vec3 uW;
uniform vec4 uLv;
uniform vec2 uObjOff;
uniform float uObjAF,uObjBF;
uniform float uHue,uHueA;   /* 自訂風格：整個畫面轉成另一個色相 */

float hash(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);vec2 u=f*f*(3.-2.*f);
 return mix(mix(hash(i),hash(i+vec2(1.,0.)),u.x),mix(hash(i+vec2(0.,1.)),hash(i+vec2(1.,1.)),u.x),u.y);}
float fbm(vec2 p){float v=0.,a=.5;mat2 m=mat2(1.6,1.2,-1.2,1.6);
 for(int i=0;i<5;i++){v+=a*noise(p);p=m*p;a*=.5;}return v;}

float obj(sampler2D s,vec2 p,float t,float fade){
 vec2 w=(vec2(fbm(p*2.2+t),fbm(p*2.2-t+3.7))-.5)*.09;
 vec2 c=p*.95+.5+w+uObjOff;
 float o=texture2D(s,c).r*.2;
 for(int i=0;i<8;i++){float a=float(i)*.7854;vec2 d=vec2(cos(a),sin(a));
  o+=texture2D(s,c+d*.014).r*.06+texture2D(s,c+d*.035).r*.04;}
 return o*fade*(.55+.75*fbm(p*3.+t*1.5));
}

float beam(vec2 uv,float cx,float lv,float t){
 float n=fbm(vec2(uv.x*7.+cx*9.,uv.y*2.5-t*.6));
 float x=(uv.x-cx+(n-.5)*.05)/(.022+.03*lv);
 float h=.08+lv*.85;
 return exp(-x*x)*smoothstep(h,0.,uv.y)*(.45+1.3*lv)*(.55+.8*n);
}
/* 五道光束，由左到右：鼓（大鼓、小鼓）、貝斯（memphis 是 808）、觀眾（中間）、Rhodes、旋律（吉他＋鐘） */
float ibeams(vec2 uv,float t){
 return beam(uv,.167,uLv.x,t)+beam(uv,.333,uLv.y,t)+beam(uv,.667,uLv.z,t)+beam(uv,.833,uLv.w,t);
}
float hbeam(vec2 uv,float t){return beam(uv,.5,uHum,t);}

vec3 pal(float t,vec3 c0,vec3 c1,vec3 c2,vec3 c3){
 t=clamp(t,0.,1.);
 if(t<.33)return mix(c0,c1,t/.33);
 if(t<.66)return mix(c1,c2,(t-.33)/.33);
 return mix(c2,c3,(t-.66)/.34);
}
vec3 w3(vec3 a,vec3 b,vec3 c){return a*uW.x+b*uW.y+c*uW.z;}
vec3 hue(vec3 c,float a){   /* 在 YIQ 色彩空間轉色相，亮度不變 */
 vec3 y=mat3(.299,.596,.211,.587,-.274,-.523,.114,-.322,.312)*c;
 float h=atan(y.z,y.y)+a,ch=length(y.yz);
 return max(mat3(1.,1.,1.,.956,-.272,-1.106,.621,-.647,1.703)*vec3(y.x,ch*cos(h),ch*sin(h)),0.);
}

vec3 neo(vec2 uv,vec2 p){
 float d0=length(p);
 p*=1.-.045*uKick;
 float t=uFlow*.055;
 vec3 c0=w3(vec3(.09,.02,.01),vec3(.10,.012,.035),vec3(.07,.035,.01));
 vec3 c1=w3(vec3(.55,.12,.04),vec3(.52,.07,.13),vec3(.40,.21,.04));
 vec3 c2=w3(vec3(.97,.48,.13),vec3(.96,.38,.31),vec3(.90,.63,.17));
 vec3 c3=w3(vec3(1.,.84,.52),vec3(1.,.78,.68),vec3(1.,.93,.70));
 vec2 sp=p*dot(uW,vec3(1.35,1.75,1.0));
 vec2 q=vec2(fbm(sp+t),fbm(sp+vec2(5.2,1.3)-t));
 vec2 r=vec2(fbm(sp+3.*q+vec2(1.7,9.2)+t*1.3),fbm(sp+3.*q+vec2(8.3,2.8)-t));
 float A=fbm(sp+(2.8+uBass*1.6)*r);
 float B=.5+.5*sin(9.*fbm(sp*1.4+r*2.-t*2.)+uFlow*.35);
 float d=length(p);
 float wn=fbm(p*2.+t)*2.5;
 float mA=sin(d*6.5-uFlow*.4-uBeat*.6+wn);
 float mB=sin(p.y*7.+p.x*1.5+uFlow*.35+wn);
 float mC=sin(atan(p.y,p.x)*3.+uFlow*.2+wn)*smoothstep(.03,.4,d);
 float m=smoothstep(.15,.85,.5+.5*dot(uW,vec3(mA,mB,mC)));
 float v=mix(A,B*.75+A*.35,m);
 float o=obj(uObjA,p,t,uObjAF);
 v=v*(1.-.35*o)+o*.5;
 vec3 col=pal(v*1.1+.14*uKeys+.05*uKick-.03,c0,c1,c2,c3);
 col*=mix(vec3(1.,.95,.8),vec3(1.06,.8,.82),uTone);
 col*=.78+.4*uKeys;
 col+=c3*pow(max(0.,A-.52),2.)*2.2;
 col+=mix(c2,c3,.5)*o*.24;
 col+=c2*exp(-d*d*2.2)*(.2+.3*uKick);
 float ry=(uMelPos-.5)*.8+.07*sin(p.x*3.+uTime*.8)+.05*(fbm(p*3.+t)-.5);
 col+=c3*exp(-pow((p.y-ry)*16.,2.))*uMel*.55;
 vec2 wq=p*2.6+vec2(uFlow*.05,uFlow*.12);
 wq+=.7*vec2(fbm(wq+t*2.),fbm(wq-t*2.+2.3));
 float wc=sin(wq.x*3.1+uTime*.5)+sin(wq.y*3.7-uTime*.4)+sin((wq.x+wq.y)*2.4+uTime*.3);
 float caus=pow(.5+.5*sin(wc*1.7),7.);
 col+=c3*caus*(.1+.45*uMel+.25*uHat)*(.5+.8*A);
 col*=1.-.5*smoothstep(.45,1.35,d0);
 col+=(c3*ibeams(uv,uTime)+vec3(.85,.95,1.)*hbeam(uv,uTime)*1.2)*1.35*uStarted;
 return col;
}

vec3 trap(vec2 uv,vec2 p){
 float d0=length(p);
 p*=1.-.07*uKick;
 float t=uFlow*.045;
 float rad=uKickAge*1.7;
 float w=exp(-pow((d0-rad)*9.,2.))*max(0.,1.-uKickAge*1.4);
 p+=normalize(p+1e-4)*w*.07;
 float sl=floor(uv.y*36.);
 float tear=step(.72,hash(vec2(sl,floor(uTime*24.))))*uSnare;
 p.x+=(hash(vec2(sl,7.))-.5)*.3*tear;
 float d=length(p);
 vec2 sp=p*dot(uW,vec3(1.7,1.4,2.2));
 vec2 q=vec2(fbm(sp+t),fbm(sp-t+4.));
 float smoke=fbm(sp+(2.2+uBass*1.2)*q-vec2(0.,uFlow*.07));
 float rid=1.-abs(fbm(sp*1.8+q*1.6+t)*2.-1.);
 rid=pow(rid,8.);
 float ang=atan(p.y,p.x);
 float seg=(ang+3.14159)/6.28318*6.+uBeat*.0625+fbm(p*2.)*.8;
 float wn=fbm(p*2.)*2.5;
 float mB=sin(max(abs(p.x),abs(p.y))*16.+uBeat*.2+wn);
 float mC=sin(p.x*11.-uFlow*.12+wn);
 float m=smoothstep(.25,.75,.5+.5*dot(uW,vec3(sin(seg*3.14159),mB,mC)));
 vec3 vein=mix(w3(vec3(.42,.015,.04),vec3(.2,.02,.38),vec3(.3,.27,.26)),w3(vec3(.2,.01,.2),vec3(.05,.06,.36),vec3(.36,.05,.07)),uTone*.7);
 vec3 smk=w3(vec3(.13,.1,.105),vec3(.09,.08,.16),vec3(.16,.16,.17));
 vec3 col=vec3(.012,.008,.01);
 col+=smk*smoothstep(.3,.9,smoke)*(1.-.6*m)*(.55+1.1*uKeys);
 col+=vein*rid*(.4+.6*m)*(.5+1.1*uBass);
 float o=obj(uObjB,p,t,uObjBF);
 col*=1.-.85*min(1.,o*1.6);
 col+=(vein*(1.1+1.2*uKick)+vec3(.1,.085,.09))*o*(.7+.5*smoke);
 col+=vein*1.25*w;
 col+=vein*.8*exp(-d*d*(16.-9.*uKick))*uKick;
 float lx=(uMelPos-.5)*1.3+.02*sin(p.y*40.+uTime*30.);
 col+=(vein*1.4+vec3(.03))*exp(-pow((p.x-lx)*70.,2.))*uMel*step(.25,noise(vec2(p.y*9.,uTime*14.)));
 col+=vein*.3*tear*smoothstep(.4,.9,smoke);
 col.gb+=vec2(.018)*tear;
 col+=(hash(uv*uRes+fract(uTime)*91.)-.5)*(.045+.06*uHat);
 col*=.86+.14*sin(uv.y*uRes.y*1.7);
 col*=1.-.85*smoothstep(.35,1.25,d0);
 col+=((vein*1.5+vec3(.05,.03,.035))*ibeams(uv,uTime)+vec3(.9,.95,1.)*hbeam(uv,uTime)*1.1)*1.6*uStarted;
 return max(col,0.);
}

void main(){
 vec2 uv=gl_FragCoord.xy/uRes;
 vec2 p=(gl_FragCoord.xy-.5*uRes)/min(uRes.x,uRes.y);
 vec3 a=vec3(0.),b=vec3(0.);
 if(uMix<.999)a=neo(uv,p);
 if(uMix>.001)b=trap(uv,p);
 vec3 col=mix(a,b,uMix);
 col=mix(col,hue(col,uHue),uHueA);
 col*=mix(.28+.07*sin(uTime*.6),1.,uStarted);   /* 音樂開始前：暗、慢慢呼吸 */
 col=col/(1.+col*.25);
 col=pow(col,vec3(.95));
 vec4 tt=texture2D(uText,vec2(uv.x,1.-uv.y));   /* r = 提示文字、a = 它的描邊、g = 光柱名稱、b = 名稱的描邊 */
 col*=1.-.88*max(tt.b*uStarted,tt.a*uTextA);
 float tx=max(tt.r*uTextA,tt.g*uStarted);
 gl_FragColor=vec4(mix(col,vec3(.96,.92,.85),tx),1.);
}
"""

UNIFORMS = ["uRes", "uTime", "uBeat", "uFlow", "uMix", "uKickAge", "uStarted", "uTextA", "uKick", "uSnare", "uHat",
            "uBass", "uKeys", "uTone", "uMel", "uMelPos", "uHum", "uObjA", "uObjB", "uText", "uW", "uLv", "uObjOff",
            "uObjAF", "uObjBF", "uHue", "uHueA"]
NEO_OBJECTS, TRAP_OBJECTS = ["candle", "glass", "leaf", "lava"], ["cassette", "pyramid", "sub"]
EXTRA_OBJECTS = ["moon", "star", "spiral", "eye"]  # 只給自訂風格隨機挑
PRESET_STYLES = ["neo soul", "memphis rap"]  # 其他名稱都是自訂風格，依名稱決定隨機的顏色和圖案


def draw_object(name: str) -> np.ndarray:
    """網頁版用 canvas 畫的朦朧小物件，這裡用 PIL 畫成 256×256 灰階（shader 會把它暈開）。"""
    img = Image.new("L", (256, 256), 0)
    d = ImageDraw.Draw(img)
    G = lambda v: int(v * 255)
    if name == "candle":
        d.ellipse((70, 26, 186, 142), fill=G(.22))
        d.rounded_rectangle((96, 122, 160, 230), 8, fill=G(.55))
        d.ellipse((96, 116, 160, 132), fill=G(.8))
        d.rectangle((126, 106, 129, 124), fill=G(.15))
        d.polygon([(128, 48), (150, 90), (140, 110), (128, 112), (116, 110), (106, 90)], fill=G(1))
    elif name == "glass":
        d.polygon([(82, 36), (78, 80), (90, 120), (110, 140), (128, 144), (146, 140), (166, 120), (178, 80), (174, 36)],
                  fill=G(.5))
        d.polygon([(84, 84), (92, 118), (110, 136), (128, 139), (146, 136), (164, 118), (172, 84)], fill=G(.95))
        d.rectangle((124, 142, 131, 208), fill=G(.6))
        d.ellipse((86, 204, 170, 220), fill=G(.6))
    elif name == "leaf":
        d.line((128, 236, 128, 190), fill=G(.5), width=5)
        d.polygon([(128, 196), (60, 180), (24, 110), (50, 52), (90, 24), (128, 46), (166, 24), (206, 52), (232, 110),
                   (196, 180)], fill=G(.75))
        for j in range(4):
            yy, w = 74 + j * 30, 92 - j * 8
            d.line((128 - w, yy - 22, 128 - 34, yy), fill=0, width=9)
            d.line((128 + w, yy - 22, 128 + 34, yy), fill=0, width=9)
        for x, y in ((104, 84), (152, 84), (108, 128), (148, 128)):
            d.ellipse((x - 6, y - 10, x + 6, y + 10), fill=0)
        d.line((128, 50, 128, 196), fill=G(1), width=3)
    elif name == "lava":
        d.polygon([(108, 22), (148, 22), (152, 40), (104, 40)], fill=G(.75))
        d.polygon([(86, 176), (170, 176), (154, 232), (102, 232)], fill=G(.75))
        d.polygon([(104, 40), (152, 40), (170, 176), (86, 176)], fill=G(.38))
        for j in range(5):
            bx, by, r = 128 + 14 * math.sin(j * 2.3), 108 + 58 * math.sin(j * 1.9), 11 + 5 * math.sin(j * 2.5)
            d.ellipse((bx - r, by - r, bx + r, by + r), fill=G(.98))
        d.ellipse((102, 156, 154, 208), fill=G(.98))
    elif name == "cassette":
        d.rounded_rectangle((28, 62, 228, 194), 12, fill=G(.55))
        d.rounded_rectangle((44, 76, 212, 150), 6, fill=G(.85))
        d.rounded_rectangle((74, 96, 182, 134), 19, fill=G(.12))
        for cx in (96, 160):
            d.ellipse((cx - 16, 99, cx + 16, 131), fill=G(.95))
            d.ellipse((cx - 6, 109, cx + 6, 121), fill=G(.1))
        d.polygon([(62, 194), (74, 162), (182, 162), (194, 194)], fill=G(.35))
    elif name == "pyramid":
        d.polygon([(128, 40), (22, 206), (156, 206)], fill=G(.8))
        d.polygon([(128, 40), (156, 206), (234, 206)], fill=G(.38))
        for j in range(1, 6):
            yy, hw = 40 + j * 27.6, j * 27.6 * .64
            d.line((128 - hw, yy, 128 + hw, yy), fill=G(.3), width=2)
    elif name == "sub":
        d.rounded_rectangle((30, 30, 226, 226), 18, fill=G(.35))
        for r, v in ((86, .7), (72, .4), (54, .6), (24, .98)):
            d.ellipse((128 - r, 128 - r, 128 + r, 128 + r), fill=G(v))
        for x, y in ((46, 46), (210, 46), (46, 210), (210, 210)):
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=G(.8))
    elif name == "moon":
        d.ellipse((40, 40, 216, 216), fill=G(.9))
        d.ellipse((86, 22, 246, 182), fill=0)
        for x, y, r in ((200, 60, 5), (222, 120, 3), (180, 200, 4)):
            d.ellipse((x - r, y - r, x + r, y + r), fill=G(.7))
    elif name == "star":
        pts = [(128 + (100 if j % 2 == 0 else 42) * math.sin(j * math.pi / 5),
                128 - (100 if j % 2 == 0 else 42) * math.cos(j * math.pi / 5)) for j in range(10)]
        d.polygon(pts, fill=G(.85))
        d.ellipse((108, 108, 148, 148), fill=G(1))
    elif name == "spiral":
        pts = [(128 + 6 * a * math.cos(a), 128 + 6 * a * math.sin(a)) for a in np.linspace(0, 6 * math.pi, 300)]
        d.line(pts, fill=G(.9), width=10)
    elif name == "eye":
        d.polygon([(20, 128), (70, 76), (128, 60), (186, 76), (236, 128), (186, 180), (128, 196), (70, 180)], fill=G(.5))
        d.ellipse((88, 88, 168, 168), fill=G(.9))
        d.ellipse((112, 112, 144, 144), fill=G(.1))
    return np.asarray(img)[::-1].copy()  # OpenGL 的貼圖第一列在下面


BEAM_X = (.167, .333, .5, .667, .833)  # 五道光柱的位置，和 shader 的 ibeams / hbeam 一致


def draw_text(size: tuple, big: str, small: str = "", labels: tuple = ()) -> np.ndarray:
    """RGBA 貼圖：R = 中間的提示文字、A = 它的陰影；G = 光柱底下的名稱、B = 名稱的陰影（柔和的暗影，亮的光柱上也看得清楚）。"""
    img, lab, edge, msg_edge = (Image.new("L", size, 0) for _ in range(4))
    d = ImageDraw.Draw(img)
    try:
        f1, f2 = ImageFont.truetype(FONT, size[1] // 14), ImageFont.truetype(FONT, size[1] // 28)
    except OSError:
        f1 = f2 = ImageFont.load_default()
    de = ImageDraw.Draw(msg_edge)
    for text, y, font, fill in ((big, .46, f1, 255), (small, .56, f2, 190)):
        if text:
            de.text((size[0] / 2, size[1] * y), text, fill=255, font=font, anchor="mm")
            d.text((size[0] / 2, size[1] * y), text, fill=fill, font=font, anchor="mm")
    for x, name in zip(BEAM_X, labels):
        pos = (size[0] * x, size[1] * .955)
        ImageDraw.Draw(edge).text(pos, name, fill=255, font=f2, anchor="mm")
        ImageDraw.Draw(lab).text(pos, name, fill=255, font=f2, anchor="mm")
    shadow = lambda m, r, k: np.minimum(255, np.asarray(m.filter(ImageFilter.GaussianBlur(r)), np.float32) * k).astype(np.uint8)
    return np.stack([np.asarray(img), np.asarray(lab), shadow(edge, 5, 6), shadow(msg_edge, 7, 4)], axis=2)


class Show:
    """畫面狀態：接收樂團的資料，每一幀更新數值（和網頁版 frame() 一樣的衰減、起音判斷）。"""

    def __init__(self) -> None:
        self.lines: "queue.Queue[str]" = queue.Queue()
        self.t0 = self.last = time.perf_counter()
        self.scene, self.mix, self.beats, self.flow = 0, 0.0, 0.0, 0.0
        self.kick = self.snare = self.hat = self.bass = self.keys = self.mel = self.hum = self.pad = 0.0  # pad：鍵盤＋弦樂（背景）
        self.kick_age, self.tone, self.tone_t, self.mel_pos, self.mel_pos_t = 9.0, .2, .2, .5, .5
        self.started, self.started_f, self.notes, self.prev = False, 0.0, set(), {}
        self.text, self.text_t, self.text_a, self.text_until = None, ("樂團準備中…", ""), 0.0, 1e9
        self.conn = None  # 連到樂團的 socket，按鍵時送指令回去
        self.labels, self.labels_on = ("鼓", "貝斯", "你", "Rhodes", "吉他＋鐘"), False  # 光柱名稱（樂團會更新），S 鍵顯示/隱藏
        self.instruments = ["Rhodes", "電吉他", "電貝斯", "鼓", "弦樂", "鐘"]  # 1–6 靜音鍵對應的樂器名稱（樂團會更新）
        self.muted, self.paused, self.typing = set(), False, None  # muted：靜音中的鍵號；typing：正在打的風格（None = 沒在打字）
        self.style, self.style_list = "neo soul", list(PRESET_STYLES)  # 目前的風格、選單裡的風格（樂團會送來）
        self.hue = self.hue_next = self.hue_a = 0.0  # 自訂風格的色相（弧度）與強度
        self.custom_objects = None  # 自訂風格的圖案（None = 用原本的）

    def onset(self, key: str, v: float, ratio: float, floor: float) -> bool:
        p = self.prev.get(key, 0.0)
        self.prev[key] = p + (v - p) * .3
        return v > p * ratio + .04 and v > floor

    def show_text(self, big: str, small: str = "", seconds: float = 1e9) -> None:
        self.text_t, self.text_until = (big, small), time.perf_counter() + seconds

    def hint(self) -> None:
        """游標移動時提醒按鍵（開始前的等待畫面本來就寫著）。"""
        if self.started:
            self.show_text(self.style, "點一下畫面（或 M）：選風格   Enter：輸入風格   1–6：樂器靜音   S：光柱名稱   P：暫停   Q：結束", 3)

    def send(self, command: str) -> None:
        """把指令送回樂團：g <風格> 切換風格、m1–m6 樂器靜音、p 暫停/繼續、q 結束、t <文字> 新風格。"""
        try:
            if self.conn:
                self.conn.sendall((command + "\n").encode())
        except OSError:
            pass

    def mute(self, n: int) -> None:
        self.muted.symmetric_difference_update({n})
        self.send(f"m{n}")
        self.show_text(f"{self.instruments[n - 1]} {'靜音' if n in self.muted else '關閉靜音'}",
                       "靜音中：" + "、".join(self.instruments[i - 1] for i in sorted(self.muted)) if self.muted else "", 3)

    def set_style(self, name: str) -> None:
        """換風格時決定畫面：原本的兩種用原本的配色和圖案；自訂風格依名稱算出固定的隨機色相和 3 個圖案。"""
        self.style = name
        if name in PRESET_STYLES:
            self.hue_next, self.custom_objects = None, None
        else:
            rng = random.Random(zlib.crc32(name.encode()))
            self.hue_next = rng.uniform(.6, 5.7)  # 避開接近原本色相的角度
            self.custom_objects = rng.sample(NEO_OBJECTS + TRAP_OBJECTS + EXTRA_OBJECTS, 3)

    def choose(self, name: str) -> None:
        """選單選了一個風格。"""
        self.send(f"g {name}")

    def submit(self) -> None:
        """打字結束（Enter）：把風格描述送給樂團，樂團準備好後在下一段切過去。"""
        text, self.typing = (self.typing or "").strip(), None
        if text:
            self.send(f"t {text}")
            self.show_text(text, "送出了，準備中…", 30)

    def pause(self) -> None:
        self.paused = not self.paused
        self.send("p")
        if self.paused:
            self.show_text("暫停", "按 P 繼續")
        else:
            self.show_text(self.style, "", 2)

    def handle(self, line: str) -> None:
        if line.startswith("f "):
            k, s, h, b, ky, m, tone, mpos, pad, _beat, sidx = map(float, line[2:].split())
            if k > .25:  # 大鼓、小鼓、hi-hat 是鼓手實際打下去的力度，只在打擊那一幀不是 0
                self.kick = max(self.kick, min(1.0, .3 + k))
                self.kick_age = 0.0
            if s > .25:
                self.snare = max(self.snare, min(.9, .2 + s))
            self.hat, self.bass = max(self.hat, min(1, h)), max(self.bass, min(1, b))
            self.keys, self.pad, self.tone_t = max(self.keys, min(1, ky)), max(self.pad, min(1, pad)), tone
            if self.onset("m", m, 1.25, .15) and not self.notes:
                self.mel_pos_t = mpos
            self.mel = max(self.mel, min(1, m))
            self.scene = int(sidx)  # 視覺的配色；換風格的文字由樂團的 c| 訊息顯示
        elif line.startswith("n "):
            down, note = map(int, line[2:].split())
            if down:
                self.notes.add(note)
                self.hum, self.mel_pos_t = 1.0, min(1.0, max(0.0, (note - 48) / 36))
            else:
                self.notes.discard(note)
        elif line.startswith("w|"):  # 等待觀眾彈和弦：w|目前選的風格|配色
            style, scene = line[2:].split("|")
            self.started, self.scene = False, int(scene)
            self.set_style(style)
            self.show_text("彈一個和弦，樂團就會跟著你", f"風格：{style}（點一下畫面選風格）")
        elif line.startswith("c|"):  # 剛切換了風格
            self.set_style(line[2:])
            self.show_text(line[2:], "風格切換", 4)
        elif line.startswith("y|"):  # 選單裡的風格
            self.style_list = line[2:].split("|")
        elif line.startswith("t|"):  # 樂團要顯示的文字：t|大字|小字|秒數
            big, small, seconds = line[2:].split("|")
            self.show_text(big, small, float(seconds))
        elif line.startswith("i|"):  # 1–6 靜音鍵對應的樂器名稱
            self.instruments = line[2:].split("|")
        elif line.startswith("p|"):  # 已經按了切換，下一段生效
            self.show_text(line[2:], "下一段開始")
        elif line.startswith("l|"):  # 五道光柱的名稱
            self.labels = tuple(line[2:].split("|"))
        elif line.startswith("k|"):  # 開始：調名|風格
            key_name, style = line[2:].split("|")
            self.started = True
            self.set_style(style)
            self.show_text(key_name, style, 4)

    def step(self) -> None:
        now = time.perf_counter()
        dt, self.last = min(.1, now - self.last), now
        while not self.lines.empty():
            self.handle(self.lines.get())
        sc = self.scene
        if self.paused:
            dt = 0.0  # 畫面凍住
        self.beats += dt * BPM * (2 if sc else 1) / 60
        self.mix += (sc - self.mix) * min(1, dt * 2.2)
        self.started_f += ((1.0 if self.started else 0.0) - self.started_f) * min(1, dt * .8)
        self.kick_age += dt
        self.flow += dt * (1 + self.bass * (2.6 if sc else 2))
        self.tone += (self.tone_t - self.tone) * min(1, dt * 1.5)
        self.mel_pos += (self.mel_pos_t - self.mel_pos) * min(1, dt * 12)
        self.text_a += ((1.0 if now < self.text_until or self.typing is not None else 0.0) - self.text_a) * min(1, dt * 2)
        if self.hue_next is not None and abs(self.hue_next - self.hue) > 1e-3 and self.hue_a > .02:
            self.hue_a += (0 - self.hue_a) * min(1, dt * 1.5)  # 換到另一個自訂色：先淡出舊的
        else:
            if self.hue_next is not None:
                self.hue = self.hue_next
            self.hue_a += ((0.0 if self.hue_next is None else 1.0) - self.hue_a) * min(1, dt * .8)
        self.kick *= math.exp(-dt * (8 if sc else 3))
        self.snare *= math.exp(-dt * 9)
        self.hat *= math.exp(-dt * 14)
        self.bass *= math.exp(-dt * (3 if sc else 2.5))
        self.keys *= math.exp(-dt * (1.5 if sc else .9))
        self.pad *= math.exp(-dt * (1.5 if sc else .9))
        self.mel *= math.exp(-dt * 5)
        self.hum = max(self.hum * math.exp(-dt * 4), .7 if self.notes else 0.0)


class VJ(OpenGLFrame):
    def __init__(self, master, show: Show, shot=None, **kw) -> None:
        super().__init__(master, **kw)
        self.show, self.shot, self.prog, self.frames = show, shot, None, 0
        self.obj_idx, self.text_shown = [-1, -1], None

    def initgl(self) -> None:
        if self.prog:
            self.make_fbo()
            return
        def compile_shader(kind, src):
            s = GL.glCreateShader(kind)
            GL.glShaderSource(s, src)
            GL.glCompileShader(s)
            if not GL.glGetShaderiv(s, GL.GL_COMPILE_STATUS):
                raise SystemExit(GL.glGetShaderInfoLog(s).decode())
            return s
        self.prog = GL.glCreateProgram()
        GL.glAttachShader(self.prog, compile_shader(GL.GL_VERTEX_SHADER, VERT))
        GL.glAttachShader(self.prog, compile_shader(GL.GL_FRAGMENT_SHADER, FRAG))
        GL.glBindAttribLocation(self.prog, 0, "a")
        GL.glLinkProgram(self.prog)
        GL.glUseProgram(self.prog)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, GL.glGenBuffers(1))
        GL.glBufferData(GL.GL_ARRAY_BUFFER, np.array([-1, -1, 3, -1, -1, 3], np.float32), GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, 0, None)
        self.U = {n: GL.glGetUniformLocation(self.prog, n) for n in UNIFORMS}
        self.tex = GL.glGenTextures(4)  # 0 = neo 物件、1 = trap 物件、2 = 文字、3 = 離屏畫布
        for unit, name in ((0, "uObjA"), (1, "uObjB"), (2, "uText")):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex[unit])
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            GL.glUniform1i(self.U[name], unit)
        self.fbo = GL.glGenFramebuffers(1)
        self.make_fbo()
        gpu = GL.glGetString(GL.GL_RENDERER).decode()
        print("GPU:", gpu, flush=True)
        if "NVIDIA" in gpu:  # 跟生成音樂搶同一張卡會斷音（測過：618 次）；run_band.bat 會把 python 設成用內顯
            print("[視覺] 正在用 RTX 3050 畫圖，會跟音樂搶 GPU。請用 run_band.bat 啟動（它會改成用內顯）。", flush=True)

    def make_fbo(self) -> None:  # 離屏畫布：寬 RENDER_WIDTH，比例跟視窗一樣
        self.rw, self.rh = RENDER_WIDTH, max(2, round(RENDER_WIDTH * self.height / max(1, self.width)))
        GL.glActiveTexture(GL.GL_TEXTURE3)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex[3])
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, self.rw, self.rh, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex[3], 0)
        self.text_shown = None

    def upload(self, unit: int, img: np.ndarray) -> None:
        GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex[unit])
        fmt = {3: GL.GL_RGB, 4: GL.GL_RGBA}.get(img.shape[2], GL.GL_RGB) if img.ndim == 3 else GL.GL_LUMINANCE
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, img.shape[1], img.shape[0], 0, fmt,
                        GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(img))

    def redraw(self) -> None:
        s = self.show
        s.step()
        t = time.perf_counter() - s.t0
        # 朦朧物件：neo 每 16 拍換一個、trap 每 32 拍，淡入淡出
        fades = []
        for unit, (objs, per) in enumerate(((NEO_OBJECTS, 16), (TRAP_OBJECTS, 32))):
            objs = s.custom_objects if s.custom_objects and unit == s.scene else objs  # 自訂風格用它自己的隨機圖案
            k, f = divmod(s.beats / per, 1)
            name = objs[int(k) % len(objs)]
            if name != self.obj_idx[unit]:
                self.obj_idx[unit] = name
                self.upload(unit, draw_object(name))
            a = min(1, f * 4, (1 - f) * 4)
            fades.append(a * a * (3 - 2 * a))
        shown = (s.typing + "▏", "輸入一個風格（英文描述效果最好）  Enter 送出、Esc 取消") if s.typing is not None else s.text_t
        text = (shown, s.labels if s.labels_on else ())
        if text != self.text_shown:
            self.text_shown = text
            self.upload(2, draw_text((self.rw, self.rh), *text[0], labels=text[1]))
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        GL.glViewport(0, 0, self.rw, self.rh)
        U = self.U
        GL.glUniform2f(U["uRes"], self.rw, self.rh)
        for n, v in (("uTime", t), ("uBeat", s.beats), ("uFlow", s.flow), ("uMix", s.mix), ("uKickAge", s.kick_age),
                     ("uStarted", s.started_f), ("uTextA", s.text_a), ("uKick", s.kick), ("uSnare", s.snare),
                     ("uHat", s.hat), ("uBass", s.bass), ("uKeys", s.pad), ("uTone", s.tone), ("uMel", s.mel),
                     ("uMelPos", s.mel_pos), ("uHum", s.hum), ("uObjAF", fades[0]), ("uObjBF", fades[1]),
                     ("uHue", s.hue), ("uHueA", s.hue_a)):
            GL.glUniform1f(U[n], v)
        vv = s.beats / (128 if s.scene else 64)  # 三種配色變化，每 64/128 拍輪替
        vi, vf = int(vv) % 3, vv % 1
        vb = min(1, max(0, (vf - .88) / .12))
        vb = vb * vb * (3 - 2 * vb)
        w3 = [0.0, 0.0, 0.0]
        w3[vi], w3[(vi + 1) % 3] = 1 - vb, vb
        GL.glUniform3f(U["uW"], *w3)
        GL.glUniform4f(U["uLv"], min(1, max(s.kick, s.snare)), min(1, s.bass), min(1, s.keys), min(1, s.mel))
        GL.glUniform2f(U["uObjOff"], math.sin(t * .21) * .04, math.cos(t * .17) * .03)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, 0)
        GL.glBlitFramebuffer(0, 0, self.rw, self.rh, 0, 0, self.width, self.height, GL.GL_COLOR_BUFFER_BIT, GL.GL_LINEAR)
        self.frames += 1
        if self.shot and s.started:  # --shot 檔名 秒數：音樂開始幾秒後存一張畫面（測試用）
            self.shot_t0 = getattr(self, "shot_t0", t)
        if self.shot and s.started and t - self.shot_t0 >= self.shot[1]:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            px = GL.glReadPixels(0, 0, self.rw, self.rh, GL.GL_RGB, GL.GL_UNSIGNED_BYTE)
            Image.frombytes("RGB", (self.rw, self.rh), px).transpose(Image.FLIP_TOP_BOTTOM).save(self.shot[0])
            print(f"shot saved, {self.frames / t:.0f} fps", flush=True)
            self.shot, self.animate = None, 0
            self.master.after(50, self.master.destroy)


def receive(show: Show, stop: threading.Event, on_lost) -> None:
    """連到 WSL 的樂團，一行一行讀進 show.lines；樂團結束（斷線）時關閉視窗。"""
    while not stop.is_set():
        try:
            conn = socket.create_connection(VISUAL_TCP, timeout=1)
        except OSError:
            stop.wait(0.5)
            continue
        conn.settimeout(0.5)
        show.conn, buf = conn, b""
        while not stop.is_set():
            try:
                data = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                show.lines.put(line.decode())
        conn.close()
        on_lost()
        return


def demo(show: Show, stop: threading.Event) -> None:
    """沒有樂團時用的假資料：等 4 秒 → 開始 → 固定鼓點，每 8 小節換風格，偶爾彈音。"""
    show.lines.put("y|neo soul|memphis rap|rainy night lo-fi")
    show.lines.put("w|neo soul|0")
    stop.wait(4)
    show.lines.put("k|D minor|neo soul")
    f = 0
    while not stop.is_set():
        step, bar = (f * BPM * 4 // 1500) % 16, f * BPM // 1500 // 4
        sc = (bar // 8) % 2
        hit = f == 0 or f * BPM * 4 // 1500 != (f - 1) * BPM * 4 // 1500  # 這一幀是 16 分音符的開頭（鼓只在打擊那一幀有值）
        kick = 1.0 if hit and step in ((0, 7, 10) if sc == 0 else (0, 6, 10)) else 0
        snare = 1.0 if hit and step in ((4, 12) if sc == 0 else (8,)) else 0
        hat = .6 if hit and (step % 2 == 0 or sc) else 0
        keys = .8 if step == 0 else .3
        mel = .9 if (step in (2, 7, 11, 14)) else .2
        show.lines.put(f"f {kick:.2f} {snare:.2f} {hat:.2f} {.8 if step in (0, 7, 10) else .2:.2f} {keys:.2f} {mel:.2f} .4 "
                       f"{.3 + .4 * (step / 16):.2f} {keys + .2:.2f} 0 {sc}")
        if f % 50 == 25:
            note = 55 + (f // 50) % 24
            show.lines.put(f"n 1 {note}")
            threading.Timer(0.6, lambda n=note: show.lines.put(f"n 0 {n}")).start()
        f += 1
        stop.wait(0.04)


def main() -> None:
    args = sys.argv[1:]
    shot = (args[args.index("--shot") + 1], float(args[args.index("--shot") + 2])) if "--shot" in args else None
    show, stop = Show(), threading.Event()
    root = tk.Tk()
    root.title("AI Duet")
    root.configure(bg="black")
    if shot:
        root.geometry("1280x720")
    else:
        root.attributes("-fullscreen", True)
    vj = VJ(root, show, shot, width=1280, height=720)
    vj.pack(fill=tk.BOTH, expand=True)
    vj.animate = 25  # 最多約 40 fps；畫得更快只是多佔 GPU
    def on_key(e) -> None:
        """所有按鍵。打字時字都進輸入框（Enter 送出、Esc 取消），其他鍵不作用。"""
        if show.typing is not None:
            if e.keysym == "Return":
                show.submit()
            elif e.keysym == "Escape":
                show.typing = None
            elif e.keysym == "BackSpace":
                show.typing = show.typing[:-1]
            elif e.char and e.char.isprintable() and len(show.typing) < 60:
                show.typing += e.char
            return
        key = e.keysym.lower()
        if key == "return":
            show.typing = ""
        elif key == "escape":
            root.attributes("-fullscreen", False)
        elif key == "f":
            root.attributes("-fullscreen", not root.attributes("-fullscreen"))
        elif key == "q":
            show.send("q")  # 連樂團一起結束
            root.destroy()
        elif key == "m":
            open_menu()
        elif key == "p":
            show.pause()
        elif key == "s":
            show.labels_on = not show.labels_on
        elif e.char in ("1", "2", "3", "4", "5", "6"):
            show.mute(int(e.char))

    def open_menu(e=None) -> None:
        """風格選單：現有的風格（● = 現在的），最後一項是輸入新風格。"""
        menu = tk.Menu(root, tearoff=0, font=("Microsoft JhengHei", 20))
        for name in show.style_list:
            menu.add_command(label=("● " if name == show.style else "    ") + name, command=lambda n=name: show.choose(n))
        menu.add_separator()
        menu.add_command(label="＋ 輸入新風格…", command=lambda: setattr(show, "typing", ""))
        x, y = (e.x_root, e.y_root) if e else (root.winfo_rootx() + root.winfo_width() // 2, root.winfo_rooty() + root.winfo_height() // 3)
        menu.tk_popup(x, y)

    root.bind("<Key>", on_key)
    root.bind("<Button-1>", open_menu)
    root.bind("<Motion>", lambda e: show.hint())
    if "--demo" in args:
        threading.Thread(target=demo, args=(show, stop), daemon=True).start()
    else:
        threading.Thread(target=receive, args=(show, stop, lambda: root.after(0, root.destroy)), daemon=True).start()
    root.mainloop()
    stop.set()


if __name__ == "__main__":
    main()
