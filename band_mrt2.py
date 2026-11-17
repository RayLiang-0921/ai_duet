"""
band_mrt2.py — AI 樂團的入口。
  WSL 端：python band_mrt2.py [--no-visual] [--record 檔名]  啟動樂團（程式在 band/ 套件裡，設定在 band/settings.py）
  Windows 端：python band_mrt2.py --midi-bridge                 由樂團自動啟動，讀 MIDI 鍵盤、發鋼琴聲
平常請雙擊 run_band.bat。
"""
import sys

if __name__ == "__main__":
    if "--midi-bridge" in sys.argv:
        from band.links import midi_bridge
        midi_bridge()
    else:
        from band.band import main
        main()
