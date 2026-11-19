# 鼓組取樣（不包含在 repo 裡）

鼓手用 one-shot 取樣發聲。取樣檔案有授權限制，所以這個資料夾在 repo 裡是空的，要自己放鼓組。

## 最簡單：每種風格一個資料夾、固定檔名

```
samples/
  neo soul/      kick.wav  snare.wav  hihat.wav  openhat.wav  rim.wav
  memphis rap/   kick.wav  snare.wav  hihat.wav  openhat.wav  rim.wav  808.wav
```

- 資料夾名稱要和 `band/settings.py` 裡 `STYLES[...]["kits"]` 寫的一樣。
- 任何取樣率、單聲道或立體聲的 wav 都可以，鼓手會自己轉成 48 kHz 立體聲、統一音量。
- `808.wav` 是 memphis 的 808 貝斯，會自動量音高、調到和弦根音。

## 整包音色包

也可以把整包音色包解壓成一個資料夾（例如 `samples/trap/`），再把名稱加進 `kits`。
鼓手會照子資料夾和檔名自動找出各個角色，每次輪到時隨機挑一個：

| 角色 | 路徑裡要有的字 |
|------|----------------|
| kick | kick |
| snare | snare（沒有就用 clap） |
| hihat | hi hat、hihat、hi-hat、closed hat |
| openhat | open hat |
| rim | rim（沒有就用 perc、stick、snap、clap） |
| 808 | 808 |

- 資料夾名稱有 loop、fill、fx、vox 這類字的會跳過。
- 太長的檔案（多半是 roll 或 loop）也會跳過。
- 規則在 `band/drummer.py` 的 `KIT_ROLES`、`KIT_SKIP`、`KIT_MAX_S`。

某種風格一個鼓組都沒有時，鼓手會安靜，其他樂器照常演奏。
