# Nico Live Subtitle

Windows 桌面实时字幕 MVP：捕获默认播放设备的系统音频，识别日语，并在置顶透明窗口中显示日文原文和中文翻译。它不读取 niconico 页面，也不会处理或遮挡弹幕数据，因此同样适用于浏览器、播放器和游戏。

> 当前状态：Alpha。纯逻辑测试已通过，但尚未在真实 Windows 声卡、Whisper 模型和 niconico 播放场景中完成端到端验证。

## 功能

- Windows WASAPI 回环录音，不使用麦克风
- `faster-whisper` 日语识别，支持 CPU / NVIDIA GPU
- 日文字幕先显示，翻译完成后补充中文字幕
- Google 在线翻译、Argos 离线翻译或仅显示原文
- 透明置顶悬浮窗、字号/透明度调整、点击穿透
- 音频设备枚举、自定义 Whisper 模型路径、日语热词

## 安装

建议使用 Python 3.11。PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

安装和首次运行都会在工作目录之外产生环境或模型缓存，请确认后再执行。按模型名称首次启动时，`faster-whisper` 会从 Hugging Face 下载模型；也可以在设置中填写已经下载好的 CTranslate2 模型目录。

## 运行

```powershell
nico-live-subtitle
```

加载示例配置：

```powershell
nico-live-subtitle --config .\config.example.json
```

只列出可用的系统音频回环设备：

```powershell
nico-live-subtitle --list-devices
```

启动后先播放一段日语视频，再点击“开始”。字幕窗口可以拖动；右键或托盘菜单可以开关点击穿透。开启点击穿透后，请通过系统托盘菜单恢复交互。

## 推荐配置

| 环境 | 模型 | 设备 | 计算类型 |
| --- | --- | --- | --- |
| 无独显、优先低延迟 | `base` | `cpu` | `int8` |
| 较新的桌面 CPU | `small` | `cpu` | `int8` |
| NVIDIA 显卡 | `small` / `medium` | `cuda` | `float16` |

`device=auto` 会在 CTranslate2 检测到 CUDA 时选 GPU，否则使用 CPU。当前版按能量检测语音并在约 1.8 秒时生成临时字幕，在停顿约 0.65 秒时确认整句；实际延迟取决于模型和硬件。

## 翻译后端

- `google`：默认，依赖网络，使用 `deep-translator`；第三方服务变化或网络限制可能导致翻译失败。
- `argos`：完全离线，但需要额外安装 `argostranslate`，并由用户自行安装日语到中文语言包。
- `none`：只显示日文原文，用于离线识别或排查性能问题。

在线翻译失败不会中断日语识别，状态栏会显示错误。软件不会保存音频和字幕历史。

## 已知边界

- 动画配乐、角色重叠说话、专有名词会降低识别和翻译质量；可以在设置中添加作品名、角色名等热词。
- WASAPI 默认采集整个默认播放设备，而不是只采集 Chrome。系统通知和其他应用声音也会进入识别。
- SoundCard 在 Windows 上单声道回环存在已知问题，因此程序先采集多声道，再在内存中混为单声道。
- 这是低延迟分段识别，不是逐字同步；长时间连续台词会按照最大句长强制切段。
- GPU 模式依赖与当前 `faster-whisper` / CTranslate2 兼容的 CUDA 与 cuDNN 运行库；无法加载时程序会报告错误，可改用 CPU。

## 测试

纯逻辑测试不需要声卡、模型或 Qt：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m unittest discover -s tests -v
```

## 参与贡献与许可证

欢迎提交 Issue 和 Pull Request。贡献前请阅读 `CONTRIBUTING.md`。本项目采用 MIT License。
