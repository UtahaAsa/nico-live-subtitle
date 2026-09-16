# Nico Live Subtitle

Windows 桌面实时字幕 MVP：捕获默认播放设备的系统音频，识别日语，并在置顶透明窗口中显示日文原文和中文翻译。它不读取 niconico 页面，也不会处理或遮挡弹幕数据，因此同样适用于浏览器、播放器和游戏。

> 当前状态：Alpha。纯逻辑测试、Windows 回环采集以及本地模型组件烟测已通过，但仍需更多 niconico 实际片段验证识别和翻译质量。

## 功能

- Windows WASAPI 回环录音，不使用麦克风
- Anime-Whisper 动画日语识别，也可切回 `faster-whisper`
- Silero 神经网络 VAD、动态停顿切句、长句低置信度位置切分
- 仅翻译已经稳定的完整句，避免临时识别结果反复改写译文
- 本地 Qwen、OpenAI 兼容接口、HY-MT、Google、Argos 多种翻译后端
- LLM 流式翻译、日中成对上下文和动画专用提示词
- 可搜索的作品词库包，自动叠加角色名、地名、招式和固定译名
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

基础安装默认提供 `faster-whisper` 和 Google 翻译。Anime-Whisper、Silero VAD、本地 LLM 和兼容接口为可选能力，安装方式见下文。

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

## 识别与切句

| 环境 | 模型 | 设备 | 计算类型 |
| --- | --- | --- | --- |
| 无独显、优先低延迟 | `base` | `cpu` | `int8` |
| 较新的桌面 CPU | `small` | `cpu` | `int8` |
| NVIDIA 显卡 | `small` / `medium` | `cuda` | `float16` |

看动画推荐使用 `anime_whisper` 和 Silero VAD。Anime-Whisper 针对动画、Galgame 式演技日语微调；该模型不适合 `initial_prompt`，因此程序不会向它传递日语热词。Silero 模式固定使用 32ms 音频块，通过神经网络概率判断语音，并随台词变长逐步缩短停顿阈值。

在 Windows + NVIDIA 显卡上的参考安装命令：

```powershell
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch==2.14.0+cu130
python -m pip install -e ".[anime-asr]"
python -c "from huggingface_hub import snapshot_download; snapshot_download('litagin/anime-whisper', local_dir='work/models/anime-whisper')"
New-Item -ItemType Directory -Force work/models/silero-vad
Invoke-WebRequest https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit -OutFile work/models/silero-vad/silero_vad.jit
```

## 作品词库

设置中的“作品词库”代替了不断增长的单行热词和术语表。程序始终加载 `anime-common` 通用词库，再叠加当前选择的一部作品；“自定义热词”和“自定义术语”只用于少量个人修正，并且自定义译名优先于内置译名。

当前内置：

- Re:从零开始的异世界生活
- BLEACH 千年血战篇
- 无职转生
- 葬送的芙莉莲
- 药屋少女的呢喃

词库位于 `lexicons/`，每部作品一个 JSON 文件。新增作品不需要修改 Python 代码，复制下面的格式后在设置中点击“刷新词库”即可：

```json
{
  "schema_version": 1,
  "id": "example-anime",
  "title": "示例动画",
  "aliases": ["作品日文标题"],
  "hotwords": ["只需识别、不需要固定翻译的词"],
  "terms": [
    {"source": ["角色全名", "角色昵称"], "target": "简体中文固定译名"}
  ]
}
```

Anime-Whisper 不直接接收热词，作品词库会进入 LLM 提示词，用于修正明显的名字识别错误并统一译名；选择 Faster-Whisper 时，同一批日文词还会自动作为 ASR 热词。不同地区的官方译名可能不一致，可在“自定义术语”中覆盖，格式仍为 `日文=中文`。

## 翻译后端

- `local_llm`：使用本地 GGUF 通用 LLM 和动画字幕提示词，支持流式输出、术语表以及最近几组日中译文上下文。
- `openai_compatible`：质量优先，可连接 Ollama、LM Studio 或远程 OpenAI 兼容服务。接口地址和模型名写入配置；密钥只从 `api_key_env` 指定的环境变量读取，不写入配置文件。
- `google`：默认，依赖网络，使用 `deep-translator`；第三方服务变化或网络限制可能导致翻译失败。
- `hunyuan`：轻量离线方案，使用 HY-MT1.5-1.8B GGUF 直接日译中。
- `argos`：轻量离线方案，需要额外安装 `argostranslate`。当前官方索引没有日语到中文直连包，只能由日语到英语、英语到中文两个包组合翻译，因此质量通常低于 HY-MT；`packages_dir` 用于指定语言包目录。
- `none`：只显示日文原文，用于离线识别或排查性能问题。

在线翻译失败不会中断日语识别，状态栏会显示错误。软件不会保存音频和字幕历史。

### 安装本地 Qwen 翻译

Windows + NVIDIA 显卡推荐安装官方 Vulkan 预编译包，避免依赖本机 CUDA Toolkit：

```powershell
python -m pip install --only-binary=:all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/vulkan "llama-cpp-python>=0.3.16,<0.4"
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen3-4B-GGUF', 'Qwen3-4B-Q4_K_M.gguf', local_dir='work/models/qwen3-4b')"
```

然后在设置中选择“本地动画 LLM”，模型路径填写：

```text
work/models/qwen3-4b/Qwen3-4B-Q4_K_M.gguf
```

`context_lines` 控制携带多少组历史日文及其中文译文；`n_gpu_layers=-1` 表示尽量把全部层放到 GPU。Qwen3 和 Anime-Whisper 均使用 MIT/Apache-2.0 兼容的开放许可，但模型权重仍不包含在本仓库中。

HY-MT 仍作为兼容后端保留。它采用独立的 Tencent HY Community License，许可地域不包括欧盟、英国和韩国；使用前请阅读[模型仓库的完整许可](https://huggingface.co/tencent/HY-MT1.5-1.8B-GGUF/blob/main/License.txt)。

## 已知边界

- 动画配乐、角色重叠说话仍会降低识别质量；Anime-Whisper 不使用热词，角色译名请配置在翻译术语表中。
- Anime-Whisper 与本地 LLM 会同时占用显存；显存不足时可减少 `n_gpu_layers`，把一部分翻译模型放到 CPU。
- 本地 LLM 首次翻译需要 Vulkan 着色器预热，之后延迟会明显降低。4B 模型仍可能误译；质量要求更高时应使用 `openai_compatible` 后端连接更强模型。
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

## 设计参考

- [LiveTranslate](https://github.com/TheDeathDragon/LiveTranslate)：Anime-Whisper、Silero VAD、稳定句提交、上下文 LLM 翻译等整体思路。
- [AutoTranslation](https://github.com/felenko/AutoTranslation)：跨片段文本去重与音频边界处理思路。

本项目没有直接打包上述项目的源码或模型；具体实现保持在本仓库内，并遵循各上游项目和模型各自的许可证。

## 参与贡献与许可证

欢迎提交 Issue 和 Pull Request。贡献前请阅读 `CONTRIBUTING.md`。本项目采用 MIT License。
