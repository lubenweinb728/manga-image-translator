# Manga Image Translator · v0.1

**该版本为自用项目，欢迎继续开发**

基于 [manga-image-translator](https://github.com/zyddnys/manga-image-translator) 的漫画图片翻译分支，支持检测、OCR、翻译、修复与排版，增加 Intel OpenVINO 适配。此处 v0.1 是首次正式整理发布的版本，忽略开发期间的计划版本编号。

[修改明细](RELEASE_NOTES_v0.1.md) · [OpenVINO 说明](OPENVINO_CN.md) · [许可证](LICENSE)

## 环境安装

建议使用 Python 3.10（本版在 Windows / Python 3.10.11 验证；其他 Python 版本需自行确认依赖兼容性）。安装 Git 和 C++ 编译工具；Windows 使用 Microsoft C++ Build Tools 的 C++ 桌面开发组件，Ubuntu 可安装 `git build-essential python3-venv libgl1 libglib2.0-0`。`pydensecrf` 从源码构建，因此使用源码 ZIP 也需要 Git。

```powershell
git clone --branch v0.1 https://github.com/lubenweinb728/manga-image-translator.git
cd manga-image-translator
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip "setuptools>=68,<82" wheel
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

以下命令中的 `python` 指已配置好依赖的环境。Windows 可执行 `.\.venv\Scripts\Activate.ps1` 激活，或直接使用 `.\.venv\Scripts\python.exe`。Linux/macOS 使用 `source .venv/bin/activate`，安装命令为 `python -m pip install -r requirements.txt`。核显模式需要 Intel 显卡驱动与 OpenVINO GPU 运行环境；CPU 模式不需要。

```powershell
# 仅在需要 OpenVINO 时安装
python -m pip install -r requirements-openvino.txt
# 仅在需要桌面界面时安装
python -m pip install -r requirements-gui.txt
```

模型权重首次使用时自动下载到 `models/`，不包含在源码中。默认字体首次排版时自动下载到 `fonts/` 并校验哈希；离线准备和自定义字体见 [字体说明](fonts/README.md)。运行目录需要可写，首次运行需要连接模型下载源和 GitHub。翻译器的密钥可参考 `examples/Example.env`，复制为根目录 `.env` 后按实际服务填写；不要提交该文件。

## 命令行运行

在项目根目录执行，将 `path/to/page.png` 替换为自己的图片路径：

```powershell
# 原版 CPU 路径
python -m manga_translator local -i path/to/page.png --config-file examples/config-openvino-ctc.json
# 启用 OpenVINO
python -m manga_translator local -i path/to/page.png --config-file examples/config-openvino-ctc.json --use-igpu
# 本项目自用配置：超分、CRAFT、CTC、Papago 简体中文、LaMa Large
python -m manga_translator local -i path/to/page.png --config-file examples/config-openvino.json --use-igpu
```

单张图片输出为 `result/final.png`；目录批量输入默认输出到输入目录旁的 `输入目录名-translated`。Papago 等在线服务需要联网；其他翻译器可能需要自己的 API 密钥。测试识别和修复而不调用在线翻译时，可在配置副本中设置 `translator.translator` 为 `original`，此时只回填原文，不生成翻译。

Windows 也可使用 `run.bat local ...`，Linux 使用 `bash run.sh local ...`。脚本只启动程序，不自动拉取或更新代码。

```powershell
python -m manga_translator --help
python -m manga_translator config-help
```

`config-help` 输出完整配置结构。正式配置示例位于 `examples/`；`config-openvino-default.json` 使用默认检测器，`config-openvino-ctc.json` 使用默认检测器和 CTC OCR。配置名不自动启用 OpenVINO，仍需传 `--use-igpu`。

## 桌面界面

安装 `requirements-gui.txt` 后，在项目根目录执行：

```powershell
python MangaStudioMain.py
```

Windows 可双击 `MangaStudioMainRun.bat`。虚拟环境选择顺序：已激活环境、`.venv`、`venv`、系统 Python。个人预设和缓存首次使用时创建；源码不携带开发者个人配置。

## 网页与 API

后端在项目根目录启动（默认监听本机 8000 端口）：

```powershell
python server/main.py --host 127.0.0.1 --port 8000 --start-instance
```

访问 http://127.0.0.1:8000 和 http://127.0.0.1:8000/docs 。可选 React 前端需要 Node.js 20 或更新版本，在另一个终端执行：

```powershell
cd front
npm ci
npm run dev
```

开发页面默认 http://localhost:5173 。生产构建使用 `npm run build`，启动使用 `npm start`。当前服务器启动参数保留原版 CPU/GPU 路径，OpenVINO 使用上面的 CLI；不要把 `--use-igpu` 直接传给服务器。Docker 相关文件保留为构建入口，本版以本机 Python 流程为主要验证范围。

## 回归测试

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest test
```

OpenVINO 用例需安装 `requirements-openvino.txt`，否则部分用例自动跳过。保留的翻译器测试使用模拟网络，不要求真实 API 密钥；排版测试需要默认字体可下载或已离线准备。测试生成的图片、缓存和日志不上传。

## 限制与授权

OpenVINO 首次转换编译开销较大，性能依赖模型、输入尺寸及硬件；不承诺所有模块均加速。详情见 [OpenVINO 说明](OPENVINO_CN.md)。默认 Noto 字体覆盖 CJK、拉丁文字与阿拉伯文，其他文字可能需要 `--font-path` 指定合适字体。

代码遵循 GPL-3.0，保留上游作者声明。模型权重和字体遵循其各自许可证，源码包不包含这些下载资源。
