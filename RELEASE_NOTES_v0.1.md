# v0.1

该版本为自用项目，欢迎继续开发

这是本分支首次整理发布，版本统一为 **v0.1**；开发阶段文档中的 v0.2 等计划编号不作为发布版本。

## 修改明细

- 在原模型实现中接入 OpenVINO，以 `--use-igpu` 启用，不增加独立模型名称，保留原有预处理、后处理与 CPU 回退。
- 支持 CTD、Paddle、默认检测器、CRAFT、48px_ctc OCR、48px 自回归 OCR、AOT、LaMa Large、MC2 和 4xUltraSharp 的对应 OpenVINO 路径。
- CTD、Paddle、LaMa Large 固定使用 OpenVINO CPU；MC2 主网络默认 CPU FP32，去噪保留 PyTorch CPU。其他支持模块按设备规则选择核显；该开关不表示全部模型都在 GPU 上运行。
- 改进参数互斥与传递、CRAFT 坐标恢复、Papago 请求处理，以及排版禁用断词行为，附对应回归测试。
- 保留兼容 API 路由翻译器与 ChatGPT GPT-6 Luna 路径及其模拟网络回归测试。
- 启动脚本移除自动 git pull，支持已激活环境、.venv、venv；桌面后端沿用界面进程的 Python。
- 补齐可选桌面依赖 PySide6，声明 pyhyphen 所需 setuptools。
- 按实际 Python 3.10 环境整理依赖版本，固定 pydensecrf 源码提交；补回被全局 lib/ 忽略规则遗漏的分镜运行代码。
- 修正网页端生产构建输出与 npm start 不一致的问题，更新前端 Docker 构建流程。
- 移除发布包中的字体二进制，默认排版改为首次下载官方 Noto CJK 与 Arabic 字体，校验 SHA256，提供许可证与离线准备方式；字体变化可能影响排版外观。
- 仅发布运行源码、必要资源、配置和回归测试；不包含模型权重、个人配置、输入输出、日志、缓存、训练资料、临时测速脚本及上游自动发布工作流。

## 运行方式

见 [README_CN.md](README_CN.md)，包括环境安装、命令行、OpenVINO、桌面端、网页端和测试命令。

## 已知限制

发布验证：Windows / Python 3.10.11 的独立虚拟环境完成依赖安装（不继承系统包），`pip check` 通过，149 项回归测试通过。CLI 帮助与配置结构输出、桌面窗口初始化、HTTP API 和前端生产 HTTP 响应已检查；前端构建与 TypeScript 检查通过。在线翻译服务和所有型号 GPU 未逐一验证。

本项目按自用场景维护。OpenVINO 支持执行不等于每种硬件均加速；首次转换与编译可能较慢。旧核显上 LaMa Large 明显不适合 GPU，因此本版固定 CPU FP32。48px FP16 可能出现颜色差异，默认精度保留 FP32。模型与默认字体首次使用需要联网；在线翻译服务可能需要密钥并产生费用。

代码基于 [zyddnys/manga-image-translator](https://github.com/zyddnys/manga-image-translator)，保留原项目 GPL-3.0 许可证；模型及字体遵循各自授权。
