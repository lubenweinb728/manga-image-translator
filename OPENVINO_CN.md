# OpenVINO · v0.1

安装 `python -m pip install -r requirements-openvino.txt` 后，用 `--use-igpu` 启用。
不传开关时保持原版推理路径；不能同时传 `--use-gpu` 或 `--use-gpu-limited`。

```powershell
python -m manga_translator local -i path/to/page.png --config-file examples/config-openvino-ctc.json --use-igpu
```

| 模块 | 配置名 | 本版执行方式 |
| --- | --- | --- |
| CTD | detector: ctd | 固定 OpenVINO CPU，失败回退 OpenCV CPU |
| Paddle | detector: paddle | 固定 OpenVINO CPU FP32，失败回退 Rust CPU |
| 默认检测器 | detector: default | OpenVINO 前向，复用原后处理 |
| CRAFT | detector: craft | 检测及 RefineNet OpenVINO 前向 |
| CTC OCR | ocr: 48px_ctc | OpenVINO 前向，原解码与颜色处理 |
| 自回归 OCR | ocr: 48px | OpenVINO 编码器及逐步解码，beam search 留在 CPU |
| AOT | inpainter: default | OpenVINO 整图前向，原遮罩合成 |
| LaMa Large | inpainter: lama_large | 固定 OpenVINO CPU FP32 |
| MC2 | colorizer: mc2 | 主网络默认 OpenVINO CPU FP32，去噪仍为 PyTorch CPU |
| 4xUltraSharp | upscaler: 4xultrasharp | 分块 OpenVINO 前向 |

未支持的模块保持 CPU 路径。OpenVINO 转换、编译或推理失败时，支持模块按各自策略回退原 CPU 实现。

可用 `MANGA_OPENVINO_DEVICE` 指定支持模块的设备，例如 `CPU`、`GPU.0`；
`MANGA_OPENVINO_PRECISION` 可设 `f32`（默认）或 `f16`，具体受模块约束。
CTD、Paddle、LaMa Large 不受全局 GPU 选择影响。MC2 独立选项以源码为准。
`MANGA_OPENVINO_CACHE_DIR` 用于运行时编译缓存，不属于源码。

本版不发布权重和编译缓存。首次推理会下载模型，并产生转换、编译开销；输入尺寸变化可能重新编译。
核显需要合适的 Intel 驱动和 OpenVINO GPU 运行环境，设备可用性可用以下命令检查：

```powershell
python -c "import openvino as ov; print(ov.Core().available_devices)"
```

性能依赖具体硬件和图片。旧 Gen9 核显上的 LaMa Large 明显更慢，因此固定 CPU FP32；
默认检测器、CRAFT、AOT 不保证加速。48px FP16 可能改变最终文本颜色，默认保留 FP32。
不要把同页预热后的速度等同于整本处理速度。开发测速报告和临时检查脚本不包含在本次发布中。

完整安装和运行方式见 [README_CN.md](README_CN.md)。
