# OpenVINO acceleration roadmap

This document tracks the OpenVINO-specific work in this fork of `zyddnys/manga-image-translator`.

## Goals

- Add OpenVINO as an **optional inference backend**, not as a replacement for existing PyTorch/ncnn paths.
- Use Intel integrated GPUs for neural-network inference when supported by the installed OpenVINO runtime and Intel graphics driver.
- Preserve CPU fallback so the application remains usable when OpenVINO GPU compilation or inference is unavailable.
- Keep upstream behavior and configuration compatible wherever practical.
- Make each migration independently testable and reversible.

## Backend policy

OpenVINO is not a PyTorch device and must not be implemented as `model.to("openvino")`.

The project should distinguish between:

- inference backend: PyTorch / OpenVINO / ONNX Runtime / ncnn
- device used by that backend: CPU / CUDA / MPS / XPU / OpenVINO GPU

OpenVINO-specific model loading, compilation and inference should therefore live behind a backend abstraction instead of being mixed into the existing PyTorch `device` variable.

## v0.1 — OCR pipeline foundation

Status: **planned / implementation pending**

Primary targets:

1. Introduce the minimal OpenVINO runtime/backend plumbing.
2. Add runtime device discovery and clear logging of selected OpenVINO device.
3. Add safe CPU fallback when OpenVINO GPU is unavailable or model compilation fails.
4. CTD text detector:
   - reuse the existing ONNX model where possible;
   - compile/infer with OpenVINO;
   - keep current NumPy/OpenCV post-processing.
5. `48px_ctc` OCR:
   - move neural-network inference to OpenVINO;
   - keep CTC decoding and other lightweight post-processing on CPU.
6. Add small compatibility/regression tests so the OpenVINO path can be compared against the existing implementation.

Success criteria:

- Existing CPU/CUDA/MPS/XPU behavior is not intentionally changed.
- CTD + `48px_ctc` can complete the normal translation pipeline using OpenVINO GPU for supported inference stages.
- Failure to use OpenVINO GPU produces a clear diagnostic and a defined fallback instead of an unexplained crash.

## v0.2 — additional high-value models

Status: **planned**

Evaluate and, where practical, add OpenVINO inference paths for:

- Default text detector
- CRAFT text detector
- `4xUltraSharp` PyTorch upscaler

These models are prioritized because their inference is comparatively suitable for accelerator offload.

## v0.3 — optional image-processing models

Status: **planned / evaluation required**

Candidates:

- AOT inpainting
- MC2 colorization

Models with more complex execution graphs, iterative decoding, special operators, or weak expected benefit will not be migrated merely for feature parity.

Examples that require separate evaluation include:

- autoregressive `32px` / `48px` OCR paths and beam search
- MangaOCR integration
- LaMa variants
- Stable Diffusion inpainting
- local Transformer translation models

## Backends intentionally kept as-is

Real-ESRGAN and Waifu2x currently use ncnn/Vulkan paths. Intel GPUs may already be usable through Vulkan on a correctly configured system, so these are not initial OpenVINO migration targets.

## Hardware and deployment direction

The OpenVINO path is intended primarily for Intel integrated graphics supported by the host's OpenVINO runtime and graphics stack. Exact device support depends on OpenVINO version, operating system and Intel driver.

Container deployment should eventually expose the Intel render device (for example through Linux DRM `/dev/dri`) without making Intel GPU access mandatory for CPU-only users.

## Compatibility principle

The fork should remain easy to compare and rebase against upstream. OpenVINO work should therefore be kept in focused modules and commits, avoiding unrelated rewrites during the initial iterations.
