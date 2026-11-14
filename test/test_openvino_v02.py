"""v0.2 original-model integration, tiling, CRAFT refinement and CPU retry."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip('openvino')
from manga_translator.upscaling.esrgan_pytorch import ESRGANUpscalerPytorch
from manga_translator.detection.craft import CRAFTDetector
from manga_translator import detection, upscaling
from manga_translator.config import Detector, Upscaler


def finish(coroutine):
    with pytest.raises(StopIteration) as result:
        coroutine.send(None)
    return result.value.value


class TinySR(torch.nn.Module):
    def forward(self, image):
        return torch.nn.functional.interpolate(image, scale_factor=4, mode='nearest')


class TinyCRAFT(torch.nn.Module):
    def forward(self, image):
        features = torch.nn.functional.avg_pool2d(image, 2)
        return features[:, :2].permute(0, 2, 3, 1), features[:, :1]


class TinyRefiner(torch.nn.Module):
    def forward(self, scores, features):
        return scores[:, :, :, :1] + features.permute(0, 2, 3, 1)


@pytest.fixture
def upscaler(monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'CPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f32')
    model = ESRGANUpscalerPytorch()
    model.model = TinySR().eval()
    model.scale, model.device, model.use_igpu = 4, 'cpu', True
    model._load_openvino()
    model.tile_size, model.tile_pad = 8, 2
    assert model.ov_model is not None
    return model


@pytest.fixture
def craft(monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'CPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f32')
    model = CRAFTDetector()
    model.model, model.model_refiner = TinyCRAFT().eval(), TinyRefiner().eval()
    model.device, model.use_igpu = 'cpu', True
    model._load_openvino()
    assert model.ov_model is not None
    return model


def test_super_resolution_tiles_preserve_edges_and_channels(upscaler):
    image = torch.rand(1, 3, 13, 19)
    with torch.inference_mode():
        torch.testing.assert_close(upscaler._forward(image), upscaler.model(image))
    assert upscaler.execution_devices == ['CPU']
    assert upscaler.ov_model is not None
    assert len(upscaler.compiled_tiles) <= 4


@pytest.mark.parametrize('ratio', [2, 3, 4])
def test_super_resolution_image_output_and_mixed_sizes(upscaler, ratio):
    red = Image.new('RGB', (5, 7), (240, 10, 20))
    blue = Image.new('RGB', (11, 3), (20, 30, 230))
    output = finish(upscaler._infer([red, blue], ratio))
    assert [im.size for im in output] == [(5 * ratio, 7 * ratio), (11 * ratio, 3 * ratio)]
    np.testing.assert_array_equal(np.asarray(output[0])[0, 0], [240, 10, 20])
    np.testing.assert_array_equal(np.asarray(output[1])[0, 0], [20, 30, 230])
    assert finish(upscaler._infer([], ratio)) == []


def test_super_resolution_compile_failure_is_sticky(upscaler):
    upscaler.ov_device = 'INVALID_DEVICE'
    image = torch.rand(1, 3, 13, 19)
    with torch.inference_mode():
        for _ in range(2):
            torch.testing.assert_close(upscaler._forward(image), upscaler.model(image))
    assert upscaler.ov_model is None
    assert upscaler.execution_devices == ['PyTorch CPU']


def test_super_resolution_runtime_failure_retries_whole_image(upscaler):
    compiled = Mock()
    compiled.output.return_value = 'out'
    compiled.outputs = ['out']
    compiled.get_property.return_value = ['GPU.0']
    calls = []

    def inference(inputs):
        calls.append(inputs[0].copy())
        if len(calls) == 2:
            raise RuntimeError('GPU device lost after first tile')
        return {'out': np.repeat(np.repeat(inputs[0], 4, 2), 4, 3)}

    compiled.side_effect = inference
    upscaler.core = Mock()
    upscaler.core.compile_model.return_value = compiled
    image = torch.rand(1, 3, 13, 19)
    with torch.inference_mode():
        torch.testing.assert_close(upscaler._forward(image), upscaler.model(image))
        torch.testing.assert_close(upscaler._forward(image), upscaler.model(image))
    assert len(calls) == 2
    assert upscaler.ov_model is None
    assert not upscaler.compiled_tiles


def test_craft_forward_includes_refiner_and_reuses_shapes(craft):
    image = torch.rand(1, 3, 32, 64)
    with torch.inference_mode():
        scores, features = craft.model(image)
        reference = scores, craft.model_refiner(scores, features)
        for _ in range(2):
            for ref, actual in zip(reference, craft._forward(image)):
                torch.testing.assert_close(actual, ref, atol=1e-6, rtol=1e-6)
    assert len(craft.compiled_shapes) == 1
    assert craft.execution_devices == ['CPU']


@pytest.mark.parametrize('failure', ['compile', 'runtime', 'layout'])
def test_craft_failure_retries_both_networks_on_cpu(craft, failure):
    image = torch.rand(1, 3, 32, 64)
    if failure == 'compile':
        craft.ov_device = 'INVALID_DEVICE'
    else:
        bad = Mock(side_effect=RuntimeError('GPU device lost')) if failure == 'runtime' else Mock(return_value={'bad': np.zeros((1, 3))})
        bad.outputs = ['bad']
        craft.compiled_shapes[tuple(image.shape)] = bad
    scores, features = craft.model(image)
    for _ in range(2):
        for ref, actual in zip((scores, craft.model_refiner(scores, features)), craft._forward(image)):
            torch.testing.assert_close(actual, ref)
    assert craft.ov_model is None
    assert craft.execution_devices == ['PyTorch CPU']
    assert not craft.compiled_shapes


def test_craft_area_filter_preserves_actual_polygons(craft, monkeypatch):
    from manga_translator.detection import craft as module
    rectangle = np.array([[10, 10], [50, 10], [50, 30], [10, 30]], dtype=np.float32)
    monkeypatch.setattr(module.craft_utils, 'getDetBoxes', lambda *args: ([rectangle], [rectangle]))
    craft._forward = Mock(return_value=(torch.ones(1, 128, 128, 2), torch.ones(1, 128, 128, 1)))
    boxes, mask, _ = finish(craft._infer(np.zeros((256, 256, 3), np.uint8), 256, .5, .7, 2.3))
    assert len(boxes) == 1
    assert boxes[0].area > 16
    assert np.count_nonzero(mask) > 0


def test_craft_unload_releases_both_networks(craft):
    finish(craft._unload())
    assert not hasattr(craft, 'model')
    assert not hasattr(craft, 'model_refiner')
    assert craft.ov_model is None
    assert not craft.compiled_shapes


@pytest.mark.parametrize('enabled', [True, False])
def test_upscaler_cache_switches_backend(monkeypatch, enabled):
    model = SimpleNamespace(is_loaded=lambda: True, use_igpu=not enabled,
                            device='cpu', unload=AsyncMock(), load=AsyncMock())
    monkeypatch.setattr(upscaling, 'get_upscaler', lambda key: model)
    finish(upscaling._load_upscaler(Upscaler.upscler4xultrasharp, 'cpu', enabled))
    model.unload.assert_awaited_once()
    model.load.assert_awaited_once_with('cpu', use_igpu=enabled)


def test_craft_dispatch_enables_integrated_backend(monkeypatch):
    model = SimpleNamespace(is_loaded=lambda: False, load=AsyncMock())
    monkeypatch.setattr(detection, 'get_detector', lambda key: model)
    finish(detection._load_detector(Detector.craft, 'cpu', True))
    model.load.assert_awaited_once_with('cpu', use_igpu=True)


@pytest.mark.parametrize('kind', ['upscaler', 'craft'])
def test_missing_optional_runtime_uses_original_cpu(monkeypatch, kind):
    import builtins
    original_import = builtins.__import__

    def missing_openvino(name, *args, **kwargs):
        if name.startswith('openvino'):
            raise ModuleNotFoundError('Optional OpenVINO is not installed')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', missing_openvino)
    model = ESRGANUpscalerPytorch() if kind == 'upscaler' else CRAFTDetector()
    model.model = TinySR() if kind == 'upscaler' else TinyCRAFT()
    model.use_igpu, model.device = True, 'cpu'
    if kind == 'upscaler':
        model.scale = 4
    else:
        model.model_refiner = TinyRefiner()
    model._load_openvino()
    assert model.ov_model is None
    assert model.execution_devices == ['PyTorch CPU']
    with torch.inference_mode():
        output = model._forward(torch.zeros(1, 3, 8, 8))
    assert output is not None


def test_original_cpu_path_does_not_initialize_openvino():
    sr, detector = ESRGANUpscalerPytorch(), CRAFTDetector()
    sr.model, detector.model, detector.model_refiner = TinySR(), TinyCRAFT(), TinyRefiner()
    sr._load_openvino = Mock(side_effect=AssertionError('Unexpected OpenVINO initialization'))
    detector._load_openvino = Mock(side_effect=AssertionError('Unexpected OpenVINO initialization'))
    image = torch.ones(1, 3, 8, 8)
    assert sr._forward(image).shape == (1, 3, 32, 32)
    assert detector._forward(image)[1].shape == (1, 4, 4, 1)
    sr._load_openvino.assert_not_called()
    detector._load_openvino.assert_not_called()
