"""MC2 normalization, crop, scoped settings and same-input CPU retry."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import numpy as np
import pytest
import torch
from PIL import Image

ov = pytest.importorskip('openvino')
from manga_translator.colorization.manga_colorization_v2 import MangaColorizationV2
from manga_translator.colorization import _load_colorizer
from manga_translator.config import Colorizer


def finish(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError('Unexpected suspension')


class TinyMC2(torch.nn.Module):
    def forward(self, tensor):
        value = tensor[:, :1] * 2 - 1
        return value.repeat(1, 3, 1, 1), value


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setenv('MANGA_MC2_OPENVINO_DEVICE', 'CPU')
    monkeypatch.setenv('MANGA_MC2_OPENVINO_PRECISION', 'f32')
    model = MangaColorizationV2()
    model.colorizer = TinyMC2().eval()
    model.device = 'cpu'
    model.use_igpu = True
    model._load_openvino()
    assert model.ov_model is not None
    return model


def test_primary_output_and_latest_shape_cache(model):
    for size in (32, 32, 64):
        tensor = torch.rand(1, 5, size, size)
        torch.testing.assert_close(model._forward(tensor), model.colorizer(tensor)[0])
        assert list(model.compiled_shapes) == [tuple(tensor.shape)]
    assert model.execution_devices == ['CPU']


@pytest.mark.parametrize('failure', ['compile', 'runtime', 'nan', 'layout'])
def test_failure_retries_and_sticks_to_cpu(model, failure):
    tensor = torch.rand(1, 5, 32, 32)
    if failure == 'compile':
        model.ov_device = 'INVALID_DEVICE'
    else:
        compiled = Mock()
        compiled.output.return_value = 0
        if failure == 'runtime':
            compiled.side_effect = RuntimeError('device lost')
        else:
            compiled.return_value = {0: np.full((1, 3, 32, 32) if failure == 'nan' else (1,), np.nan if failure == 'nan' else 0, np.float32)}
        model.compiled_shapes[tuple(tensor.shape)] = compiled
    for _ in range(2):
        torch.testing.assert_close(model._forward(tensor), model.colorizer(tensor)[0])
    assert model.ov_model is None and not model.compiled_shapes
    assert model.execution_devices == ['PyTorch CPU']


@pytest.mark.parametrize('sigma', [-1, 25, 256])
@pytest.mark.parametrize('shape', [(65, 95), (95, 65)])
def test_denoise_skip_grayscale_hints_and_crop_match_original(model, sigma, shape):
    seen = []
    class Observe(TinyMC2):
        def forward(self, tensor):
            seen.append(tensor.clone())
            return super().forward(tensor)
    model.colorizer = Observe()
    model.denoiser = SimpleNamespace(get_denoised_image=Mock(side_effect=lambda image, sigma: image[:, :, :3]))
    image = Image.fromarray(np.random.default_rng(2).integers(0, 256, (*shape, 4), np.uint8), 'RGBA')
    model.use_igpu = False
    expected = finish(model._infer(image, 64, denoise_sigma=sigma))
    assert seen[0].shape[1] == 5 and not seen[0][:, 1:].any()
    assert 0 <= seen[0][:, :1].min() <= seen[0][:, :1].max() <= 1
    model.use_igpu = True
    actual = finish(model._infer(image, 64, denoise_sigma=sigma))
    np.testing.assert_array_equal(np.array(actual), np.array(expected))
    assert actual.size == expected.size and actual.mode == 'RGB'
    assert model.denoiser.get_denoised_image.call_count == (2 if sigma == 25 else 0)


def test_scoped_settings_ignore_global_precision(model, monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'GPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f16')
    model._load_openvino()
    assert model.ov_device == 'CPU' and model.ov_precision == 'f32'


def test_missing_optional_runtime(model, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'openvino', None)
    model._load_openvino()
    assert model.ov_model is None and model.execution_devices == ['PyTorch CPU']


def test_unload_releases_both_networks(model):
    model.denoiser = Mock()
    finish(model._unload())
    assert not hasattr(model, 'colorizer') and not hasattr(model, 'denoiser')
    assert model.ov_model is None and model.core is None and not model.compiled_shapes


@pytest.mark.parametrize('enabled', [False, True])
def test_dispatch_switches_cached_backend(monkeypatch, enabled):
    model = SimpleNamespace(is_loaded=lambda: True, use_igpu=not enabled, device='cpu', unload=AsyncMock(), load=AsyncMock())
    monkeypatch.setattr('manga_translator.colorization.get_colorizer', lambda key: model)
    finish(_load_colorizer(Colorizer.mc2, 'cpu', enabled))
    model.unload.assert_awaited_once()
    model.load.assert_awaited_once_with('cpu', use_igpu=enabled)
