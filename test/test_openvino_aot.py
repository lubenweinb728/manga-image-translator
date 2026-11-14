"""AOT two-input inference, shape eviction, composition and CPU retry."""
from unittest.mock import Mock, AsyncMock
import numpy as np
import pytest
import torch

pytest.importorskip('openvino')
from manga_translator.inpainting.inpainting_aot import AotInpainter
from manga_translator.inpainting import _load_inpainter
from manga_translator.config import Inpainter, InpainterConfig


def finish(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError('Unexpected suspension')


class TinyAOT(torch.nn.Module):
    def forward(self, image, mask):
        return image * (1-mask) + mask * 0.25


@pytest.fixture
def painter(monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'CPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f32')
    model = AotInpainter()
    model.model = TinyAOT().eval()
    model.device = 'cpu'
    model.use_igpu = True
    model._load_openvino()
    assert model.ov_model is not None
    return model


def test_two_inputs_and_latest_shape_cache(painter):
    for size in (24, 24, 32):
        image = torch.rand(1, 3, size, size)
        mask = torch.zeros(1, 1, size, size)
        mask[:, :, 4:12, 4:12] = 1
        torch.testing.assert_close(painter._forward(image, mask), painter.model(image, mask))
        assert list(painter.compiled_shapes) == [tuple(image.shape)]
    assert painter.execution_devices == ['CPU']


@pytest.mark.parametrize('failure', ['compile', 'runtime', 'nan', 'shape'])
def test_failure_retries_same_input_and_stays_cpu(painter, failure):
    image = torch.rand(1, 3, 24, 24)
    mask = torch.ones(1, 1, 24, 24)
    if failure == 'compile':
        painter.ov_device = 'INVALID_DEVICE'
    else:
        compiled = Mock()
        compiled.outputs = [0]
        compiled.output.return_value = 0
        if failure == 'runtime':
            compiled.side_effect = RuntimeError('device lost')
        else:
            array = np.full(tuple(image.shape) if failure == 'nan' else (1,), np.nan if failure == 'nan' else 0, np.float32)
            compiled.return_value = {0: array}
        painter.compiled_shapes[tuple(image.shape)] = compiled
    for _ in range(2):
        torch.testing.assert_close(painter._forward(image, mask), painter.model(image, mask))
    assert painter.ov_model is None
    assert not painter.compiled_shapes
    assert painter.execution_devices == ['PyTorch CPU']


def test_original_preprocessing_and_compositing(painter):
    image = np.random.default_rng(42).integers(0, 256, (75, 99, 3), dtype=np.uint8)
    mask = np.zeros((75, 99), np.uint8)
    mask[20:40, 30:50] = 255
    mask[40, 40] = 127
    painter.use_igpu = False
    expected = finish(painter._infer(image, mask, InpainterConfig(), 64))
    painter.use_igpu = True
    actual = finish(painter._infer(image, mask, InpainterConfig(), 64))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[mask < 127], image[mask < 127])
    assert actual.shape == image.shape and actual.dtype == np.uint8


def test_missing_runtime_falls_back(painter, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'openvino', None)
    painter._load_openvino()
    assert painter.ov_model is None
    assert painter.execution_devices == ['PyTorch CPU']


@pytest.mark.parametrize('key, enabled', [(Inpainter.default, True), (Inpainter.lama_large, True), (Inpainter.lama_mpe, False)])
def test_dispatch_cpu_and_backend_switch(monkeypatch, key, enabled):
    painter = Mock()
    painter.use_igpu = False
    painter.device = 'cpu'
    painter.is_loaded.return_value = True
    painter.load = AsyncMock()
    painter.unload = AsyncMock()
    monkeypatch.setattr('manga_translator.inpainting.get_inpainter', lambda key: painter)
    assert finish(_load_inpainter(key, 'cuda', True)) is painter
    if enabled:
        painter.unload.assert_awaited_once()
        painter.load.assert_awaited_once_with('cpu', use_igpu=True)
    else:
        painter.unload.assert_not_awaited()
        painter.load.assert_awaited_once_with('cpu')


def test_unload_releases_models(painter):
    finish(painter._unload())
    assert not hasattr(painter, 'model')
    assert painter.ov_model is None and painter.core is None
    assert not painter.compiled_shapes
