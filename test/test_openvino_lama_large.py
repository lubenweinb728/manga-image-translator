"""Exercise LaMa's actual FourierUnit and its distinct normalization/composition."""
from unittest.mock import Mock
import numpy as np
import pytest
import torch

pytest.importorskip('openvino')
from manga_translator.inpainting.inpainting_lama_mpe import (
    LamaLargeInpainter, LamaFourier, FourierUnit,
)
from manga_translator.config import InpainterConfig


def finish(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError('Unexpected suspension')


class TinyGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fourier = FourierUnit(4, 3)

    def forward(self, image, mask, rel_pos=None, direct=None):
        return self.fourier(torch.cat([image * (1-mask), mask], 1)).sigmoid()


@pytest.fixture
def painter(monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'CPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f32')
    painter = LamaLargeInpainter()
    painter.model = object.__new__(LamaFourier)
    painter.model.generator = TinyGenerator().eval()
    painter.model.mpe = None
    painter.model.inpaint_only = True
    painter.device = 'cpu'
    painter.use_igpu = True
    painter._load_openvino()
    assert painter.ov_model is not None
    return painter


def test_fourier_and_composition_across_shapes(painter):
    for height, width in [(24, 32), (32, 40), (24, 32)]:
        image = torch.rand(1, 3, height, width)
        mask = torch.zeros(1, 1, height, width)
        mask[:, :, 8:16, 8:16] = 1
        with torch.inference_mode():
            actual = painter._forward(image, mask)
            expected = painter.model(image, mask)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(actual*(1-mask), image*(1-mask), atol=0, rtol=0)
        assert list(painter.compiled_shapes) == [tuple(image.shape)]
    assert painter.execution_devices == ['CPU']


def test_resize_and_original_pixel_preservation(painter):
    image = np.random.default_rng(7).integers(0, 256, (75, 99, 3), dtype=np.uint8)
    mask = np.zeros((75, 99), np.uint8)
    mask[20:40, 30:50] = 255
    mask[40, 40] = 127
    painter.use_igpu = False
    expected = finish(painter._infer(image, mask, InpainterConfig(), 64))
    painter.use_igpu = True
    actual = finish(painter._infer(image, mask, InpainterConfig(), 64))
    assert actual.shape == image.shape and actual.dtype == np.uint8
    np.testing.assert_array_equal(actual[mask < 127], image[mask < 127])
    assert np.abs(actual.astype(int)-expected.astype(int)).max() <= 1


@pytest.mark.parametrize('failure', ['compile', 'runtime'])
def test_cpu_retry_preserves_lama_semantics(painter, failure):
    image = torch.rand(1, 3, 24, 32)
    mask = torch.zeros(1, 1, 24, 32)
    mask[:, :, 8:16, 8:16] = 1
    if failure == 'compile':
        painter.ov_device = 'INVALID_DEVICE'
    else:
        broken = Mock(side_effect=RuntimeError('device lost'))
        painter.compiled_shapes[tuple(image.shape)] = broken
    for _ in range(2):
        with torch.inference_mode():
            torch.testing.assert_close(painter._forward(image, mask), painter.model(image, mask))
    assert painter.ov_model is None and not painter.compiled_shapes
    assert painter.execution_devices == ['PyTorch CPU']


def test_unload(painter):
    finish(painter._unload())
    assert not hasattr(painter, 'model')
    assert painter.ov_model is None and painter.core is None


def test_lama_keeps_fp32(painter, monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f16')
    painter._load_openvino()
    assert painter.precision == 'f32' and painter.ov_model is not None


def test_lama_ignores_global_gpu_and_fp16(painter, monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'GPU.0')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f16')
    painter._load_openvino()
    assert painter.ov_device == 'CPU' and painter.precision == 'f32'
    image = torch.rand(1, 3, 24, 32)
    mask = torch.ones(1, 1, 24, 32)
    with torch.inference_mode():
        torch.testing.assert_close(painter._forward(image, mask), painter.model(image, mask), atol=1e-5, rtol=1e-5)
    assert painter.execution_devices == ['CPU']
    compiled = next(iter(painter.compiled_shapes.values()))
    import openvino as ov
    assert compiled.get_property('INFERENCE_PRECISION_HINT') == ov.Type.f32
