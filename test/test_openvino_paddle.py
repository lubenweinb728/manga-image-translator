"""Paddle CPU-only selection, geometry and full-input fallback contracts."""
from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
import numpy as np
import pytest

ov = pytest.importorskip('openvino')
from openvino import opset13 as ops
from manga_translator.detection.paddle_rust import PaddleDetector
from manga_translator.detection import _load_detector
from manga_translator.config import Detector


def finish(coro):
    try:
        coro.send(None)
    except StopIteration as done:
        return done.value
    raise AssertionError('Unexpected suspension')


@pytest.fixture
def detector():
    detector = PaddleDetector()
    detector.use_igpu = True
    detector.device = 'cpu'
    detector._loaded = True
    detector.core = ov.Core()
    parameter = ops.parameter([1, 3, -1, -1], np.float32)
    output = ops.sigmoid(ops.reduce_mean(parameter, ops.constant([1], np.int64), True))
    detector.ov_model = ov.Model([output], [parameter])
    return detector


def test_always_cpu_f32_and_cache_is_bounded(detector, monkeypatch):
    monkeypatch.setenv('MANGA_OPENVINO_DEVICE', 'GPU')
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'f16')
    for size in (32, 32, 64, 96):
        tensor = np.zeros((1, 3, size, size), np.float32)
        np.testing.assert_allclose(detector._forward(tensor), 0.5)
    assert list(detector.compiled_shapes) == [(1, 3, 64, 64), (1, 3, 96, 96)]
    assert detector.execution_devices == ['CPU']
    compiled = list(detector.compiled_shapes.values())[0]
    assert compiled.get_property('INFERENCE_PRECISION_HINT') == ov.Type.f32


def test_rgb_normalization_and_native_960_limit(detector):
    observed = []
    detector._forward = lambda x: observed.append(x) or np.zeros(x.shape[2:], np.float32)
    image = np.zeros((999, 781, 3), np.uint8)
    image[:, :, 0] = 255
    result = finish(detector._infer(image, 2560, 0.35, 0.5, 2.4))
    tensor = observed[0]
    assert tensor.shape == (1, 3, 960, 736) and tensor.flags.c_contiguous
    np.testing.assert_allclose(tensor[0, :, 0, 0], [(1-.485)/.229, -.456/.224, -.406/.225], rtol=1e-6)
    assert result[1].shape == image.shape[:2]
    assert not result[0] and result[2] is None


def test_empty_and_scored_rectangles():
    probability = np.zeros((64, 64), np.float32)
    lines, mask, refined = PaddleDetector._postprocess(probability, (128, 96), .35, .5, 1.5)
    assert not lines and not mask.any() and refined is None
    probability[20:40, 15:45] = .9
    lines, mask, _ = PaddleDetector._postprocess(probability, (128, 96), .35, .5, 1.5)
    assert len(lines) == 1 and 0.7 < lines[0].prob <= .9
    assert mask.dtype == np.uint8 and set(np.unique(mask)) == {0, 255}
    assert (lines[0].pts >= 0).all()


@pytest.mark.parametrize('rotate, auto_rotate', [(False, False), (True, False), (False, True)])
def test_common_filters_restore_original_dimensions(detector, rotate, auto_rotate):
    def predict(tensor):
        output = np.zeros(tensor.shape[2:], np.float32)
        output[20:40, 20:70] = .9
        return output
    detector._forward = predict
    image = np.full((100, 160, 3), 128, np.uint8)
    original = image.copy()
    lines, mask, _ = finish(detector.detect(image, 2560, .35, .5, 1.5, True, True, rotate, auto_rotate))
    assert mask.shape == image.shape[:2] and mask.dtype == np.uint8
    np.testing.assert_array_equal(image, original)


def test_failure_retries_original_image_and_sticks_to_native(detector, monkeypatch):
    native = SimpleNamespace(load=AsyncMock(), detect=AsyncMock(return_value=([], np.zeros((80, 96), np.uint8), None)))
    monkeypatch.setattr('manga_translator.detection.paddle_rust.RustDetector', lambda det: native)
    session = Mock()
    factory = Mock(return_value=session)
    monkeypatch.setattr('rusty_manga_image_translator.Session', factory)
    detector._forward = Mock(side_effect=RuntimeError('CPU compilation failed'))
    image = np.zeros((80, 96, 3), np.uint8)
    args = (image, 2560, .35, .5, 2.4, True, True, True, False, False)
    finish(detector.detect(*args))
    finish(detector.detect(*args))
    factory.assert_called_once_with([])
    assert detector._forward.call_count == 1
    assert native.detect.await_count == 2
    assert native.detect.call_args.args[0] is image
    assert native.detect.call_args.args[1:] == args[1:]
    assert detector.ov_model is None and detector.execution_devices == ['Rust CPU']


def test_missing_runtime_falls_back(detector, monkeypatch):
    monkeypatch.setitem(__import__('sys').modules, 'openvino', None)
    detector._fallback = AsyncMock()
    finish(detector._load('cpu', use_igpu=True))
    detector._fallback.assert_awaited_once()


def test_without_flag_retains_native_path(monkeypatch):
    detector = PaddleDetector()
    native = SimpleNamespace(load=AsyncMock(), detect=AsyncMock(return_value='native'))
    monkeypatch.setattr('manga_translator.detection.paddle_rust.RustDetector', lambda det: native)
    monkeypatch.setattr('manga_translator.detection.paddle_rust.get_session', Mock())
    monkeypatch.setitem(__import__('sys').modules, 'openvino', None)
    finish(detector._load('cpu', use_igpu=False))
    assert finish(detector.detect(np.zeros((32, 32, 3), np.uint8), 512, .35, .5, 2.4, False, False, False)) == 'native'
    native.load.assert_awaited_once_with('cpu')


def test_dispatch_switches_cached_backend(monkeypatch):
    detector = SimpleNamespace(is_loaded=lambda: True, use_igpu=False, device='cpu', unload=AsyncMock(), load=AsyncMock())
    monkeypatch.setattr('manga_translator.detection.get_detector', lambda key: detector)
    finish(_load_detector(Detector.paddle, 'cuda', True))
    detector.unload.assert_awaited_once()
    detector.load.assert_awaited_once_with('cpu', use_igpu=True)


def test_unload_releases_both_runtimes(detector):
    detector.native = SimpleNamespace(unload=AsyncMock())
    native = detector.native
    finish(detector._unload())
    native.unload.assert_awaited_once()
    assert detector.ov_model is None and detector.core is None and detector.native is None
    assert not detector.compiled_shapes
