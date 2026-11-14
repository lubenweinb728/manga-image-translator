"""Runtime contract tests without loading translators or downloading weights."""
import logging
from unittest.mock import Mock

import numpy as np
import pytest

from unittest.mock import patch
from manga_translator.detection.ctd_utils.basemodel import TextDetBaseDNN


def make_runner(size, path, fallback, logger, device='GPU', precision='f32'):
    with patch.dict('os.environ', {'MANGA_OPENVINO_DEVICE': device,
                                   'MANGA_OPENVINO_PRECISION': precision}):
        runner = TextDetBaseDNN(size, path, use_igpu=True, logger=logger)
    runner._forward_opencv = fallback
    return runner


@pytest.fixture
def model_path(tmp_path):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops
    image = ops.parameter([1, 3, 8, 8], np.float32)
    mask = ops.reduce_mean(image, ops.constant([1], np.int64), True)
    lines = ops.concat([mask, mask], 1)
    blocks = ops.reshape(mask, ops.constant([1, 64, 1], np.int64), False)
    # Deliberately permute outputs to check semantic matching.
    model = ov.Model([lines, blocks, mask], [image])
    path = tmp_path / 'ctd.xml'
    ov.save_model(model, path, compress_to_fp16=False)
    return path


def test_real_cpu_inference_matches_preprocessing(model_path):
    fallback = Mock()
    runner = make_runner(
        8, model_path, fallback, logging.getLogger(__name__), device='CPU')
    image = np.zeros((8, 8, 3), np.uint8)
    image[:, :, 0] = 255
    blocks, mask, lines = runner(image)
    assert runner.compiled is not None
    assert 'CPU' in runner.execution_devices
    assert blocks.shape == (1, 64, 1)
    assert lines.shape == (1, 2, 8, 8)
    np.testing.assert_allclose(mask, 1 / 3, atol=1e-6)
    fallback.assert_not_called()


def test_compile_failure_retries_on_cpu(model_path):
    image = np.zeros((8, 8, 3), np.uint8)
    fallback = Mock(return_value=('blocks', 'mask', 'lines'))
    with patch('openvino.Core') as core:
        core.return_value.compile_model.side_effect = RuntimeError('CPU compilation failed')
        runner = make_runner(8, model_path, fallback, logging.getLogger(__name__))
    assert runner(image) == ('blocks', 'mask', 'lines')
    fallback.assert_called_once_with(image)
    assert runner.execution_devices == ['OpenCV CPU']


@pytest.mark.parametrize('requested_device', ['GPU', 'GPU.0'])
def test_ctd_uses_openvino_cpu_even_when_other_models_use_gpu(model_path, requested_device):
    fallback = Mock()
    runner = make_runner(8, model_path, fallback, Mock(), device=requested_device)
    runner(np.zeros((8, 8, 3), np.uint8))
    assert runner.compiled is not None
    assert runner.execution_devices == ['CPU']
    fallback.assert_not_called()


def test_inference_failure_is_sticky(model_path):
    image = np.zeros((8, 8, 3), np.uint8)
    fallback = Mock(return_value=('blocks', 'mask', 'lines'))
    runner = make_runner(
        8, model_path, fallback, logging.getLogger(__name__), device='CPU')
    broken = Mock(side_effect=RuntimeError('GPU device lost'))
    runner.compiled = broken
    assert runner(image) == fallback.return_value
    assert runner(image) == fallback.return_value
    assert broken.call_count == 1
    assert fallback.call_count == 2


def test_invalid_precision_is_reported():
    with pytest.raises(ValueError, match='precision'):
        make_runner(8, 'unused', Mock(), Mock(), precision='typo')


def test_invalid_output_layout_retries_same_image(model_path):
    fallback = Mock(return_value=('blocks', 'mask', 'lines'))
    runner = make_runner(8, model_path, fallback, Mock(), device='CPU')
    broken = Mock(return_value={'unexpected': np.zeros((1, 5))})
    broken.outputs = ['unexpected']
    runner.compiled = broken
    image = np.zeros((8, 8, 3), np.uint8)
    assert runner(image) == fallback.return_value
    fallback.assert_called_once_with(image)
    assert runner.compiled is None


def test_original_opencv_path_ignores_openvino_settings(monkeypatch):
    import builtins
    import cv2
    original_import = builtins.__import__

    def no_openvino(name, *args, **kwargs):
        if name.startswith('openvino'):
            raise AssertionError('CPU mode must not import OpenVINO')
        return original_import(name, *args, **kwargs)

    net = Mock()
    monkeypatch.setattr(cv2.dnn, 'readNetFromONNX', Mock(return_value=net))
    monkeypatch.setattr(builtins, '__import__', no_openvino)
    monkeypatch.setenv('MANGA_OPENVINO_PRECISION', 'unused-invalid-setting')
    runner = TextDetBaseDNN(8, 'unused.onnx')
    image = np.zeros((8, 8, 3), np.uint8)
    assert runner(image) is net.forward.return_value
    assert runner.compiled is None
    assert runner.execution_devices == ['OpenCV CPU']
