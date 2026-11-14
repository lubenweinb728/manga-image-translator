"""Exercise default-detector normalization, shape specialization and CPU retry."""
import logging
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ov = pytest.importorskip('openvino')
from openvino import opset13 as ops
from manga_translator.detection.default import DefaultDetector, det_batch_forward_default


class TinyDetector(torch.nn.Module):
    def forward(self, image):
        logits = image.mean(dim=1, keepdim=True)
        return torch.cat([logits, logits], dim=1), logits.sigmoid()


@pytest.fixture
def detector():
    detector = DefaultDetector()
    detector.use_igpu = True
    detector.model = TinyDetector().eval()
    detector.device = 'cpu'
    detector.core = ov.Core()
    image = ops.parameter([1, 3, -1, -1], np.float32)
    logits = ops.reduce_mean(image, ops.constant([1], np.int64), True)
    detector.ov_model = ov.Model([ops.concat([logits, logits], 1), ops.sigmoid(logits)], [image])
    detector.ov_device = 'CPU'
    detector.precision = 'f32'
    detector.compiled_shapes = OrderedDict()
    detector.execution_devices = ['PyTorch CPU']
    detector.logger = logging.getLogger(__name__)
    return detector


def test_batch_and_normalization_match_original(detector):
    batch = np.zeros((2, 32, 64, 3), np.uint8)
    batch[1] = 255
    expected = det_batch_forward_default(batch, 'cpu', detector.model)
    actual = detector._forward(batch, 'cpu')
    for ref, out in zip(expected, actual):
        np.testing.assert_allclose(out, ref, atol=1e-6)
    assert detector.ov_model is not None
    assert detector.execution_devices == ['CPU']
    assert len(detector.compiled_shapes) == 1


def test_shape_cache_reuses_and_evicts(detector):
    for size in (32, 64, 32, 96):
        detector._forward([np.zeros((size, size, 3), np.uint8)], 'cpu')
    assert list(detector.compiled_shapes) == [(1, 3, 32, 32), (1, 3, 96, 96)]
    assert detector.ov_model is not None


def test_compile_failure_retries_same_batch(detector):
    detector.ov_device = 'INVALID_DEVICE'
    batch = np.full((2, 32, 32, 3), 127, np.uint8)
    expected = det_batch_forward_default(batch, 'cpu', detector.model)
    actual = detector._forward(batch, 'cpu')
    for ref, out in zip(expected, actual):
        np.testing.assert_array_equal(ref, out)
    assert detector.ov_model is None
    assert detector.execution_devices == ['PyTorch CPU']
    assert not detector.compiled_shapes


def test_runtime_failure_clears_cache_and_stays_on_cpu(detector):
    batch = np.zeros((2, 32, 32, 3), np.uint8)
    broken = Mock(side_effect=RuntimeError('GPU device lost'))
    detector.compiled_shapes[(1, 3, 32, 32)] = broken
    expected = det_batch_forward_default(batch, 'cpu', detector.model)
    for _ in range(2):
        for ref, actual in zip(expected, detector._forward(batch, 'cpu')):
            np.testing.assert_array_equal(ref, actual)
    assert broken.call_count == 1
    assert not detector.compiled_shapes
    assert detector.ov_model is None


def test_default_instances_use_their_own_models():
    first, second = DefaultDetector(), DefaultDetector()
    first.model = TinyDetector()

    class OtherDetector(TinyDetector):
        def forward(self, image):
            db, mask = super().forward(image)
            return db + 1, mask

    second.model = OtherDetector()
    batch = [np.zeros((32, 32, 3), np.uint8)]
    before = first._forward(batch, 'cpu')[0]
    other = second._forward(batch, 'cpu')[0]
    after = first._forward(batch, 'cpu')[0]
    np.testing.assert_array_equal(before, after)
    assert not np.array_equal(before, other)
