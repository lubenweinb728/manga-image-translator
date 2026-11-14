"""CTC forward contract, width caching, original decoder and sticky fallback."""
import logging
from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ov = pytest.importorskip('openvino')
from openvino import opset13 as ops
from manga_translator.ocr.model_48px_ctc import OCR
from manga_translator.ocr.model_48px_ctc import Model48pxCTCOCR


class TinyOCR(torch.nn.Module):
    dict_size = 4
    dictionary = ['<blank>', 'a', 'b', '<SP>']
    decode_ctc_top1 = OCR.decode_ctc_top1

    def forward(self, images):
        features = images.mean(dim=(1, 2)).unsqueeze(-1)
        return (torch.cat([features, -features, features * 0, features * 0 + 1], -1),
                features.expand(-1, -1, 6).clone())


@pytest.fixture
def ocr():
    reader = Model48pxCTCOCR()
    reader.use_igpu = True
    reader.model = TinyOCR().eval()
    reader.logger = logging.getLogger(__name__)
    reader.ov_device, reader.precision = 'CPU', 'f32'
    reader.core = ov.Core()
    image = ops.parameter([1, 3, 48, -1], np.float32)
    features = ops.unsqueeze(ops.reduce_mean(image, ops.constant([1, 2], np.int64), False),
                             ops.constant([-1], np.int64))
    zero = ops.multiply(features, ops.constant(0, np.float32))
    logits = ops.concat([features, ops.negative(features), zero,
                         ops.add(zero, ops.constant(1, np.float32))], -1)
    reader.ov_model = ov.Model([logits, ops.concat([features] * 6, -1)], [image])
    reader.compiled_widths = OrderedDict()
    reader.execution_devices = ['PyTorch CPU']
    return reader


def test_forward_preserves_batch_and_exact_width(ocr):
    images = torch.zeros(2, 3, 48, 160)
    images[0] = -.8
    images[1] = .7
    expected = ocr.model(images)
    actual = ocr._forward_openvino(images)
    for ref, out in zip(expected, actual):
        torch.testing.assert_close(ref, out, atol=1e-6, rtol=1e-6)
    assert actual[0].shape == (2, 160, 4)
    assert ocr.execution_devices == ['CPU']
    assert ocr.ov_model is not None
    assert list(ocr.compiled_widths) == [160]
    ref = ocr.model.decode_ctc_top1(*expected, 0)
    decoded = ocr._decode(images, [100, 120])
    assert [[int(c[0]) for c in row] for row in decoded] == [[int(c[0]) for c in row] for row in ref]


def test_width_cache_reuses_and_evicts(ocr):
    for width in (32, 64, 96, 128, 32, 160):
        ocr._forward_openvino(torch.zeros(1, 3, 48, width))
    assert list(ocr.compiled_widths) == [96, 128, 32, 160]
    assert ocr.ov_model is not None


@pytest.mark.parametrize('failure', ['compile', 'runtime', 'layout'])
def test_failure_retries_entire_batch_and_stays_on_cpu(ocr, failure):
    images = torch.rand(2, 3, 48, 160)
    if failure == 'compile':
        ocr.ov_device = 'INVALID_DEVICE'
    elif failure == 'runtime':
        # First sample completes; the second fails. CPU must retry both.
        compiled = Mock(side_effect=[{'a': np.zeros((1, 160, 4), np.float32),
                                     'b': np.zeros((1, 160, 6), np.float32)},
                                    RuntimeError('GPU device lost')])
        compiled.outputs = ['a', 'b']
        ocr.compiled_widths[160] = compiled
    else:
        compiled = Mock(return_value={'a': np.zeros((1, 160, 9), np.float32)})
        compiled.outputs = ['a']
        ocr.compiled_widths[160] = compiled
    for _ in range(2):
        for ref, actual in zip(ocr.model(images), ocr._forward_openvino(images)):
            torch.testing.assert_close(ref, actual, atol=0, rtol=0)
    assert ocr.ov_model is None
    assert not ocr.compiled_widths
    assert ocr.execution_devices == ['PyTorch CPU']
