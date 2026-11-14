from collections import OrderedDict
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn

ov = pytest.importorskip('openvino')
from manga_translator.ocr.model_48px import OCR, AR48DecoderStep, Model48pxOCR, XposMultiheadAttention


class TinyDecoder(nn.Module):
    decoder_forward = OCR.decoder_forward

    def __init__(self):
        super().__init__()
        self.embd = nn.Embedding(10, 16)
        self.decoders = nn.ModuleList()
        for _ in range(2):
            layer = nn.TransformerDecoderLayer(16, 4, 32, dropout=0,
                                                batch_first=True, norm_first=True)
            layer.self_attn = XposMultiheadAttention(16, 4, self_attention=True)
            layer.multihead_attn = XposMultiheadAttention(16, 4, encoder_decoder_attention=True)
            self.decoders.append(layer)
        self.pred1 = nn.Sequential(nn.Linear(16, 16), nn.GELU())
        self.pred = nn.Linear(16, 10)


@pytest.fixture(scope='module')
def models():
    torch.manual_seed(42)
    model = TinyDecoder().eval()
    wrapper = AR48DecoderStep(model).eval()
    with torch.no_grad():
        graph = ov.convert_model(wrapper, example_input=(
            torch.ones(2, 1, dtype=torch.long), torch.zeros(2, 3, 1, 16),
            torch.zeros(2, 9, 16), torch.zeros(2, 9, dtype=torch.bool)),
            input=[ov.PartialShape([-1, 1]), ov.PartialShape([-1, 3, -1, 16]),
                   ov.PartialShape([-1, -1, 16]), ov.PartialShape([-1, -1])])
    return model, wrapper, ov.Core().compile_model(graph, 'CPU', {'INFERENCE_PRECISION_HINT': 'f32'})


@pytest.mark.parametrize('batch,step,length', [(2, 0, 9), (2, 1, 9), (1, 4, 13)])
def test_dynamic_decoder_matches_original_cache_and_logits(models, batch, step, length):
    model, wrapper, compiled = models
    tokens = torch.ones(batch, 1, dtype=torch.long)
    cache = torch.randn(batch, 3, 12, 16)
    memory = torch.randn(batch, length, 16)
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, -2:] = True
    with torch.no_grad():
        decoded, expected_cache = model.decoder_forward(model.embd(tokens), cache.clone(), memory, mask, step)
        expected_logits = model.pred(model.pred1(decoded))
        prefix = cache[:, :, :step].contiguous()
        eager_logits, eager_cache = wrapper(tokens, prefix, memory, mask)
        torch.testing.assert_close(eager_logits, expected_logits)
        torch.testing.assert_close(eager_cache, expected_cache[:, :, :step+1])
        result = compiled([tokens.numpy(), prefix.numpy(), memory.numpy(), mask.numpy()])
        np.testing.assert_allclose(result[compiled.output(0)], expected_logits.numpy(), atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(result[compiled.output(1)], expected_cache[:, :, :step+1].numpy(), atol=2e-5, rtol=2e-5)


def test_failure_restarts_whole_beam_batch_on_cpu():
    reader = Model48pxOCR()
    reader.use_igpu = True
    reader.ov_encoder = reader.ov_decoder = Mock()
    reader.ov_compiled = OrderedDict()
    reader.model = Mock()
    reader.model.infer_beam_batch_tensor.side_effect = [RuntimeError('decoder failed'), ['cpu result']]
    reader.logger = Mock()
    assert reader._infer_beam(torch.zeros(2, 3, 48, 160), [100, 144]) == ['cpu result']
    assert reader.model.infer_beam_batch_tensor.call_count == 2
    assert reader.ov_encoder is None and reader.ov_decoder is None
    assert reader.model._ov_encode is None and reader.model._ov_step is None
    assert reader.execution_devices == ['PyTorch CPU']
