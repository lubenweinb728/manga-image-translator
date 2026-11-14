"""The public device flag must reach existing models without changing their names."""
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from manga_translator.args import parser, reparse
from manga_translator.config import Detector, Ocr
from manga_translator.manga_translator import MangaTranslator
from manga_translator import detection, ocr


def parse(argv):
    args, trailing = parser.parse_known_args(argv)
    return Namespace(**{**vars(args), **vars(reparse(trailing))})


def finish(coroutine):
    # These mocked I/O operations complete synchronously, without an OS event loop.
    with pytest.raises(StopIteration) as result:
        coroutine.send(None)
    return result.value.value


@pytest.mark.parametrize('before', [True, False])
def test_flag_before_or_after_mode(tmp_path, before):
    argv = ['local', '-i', str(tmp_path)]
    argv = ['--use-igpu'] + argv if before else argv + ['--use-igpu']
    args = parse(argv)
    assert args.use_igpu is True
    translator = MangaTranslator({'kernel_size': 3, 'use_igpu': args.use_igpu})
    assert translator.use_igpu is True
    assert translator.device == 'cpu'


@pytest.mark.parametrize('other', ['--use-gpu', '--use-gpu-limited'])
def test_conflicting_flags_rejected(tmp_path, other):
    with pytest.raises(SystemExit):
        parse(['local', '-i', str(tmp_path), '--use-igpu', other])
    with pytest.raises(ValueError, match='cannot be combined'):
        MangaTranslator({'kernel_size': 3, 'use_igpu': True,
                         other.lstrip('-').replace('-', '_'): True})


def test_default_is_original_cpu(tmp_path):
    args = parse(['local', '-i', str(tmp_path)])
    assert not args.use_igpu
    assert not reparse([]).__dict__
    assert [v.value for v in Detector] == ['default', 'dbconvnext', 'ctd', 'craft', 'paddle', 'none']
    assert [v.value for v in Ocr] == ['32px', '48px', '48px_ctc', 'mocr']


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('ocr_key', [Ocr.ocr48px, Ocr.ocr48px_ctc])
def test_cached_ocr_reloads_when_device_mode_changes(monkeypatch, enabled, ocr_key):
    reader = SimpleNamespace(is_loaded=lambda: True, use_igpu=not enabled,
                             device='cpu', unload=AsyncMock(), load=AsyncMock())
    monkeypatch.setattr(ocr, 'get_ocr', lambda key: reader)
    finish(ocr._load_ocr(ocr_key, 'cpu', enabled))
    reader.unload.assert_awaited_once()
    reader.load.assert_awaited_once_with('cpu', use_igpu=enabled)


def test_unsupported_ocr_uses_cpu_and_warns(monkeypatch):
    reader = SimpleNamespace(is_loaded=lambda: False, load=AsyncMock(), logger=Mock())
    monkeypatch.setattr(ocr, 'get_ocr', lambda key: reader)
    finish(ocr._load_ocr(Ocr.ocr32px, 'xpu', True))
    reader.load.assert_awaited_once_with('cpu')
    reader.logger.warning.assert_called_once()


def test_ctd_igpu_prepares_onnx_without_pytorch_weights(monkeypatch):
    detector = SimpleNamespace(is_loaded=lambda: False, download_onnx=AsyncMock(),
                               load=AsyncMock(), logger=Mock())
    monkeypatch.setattr(detection, 'get_detector', lambda key: detector)
    finish(detection._load_detector(Detector.ctd, 'cpu', True))
    detector.download_onnx.assert_awaited_once()
    detector.load.assert_awaited_once_with('cpu', use_igpu=True)


def test_openvino_missing_falls_back_inside_original_detector(monkeypatch):
    import builtins
    import torch
    from manga_translator.detection import default

    class TinyModel(torch.nn.Module):
        def forward(self, image):
            return image[:, :2], image[:, :1]

    original_import = builtins.__import__

    def missing_openvino(name, *args, **kwargs):
        if name.startswith('openvino'):
            raise ModuleNotFoundError('Optional dependency is not installed')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(default, 'TextDetectionDefault', TinyModel)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: {})
    monkeypatch.setattr(builtins, '__import__', missing_openvino)
    detector = default.DefaultDetector()
    finish(detector._load('cpu'))
    assert not detector.use_igpu
    assert not hasattr(detector, 'ov_model')
    finish(detector._load('cpu', use_igpu=True))
    assert detector.use_igpu
    assert detector.ov_model is None
    assert detector.execution_devices == ['PyTorch CPU']
