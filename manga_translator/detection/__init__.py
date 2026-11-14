import numpy as np

from .default import DefaultDetector
from .dbnet_convnext import DBConvNextDetector
from .ctd import ComicTextDetector
from .craft import CRAFTDetector
from .paddle_rust import PaddleDetector
from .none import NoneDetector
from .common import CommonDetector, OfflineDetector
from ..config import Detector

DETECTORS = {
    Detector.default: DefaultDetector,
    Detector.dbconvnext: DBConvNextDetector,
    Detector.ctd: ComicTextDetector,
    Detector.craft: CRAFTDetector,
    Detector.paddle: PaddleDetector,
    Detector.none: NoneDetector,
}
detector_cache = {}

def get_detector(key: Detector, *args, **kwargs) -> CommonDetector:
    if key not in DETECTORS:
        raise ValueError(f'Could not find detector for: "{key}". Choose from the following: %s' % ','.join(DETECTORS))
    if not detector_cache.get(key):
        detector = DETECTORS[key]
        detector_cache[key] = detector(*args, **kwargs)
    return detector_cache[key]

async def prepare(detector_key: Detector, use_igpu: bool = False):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        if detector_key == Detector.ctd and use_igpu:
            await detector.download_onnx()
        else:
            await detector.download()

async def _load_detector(detector_key, device, use_igpu):
    detector = get_detector(detector_key)
    supported = detector_key in (Detector.default, Detector.ctd, Detector.craft, Detector.paddle)
    enabled = use_igpu and supported
    target_device = 'cpu' if use_igpu else device
    if use_igpu and not supported and not getattr(detector, '_igpu_warned', False):
        detector.logger.warning('--use-igpu is not supported by detector %s; using CPU', detector_key.value)
        detector._igpu_warned = True
    if detector.is_loaded() and (getattr(detector, 'use_igpu', False) != enabled
                                 or getattr(detector, 'device', target_device) != target_device):
        await detector.unload()
    if detector_key == Detector.ctd:
        if enabled:
            await detector.download_onnx()
        else:
            detector._downloaded = detector._check_downloaded()
    if supported:
        await detector.load(target_device, use_igpu=enabled)
    else:
        await detector.load(target_device)
    return detector

async def dispatch(detector_key: Detector, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                   invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, device: str = 'cpu', verbose: bool = False, use_igpu: bool = False):
    detector = get_detector(detector_key)
    if isinstance(detector, OfflineDetector):
        detector = await _load_detector(detector_key, device, use_igpu)
    return await detector.detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct, rotate, auto_rotate, verbose)

async def unload(detector_key: Detector):
    detector_cache.pop(detector_key, None)
