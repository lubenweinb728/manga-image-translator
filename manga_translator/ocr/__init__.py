import numpy as np
from typing import List, Optional
from .common import CommonOCR, OfflineOCR
from .model_32px import Model32pxOCR
from .model_48px import Model48pxOCR
from .model_48px_ctc import Model48pxCTCOCR
from .model_manga_ocr import ModelMangaOCR
from ..config import Ocr, OcrConfig
from ..utils import Quadrilateral

OCRS = {
    Ocr.ocr32px: Model32pxOCR,
    Ocr.ocr48px: Model48pxOCR,
    Ocr.ocr48px_ctc: Model48pxCTCOCR,
    Ocr.mocr: ModelMangaOCR,
}
ocr_cache = {}

def get_ocr(key: Ocr, *args, **kwargs) -> CommonOCR:
    if key not in OCRS:
        raise ValueError(f'Could not find OCR for: "{key}". Choose from the following: %s' % ','.join(OCRS))
    if not ocr_cache.get(key):
        ocr = OCRS[key]
        ocr_cache[key] = ocr(*args, **kwargs)
    return ocr_cache[key]

async def _load_ocr(ocr_key, device, use_igpu):
    ocr = get_ocr(ocr_key)
    supported = ocr_key in (Ocr.ocr48px, Ocr.ocr48px_ctc)
    enabled = use_igpu and supported
    target_device = 'cpu' if use_igpu else device
    if use_igpu and not supported and not getattr(ocr, '_igpu_warned', False):
        ocr.logger.warning('--use-igpu is not supported by OCR %s; using CPU', ocr_key.value)
        ocr._igpu_warned = True
    if ocr.is_loaded() and (getattr(ocr, 'use_igpu', False) != enabled
                            or getattr(ocr, 'device', target_device) != target_device):
        await ocr.unload()
    if supported:
        await ocr.load(target_device, use_igpu=enabled)
    else:
        await ocr.load(target_device)
    return ocr

async def prepare(ocr_key: Ocr, device: str = 'cpu', use_igpu: bool = False):
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.download()
        await _load_ocr(ocr_key, device, use_igpu)

async def dispatch(ocr_key: Ocr, image: np.ndarray, regions: List[Quadrilateral], config:Optional[OcrConfig] = None, device: str = 'cpu', verbose: bool = False, use_igpu: bool = False) -> List[Quadrilateral]:
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        ocr = await _load_ocr(ocr_key, device, use_igpu)
    config = config or OcrConfig()
    return await ocr.recognize(image, regions, config, verbose)

async def unload(ocr_key: Ocr):
    ocr_cache.pop(ocr_key, None)
