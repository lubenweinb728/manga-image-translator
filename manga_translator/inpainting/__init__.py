from typing import Optional

import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
from .inpainting_sd import StableDiffusionInpainter
from .none import NoneInpainter
from .original import OriginalInpainter
from ..config import Inpainter, InpainterConfig

INPAINTERS = {
    Inpainter.default: AotInpainter,
    Inpainter.lama_large: LamaLargeInpainter,
    Inpainter.lama_mpe: LamaMPEInpainter,
    Inpainter.sd: StableDiffusionInpainter,
    Inpainter.none: NoneInpainter,
    Inpainter.original: OriginalInpainter,
}
inpainter_cache = {}

def get_inpainter(key: Inpainter, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]

async def _load_inpainter(inpainter_key, device, use_igpu):
    inpainter = get_inpainter(inpainter_key)
    supported = inpainter_key in (Inpainter.default, Inpainter.lama_large)
    enabled = use_igpu and supported
    target_device = 'cpu' if use_igpu else device
    if use_igpu and not supported and not getattr(inpainter, '_igpu_warned', False):
        inpainter.logger.warning('--use-igpu is not supported by inpainter %s; using CPU', inpainter_key.value)
        inpainter._igpu_warned = True
    if inpainter.is_loaded() and (getattr(inpainter, 'use_igpu', False) != enabled
                                  or getattr(inpainter, 'device', target_device) != target_device):
        await inpainter.unload()
    if supported:
        await inpainter.load(target_device, use_igpu=enabled)
    else:
        await inpainter.load(target_device)
    return inpainter

async def prepare(inpainter_key: Inpainter, device: str = 'cpu', use_igpu: bool = False):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.download()
        await _load_inpainter(inpainter_key, device, use_igpu)

async def dispatch(inpainter_key: Inpainter, image: np.ndarray, mask: np.ndarray, config: Optional[InpainterConfig], inpainting_size: int = 1024, device: str = 'cpu', verbose: bool = False, use_igpu: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        inpainter = await _load_inpainter(inpainter_key, device, use_igpu)
    config = config or InpainterConfig()
    return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)

async def unload(inpainter_key: Inpainter):
    inpainter_cache.pop(inpainter_key, None)