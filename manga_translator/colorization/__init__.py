from PIL import Image

from .common import CommonColorizer, OfflineColorizer
from .manga_colorization_v2 import MangaColorizationV2
from ..config import Colorizer

COLORIZERS = {
    Colorizer.mc2: MangaColorizationV2,
}
colorizer_cache = {}

def get_colorizer(key: Colorizer, *args, **kwargs) -> CommonColorizer:
    if key not in COLORIZERS:
        raise ValueError(f'Could not find colorizer for: "{key}". Choose from the following: %s' % ','.join(COLORIZERS))
    if not colorizer_cache.get(key):
        upscaler = COLORIZERS[key]
        colorizer_cache[key] = upscaler(*args, **kwargs)
    return colorizer_cache[key]

async def prepare(key: Colorizer):
    upscaler = get_colorizer(key)
    if isinstance(upscaler, OfflineColorizer):
        await upscaler.download()

async def _load_colorizer(key: Colorizer, device: str, use_igpu: bool):
    colorizer = get_colorizer(key)
    target = 'cpu' if use_igpu else device
    if colorizer.is_loaded() and (getattr(colorizer, 'use_igpu', False) != use_igpu
                                  or getattr(colorizer, 'device', target) != target):
        await colorizer.unload()
    await colorizer.load(target, use_igpu=use_igpu)
    return colorizer

async def dispatch(key: Colorizer, device: str = 'cpu', use_igpu: bool = False, **kwargs) -> Image.Image:
    colorizer = get_colorizer(key)
    if isinstance(colorizer, OfflineColorizer):
        colorizer = await _load_colorizer(key, device, use_igpu)
    return await colorizer.colorize(**kwargs)

async def unload(key: Colorizer):
    colorizer_cache.pop(key, None)
