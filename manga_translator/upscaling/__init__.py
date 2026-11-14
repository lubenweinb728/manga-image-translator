from typing import List
from PIL import Image

from .common import CommonUpscaler, OfflineUpscaler
from .waifu2x import Waifu2xUpscaler
from .esrgan import ESRGANUpscaler
from .esrgan_pytorch import ESRGANUpscalerPytorch
from ..config import Upscaler

UPSCALERS = {
    Upscaler.waifu2x: Waifu2xUpscaler,
    Upscaler.esrgan: ESRGANUpscaler,
    Upscaler.upscler4xultrasharp: ESRGANUpscalerPytorch,
}
upscaler_cache = {}

def get_upscaler(key: Upscaler, *args, **kwargs) -> CommonUpscaler:
    if key not in UPSCALERS:
        raise ValueError(f'Could not find upscaler for: "{key}". Choose from the following: %s' % ','.join(UPSCALERS))
    if not upscaler_cache.get(key):
        upscaler = UPSCALERS[key]
        upscaler_cache[key] = upscaler(*args, **kwargs)
    return upscaler_cache[key]

async def prepare(upscaler_key: Upscaler):
    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        await upscaler.download()

async def _load_upscaler(upscaler_key, device, use_igpu):
    upscaler = get_upscaler(upscaler_key)
    supported = upscaler_key == Upscaler.upscler4xultrasharp
    enabled = use_igpu and supported
    target_device = 'cpu' if use_igpu else device
    if upscaler.is_loaded() and (getattr(upscaler, 'use_igpu', False) != enabled
                                 or getattr(upscaler, 'device', target_device) != target_device):
        await upscaler.unload()
    if supported:
        await upscaler.load(target_device, use_igpu=enabled)
    else:
        # ncnn/Vulkan backends keep their existing independent device handling.
        await upscaler.load(target_device)
    return upscaler

async def dispatch(upscaler_key: Upscaler, image_batch: List[Image.Image], upscale_ratio: int, device: str = 'cpu', use_igpu: bool = False) -> List[Image.Image]:
    if upscale_ratio == 1:
        return image_batch
    upscaler = get_upscaler(upscaler_key)
    if isinstance(upscaler, OfflineUpscaler):
        upscaler = await _load_upscaler(upscaler_key, device, use_igpu)
    return await upscaler.upscale(image_batch, upscale_ratio)

async def unload(upscaler_key: Upscaler):
    upscaler_cache.pop(upscaler_key, None)
