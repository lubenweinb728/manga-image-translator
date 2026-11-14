import os
from collections import OrderedDict
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor

from .common import OfflineColorizer
from .manga_colorization_v2_utils.networks.models import Colorizer
from .manga_colorization_v2_utils.denoising.denoiser import FFDNetDenoiser
from .manga_colorization_v2_utils.utils.utils import resize_pad


# https://github.com/qweasdd/manga-colorization-v2
class MangaColorizationV2(OfflineColorizer):
    _MODEL_SUB_DIR = os.path.join(OfflineColorizer._MODEL_SUB_DIR, 'manga-colorization-v2')
    _MODEL_MAPPING = {
        # Models were in google drive so had to upload to github
        'generator': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/manga-colorization-v2-generator.zip',
            'file': 'generator.zip',
            'hash': '087e6a0bc02770e732a52f33878b71a272a6123c9ac649e9b5bfb75e39e5c1d5',
        },
        'denoiser': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/manga-colorization-v2-net_rgb.pth',
            'file': 'net_rgb.pth',
            'hash': '0fe98bfd2ac870b15f360661b1c4789eecefc6dc2e4462842a0dd15e149a0433',
        }
    }

    async def _load(self, device: str, use_igpu: bool = False):
        self.use_igpu = use_igpu
        if use_igpu:
            device = 'cpu'
        self.device = device
        self.colorizer = Colorizer().to(device)
        self.colorizer.generator.load_state_dict(
            torch.load(self._get_file_path('generator.zip'), map_location=self.device))
        self.colorizer = self.colorizer.eval()
        self.denoiser = FFDNetDenoiser(device, _weights_dir=self.model_dir)
        self.execution_devices = ['PyTorch CPU'] if device == 'cpu' else [device]
        if use_igpu:
            self._load_openvino()

    def _load_openvino(self):
        self.ov_model = None
        self.compiled_shapes = OrderedDict()
        try:
            # Scoped settings avoid changing detection/OCR/inpainting precision.
            self.ov_device = os.environ.get('MANGA_MC2_OPENVINO_DEVICE', 'CPU')
            self.ov_precision = os.environ.get('MANGA_MC2_OPENVINO_PRECISION', 'f32')
            if self.ov_device not in ('CPU', 'GPU', 'GPU.0') or self.ov_precision not in ('f32', 'f16'):
                raise ValueError('MC2 OpenVINO supports CPU/GPU/GPU.0 and f32/f16')
            if self.ov_device == 'CPU' and self.ov_precision != 'f32':
                raise ValueError('MC2 CPU path requires f32')
            import openvino as ov
            self.core = ov.Core()
            with torch.no_grad():
                self.ov_model = ov.convert_model(self.colorizer,
                    example_input=torch.zeros(1, 5, 128, 128), input=[[1, 5, -1, -1]])
            self.logger.info('MC2 OpenVINO %s %s; FFDNet remains PyTorch CPU', self.ov_device, self.ov_precision)
        except Exception as exc:
            self._fallback_openvino(exc)

    def _fallback_openvino(self, exc):
        self.compiled_shapes.clear()
        self.ov_model = None
        self.core = None
        self.execution_devices = ['PyTorch CPU']
        self.logger.warning('MC2 OpenVINO failed; using original PyTorch CPU: %s', exc)

    def _forward_openvino(self, tensor):
        shape = tuple(tensor.shape)
        if shape not in self.compiled_shapes:
            model = self.ov_model.clone()
            model.reshape({model.input(0): list(shape)})
            config = {'INFERENCE_PRECISION_HINT': self.ov_precision, 'PERFORMANCE_HINT': 'LATENCY'}
            cache = os.environ.get('MANGA_OPENVINO_CACHE_DIR')
            if cache:
                config['CACHE_DIR'] = cache
            # Full-page colorization retains only the latest compiled shape.
            self.compiled_shapes.clear()
            compiled = self.core.compile_model(model, self.ov_device, config)
            self.compiled_shapes[shape] = compiled
            self.execution_devices = list(compiled.get_property('EXECUTION_DEVICES'))
            self.logger.info('MC2 OpenVINO %s shape %s execution devices: %s', self.ov_precision, shape, self.execution_devices)
        compiled = self.compiled_shapes[shape]
        result = compiled([np.ascontiguousarray(tensor.detach().cpu().numpy())])
        output = np.array(result[compiled.output(0)], copy=True)
        if output.shape != (shape[0], 3, *shape[2:]) or not np.isfinite(output).all():
            raise ValueError('Unexpected MC2 output layout or nonfinite values')
        return torch.from_numpy(output)

    def _forward(self, tensor):
        if getattr(self, 'use_igpu', False) and self.ov_model is not None:
            try:
                return self._forward_openvino(tensor)
            except Exception as exc:
                self._fallback_openvino(exc)
        return self.colorizer(tensor)[0]

    async def _unload(self):
        if getattr(self, 'use_igpu', False):
            self.compiled_shapes.clear()
            self.ov_model = None
            self.core = None
        del self.colorizer
        del self.denoiser

    async def _infer(self, image: Image.Image, colorization_size: int, denoise_sigma=25, **kwargs) -> Image.Image:
        # Size has to be multiple of 32
        img = np.array(image.convert('RGBA'))
        max_size = min(*img.shape[:2])
        max_size -= max_size % 32
        if colorization_size > 0:
            size = min(max_size, colorization_size - (colorization_size % 32))
        else:
            # size<=576 gives best results
            size = min(max_size, 576)

        if 0 <= denoise_sigma and denoise_sigma <= 255:
            with torch.no_grad():
                img = self.denoiser.get_denoised_image(img, sigma=denoise_sigma)

        img, current_pad = resize_pad(img, size)

        transform = ToTensor()
        current_image = transform(img).unsqueeze(0).to(self.device)
        current_hint = torch.zeros(1, 4, current_image.shape[2], current_image.shape[3]).float().to(self.device)

        with torch.no_grad():
            fake_color = self._forward(torch.cat([current_image, current_hint], 1))
            fake_color = fake_color.detach()

        result = fake_color[0].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5

        if current_pad[0] != 0:
            result = result[:-current_pad[0]]
        if current_pad[1] != 0:
            result = result[:, :-current_pad[1]]

        colored_image = result.numpy() * 255
        return Image.fromarray(colored_image.astype(np.uint8))
