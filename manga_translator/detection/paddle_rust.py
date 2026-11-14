from collections import OrderedDict
import os

import cv2
import numpy as np
import pyclipper

from .common import OfflineDetector
from .common_rust import RustDetector, get_session
from ..utils import Quadrilateral


class PaddleDetector(OfflineDetector):
    # Share the original Rust download location and weights.
    _MODEL_SUB_DIR = 'detector/paddle'
    _MODEL_MAPPING = {'det': {
        'url': 'https://github.com/frederik-uni/manga-image-translator-rust/releases/download/paddle-ocr-chinese-v4/det.onnx',
        'hash': 'b21a993484b367c0ea29d4a703c038d6ee3212173e6abf962b09188b032a9483',
        'file': 'det.onnx',
    }}

    def __init__(self):
        super().__init__()
        self.native = None
        self.compiled_shapes = OrderedDict()

    async def _load(self, device='cpu', use_igpu=False):
        self.device = 'cpu' if use_igpu else device
        self.use_igpu = use_igpu
        self.ov_model = None
        self.core = None
        if not use_igpu:
            self.native = RustDetector(get_session().paddle_detector())
            await self.native.load(device)
            self.execution_devices = ['Rust original']
            return
        try:
            import openvino as ov
            self.core = ov.Core()
            self.ov_model = self.core.read_model(self._get_file_path('det.onnx'))
            self.execution_devices = ['OpenVINO CPU (not yet compiled)']
            self.logger.info('Paddle uses OpenVINO CPU FP32 (measured faster than this machine\'s iGPU)')
        except Exception as exc:
            await self._fallback(exc)

    async def _fallback(self, exc):
        self.logger.warning('Paddle OpenVINO failed; retrying original Rust CPU: %s', exc)
        self.compiled_shapes.clear()
        self.ov_model = None
        self.core = None
        # An empty provider list explicitly disables native GPU providers.
        from rusty_manga_image_translator import Session
        self.native_session = Session([])
        self.native = RustDetector(self.native_session.paddle_detector())
        await self.native.load('cpu')
        self.execution_devices = ['Rust CPU']

    async def _unload(self):
        self.compiled_shapes.clear()
        self.ov_model = None
        self.core = None
        if self.native is not None:
            await self.native.unload()
        self.native = None
        self.native_session = None

    async def detect(self, image, detect_size, text_threshold, box_threshold, unclip_ratio,
                     invert, gamma_correct, rotate, auto_rotate=False, verbose=False):
        args = (image, detect_size, text_threshold, box_threshold, unclip_ratio,
                invert, gamma_correct, rotate, auto_rotate, verbose)
        if self.ov_model is not None:
            try:
                # Pad before rotation, then crop only after coordinates are restored.
                # CommonDetector's automatic crop otherwise precedes unrotation.
                h, w = image.shape[:2]
                if min(h, w) < 400:
                    padded = self._add_border(image, 400)
                    result = await super().detect(padded, *args[1:])
                    return self._remove_border(padded, w, h, *result)
                return await super().detect(*args)
            except Exception as exc:
                await self._fallback(exc)
        # Retry the original image and all options, not a partly transformed image.
        return await self.native.detect(*args)

    def _forward(self, tensor):
        shape = tuple(tensor.shape)
        if shape not in self.compiled_shapes:
            model = self.ov_model.clone()
            model.reshape({model.input(0): list(shape)})
            config = {'INFERENCE_PRECISION_HINT': 'f32', 'PERFORMANCE_HINT': 'LATENCY'}
            cache = os.environ.get('MANGA_OPENVINO_CACHE_DIR')
            if cache:
                config['CACHE_DIR'] = cache
            if len(self.compiled_shapes) >= 2:
                self.compiled_shapes.popitem(last=False)
            self.compiled_shapes[shape] = self.core.compile_model(model, 'CPU', config)
            self.execution_devices = list(self.compiled_shapes[shape].get_property('EXECUTION_DEVICES'))
            self.logger.info('Paddle OpenVINO FP32 shape %s execution devices: %s', shape, self.execution_devices)
        self.compiled_shapes.move_to_end(shape)
        compiled = self.compiled_shapes[shape]
        result = np.array(compiled([tensor])[compiled.output(0)], copy=True)
        if result.shape != (1, 1, *shape[2:]) or not np.isfinite(result).all():
            raise ValueError('Unexpected Paddle output layout or nonfinite values')
        return result[0, 0]

    async def _infer(self, image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose=False):
        # Preserve Rust Paddle's fixed 960 maximum, including its no-upscale rule.
        h, w = image.shape[:2]
        ratio = min(1.0, 960 / max(h, w))
        rh, rw = [max(32, int(v * ratio) // 32 * 32) for v in (h, w)]
        resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        mean = np.array([0.485, 0.456, 0.406], np.float32) * np.float32(255)
        norm = np.float32(1) / np.array([0.229, 0.224, 0.225], np.float32) / np.float32(255)
        tensor = np.ascontiguousarray((resized * norm - mean * norm).transpose(2, 0, 1)[None])
        prediction = self._forward(tensor)
        return self._postprocess(prediction, (h, w), text_threshold, box_threshold, unclip_ratio)

    @staticmethod
    def _postprocess(prediction, original_shape, text_threshold, box_threshold, unclip_ratio):
        # Rust passes box_threshold to bitmap generation and text_threshold to scoring.
        bitmap = (np.clip(prediction * 255, 0, 255).astype(np.uint8)
                  > int(box_threshold * 255)).astype(np.uint8)
        bitmap = cv2.dilate(bitmap, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        h, w = original_shape
        ph, pw = prediction.shape
        textlines = []
        mask = np.zeros((h, w), np.uint8)
        for contour in reversed(contours):
            if len(contour) < 3:
                continue
            rect = cv2.minAreaRect(contour)
            if min(rect[1]) < 3:
                continue
            x, y, cw, ch = cv2.boundingRect(contour)
            region = np.zeros((ch, cw), np.uint8)
            cv2.fillPoly(region, [contour - [x, y]], 1)
            score = float(cv2.mean(prediction[y:y+ch, x:x+cw], region)[0])
            if score < text_threshold:
                continue
            # imageproc's minimum rectangle has integer corners before unclipping.
            points = np.rint(cv2.boxPoints(rect)).astype(np.float32)
            area, perimeter = cv2.contourArea(points), cv2.arcLength(points, True)
            if perimeter <= 0:
                continue
            offset = pyclipper.PyclipperOffset()
            offset.AddPath(np.rint(points).astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            expanded = offset.Execute(area * unclip_ratio / perimeter)
            if not expanded:
                continue
            rect = cv2.minAreaRect(np.array(expanded[0], np.float32))
            if min(rect[1]) < 5:
                continue
            points = np.rint(cv2.boxPoints(rect)).astype(np.float32)
            points[:, 0] = np.clip(points[:, 0] * w / pw, 0, w)
            points[:, 1] = np.clip(points[:, 1] * h / ph, 0, h)
            points = points.astype(np.int64)
            textlines.append(Quadrilateral(points, '', score))
            cv2.fillPoly(mask, [points.astype(np.int32)], 255)
        return textlines, mask, None
