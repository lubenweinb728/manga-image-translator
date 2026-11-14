import os
from collections import OrderedDict
import shutil
import numpy as np
import torch
import cv2
import einops
from typing import List, Tuple

from .default_utils.DBNet_resnet34 import TextDetection as TextDetectionDefault
from .default_utils import imgproc, dbnet_utils, craft_utils
from .common import OfflineDetector
from ..utils import TextBlock, Quadrilateral, det_rearrange_forward

def det_batch_forward_default(batch: np.ndarray, device: str, model):
    if isinstance(batch, list):
        batch = np.array(batch)
    batch = einops.rearrange(batch.astype(np.float32) / 127.5 - 1.0, 'n h w c -> n c h w')
    batch = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        db, mask = model(batch)
        db = db.sigmoid().cpu().numpy()
        mask = mask.cpu().numpy()
    return db, mask

class DefaultDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'model': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/detect-20241225.ckpt',
            'hash': '67ce1c4ed4793860f038c71189ba9630a7756f7683b1ee5afb69ca0687dc502e',
            'file': '.',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('detect-20241225.ckpt'):
            shutil.move('detect-20241225.ckpt', self._get_file_path('detect-20241225.ckpt'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str, use_igpu: bool = False):
        self.use_igpu = use_igpu
        if use_igpu:
            device = 'cpu'
        self.model = TextDetectionDefault()
        sd = torch.load(self._get_file_path('detect-20241225.ckpt'), map_location='cpu')
        self.model.load_state_dict(sd['model'] if 'model' in sd else sd)
        self.model.eval()
        self.device = device
        self.execution_devices = ['PyTorch CPU'] if device == 'cpu' else [device]
        if device.startswith('cuda') or device == 'mps' or device == 'xpu':
            self.model = self.model.to(self.device)

        if use_igpu:
            self._load_openvino()

    async def _unload(self):
        if getattr(self, 'use_igpu', False):
            self.compiled_shapes.clear()
            self.ov_model = None
            self.core = None
        del self.model

    def _forward(self, batch, device):
        if getattr(self, 'use_igpu', False):
            return self._forward_openvino(batch, device)
        return det_batch_forward_default(batch, device, model=self.model)

    def _load_openvino(self):
        self.ov_device = os.environ.get('MANGA_OPENVINO_DEVICE', 'GPU')
        self.precision = os.environ.get('MANGA_OPENVINO_PRECISION', 'f32')
        if self.precision not in ('f32', 'f16'):
            raise ValueError('OpenVINO precision must be f32 or f16')
        self.compiled_shapes = OrderedDict()
        self.execution_devices = ['PyTorch CPU']
        self.ov_model = None
        try:
            import openvino as ov
            self.core = ov.Core()
            # The fully convolutional network supports spatial dimensions divisible by 256.
            # Runtime specialization avoids dynamic-shape GPU requirements on old drivers.
            with torch.inference_mode():
                self.ov_model = ov.convert_model(
                    self.model, example_input=torch.zeros(1, 3, 256, 256),
                    input=[1, 3, -1, -1])
        except Exception as exc:
            self._fallback_openvino(exc)

    def _fallback_openvino(self, exc):
        self.ov_model = None
        self.compiled_shapes.clear()
        self.execution_devices = ['PyTorch CPU']
        self.logger.warning('Default detector OpenVINO failed; using PyTorch CPU: %s', exc)

    def _forward_openvino(self, batch, device):
        if self.ov_model is None:
            return det_batch_forward_default(batch, 'cpu', self.model)
        try:
            inputs = np.asarray(batch, dtype=np.float32) / 127.5 - 1.0
            inputs = np.ascontiguousarray(inputs.transpose(0, 3, 1, 2))
            outputs = []
            # Batch one limits memory use and compiled-model variants on integrated GPUs.
            for sample in inputs:
                shape = (1, *sample.shape)
                if shape not in self.compiled_shapes:
                    model = self.ov_model.clone()
                    model.reshape({model.input(0): list(shape)})
                    config = {'INFERENCE_PRECISION_HINT': self.precision}
                    cache = os.environ.get('MANGA_OPENVINO_CACHE_DIR')
                    if cache:
                        config['CACHE_DIR'] = cache
                    # Evict before compiling to bound retained compiled models.
                    if len(self.compiled_shapes) >= 2:
                        self.compiled_shapes.popitem(last=False)
                    compiled = self.core.compile_model(model, self.ov_device, config)
                    self.compiled_shapes[shape] = compiled
                    self.execution_devices = list(compiled.get_property('EXECUTION_DEVICES'))
                    self.logger.info('Default OpenVINO shape %s execution devices: %s',
                                     shape, self.execution_devices)
                self.compiled_shapes.move_to_end(shape)
                compiled = self.compiled_shapes[shape]
                result = compiled([sample[None]])
                values = [np.array(result[port], copy=True) for port in compiled.outputs]
                if (len(values) != 2 or any(x.ndim != 4 for x in values)
                        or values[0].shape[1] != 2 or values[1].shape[1] != 1):
                    raise ValueError('Unexpected default detector output layout')
                # Conversion preserves the source tuple: DB logits, then sigmoid mask.
                logits, mask = values
                db = torch.from_numpy(logits).sigmoid().numpy()
                outputs.append((db, mask))
            return tuple(np.concatenate([x[i] for x in outputs], axis=0) for i in (0, 1))
        except Exception as exc:
            self._fallback_openvino(exc)
            return det_batch_forward_default(batch, 'cpu', self.model)


    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):

        # TODO: Move det_rearrange_forward to common.py and refactor
        db, mask = det_rearrange_forward(image, self._forward, detect_size, 4, device=self.device, verbose=verbose)

        if db is None:
            # rearrangement is not required, fallback to default forward
            img_resized, target_ratio, _, pad_w, pad_h = imgproc.resize_aspect_ratio(cv2.bilateralFilter(image, 17, 80, 80), detect_size, cv2.INTER_LINEAR, mag_ratio = 1)
            img_resized_h, img_resized_w = img_resized.shape[:2]
            ratio_h = ratio_w = 1 / target_ratio
            db, mask = self._forward([img_resized], self.device)
        else:
            img_resized_h, img_resized_w = image.shape[:2]
            ratio_w = ratio_h = 1
            pad_h = pad_w = 0
        self.logger.info(f'Detection resolution: {img_resized_w}x{img_resized_h}')

        mask = mask[0, 0, :, :]
        det = dbnet_utils.SegDetectorRepresenter(text_threshold, box_threshold, unclip_ratio=unclip_ratio)
        # boxes, scores = det({'shape': [(img_resized.shape[0], img_resized.shape[1])]}, db)
        boxes, scores = det({'shape':[(img_resized_h, img_resized_w)]}, db)
        boxes, scores = boxes[0], scores[0]
        if boxes.size == 0:
            polys = []
        else:
            idx = boxes.reshape(boxes.shape[0], -1).sum(axis=1) > 0
            polys, _ = boxes[idx], scores[idx]
            polys = polys.astype(np.float64)
            polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h, ratio_net=1)
            polys = polys.astype(np.int64)

        textlines = [Quadrilateral(pts.astype(int), '', score) for pts, score in zip(polys, scores)]
        textlines = list(filter(lambda q: q.area > 16, textlines))
        mask_resized = cv2.resize(mask, (mask.shape[1] * 2, mask.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        if pad_h > 0:
            mask_resized = mask_resized[:-pad_h, :]
        elif pad_w > 0:
            mask_resized = mask_resized[:, :-pad_w]
        raw_mask = np.clip(mask_resized * 255, 0, 255).astype(np.uint8)

        # if verbose:
        #     img_bbox_raw = np.copy(image)
        #     for txtln in textlines:
        #         cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
        #     cv2.imwrite(f'result/bboxes_unfiltered.png', cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        return textlines, raw_mask, None
