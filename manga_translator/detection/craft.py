"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
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
from shapely.geometry import Polygon, MultiPoint
from shapely import affinity

from .craft_utils.vgg16_bn import vgg16_bn, init_weights
from .craft_utils.refiner import RefineNet

class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())
        
    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0,2,3,1), feature


from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

class CRAFTDetector(OfflineDetector):
    _MODEL_MAPPING = {
        'refiner': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/craft_refiner_CTW1500.pth',
            'hash': 'f7000cd3e9c76f2231b62b32182212203f73c08dfaa12bb16ffb529948a01399',
            'file': 'craft_refiner_CTW1500.pth',
        },
        'craft': {
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/craft_mlt_25k.pth',
            'hash': '4a5efbfb48b4081100544e75e1e2b57f8de3d84f213004b14b85fd4b3748db17',
            'file': 'craft_mlt_25k.pth',
        }
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        if os.path.exists('craft_mlt_25k.pth'):
            shutil.move('craft_mlt_25k.pth', self._get_file_path('craft_mlt_25k.pth'))
        if os.path.exists('craft_refiner_CTW1500.pth'):
            shutil.move('craft_refiner_CTW1500.pth', self._get_file_path('craft_refiner_CTW1500.pth'))
        super().__init__(*args, **kwargs)

    async def _load(self, device: str, use_igpu: bool = False):
        self.use_igpu = use_igpu
        if use_igpu:
            device = 'cpu'
        self.model = CRAFT()
        self.model.load_state_dict(copyStateDict(torch.load(self._get_file_path('craft_mlt_25k.pth'), map_location='cpu')))
        self.model.eval()
        self.model_refiner = RefineNet()
        self.model_refiner.load_state_dict(copyStateDict(torch.load(self._get_file_path('craft_refiner_CTW1500.pth'), map_location='cpu')))
        self.model_refiner.eval()
        self.device = device
        if device.startswith('cuda') or device == 'mps' or device == 'xpu':
            self.model = self.model.to(self.device)
            self.model_refiner = self.model_refiner.to(self.device)
        self.execution_devices = ['PyTorch CPU'] if device == 'cpu' else [device]
        if use_igpu:
            self._load_openvino()

    async def _unload(self):
        if getattr(self, 'use_igpu', False):
            self.compiled_shapes.clear()
            self.ov_model = None
            self.core = None
        del self.model
        del self.model_refiner

    def _load_openvino(self):
        self.ov_device = os.environ.get('MANGA_OPENVINO_DEVICE', 'GPU')
        self.precision = os.environ.get('MANGA_OPENVINO_PRECISION', 'f32')
        if self.precision not in ('f32', 'f16'):
            raise ValueError('OpenVINO precision must be f32 or f16')
        self.compiled_shapes = OrderedDict()
        self.ov_model = None
        try:
            import openvino as ov

            class DetectionWithRefiner(nn.Module):
                def __init__(self, detector, refiner):
                    super().__init__()
                    self.detector, self.refiner = detector, refiner

                def forward(self, image):
                    scores, features = self.detector(image)
                    return scores, self.refiner(scores, features)

            self.core = ov.Core()
            network = DetectionWithRefiner(self.model, self.model_refiner).eval()
            with torch.inference_mode():
                self.ov_model = ov.convert_model(network,
                    example_input=torch.zeros(1, 3, 64, 64), input=[1, 3, -1, -1])
        except Exception as exc:
            self._fallback_openvino(exc)

    def _fallback_openvino(self, exc):
        self.ov_model = None
        self.compiled_shapes.clear()
        self.execution_devices = ['PyTorch CPU']
        self.logger.warning('CRAFT OpenVINO failed; using PyTorch CPU: %s', exc)

    def _forward(self, image):
        if getattr(self, 'use_igpu', False) and self.ov_model is not None:
            try:
                shape = tuple(image.shape)
                if shape not in self.compiled_shapes:
                    model = self.ov_model.clone()
                    model.reshape({model.input(0): list(shape)})
                    config = {'INFERENCE_PRECISION_HINT': self.precision}
                    cache = os.environ.get('MANGA_OPENVINO_CACHE_DIR')
                    if cache:
                        config['CACHE_DIR'] = cache
                    if len(self.compiled_shapes) >= 2:
                        self.compiled_shapes.popitem(last=False)
                    compiled = self.core.compile_model(model, self.ov_device, config)
                    self.compiled_shapes[shape] = compiled
                    self.execution_devices = list(compiled.get_property('EXECUTION_DEVICES'))
                    self.logger.info('CRAFT OpenVINO shape %s execution devices: %s', shape, self.execution_devices)
                self.compiled_shapes.move_to_end(shape)
                compiled = self.compiled_shapes[shape]
                result = compiled([np.ascontiguousarray(image.detach().cpu().numpy())])
                values = [np.array(result[p], copy=True) for p in compiled.outputs]
                expected = (shape[0], shape[2] // 2, shape[3] // 2)
                if (len(values) != 2 or values[0].shape != (*expected, 2)
                        or values[1].shape != (*expected, 1)
                        or not all(np.isfinite(v).all() for v in values)):
                    raise ValueError('Unexpected CRAFT output layout or nonfinite values')
                return tuple(torch.from_numpy(v) for v in values)
            except Exception as exc:
                self._fallback_openvino(exc)
        with torch.no_grad():
            scores, features = self.model(image)
            return scores, self.model_refiner(scores, features)

    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                     unclip_ratio: float, verbose: bool = False):
        
        img_resized, target_ratio, size_heatmap, pad_w, pad_h = imgproc.resize_aspect_ratio(image, detect_size, interpolation = cv2.INTER_CUBIC, mag_ratio = 1)
        ratio_h = ratio_w = 1 / target_ratio

        # preprocessing
        x = imgproc.normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
        x = x.unsqueeze(0).to(self.device)                # [c, h, w] to [b, c, h, w]

        y, y_refiner = self._forward(x)

        # make score and link map
        score_text = y[0,:,:,0].cpu().data.numpy()
        score_link = y[0,:,:,1].cpu().data.numpy()

        # refine link
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

        # Post-processing
        boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, box_threshold, box_threshold, True)

        # coordinate adjustment
        boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
        polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
        for k in range(len(polys)):
            if polys[k] is None: polys[k] = boxes[k]

        mask = np.zeros(shape = (image.shape[0], image.shape[1]), dtype = np.uint8)

        for poly in polys :
            mask = cv2.fillPoly(mask, [poly.reshape((-1, 1, 2)).astype(np.int32)], color = 255)
        
        polys_ret = []
        for i in range(len(polys)) :
            poly = Polygon(polys[i])
            if poly.area > 10 :
                rect = poly.minimum_rotated_rectangle
                rect = affinity.scale(rect, xfact = 1.2, yfact = 1.2)
                polys_ret.append(np.roll(np.asarray(list(rect.exterior.coords)[:4]), 2))

        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.dilate(mask, kern)

        textlines = [Quadrilateral(pts.astype(int), '', 1) for pts in polys_ret]
        textlines = list(filter(lambda q: q.area > 16, textlines))

        return textlines, mask, None
