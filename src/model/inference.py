from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.nn.functional import softmax
from torchvision.transforms import Compose
from transformers import ResNetConfig, ResNetForImageClassification

from src import utils


class CellPainingModel:
    def __init__(self, config: ResNetConfig, transforms: Compose = None) -> None:
        self.model = ResNetForImageClassification(config)
        self.model.load_state_dict(torch.load(config.pretrained_model_name, map_location=torch.device(config.device)))
        self._transforms = transforms

    @staticmethod
    def _patch_images(images: npt.NDArray[Any]) -> npt.NDArray[Any]:
        patches = np.array([utils.view_as_blocks(i, (540, 540, 4)) for i in images])
        return patches

    def transform(self, images: npt.NDArray[Any]) -> Any:
        images = torch.stack([self._transforms(i) for i in images])
        return images

    def __call__(self, pixel_values: npt.NDArray[Any]) -> npt.NDArray[Any]:
        images = self._patch_images(pixel_values)
        input_shape = images.shape
        images = images.reshape((-1, 540, 540, 4))
        if self._transforms:
            images = self.transform(images)
        output = self.model(pixel_values=images, labels=None)
        output_shape = (input_shape[0], input_shape[1] * input_shape[2])
        logits = output.logits.reshape((*output_shape, -1))
        mean_logits: npt.NDArray[Any] = softmax(torch.mean(logits, axis=1), -1).detach().cpu().numpy()
        return mean_logits
