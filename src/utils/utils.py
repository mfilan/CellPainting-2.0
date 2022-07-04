import os
import re
from typing import Any, Dict, Union

import numpy as np
import numpy.typing as npt
import torch
from tifffile import imread


def to_numpy(vector: Union[torch.Tensor, np.ndarray]) -> Any:
    if isinstance(vector, torch.Tensor):
        return vector.detach().cpu().numpy()
    return vector


def to_list(**kwargs):
    return list(kwargs.values())


def read_image(image_name: str, image_folder: str, data_dir: str) -> npt.NDArray[np.uint16]:
    image: npt.NDArray[np.uint16] = imread(os.path.join(data_dir, image_folder, image_name))
    return image


def read_all_channels(image_info: Dict[str, Any], data_dir: str) -> npt.NDArray[np.float32]:
    per_channel_image_names = sorted(
        [image_info[attribute] for attribute in image_info if re.match(r"file_name\d", attribute) is not None]
    )
    per_channel_images = []
    for channel_image_name in per_channel_image_names:
        channel_image = read_image(channel_image_name, image_info["folder_name"], data_dir)
        per_channel_images.append(channel_image)
    stacked_image = np.stack(per_channel_images).transpose((1, 2, 0))
    image = np.array(stacked_image, dtype=np.float32)
    return image
