import os
import re
from enum import Enum
from typing import Any, Dict, List, Tuple, Type, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
from numpy.lib.stride_tricks import as_strided
from PIL import Image
from tifffile import imread, imwrite


def to_numpy(vector: Union[torch.Tensor, npt.NDArray[Any]]) -> Any:
    if isinstance(vector, torch.Tensor):
        return vector.detach().cpu().numpy()
    return vector


def to_list(**kwargs: Any) -> List[Any]:
    return list(kwargs.values())


def read_image(image_name: str, image_folder: str, data_dir: str) -> npt.NDArray[np.uint16]:
    image = imread(os.path.join(data_dir, image_folder, image_name))
    return image


def read_all_channels(image_info: Dict[str, Any], data_dir: str) -> npt.NDArray[np.uint16]:
    per_channel_image_names = sorted(
        [image_info[attribute] for attribute in image_info if re.match(r"file_name\d", attribute) is not None]
    )
    per_channel_images = []
    for channel_image_name in per_channel_image_names:
        channel_image = read_image(channel_image_name, image_info["folder_name"], data_dir)
        per_channel_images.append(channel_image)
    stacked_image = np.stack(per_channel_images).transpose((1, 2, 0))
    return stacked_image


def view_as_blocks(arr_in: npt.NDArray[Any], block_shape: Tuple[int, int, int]) -> Any:
    if arr_in.shape[0] % block_shape[0] or arr_in.shape[1] % block_shape[1] or arr_in.shape[2] != block_shape[2]:
        raise ValueError("Incompatible block shape!")
    bs = np.array(block_shape)
    arr_shape = np.array(arr_in.shape)
    new_shape = tuple(arr_shape // bs) + tuple(bs)
    new_strides = tuple(arr_in.strides * bs) + arr_in.strides
    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)
    arr_out = np.squeeze(arr_out)
    return arr_out


def save_merged_image(image_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    image = read_all_channels(image_info, image_info["dataset_path"])
    new_image_file_name = image_info["folder_name"] + "_" + image_info["file_name"]
    save_path = os.path.join(image_info["cached_dataset_path"], image_info["subset"], new_image_file_name)
    imwrite(save_path, image)
    new_image_metadata = dict(
        folder_name=image_info["folder_name"],
        file_name=image_info["file_name"],
        compound_id=image_info["compound_id"],
        concentration_id=image_info["concentration_id"],
        compound_name=image_info["compound_name"],
        concentration=image_info["concentration"],
        subset=image_info["subset"],
    )
    return [new_image_metadata]


def save_patched_image(image_info: Dict[str, Any], patch_size: int = 540) -> List[Dict[str, Any]]:
    image = read_all_channels(image_info, image_info["dataset_path"])
    patches = view_as_blocks(image, (patch_size, patch_size, int(image.shape[-1])))
    patches_metadata = []
    for row in range(patches.shape[0]):
        for column in range(patches.shape[1]):
            new_image_file_name = (
                image_info["folder_name"] + "_" + str(row) + "_" + str(column) + "_" + image_info["file_name"]
            )
            save_path = os.path.join(image_info["cached_dataset_path"], image_info["subset"], new_image_file_name)
            patch = cv2.resize(patches[row, column], (224, 224), cv2.INTER_LANCZOS4)
            imwrite(save_path, patch)
            new_image_metadata = dict(
                folder_name=image_info["folder_name"],
                file_name=str(row) + "_" + str(column) + "_" + image_info["file_name"],
                compound_id=image_info["compound_id"],
                concentration_id=image_info["concentration_id"],
                compound_name=image_info["compound_name"],
                concentration=image_info["concentration"],
                subset=image_info["subset"],
            )
            patches_metadata.append(new_image_metadata)
    return patches_metadata


class SaveMethod(str, Enum):
    BASIC = "basic"
    PATCHED = "patched"


SAVE_METHODS: Dict[SaveMethod, Any] = {
    SaveMethod.BASIC: save_merged_image,
    SaveMethod.PATCHED: save_patched_image,
}
