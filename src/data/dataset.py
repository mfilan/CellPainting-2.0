from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Dict, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from tifffile import imread
from torch.utils.data import Dataset
from torchvision.transforms import transforms

if TYPE_CHECKING:
    from src.conf import DataConfig


class CellPaintingDataset(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, config: DataConfig, transform: transforms.Compose = None) -> None:
        self.dataset_df = dataset_df
        self.config = config
        self.transform = transform

    def __len__(self) -> int:
        num_of_images = int(self.dataset_df.shape[0])
        return num_of_images

    @staticmethod
    def _read_image_channel(image_name: str, image_folder: str, data_dir: str) -> npt.NDArray[np.uint16]:
        image: npt.NDArray[np.uint16] = imread(os.path.join(data_dir, image_folder, image_name))
        return image

    def read_image(self, image_info: Dict[str, Any], data_dir: str) -> npt.NDArray[np.float32]:
        per_channel_image_names = sorted(
            [image_info[attribute] for attribute in image_info if "file_name" in attribute]
        )
        per_channel_images = []
        for channel_image_name in per_channel_image_names:
            channel_image = self._read_image_channel(channel_image_name, image_info["folder_name"], data_dir)
            per_channel_images.append(channel_image)
        stacked_image = np.stack(per_channel_images).transpose((1, 2, 0))
        image = np.array(stacked_image, dtype=np.float32)
        return image

    def __getitem__(self, idx: int) -> Tuple[npt.NDArray[np.float32], int]:
        image_info = self.dataset_df.iloc[idx, :].to_dict()
        label = self.config.label2id[image_info["compound_name"]]
        image = self.read_image(image_info, self.config.dataset_path)
        if self.transform:
            image = self.transform(image)
        return image, label
