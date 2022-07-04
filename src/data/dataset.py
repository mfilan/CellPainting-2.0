from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from src import utils

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

    def __getitem__(self, idx: int) -> Dict[str, npt.NDArray[np.float32] | int]:
        image_info = self.dataset_df.iloc[idx, :].to_dict()
        label = self.config.label2id[image_info["compound_name"]]
        image = utils.read_all_channels(image_info, self.config.dataset_path)
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "label": label}


class CellPaintingDatasetCached(Dataset):
    def __init__(self, dataset_df: pd.DataFrame, config: DataConfig, transform: transforms.Compose = None) -> None:
        self.dataset_df = dataset_df
        self.config = config
        self.transform = transform

    def __len__(self) -> int:
        num_of_images = int(self.dataset_df.shape[0])
        return num_of_images

    def __getitem__(self, idx: int) -> Dict[str, npt.NDArray[np.uint16] | int]:
        image_info = self.dataset_df.iloc[idx, :].to_dict()
        label = self.config.label2id[image_info["compound_name"]]
        image = utils.read_image(image_info["file_name"], image_info["folder_name"], self.config.cached_dataset_path)
        if self.transform:
            image = self.transform(image)
        return {"pixel_values": image, "label": label}
