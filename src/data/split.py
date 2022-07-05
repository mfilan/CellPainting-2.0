from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Type

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data.sampler import SubsetRandomSampler


class SplitStrategy(ABC):
    def __init__(self, label2id: Dict[str, int]) -> None:
        self.splitter = StratifiedGroupKFold(n_splits=8, random_state=0, shuffle=True)
        self.label2id = label2id
        self.generator = torch.Generator(device="cpu").manual_seed(0)

    @staticmethod
    def get_split_params(dataset_df: pd.DataFrame, label2id: Dict[str, int]) -> Dict[str, npt.NDArray[np.int64]]:
        labels = dataset_df["compound_name"].map(label2id).to_numpy()
        dataset_indices = dataset_df.index.to_numpy().reshape(-1, 1)
        groups = dataset_df[["well_id"]].to_numpy().flatten()
        return {"dataset_indices": dataset_indices, "labels": labels, "groups": groups}

    @abstractmethod
    def split_dataset_indices(self, dataset_df: pd.DataFrame) -> Dict[str, npt.NDArray[np.int64]]:
        ...

    def get_subset_sampler(self, dataset_df: pd.DataFrame) -> Dict[str, SubsetRandomSampler]:
        samplers = {}
        for subset_name, subset_indices in self.split_dataset_indices(dataset_df).items():
            samplers[subset_name.replace("indices", "sampler")] = torch.utils.data.SubsetRandomSampler(
                subset_indices, generator=self.generator
            )
        return samplers


class LeaveOneCartridgeStratifiedSplit(SplitStrategy):
    def split_dataset_indices(self, dataset_df: pd.DataFrame) -> Dict[str, npt.NDArray[np.int64]]:
        cartridges_names = list(dataset_df.folder_name.unique())
        test_cartridge = cartridges_names.pop(np.random.randint(0, len(cartridges_names)))
        test_cartridge_mask = dataset_df.folder_name.str.match(test_cartridge)
        test_df, train_df = dataset_df[test_cartridge_mask], dataset_df[~test_cartridge_mask]
        test_indices = test_df.index.to_numpy()
        dataset_indices, labels, groups = list(self.get_split_params(train_df, self.label2id).values())
        (train_indices, val_indices) = next(self.splitter.split(dataset_indices, labels, groups))
        return {
            "train_indices": dataset_indices.flatten()[train_indices],
            "val_indices": dataset_indices.flatten()[val_indices],
            "test_indices": test_indices,
        }


class StratifiedSplit(SplitStrategy):
    def split_dataset_indices(self, dataset_df: pd.DataFrame) -> Dict[str, npt.NDArray[np.int64]]:
        dataset_indices, labels, groups = list(self.get_split_params(dataset_df, self.label2id).values())
        (train_indices, test_indices) = next(self.splitter.split(X=dataset_indices, y=labels, groups=groups))
        train_indices = dataset_indices.flatten()[train_indices]
        test_df = dataset_df.iloc[dataset_indices.flatten()[test_indices]]
        dataset_indices, labels, groups = list(self.get_split_params(test_df, self.label2id).values())
        (val_indices, test_indices) = next(self.splitter.split(X=dataset_indices, y=labels, groups=groups))
        val_indices = dataset_indices.flatten()[val_indices]
        test_indices = dataset_indices.flatten()[test_indices]
        return {"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices}


class SplitMethod(str, Enum):
    STRATIFIED_SPLIT = "Stratified Split"
    LOC_STRATIFIED_SPLIT = "Leave One Cartridge Stratified Split"


SPLIT_METHODS: Dict[SplitMethod, Type[SplitStrategy]] = {
    SplitMethod.STRATIFIED_SPLIT: StratifiedSplit,
    SplitMethod.LOC_STRATIFIED_SPLIT: LeaveOneCartridgeStratifiedSplit,
}
