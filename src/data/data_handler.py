import multiprocessing as mp
import os
from itertools import chain
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from pandas.testing import assert_frame_equal
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from src import utils
from src.conf import DataConfig
from src.data.dataset import CellPaintingDataset
from src.data.metadata_extractor import MetadataExtractor
from src.data.split import SPLIT_METHODS, SplitMethod


class DataHandler:
    def __init__(self, config: DataConfig):
        self.config = config
        self.metadata_extractor = MetadataExtractor(self.config)
        self.dataset = CellPaintingDataset
        self.split_strategy = SPLIT_METHODS[SplitMethod[config.split_method]](self.config.label2id)

    def create_data_loaders(
        self, dataset: CellPaintingDataset, samplers: Dict[str, SubsetRandomSampler]
    ) -> Dict[str, torch.utils.data.DataLoader]:
        loader_params = dict(
            dataset=dataset,
            batch_size=self.config.batch_size,
            num_workers=0,
            pin_memory=False,
            generator=self.split_strategy.generator,
        )
        data_loaders = {}
        for sampler in samplers:
            data_loaders[sampler.replace("sampler", "data_loader")] = DataLoader(
                **loader_params, sampler=samplers[sampler]
            )
        return data_loaders

    def get_data_loaders(self) -> Dict[str, torch.utils.data.DataLoader]:
        dataset_df = self.metadata_extractor.get_data_frame()
        samplers = self.split_strategy.get_subset_sampler(dataset_df)
        dataset = self.dataset(dataset_df, self.config, self.config.transforms)
        data_loaders = self.create_data_loaders(dataset, samplers)
        return data_loaders

    @staticmethod
    def create_directories(dataset_df: pd.DataFrame) -> None:
        subset_folders = dataset_df.subset.unique()
        cached_dataset_path = dataset_df.cached_dataset_path.values[0]
        for subset_folder in subset_folders:
            path = os.path.join(cached_dataset_path, subset_folder)
            Path(path).mkdir(parents=True, exist_ok=True)

    def prepare_dataset_to_cache(self) -> pd.DataFrame:
        dataset_df = self.metadata_extractor.get_data_frame()
        dataset_df["file_name"] = dataset_df["file_name1"].str.replace("ch1", "")
        dataset_df["dataset_path"] = self.config.dataset_path
        dataset_df["cached_dataset_path"] = self.config.cached_dataset_path
        indices = self.split_strategy.split_dataset_indices(dataset_df)
        dataset_df.loc[indices["train_indices"], "subset"] = "train"
        dataset_df.loc[indices["test_indices"], "subset"] = "test"
        dataset_df.loc[indices["val_indices"], "subset"] = "val"
        self.create_directories(dataset_df)
        return dataset_df

    @staticmethod
    def dataset_exists(new_dataset_df: pd.DataFrame, dataset_path: str) -> bool:
        if Path(dataset_path).is_file():
            existing_dataset_df = pd.read_csv(dataset_path)
            try:
                assert_frame_equal(
                    existing_dataset_df,
                    new_dataset_df,
                    check_dtype=False,
                    check_column_type=False,
                    check_frame_type=False,
                )
                return True
            except AssertionError:
                return False
        return False

    @staticmethod
    def save_dataset_data_frame(dataset_df: pd.DataFrame, dataset_df_save_path: str) -> None:
        dataset_df.drop_duplicates(inplace=True)
        new_file_name = dataset_df["folder_name"] + "_" + dataset_df["file_name"]
        dataset_df.loc[:, "file_name"] = new_file_name
        dataset_df.drop(columns="folder_name", inplace=True)
        dataset_df = dataset_df.rename(columns={"subset": "folder_name"})
        dataset_df.to_csv(dataset_df_save_path, index=False)

    def cache_dataset(self, save_method_name: utils.SaveMethod = utils.SaveMethod.BASIC) -> None:
        save_method = utils.SAVE_METHODS[save_method_name]
        dataset_df = self.prepare_dataset_to_cache()
        dataset_df_save_path = os.path.join(self.config.cached_dataset_path, "meta_data.csv")

        if self.dataset_exists(dataset_df, dataset_df_save_path):
            print("Using cached dataset!")
            return None

        dataset_dict = dataset_df.to_dict(orient="index")
        images_info = list(dataset_dict.values())
        with mp.Pool() as pool:
            new_meta_data = pool.map(save_method, images_info)
        new_dataset_df = pd.DataFrame(chain.from_iterable(new_meta_data))
        self.save_dataset_data_frame(new_dataset_df, dataset_df_save_path)
        return None
