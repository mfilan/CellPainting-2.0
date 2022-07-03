from typing import Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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
