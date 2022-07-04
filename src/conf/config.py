from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List

import torch
import torchvision.transforms
from torch.optim import Optimizer

if TYPE_CHECKING:
    from src.data.split import SplitMethod


@dataclass
class DataConfig:
    batch_size: int
    split_method: SplitMethod
    dataset_path: str
    cached_dataset_path: str
    compound_mapping: Dict[int, str]
    concentration_mapping: Dict[int, float]
    label2id: Dict[str, int]
    metadata_output_columns: List[str]
    transforms: torchvision.transforms.Compose = None


@dataclass
class TrainConfig:
    model: torch.nn.Module
    id2label: Dict[int, str]
    criterion: Any
    optimizer: Optimizer
    optimizer_params: Dict[str, Any]
    device: torch.device
    epochs: int = 1
    use_wandb: bool = False
    save_model_name: str = "model.pth"
    project_name: str = "CellPainting"


@dataclass
class Config:
    data_config: DataConfig
