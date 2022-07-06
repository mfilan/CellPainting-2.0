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
    split_method: Any  # SplitMethod
    dataset_path: str
    # cached_dataset_path: str
    label2id: Dict[str, int]
    train_transforms: Any  # torchvision.transforms.Compose = None
    test_transforms: Any  # torchvision.transforms.Compose = None
    compound_mapping: Dict[int, str] | None = None
    concentration_mapping: Dict[int, float] | None = None
    metadata_output_columns: List[str] | None = None


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
