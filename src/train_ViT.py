import multiprocessing as mp
import os

os.chdir("/home/azureuser/cloudfiles/code/Users/maciej.filanowicz/CellPainting-2.0")
print(os.getcwd())
import os

import hydra
import torch
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

import src.utils as utils
from src.conf import Config, DataConfig
from src.data import DataHandler
from src.data.dataset import CellPaintingDatasetCached
from src.model import ConvNext, Dummy, ModelTrainer, ResNet, ViT

os.chdir("/home/azureuser/cloudfiles/code/Users/maciej.filanowicz/CellPainting-2.0/src")
initialize(version_base="1.2", config_path="conf", job_name="test_app")
cfg = compose(config_name="config", return_hydra_config=True)
dataset_config = instantiate(cfg.dataset)
train_config = instantiate(cfg.train)

import pandas as pd

metadata = pd.read_csv("../data/processed/meta_data.csv")
train_dataset = CellPaintingDatasetCached(
    metadata[metadata.folder_name == "train"], dataset_config, dataset_config.train_transforms
)
test_dataset = CellPaintingDatasetCached(
    metadata[metadata.folder_name == "test"], dataset_config, dataset_config.test_transforms
)
val_dataset = CellPaintingDatasetCached(
    metadata[metadata.folder_name == "val"], dataset_config, dataset_config.test_transforms
)

import numpy as np
from datasets import load_metric
from transformers import default_data_collator

metric = load_metric("accuracy")
data_collator = default_data_collator


def compute_metrics(eval_pred):

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


args = TrainingArguments(
    "ViT",
    do_train=True,
    do_eval=True,
    logging_steps=100,
    save_steps=100,
    evaluation_strategy="steps",
    learning_rate=5e-4,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=13,
    dataloader_num_workers=24,
    weight_decay=0.01,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="wandb",
    logging_dir="logs",
)

model = ViT().to("cuda")

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
