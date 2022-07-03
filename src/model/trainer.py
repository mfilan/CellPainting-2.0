from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from src.conf import TrainConfig
from src.utils import to_numpy


class ModelTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainConfig,
        train_data_loader: DataLoader,
        val_data_loader: DataLoader = None,
        test_data_loader: DataLoader = None,
    ):

        self.model = model
        self.config = config
        self.optimizer = config.optimizer(model.parameters(), **config.optimizer_params)
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.__post_initialize__()

    def __post_initialize__(self) -> None:
        self.training_step = 0
        self.epoch = 0
        self.training_loss: List[Any] = []
        self.validation_loss: List[Any] = []
        if self.config.use_wandb:
            wandb.init(project=self.config.project_name)
            wandb.define_metric("validation_loss", step_metric="step")

    def train_log(self, loss: float) -> None:
        """
        Function responsible for logging training loss to wandb

        Parameters
        ----------
            loss (float): running training loss

        """
        wandb.log({"epoch": self.epoch, "loss": loss}, step=self.training_step)

    def validation_log(self, loss: float) -> None:
        """
        Function responsible for logging validation loss to wandb

        Parameters
        ----------
            loss (float): running validation loss

        """
        wandb.log({"epoch": self.epoch, "validation_loss": loss}, step=self.training_step)

    def run_trainer(self) -> Union[None, Dict[str, List[Any]]]:
        """
        Wrapper of _run_trainer responsible for error handling
        """
        try:
            train_output = self._run_trainer()
            if self.config.use_wandb:
                wandb.finish()
            torch.save(self.model, self.config.save_model_name)
            return train_output
        except KeyboardInterrupt:
            if self.config.use_wandb:
                wandb.finish()
            torch.save(self.model, str(self.config.save_model_name))
            return None

    def _run_trainer(self) -> Dict[str, List[Any]]:
        """
        Function responsible for running _train() and _validate
        """

        progressbar = trange(self.config.epochs, desc="Progress")
        for _ in progressbar:
            self.epoch += 1  # epoch counter

            self._train()

            if self.val_data_loader is not None:
                self._validate()

        if self.test_data_loader is not None:
            self._test()

        return {
            "training_loss": self.training_loss,
            "validation_loss": self.validation_loss,
        }

    def _train(self) -> None:
        """
        Function responsible for training model
        """
        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.train_data_loader), "Training", total=len(self.train_data_loader), leave=False)

        for _, (images, labels) in batch_iter:
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(images)  # one forward pass
            loss = self.config.criterion(out, labels)
            loss_value = loss.item()
            if self.config.use_wandb:
                self.train_log(loss_value)
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            batch_iter.set_description(f"Training: (loss {loss_value:.4f})")  # update progressbar
            self.training_step += 1
        self.training_loss.append(np.mean(train_losses))
        batch_iter.close()

    def _validate(self) -> None:
        """
        Function responsible for validating model
        """
        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        batch_iter = tqdm(enumerate(self.val_data_loader), "Validation", total=len(self.val_data_loader), leave=False)

        for _, (images, labels) in batch_iter:
            with torch.no_grad():
                out = self.model(images)
                loss = self.config.criterion(out, labels)
                loss_value = loss.item()
                valid_losses.append(loss_value)
                batch_iter.set_description(f"Validation: (loss {loss_value:.4f})")
        mean_losses = np.mean(valid_losses)
        if self.config.use_wandb:
            self.validation_log(float(mean_losses))
        self.validation_loss.append(mean_losses)

        batch_iter.close()

    def _test(self) -> pd.DataFrame:

        self.model.eval()
        all_labels = []
        all_predictions = []

        for (images, labels) in tqdm(self.test_data_loader, desc="Evaluating"):
            with torch.no_grad():
                out = self.model(images)
                predictions = out.argmax(dim=-1)
                all_labels += list(to_numpy(labels))
                all_predictions += list(to_numpy(predictions))

        report = classification_report(
            y_true=all_labels,
            y_pred=all_predictions,
            labels=list(self.config.id2label.keys()),
            target_names=list(self.config.id2label.values()),
        )
        return report
