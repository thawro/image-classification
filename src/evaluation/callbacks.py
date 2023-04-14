import random

import pytorch_lightning as pl
import torch

import wandb
from src.module.base import BaseImageClassifier
from src.utils.types import Literal

from .visualisations import plot_imgs_preds_plotly, plot_imgs_probs_plotly


class ExamplePredictionsLogger(pl.Callback):
    def __init__(
        self,
        num_examples: int = 8,
        multilabel: bool = False,
        modes: list[Literal["random", "worst", "best"]] = ["random"],
    ):
        super().__init__()
        self.num_examples = num_examples
        self.modes = modes
        self.multilabel = multilabel
        self.plot_fn = plot_imgs_preds_plotly if multilabel else plot_imgs_probs_plotly
        self.every_n_epochs = 5

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseImageClassifier):
        if pl_module.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        examples = pl_module.examples["val"]
        loss, probs, targets = examples["loss"], examples["probs"], examples["targets"]
        imgs = examples["data"].permute(0, 2, 3, 1)

        if self.multilabel:
            loss = loss.mean(dim=1)
        for mode in self.modes:
            if mode == "random":
                idxs = random.choices(range(len(targets)), k=self.num_examples)
            else:
                multiplier = -1 if mode == "best" else 1
                sorted_metric = loss * multiplier
                idxs = torch.topk(sorted_metric, self.num_examples).indices.tolist()
            img_probabilities_plot = self.plot_fn(imgs[idxs], targets[idxs], probs[idxs], pl_module.classes)
            pl_module.logged_metrics[f"examples/{mode}"] = wandb.Plotly(img_probabilities_plot)
