import random

import pytorch_lightning as pl
import torch

import wandb
from src.data.transforms import imagenet_unnormalizer, mnist_unnormalizer
from src.module.base import BaseImageClassifier
from src.utils.types import Literal


class ExamplePredictionsLogger(pl.Callback):
    def __init__(
        self,
        num_examples: int = 8,
        modes: list[Literal["random", "worst", "best"]] = ["random"],
    ):
        super().__init__()
        self.num_examples = num_examples
        self.modes = modes
        self.every_n_epochs = 5

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseImageClassifier):
        if pl_module.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        examples = pl_module.examples["val"]
        loss, probs, targets = examples["loss"], examples["probs"], examples["targets"]
        images = examples["images"]
        unnormalizer = imagenet_unnormalizer if images.shape[1] == 3 else mnist_unnormalizer
        images = unnormalizer(images).permute(0, 2, 3, 1)

        if pl_module.task == "multilabel":
            loss = loss.mean(dim=1)
        for mode in self.modes:
            if mode == "random":
                idxs = random.choices(range(len(targets)), k=self.num_examples)
            else:
                multiplier = -1 if mode == "best" else 1
                sorted_metric = loss * multiplier
                idxs = torch.topk(sorted_metric, self.num_examples).indices.tolist()
            examples_plot = pl_module.visualizer.example_predictions(
                images=images[idxs],
                targets=targets[idxs],
                probs=probs[idxs],
                labels=pl_module.classes,
            )
            pl_module.logged_metrics[f"examples/{mode}"] = wandb.Plotly(examples_plot)


class ConfusionMatrixLogger(pl.Callback):
    def __init__(self):
        super().__init__()
        self.every_n_epochs = 5

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: BaseImageClassifier):
        if pl_module.current_epoch % self.every_n_epochs != 0:
            return
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        outputs = pl_module.outputs["val"]
        probs = torch.concat([output["probs"] for output in outputs]).cpu()
        targets = torch.concat([output["targets"] for output in outputs]).cpu()
        confusion_matrix_plot = pl_module.visualizer.confusion_matrix(
            targets=targets, probs=probs, labels=pl_module.classes
        )
        pl_module.logged_metrics["confusion_matrix"] = wandb.Plotly(confusion_matrix_plot)
