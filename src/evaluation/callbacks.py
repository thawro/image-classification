import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .visualisations import plot_images_probabilities_plotly
import random
from src.architectures.model import ImageClassifier
from src.utils.types import Literal


class ExamplePredictionsLogger(pl.Callback):
    def __init__(self, num_examples: int = 8, mode: Literal["random", "worst", "best"] = "random"):
        self.num_examples = num_examples
        self.mode = mode
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: ImageClassifier):
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        outputs = pl_module.val_step_outputs
        imgs = torch.concat([output["data"] for output in outputs]).permute(0, 2, 3, 1)
        probs = torch.concat([output["probs"] for output in outputs])
        targets = torch.concat([output["targets"] for output in outputs])
        if self.mode == "random":
            idxs = random.choices(range(len(targets)), k=self.num_examples)
        else:
            multiplier = -1 if self.mode == "best" else 1
            sorted_metric = F.nll_loss(torch.log(probs), targets, reduction="none") * multiplier
            idxs = torch.topk(sorted_metric, self.num_examples).indices.tolist()

        imgs, probs, targets = imgs[idxs], probs[idxs], targets[idxs]
        img_probabilities_plot = plot_images_probabilities_plotly(
            imgs, targets, probs, pl_module.classes
        )
        pl_module.metrics[f"{self.mode}_examples"] = img_probabilities_plot
