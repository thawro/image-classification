import wandb
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from .visualisations import plot_images_probabilities_plotly
from typing import Literal
import random


class ExamplePredictionsLogger(pl.Callback):
    def __init__(self, num_examples: int = 8, mode: Literal["random", "worst", "best"] = "random"):
        self.num_examples = num_examples
        self.mode = mode
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
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
        img_probabilities_plot = plot_images_probabilities_plotly(imgs, targets, probs, pl_module.classes)
        pl_module.metrics[f"{self.mode}_examples"] = img_probabilities_plot


class FeatureActivationsLogger(pl.Callback):
    def __init__(self, val_samples, num_samples=4):
        super().__init__()
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]

    def on_validation_epoch_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        imgage_classifier = trainer.model
        cnn_layer = image_classifier.classifier.model.cnn_block_1[0]
        kernels = cnn_layer._parameters["weight"]
        kenerls_min, kernels_max = kernels.min(), kernels.max()
        n_samples = len(val_imgs)
        n_channels = len(kernels)
        fig, axes = plt.subplots(n_samples + 1, n_channels + 1, figsize=(20, 10))
        axes[0, 0].axis("off")
        axes[0, 1].set_ylabel("Kernels", fontsize=16, fontweight="bold")
        axes[1, 0].set_title("Input to Conv", fontsize=16, fontweight="bold")
        for filt_idx, ax in enumerate(axes[0, 1:]):
            ax.imshow(kernels[filt_idx].squeeze().cpu(), cmap="gray", vmin=kenerls_min, vmax=kernels_max)
        for x, ax in zip(val_imgs, axes[1:, 0]):
            ax.imshow(x.cpu().numpy().transpose(1, 2, 0), cmap="gray")

        for sample_idx, x in enumerate(val_imgs):
            for filt_idx in range(n_channels):
                ax = axes[sample_idx + 1, filt_idx + 1]
                out_min, out_max = cnn_layer(x).min(), cnn_layer(x).max()
                ax.imshow(cnn_layer(x)[filt_idx].cpu(), cmap="gray", vmin=out_min, vmax=out_max)
        for ax in axes.flatten():
            ax.tick_params(
                axis="both", which="both", left=False, right=False, labelleft=False, labelbottom=False, bottom=False
            )
        plt.tight_layout()
        trainer.logger.experiment.log({"feature_activations": fig, "global_step": trainer.global_step})
        plt.close()
