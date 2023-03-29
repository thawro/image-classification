import wandb
import pytorch_lightning as pl
import torch
from .visualisations import plot_images_probabilities_plotly


class ExamplePredictionsLogger(pl.Callback):
    def __init__(self, num_examples=8):
        self.num_examples = num_examples
        super().__init__()

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if trainer.state.fn != "fit" or trainer.sanity_checking:
            return
        outputs = pl_module.val_step_outputs
        imgs = torch.concat([output["data"] for output in outputs])[: self.num_examples].permute(0, 2, 3, 1)
        probs = torch.concat([output["probs"] for output in outputs])[: self.num_examples]
        targets = torch.concat([output["targets"] for output in outputs])[: self.num_examples]
        img_probabilities_plot = plot_images_probabilities_plotly(imgs, targets, probs, pl_module.classes)
        pl_module.metrics["img_probabilities_plot"] = img_probabilities_plot


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
