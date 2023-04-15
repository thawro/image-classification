from src.evaluation.visualisations import (
    plot_images_probs_matplotlib,
    plot_imgs_preds_plotly,
    plot_imgs_probs_plotly,
    plot_multiclass_confusion_matrix,
    plot_multilabel_confusion_matrix,
)
from src.utils.types import Literal, Tensor, _Figure, _task


class ClassificationVisualizer:
    def __init__(self, task: _task, backend: Literal["matplotlib", "plotly"] = "plotly") -> None:
        self.task = task
        self.backend = backend

    def example_predictions(
        self, images: Tensor, targets: Tensor, probs: Tensor, labels: list[str], **kwargs
    ) -> _Figure:
        match (self.backend, self.task):
            case ("matplotlib", "multiclass"):
                plot_fn = plot_images_probs_matplotlib
            case ("matplotlib", "multilabel"):
                raise NotImplementedError()
            case ("plotly", "multiclass"):
                plot_fn = plot_imgs_probs_plotly
            case ("plotly", "multilabel"):
                plot_fn = plot_imgs_preds_plotly
        return plot_fn(images, targets, probs, labels)

    def confusion_matrix(self, targets: Tensor, probs: Tensor, labels: list[str], **kwargs) -> _Figure:
        match (self.backend, self.task):
            case ("matplotlib", "multiclass"):
                raise NotImplementedError()
            case ("matplotlib", "multilabel"):
                raise NotImplementedError()
            case ("plotly", "multiclass"):
                plot_fn = plot_multiclass_confusion_matrix
            case ("plotly", "multilabel"):
                plot_fn = plot_multilabel_confusion_matrix
        return plot_fn(targets, probs, labels)
