import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.express as px
from torchtyping import TensorType
from src.utils.types import Optional


def plot_probabilities(
    probs: TensorType["n_classes"],
    target: int,
    labels: list[str],
    n_best: int = 5,
    ax: Optional[plt.Axes] = None,
):
    palette = sns.color_palette()
    if ax is None:
        fig, ax = plt.subplots()
    pred = int(probs.argmax().item())
    colors = [palette[0]] * len(labels)
    if pred == target:
        colors[pred] = palette[2]  # green
    else:
        colors[pred] = palette[3]  # red
        colors[target] = palette[8]

    sorted_probs = sorted(probs, reverse=False)[-n_best:]
    sorted_labels = [label for _, label in sorted(zip(probs, labels), reverse=False)][-n_best:]
    sorted_colors = [color for _, color in sorted(zip(probs, colors), reverse=False)][-n_best:]

    hbars = ax.barh(sorted_labels, sorted_probs, color=sorted_colors, alpha=0.8)
    max_prob = max(sorted_probs)
    for i, bar in enumerate(hbars):
        ax.annotate(
            sorted_labels[i],
            xy=(0.05 * max_prob, bar.get_y() + bar.get_height() / 3),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=12,
        )

    ax.bar_label(hbars, fmt="%.2f")
    ax.set_xlim(right=1.1 * max_prob)  # adjust xlim to fit labels
    ax.grid(False)


def plot_images_probabilities_matplotlib(
    images: TensorType["batch", "height", "width", "channels"],
    targets: TensorType["batch"],
    probs: TensorType["batch", "n_classes"],
    labels: list[str],
):
    fig, axes = plt.subplots(
        2, len(images), figsize=(18, 7), gridspec_kw={"height_ratios": [0.5, 1]}
    )
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    for i, (ax_top, ax_bottom) in enumerate(axes.T):
        ax_bottom.imshow(images[i])
        plot_probabilities(probs[i], targets[i], labels, ax=ax_top)
    plt.tight_layout()
    return fig


def plot_images_probabilities_plotly(
    images: TensorType["batch", "height", "width", "channels"],
    targets: TensorType["batch"],
    probs: TensorType["batch", "n_classes"],
    labels: list[str],
    n_best: int = 5,
):
    if images.shape[-1] == 1:  # GREYSCALE
        images = images.repeat(1, 1, 1, 3)
    palette = px.colors.qualitative.Plotly
    n_examples = len(images)
    fig = make_subplots(
        rows=2,
        cols=n_examples,
        vertical_spacing=0.05,
        horizontal_spacing=0.05,
    )
    for col, (img, target, prob) in enumerate(zip(images, targets, probs)):
        pred = int(prob.argmax().item())
        colors = [palette[0]] * len(labels)
        if pred == target:
            colors[pred] = palette[2]
        else:
            colors[pred] = palette[1]
            colors[target] = palette[9]
        sorted_probs = sorted(prob, reverse=False)[-n_best:]
        sorted_labels = [label for _, label in sorted(zip(prob, labels), reverse=False)][-n_best:]
        sorted_colors = [color for _, color in sorted(zip(prob, colors), reverse=False)][-n_best:]

        fig.add_bar(
            x=sorted_probs,
            y=sorted_labels,
            orientation="h",
            marker_color=sorted_colors,
            row=1,
            col=col + 1,
            text=sorted_labels,
            textposition="inside",
            insidetextanchor="start",
            insidetextfont=dict(family="Arial", size=14, color="black"),
            outsidetextfont=dict(family="Arial", size=14, color="black"),
        )
        fig.add_image(z=img, zmin=[0] * 4, zmax=[1] * 4, row=2, col=col + 1, name=labels[target])
        fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
    return fig
