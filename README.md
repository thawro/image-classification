# Basic Image Classification Models Comparison

## Available Datasets:
* MNIST
* EMNIST
* FashionMNIST
* CIFAR10
* CIFAR100
* SVHN
* CelebA

## Available Models:
* MLP
* DeepCNN
* ResNet

# Analysis

## Datasets
* FashionMNIST
* CIFAR100
* SVHN

## Models - grid search over hyperparameters space
* MLP
* DeepCNN
* ResNet

### MLP
* Depth (`hidden_dims`): `[64], [128], [512, 256, 128]`
* Activation function (`activation`): `ReLU, LeakyReLU, Tanh`
* Batch Normalization (`use_batch_norm`): `True, False`
* Dropout (`dropout`): `0, 0.2, 0.4`

### DeepCNN
* Depth (`out_channels`): `[32], [16, 32, 64]`
* Kernels (`kernels`): `3, 5, 7`
* Pooling kernels (`pool_kernels`): `2`
* Batch Normalization (`use_batch_norm`): `True`
* Dropout (`dropout`): `0, 0.2, 0.4`
* Activation function (`activation`): `ReLU, LeakyReLU`

### ResNet
* Depth (`stages_n_blocks`): `[2], [2, 2, 2], [3, 3, 3]`
* Stem kernel (`stem_kernel_size`): `3, 5, 7`
* Stem channels (`stem_channels`): `16, 32, 64`
* Pooling kernel (`pool_kernel_size`): `2`
* Block type (`block_type`): `bottleneck, basic`

# TODO

- [ ] Training and evaluation
    - [ ] Logging
        [x] Metrics
        [x] Visualizations
        [ ] Model checkpoints + onnx
    - [ ] Training resuming
- [x] Classification (multiclass and multilabel)
    - [x] PytorchLightning compatible API (data and models)
    - [x] Visualizations
- [ ] Object detection
- [ ] Segmentation
- [ ] Architectures
    - [x] MLP
    - [x] DeepCNN
    - [x] [ResNet](https://arxiv.org/abs/1512.03385)
    - [ ] [Inception-v4](https://arxiv.org/abs/1602.07261)
    - [ ] [MobileNetV3](https://arxiv.org/abs/1905.02244v5)
    - [ ] [GhostNet](https://arxiv.org/abs/1911.11907)
    - [ ] [ShuffleNet](https://arxiv.org/abs/1707.01083v2)
    - [ ] [CBAM (Convolutional Block Attention Module)](https://arxiv.org/abs/1807.06521)
    - [ ] [ViT (Vision Transformer)](https://arxiv.org/abs/2010.11929v2)
    - [ ] [Swin Transformer](https://arxiv.org/abs/2103.14030)
    - [ ] [ConvNext (A ConvNet for the 2020s)](https://arxiv.org/abs/2201.03545)
    - [ ] [R-CNN](https://arxiv.org/abs/1311.2524v5)
