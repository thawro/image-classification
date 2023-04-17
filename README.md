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
