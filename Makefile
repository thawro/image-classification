mlp_experiments:
	python src/bin/train.py \
		experiment=fashion_mnist,cifar100,svhn \
		feature_extractor=mlp \
		feature_extractor.hidden_dims='[64], [128], [512, 256, 128]' \
		feature_extractor.activation=ReLU,LeakyReLU,Tanh \
		feature_extractor.use_batch_norm=True,False \
		feature_extractor.dropout=0,0.2,0.4 \
		--multirun

deep_cnn_experiments:
	python src/bin/train.py \
		experiment=fashion_mnist,cifar100,svhn \
		feature_extractor=deep_cnn \
		feature_extractor.out_channels='[32], [16, 32, 64]' \
		feature_extractor.kernels=3,5,7  \
		feature_extractor.dropout=0,0.2,0.4 \
		feature_extractor.activation=ReLU,LeakyReLU \
		--multirun

resnet_experiments:
	python src/bin/train.py \
		experiment=fashion_mnist,cifar100,svhn \
		feature_extractor=resnet \
		feature_extractor.stages_n_blocks='[2], [2, 2, 2], [3, 3, 3]' \
		feature_extractor.stem_kernel_size=3,5,7 \
		feature_extractor.stem_channels=16,32,64 \
		feature_extractor.block_type=bottleneck,basic \
		--multirun
