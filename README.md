# Watermark classification dataset

This tutorial shows how to set up an environment to conduct experiments with watermarking classification models.

## How to create a dataset

The subset of Imagenet called [Imagenette](https://github.com/fastai/imagenette) will be used as a base to create a watermarking dataset for the experiments.

First we need to download and extract the images:

> curl -O https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz
>
> tar -xzvf imagenette2.tgz

Then this library [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark) will be used to embed the watermarks into Imagenet images. This library can be installed with the following command (it's also included in the requirements.txt):

> pip install invisible-watermark

After that the new dataset can be created using the following command:

> python create_dataset.py -c configs/main_config.yaml

The dataset generation variables can be changed inside of this config file.

## How to train a model

Config examples can be found in the *configs/train_configs* folder. The training pipeline can be launched using the following command:

> python train.py -c *[path_to_the_config_file]* -t *[path_to_the_results_folder]*