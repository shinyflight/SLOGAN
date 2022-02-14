# Stein Latent Optimization for GANs

#### A Tensorflow implementation of "Stein Latent Optimization for GANs (SLOGAN)"

![SLOGAN_model](https://user-images.githubusercontent.com/85322658/120757112-13c79c80-c54b-11eb-9f88-1f20de57158f.png)


### Requirements

    conda env create --file environment.yaml
    conda activate slogan

<br />

## Synthetic dataset

<img src="https://user-images.githubusercontent.com/85322658/120757216-36f24c00-c54b-11eb-8b73-415f88ef5ab3.gif" width="400">


### Model training

SLOGAN can be trained with the synthetic dataset using following commands:

    python slogan_synthetic.py --gpu "GPU_NUMBER"

Generated data are stored in './logs/synthetic'


<br />

## CIFAR-2 dataset

<img src="https://user-images.githubusercontent.com/85322658/120780612-4087ae00-c563-11eb-87ef-642eb8453099.png" width="600">


### Pretrained models

We release pretrained model weights and training log files of CIFAR-2 (7:3) and CIFAR-2 (5:5).

You can download pretrained model files from [This URL](https://drive.google.com/drive/folders/1WAFfHpXM-YdywH2PkZyptv9jetcINsFk?usp=sharing) and put them into './logs/cifar2/3/pretrained/' for CIFAR-2 (7:3), and './logs/cifar2/5/pretrained/' for CIFAR-2 (5:5)

Training logs and generated images of pretrained models can be viewed using the following command:
    
    tensorboard --logdir ./logs/cifar2/"3 or 5"/pretrained

Pretrained model weights can be loaded and used to calculate evaluation metrics using the following command:

    python slogan_cifar2.py --gpu "GPU_NUMBER" --pretrained ./logs/cifar2/"3 or 5"/pretrained/model-100000


### Model training

SLOGAN can be trained with the CIFAR-2 (7:3) dataset using the following command:

    python slogan_cifar2.py --gpu "GPU_NUMBER" --ratio_plane 3
   
Log files are stored in './logs/cifar2/"RATIO_PLANE"', and training logs and generated images can be viewed using Tensorboard.
