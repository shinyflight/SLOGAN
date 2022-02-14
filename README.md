# Stein Latent Optimization for Generative Adversarial Networks

![SLOGAN_model](https://user-images.githubusercontent.com/25117385/153810087-17833971-a2dd-4a6a-b8a4-d8d28a8f3ed0.PNG)

**Stein Latent Optimization for Generative Adversarial Networks** (ICLR 2022) <br>
Uiwon Hwang, Heeseung Kim, Dahuin Jung, Hyemi Jang, Hyungyu Lee, Sungroh Yoon <br>
Seoul National University

Paper: [https://openreview.net/forum?id=2-mkiUs9Jx7](https://openreview.net/forum?id=2-mkiUs9Jx7)

Abstract: _Generative adversarial networks (GANs) with clustered latent spaces can perform conditional generation in a completely unsupervised manner. In the real world, the salient attributes of unlabeled data can be imbalanced. However, most of existing unsupervised conditional GANs cannot cluster attributes of these data in their latent spaces properly because they assume uniform distributions of the attributes. To address this problem, we theoretically derive Stein latent optimization that provides reparameterizable gradient estimations of the latent distribution parameters assuming a Gaussian mixture prior in a continuous latent space. Structurally, we introduce an encoder network and novel unsupervised conditional contrastive loss to ensure that data generated from a single mixture component represent a single attribute. We confirm that the proposed method, named **Stein Latent Optimization for GANs (SLOGAN)**, successfully learns balanced or imbalanced attributes and achieves state-of-the-art unsupervised conditional generation performance even in the absence of attribute information (e.g., the imbalance ratio). Moreover, we demonstrate that the attributes to be learned can be manipulated using a small amount of probe data._

<!-- This is a Tensorflow implementation of "Stein Latent Optimization for Generative Adversarial Networks" accepted at ICLR 2022 ([paper](https://openreview.net/forum?id=2-mkiUs9Jx7)).
Uiwon Hwang, Heeseung Kim, Dahuin Jung, Hyemi Jang, Hyungyu Lee, Sungroh Yoon. “[Stein Latent Optimization for Generative Adversarial Networks.](https://openreview.net/forum?id=2-mkiUs9Jx7)” International Conference on Learning Representations (ICLR), 2022 -->

## A Tensorflow implementation of SLOGAN

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

### Citation 
    @inproceedings{
    hwang2022stein,
    title={Stein Latent Optimization for Generative Adversarial Networks},
    author={Uiwon Hwang and Heeseung Kim and Dahuin Jung and Hyemi Jang and Hyungyu Lee and Sungroh Yoon},
    booktitle={International Conference on Learning Representations},
    year={2022},
    url={https://openreview.net/forum?id=2-mkiUs9Jx7}
    }
