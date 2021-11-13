# EfficientNetV2-pytorch
Unofficial EfficientNetV2 pytorch implementation repository. This repository is currently under the develop.

- Title: EfficientNetV2: Smaller models and Faster Training

- Author: Minxing Tan
- Publication: ICML, 2021
- Link: [Paper](https://arxiv.org/abs/2104.00298) | [official tensorflow repo](https://github.com/google/automl/tree/master/efficientnetv2) | [other pytorch repo](https://github.com/d-li14/efficientnetv2.pytorch)
- Other references: 
  - [Training ImageNet in 3 hours for USD 25; and CIFAR10 for USD 0.26](https://www.fast.ai/2018/04/30/dawnbench-fastai/)



This repository contains:

1. Pytorch version EfficientNetV2.
2. Pytorch lightning version training code on cifar10/100 using progressive learning with autoaug and mixup. 
   If you are unfamiliar with pytorch-lightning, We recommend you to read [Lightning-in-2-steps](https://pytorch-lightning.readthedocs.io/en/latest/starter/new-project.html).



## Tutorial

1. Clone this repo and install dependency

   ```sh
   git clone https://github.com/hankyul2/EfficientNetV2-pytorch.git
   pip3 install requirements.txt
   ```

2. Train & Test model

   ```sh
   python3 main.py fit --config config/base.yaml --trainer.gpus 0,3,6,7, --data.dataset_name cifar100 --seed_everything 2021
   ```





## Experiment Results

*coming soon*





## Experiment Setup

*ImageNet Setup*

| Setup         | Contents                                                     |
| ------------- | ------------------------------------------------------------ |
| Data          | ImageNet(ImgeSize=128, RandAugmentation=5, Mixup=0)          |
| Model         | EfficientNetV2(Dropout=0.1, Stochastic_depth=0.2)            |
| Optimizer     | RMSProp(decay=0.9, batch_norm_momentum=0.99, weight_decay=1e-5, momentum=0.9) |
| Learning rate | (epoch=350, batch_size=4096, lr=0.256, warmup=?) learning rate decay by 0.97 every 2.4 epochs |
| EMA           | decay_rate=0.9999                                            |

*Cifar Setup*

| Setup         | Contents                                                     |
| ------------- | ------------------------------------------------------------ |
| Data          | Cifar(ImgeSize=224, Cutout)                                  |
| Model         | EfficientNetV2(Dropout=0.1, Stochastic_depth=0.2)            |
| Optimizer     | SGD(weight_decay=0, momentum=True)                           |
| Learning rate | CosineLearningRate(epoch=100, batch_size=512, lr=0.001, warmup=0) |

*Note*

1. For progressive learning, `ImageSize`, `RandAugmentation`, `Mixup`, `Dropout` are going to be changed along with epoch.
2. Evaluation Size is different for each model
3. `epoch=100` in *Cifar Stepup* is calculated from paper like this: `10,000 step * 512 batch size / 50,000 images = 102.4`
4. To see more model specific details, check [efficientnet_v2_config.py](src/efficientnet_v2_config.py)
5. To see more train hyperparameter, check [cifar.yaml](config/cifar.yaml)

