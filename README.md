# EfficientNetV2-pytorch
Unofficial EfficientNetV2 pytorch implementation repository.

It contains:

- Simple Implementation of model ([here](efficientnetv2/efficientnetv2.py))
- Pretrained Model ([numpy weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases), we upload numpy files converted from official tensorflow checkout point)
- Training code ([here](main.py))
- Tutorial ([Colab EfficientNetV2-predict tutorial](https://colab.research.google.com/drive/1BYUeRVsVmBC4AuMyW-gkDboUVDX_jFrI?usp=sharing), [Colab EfficientNetV2-finetuning tutorial](https://colab.research.google.com/drive/1khaZWJDQJToR5GPNBJ01V6TXh8DXbKC_?usp=sharing))
- Experiment results



#### Index

1. Tutorial
2. Experiment results
3. Experiment Setup
4. References





## Tutorial

Colab Tutorial

- How to use model on colab? please check [Colab EfficientNetV2-predict tutorial](https://colab.research.google.com/drive/1BYUeRVsVmBC4AuMyW-gkDboUVDX_jFrI?usp=sharing)

- How to train model on colab? please check [Colab EfficientNetV2-finetuning tutorial](https://colab.research.google.com/drive/1khaZWJDQJToR5GPNBJ01V6TXh8DXbKC_?usp=sharing)

- See how cutmix, cutout, mixup works in [Colab Data augmentation tutorial](https://colab.research.google.com/drive/1L-vSgoPEuzdyD4W6hd5ChrgO9z4G1oue?usp=sharing)



#### How to load pretrained model?

If you just want to use pretrained model, load model by `torch.hub.load`

```python
import torch

model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, nclass=1000)
print(model)
```

*Available Model Names: `efficientnet_v2_{s|m|l}`(ImageNet), `efficientnet_v2_{s|m|l}_in21k`(ImageNet21k)*



#### How to fine-tuning model?

If you want to finetuning on cifar, use this repository.

1. Clone this repo and install dependency

   ```sh
   git clone https://github.com/hankyul2/EfficientNetV2-pytorch.git
   pip3 install requirements.txt
   ```

2. Train & Test model

   ```sh
   python3 main.py fit --config config/base.yaml --trainer.gpus 2, --data.dataset_name cifar100 --model.model_name efficientnet_v2_s  --seed_everything 2021
   ```





## Experiment Results

| Model Name              | Pretrained Dataset | Cifar10                                                      | Cifar100                                                     |
| ----------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| EfficientNetV2-S        | ImageNet           | 98.46 ([tf.dev](https://tensorboard.dev/experiment/HQqb9kYXQ1yLCdfLGQT7yQ/),) | 90.05 ([tf.dev](https://tensorboard.dev/experiment/euwy6Rv6RR2RUlLw6Dqi2g/),) |
| EfficientNetV2-M        | ImageNet           | 98.89 ([tf.dev](https://tensorboard.dev/experiment/GyJwToamQ5q5nHZARL5n2Q/),) | 91.54 ([tf.dev](https://tensorboard.dev/experiment/mVj4XfD4QwyGdGv5EV3H0A/),) |
| EfficientNetV2-L        | ImageNet           | 98.80 ([tf.dev](https://tensorboard.dev/experiment/BGRZvE0OS6WU3CqybE25vg/),) | 91.88 ([tf.dev](https://tensorboard.dev/experiment/QYjNoNKyTwmHBvBeL5NRqQ/),) |
| EfficientNetV2-S-in21k  | ImageNet21k        | 98.50 ([tf.dev](https://tensorboard.dev/experiment/f44EqAzLR2S2831tqfrZEw/),) | 90.96 ([tf.dev](https://tensorboard.dev/experiment/PnByKdA4RKeiaJ8YH2nr5Q/),) |
| EfficientNetV2-M-in21k  | ImageNet21k        | 98.70 ([tf.dev](https://tensorboard.dev/experiment/b0pd5LxeRTOmXMOibaFz7Q/),) | 92.06 ([tf.dev](https://tensorboard.dev/experiment/NZhXuDFmRH6k9as5D7foBg/),) |
| EfficientNetV2-L-in21k  | ImageNet21k        | 98.78 ([tf.dev](https://tensorboard.dev/experiment/GngI0UD5QbanKHKnLdVCWA/),) | 92.08 ([tf.dev](https://tensorboard.dev/experiment/99VVMfMORYC3UmOePzRakg/),) |
| EfficientNetV2-XL-in21k | ImageNet21k        | -                                                            | -                                                            |

*Note*

1. Training Results are not good enough to match with paper results
2. All model weights and code will be updated soon! (winter vacation begin!!)





## Experiment Setup

1. *Cifar setup*

   | Category           | Contents                                                     |
   | ------------------ | ------------------------------------------------------------ |
   | Dataset            | CIFAR10 \| CIFAR100                                          |
   | Batch_size per gpu | (s, m, l) = (256, 128, 64)                                   |
   | Train Augmentation | image_size = 224, horizontal flip, random_crop (pad=4), CutMix(prob=1.0) |
   | Test Augmentation  | image_size = 224, center_crop                                |
   | Model              | EfficientNetV2 s \| m \| l (pretrained on in1k or in21k)     |
   | Regularization     | Dropout=0.0, Stochastic_path=0.2, BatchNorm                  |
   | Optimizer          | AdamW(weight_decay=0.005)                                    |
   | Criterion          | Label Smoothing (CrossEntropyLoss)                           |
   | LR Scheduler       | LR: (s, m, l) = (0.001, 0.0005, 0.0003), LR scheduler: OneCycle Learning Rate(epoch=20) |
   | GPUs & ETC         | 16 precision<br />EMA(decay=0.999, 0.9993, 0.9995)<br />S - 2 * 3090 (batch size 512)<br />M - 2 * 3090 (batch size 256)<br />L - 2 * 3090 (batch size 128) |

   *Notes*

   1. LR, EMA decay, rand_augmentation are affected by batch_size and epoch. So if you change batch size, you also change mentioned parameters. 





## References

EfficientNetV2

- Title: EfficientNetV2: Smaller models and Faster Training

- Author: Minxing Tan
- Publication: ICML, 2021
- Link: [Paper](https://arxiv.org/abs/2104.00298) | [official tensorflow repo](https://github.com/google/automl/tree/master/efficientnetv2) | [other pytorch repo](https://github.com/d-li14/efficientnetv2.pytorch)
- Other references: 
  - [Training ImageNet in 3 hours for USD 25; and CIFAR10 for USD 0.26](https://www.fast.ai/2018/04/30/dawnbench-fastai/)

