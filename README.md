# EfficientNetV2-pytorch
Unofficial EfficientNetV2 pytorch implementation repository.

It contains:

- Simple Implementation of model ([here](efficientnetv2/efficientnet_v2.py))
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

2. Train & Test model (see more examples in [tmuxp/cifar.yaml](tmuxp/cifar.yaml))

   ```sh
   python3 main.py fit --config config/efficientnetv2_s/cifar10.yaml --trainer.gpus 2,3,
   ```





## Experiment Results

| Model Name              | Pretrained Dataset | Cifar10                                                      | Cifar100                                                     |
| ----------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| EfficientNetV2-S        | ImageNet           | 98.46 ([tf.dev](https://tensorboard.dev/experiment/HQqb9kYXQ1yLCdfLGQT7yQ/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_s_cifar10.pth)) | 90.05 ([tf.dev](https://tensorboard.dev/experiment/euwy6Rv6RR2RUlLw6Dqi2g/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_s_cifar100.pth)) |
| EfficientNetV2-M        | ImageNet           | 98.89 ([tf.dev](https://tensorboard.dev/experiment/GyJwToamQ5q5nHZARL5n2Q/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_m_cifar10.pth)) | 91.54 ([tf.dev](https://tensorboard.dev/experiment/mVj4XfD4QwyGdGv5EV3H0A/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_m_cifar100.pth)) |
| EfficientNetV2-L        | ImageNet           | 98.80 ([tf.dev](https://tensorboard.dev/experiment/BGRZvE0OS6WU3CqybE25vg/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_l_cifar10.pth)) | 91.88 ([tf.dev](https://tensorboard.dev/experiment/QYjNoNKyTwmHBvBeL5NRqQ/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_l_cifar100.pth)) |
| EfficientNetV2-S-in21k  | ImageNet21k        | 98.50 ([tf.dev](https://tensorboard.dev/experiment/f44EqAzLR2S2831tqfrZEw/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_s_in21k_cifar10.pth)) | 90.96 ([tf.dev](https://tensorboard.dev/experiment/PnByKdA4RKeiaJ8YH2nr5Q/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_s_in21k_cifar100.pth)) |
| EfficientNetV2-M-in21k  | ImageNet21k        | 98.70 ([tf.dev](https://tensorboard.dev/experiment/b0pd5LxeRTOmXMOibaFz7Q/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_m_21k_cifar100.pth)) | 92.06 ([tf.dev](https://tensorboard.dev/experiment/NZhXuDFmRH6k9as5D7foBg/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_m_in21k_cifar100.pth)) |
| EfficientNetV2-L-in21k  | ImageNet21k        | 98.78 ([tf.dev](https://tensorboard.dev/experiment/GngI0UD5QbanKHKnLdVCWA/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_l_in21k_cifar10.pth)) | 92.08 ([tf.dev](https://tensorboard.dev/experiment/99VVMfMORYC3UmOePzRakg/), [weight](https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch-cifar/efficientnet_v2_l_in21k_cifar100.pth)) |
| EfficientNetV2-XL-in21k | ImageNet21k        | -                                                            | -                                                            |

*Note*

1. The results are combination of
   - `Half precision` 
   - `Super Convergence(epoch=20)` 
   - `AdamW(weight_decay=0.005)`
   - `EMA(decay=0.999)` 
   - `cutmix(prob=1.0)`
2. Changes from original paper (CIFAR)
   1. We just run 20 epochs to got above results. If you run more epochs, you can get more higher accuracy.
   2. What we changed from original setup are: optimizer(`SGD` to `AdamW`), LR scheduler(`cosinelr` to `onecylelr`), augmentation(`cutout` to `cutmix`), image size (384 to 224), epoch (105 to 20).
   3. Important hyper-parameter(most important to least important): LR->weigth_decay->ema-decay->cutmix_prob->epoch.
3. you can get same results by running `tmuxp/cifar.yaml`





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






## References

EfficientNetV2

- Title: EfficientNetV2: Smaller models and Faster Training

- Author: Minxing Tan
- Publication: ICML, 2021
- Link: [Paper](https://arxiv.org/abs/2104.00298) | [official tensorflow repo](https://github.com/google/automl/tree/master/efficientnetv2) | [other pytorch repo](https://github.com/d-li14/efficientnetv2.pytorch)
- Other references: 
  - [Training ImageNet in 3 hours for USD 25; and CIFAR10 for USD 0.26](https://www.fast.ai/2018/04/30/dawnbench-fastai/)
  - [AdamW and Super-convergence is now the fastest way to train neural nets](https://www.fast.ai/2018/07/02/adam-weight-decay/)

