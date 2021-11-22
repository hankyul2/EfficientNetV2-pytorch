# EfficientNetV2-pytorch
Unofficial EfficientNetV2 pytorch implementation repository.

It contains:

- Simple Implementation of model ([here](src/efficientnetv2.py))
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

| Model Name              | Pretrained Dataset | Cifar10 | Cifar100  |
| ----------------------- | ------------------ | ------- | --------- |
| EfficientNetV2-S        | ImageNet           | 97.98   | 88.53     |
| EfficientNetV2-M        | ImageNet           | 98.38   | 85.81 (🤔) |
| EfficientNetV2-L        | ImageNet           | 98.4    | -         |
| EfficientNetV2-S-in21k  | ImageNet21k        | 98.1    | 89.2      |
| EfficientNetV2-M-in21k  | ImageNet21k        | 98.2    | 89.5      |
| EfficientNetV2-L-in21k  | ImageNet21k        | 98.2    | 90.1      |
| EfficientNetV2-XL-in21k | ImageNet21k        | -       | -         |

*Note*

1. Training Results are not good enough to match with paper results
2. All models are trained using same setup in experiment setup section (which is adapted from paper)





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
| Data          | Cifar(ImgeSize=224, Cutmix)                                  |
| Model         | EfficientNetV2(Dropout=0.0, Stochastic_depth=0.2)            |
| Optimizer     | SGD(weight_decay=1e-5, momentum=True)                        |
| Learning rate | CosineLearningRate(epoch=100, batch_size=32, lr=0.001, warmup=1) |

*Note*

1. For progressive learning, `ImageSize`, `RandAugmentation`, `Mixup`, `Dropout` are going to be changed along with epoch.
2. Evaluation Size is different for each model
3. `epoch=100` in *Cifar Stepup* is calculated from paper like this: `10,000 step * 512 batch size / 50,000 images = 102.4`
4. To see more model specific details, check [efficientnet_v2_config.py](src/efficientnetv2_config.py)
5. To see more train hyperparameter, check [cifar.yaml](config/base.yaml)





## References

EfficientNetV2

- Title: EfficientNetV2: Smaller models and Faster Training

- Author: Minxing Tan
- Publication: ICML, 2021
- Link: [Paper](https://arxiv.org/abs/2104.00298) | [official tensorflow repo](https://github.com/google/automl/tree/master/efficientnetv2) | [other pytorch repo](https://github.com/d-li14/efficientnetv2.pytorch)
- Other references: 
  - [Training ImageNet in 3 hours for USD 25; and CIFAR10 for USD 0.26](https://www.fast.ai/2018/04/30/dawnbench-fastai/)

