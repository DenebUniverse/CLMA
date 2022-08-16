# Complementary Label Model Attack(CLMA)

## Preface

Since our reverse complementary label model (RCLM) is trained on **PyTorch**,  the parameters of RCLM are saved as .pt file, so our attack experiment is divided into two parts. 

- Fist step: With the **PyTorch** environment, save the gradient of internal feature as weights of the CLMA section.
- Second step: With the **TensorFlow** environment, load the weights of CLMA, combine with the weights of FIA, execute the subsequent attack progress.

## Requirements

##### TensorFlow Environment

- Python 3.8.0
- Keras 2.4.3
- Tensorflow 2.5.0
- Numpy 1.21.2
- Pillow 8.4.0
- Scipy 1.2.1
- Torch 1.9.0

##### PyTorch Environment

- Python 3.7.0
- Torch 1.10.2
- Torchvision 0.11.3
- Numpy 1.21.2
- Pillow 6.2.1

## Experiments

#### Introduction

- `attack.py` : the implementation for different attacks.

- `verify.py` : the code for evaluating generated adversarial examples on different models.

  You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim,  https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./FIA/models_tf`.

#### Example Usage

##### Generate and save CLMA weights:

- CLMA

```
python CLMA.py
```

Some parameters need to be adjusted inside CLMA.py, as explained in the annotation.

##### Generate adversarial examples:

- FIA

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv4/conv4_3/Relu --ens 30 --probb 0.7 --output_dir ./adv/FIA/ --beta 0
```

- PIM

```
python attack.py --model_name vgg_16 --attack_method PIM --amplification_factor 10 --gamma 1 --Pkern_size 3 --output_dir ./adv/PIM/
```

- FIA+CLMA

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv4/conv4_3/Relu --ens 30 --probb 0.7 --output_dir ./adv/FIA/ --beta 0.05
```

In the TensorFlow environment, our approach CLMA can be combined with FIA only by changing the beta (>0).

##### Evaluate the attack success rate

```
python verify.py --ori_path ./dataset/images/ --adv_path ./adv/FIA/ --output_file ./log.csv
```

