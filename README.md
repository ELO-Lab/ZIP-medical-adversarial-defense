# Diffusion-Based Purification for Adversarial Defense in Medical Image Classification

This code is based on [ZIP](https://github.com/sycny/ZIP.git).

## Dataset

Dataset is from Kaggle [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset)

Data format for training and evaluating will be like this:

```
Alzheimer
├─  train
│  ├─ MildDemented
│  ├─ ModerateDemented
│  ├─ NonDemented
│  ├─ VeryMildDemented
├─  val
│  ├─ MildDemented
│  ├─ ModerateDemented
│  ├─ NonDemented
│  ├─ VeryMildDemented
ChestCancer
├─  train
│  ├─ Adenocarcinoma
│  ├─ Large cell carcinoma
│  ├─ Normal
│  ├─ Squamous Cell Carcinoma
├─  val
│  ├─ Adenocarcinoma
│  ├─ Large cell carcinoma
│  ├─ Normal
│  ├─ Squamous Cell Carcinoma
...
```

## Pretrained Models

The diffusion models are trained using [NVlabs/edm](https://github.com/NVlabs/edm). Download [Link](https://drive.google.com/file/d/1psJ0mHau7GPq_tHsH6ZUWVtWmg5dpsCD/view?usp=sharing) for the pretrained models.

This [Link](https://drive.google.com/file/d/1wAG5ubuinY185-UNHtrlBB1tV3sGZ5LI/view?usp=sharing) is for medical classification models.


# Installation

## Requirements

Python 3.12

    conda create <env name> python-3.12
    conda activate <env  name>
    pip install -r requirements.txt

## Purify

    python main.py --dataset <dataset name> --foolbox_attack <attacks from Foolbox> --img_size <img size> --deg Demo --deg_scale 4 --gpulist 0 -pctes -pptes -upctes -upptes --sampling 50 --classes <num classes>

`dataset`: dataset name including Alzheimer, ChestCancer, KidneyCancer.

`foolbox_attack`: adversarical attacks including FGSM, PGD, DeepFool.

`deg_scale`: scale ratio for image size when using pooling transfom.

`pctes`: purify clean dataset.

`pptes`: purify poisoned dataset.

`upctes`: evaluate the purified clean dataset.

`upptes`: evaluate the purified poisoned dataset.

`sampling`: Diffusion sampling steps.

`classes`: number of classes.

## Classification

### Training a dataset

    python classification.py --root /home/chaunm/Projects/ZIP/pur/Mode3/KidneyCancer/Foolbox/Demo/4.0/50/val_copy --train

`root`: dataset directory.

`train`: training mode.

### Evaluate a dataset

    python classification.py --root /home/chaunm/Projects/dataset/medical-scan-classification 

`root`: dataset directory.