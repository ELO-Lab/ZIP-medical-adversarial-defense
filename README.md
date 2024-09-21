# Installation

## Requirements

python 3.12

pip install -r requirements.txt

# run

## Purify

python main.py --dataset ChestCancer --attack_method Foolbox --img_size 64 --deg Demo --at_threshold 50  --deg_scale 4 --gpulist 0 -pctes -pptes -upctes -upptes --sampling 50 --foolbox_attack PGD

## Classification

### Training a dataset

python classification.py --root /home/chaunm/Projects/ZIP/pur/Mode3/KidneyCancer/Foolbox/Demo/4.0/50/val_copy --train

### Evaluate a dataset

python classification.py --root /home/chaunm/Projects/dataset/medical-scan-classification <data path>