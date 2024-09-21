'''
ZIP
This is the implement of BadNets [1].

Reference:
[1] Badnets: Evaluating Backdooring Attacks on Deep Neural Networks. IEEE Access 2019.
'''

import copy
import random
import torch
import numpy as np
import PIL
import os
from PIL import Image
from torchvision.transforms import functional as F
from torchvision.transforms import Compose
# from attack.originalimagenet import Origdataset
from torchvision.datasets import CIFAR10, MNIST, CelebA, DatasetFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip,  Resize, Normalize
from torchvision import transforms
import foolbox
import torchvision
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from .base import *
from settings import base_args, base_config
args, config = base_args, base_config


import os
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

class FoolBoxLib():
    def __init__(self, fmodel, attack_type):
        self.fmodel = fmodel
        self.epsilons = 8/255

        print('Creating', attack_type, 'method.')
        if attack_type == 'FGSM':
            self.attack = foolbox.attacks.LinfFastGradientAttack()
        elif attack_type == 'PGD':
            self.attack = foolbox.attacks.LinfPGD()
        elif attack_type == 'DeepFool':
            self.attack = foolbox.attacks.LinfDeepFoolAttack()
        

    def __call__(self, images, targets):
        # images = images.unsqueeze(0)
        # targets = targets.unsqueeze(0)
        _, advs, is_adv = self.attack(self.fmodel, images, targets, epsilons=self.epsilons)
        return advs
    
class PoisonedDataset(DatasetFolder):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 model,
                 attack_type,
                 batch_size=4):
        super(PoisonedDataset, self).__init__(
            benign_dataset.root,
            benign_dataset.loader,
            benign_dataset.extensions,
            benign_dataset.transform,
            benign_dataset.target_transform,
            None)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # self.data = torch.stack([self.poisoned_transform(img) for img in self.data])
        # self.targets = torch.tensor(self.targets)
        self.model = model
        # Add trigger to targets
        bounds = (0, 1)
        # model = torchvision.models.resnet18(pretrained=True)
        fmodel = foolbox.PyTorchModel(self.model, bounds=bounds, device='cuda')
        self.fb = FoolBoxLib(fmodel, attack_type)
        
        images_lst = []
        labels_lst = []
        for i, (path, target) in enumerate(self.samples):
            img = self.loader(path)
            if i in self.poisoned_set:
                img = self.poisoned_transform(img)
            else:
                if self.transform is not None:
                    img = self.transform(img)
            img = img.cuda()
            images_lst.append(img)
            labels_lst.append(target)
        
        images_lst = torch.stack(images_lst)
        labels_lst = torch.tensor(labels_lst)

        dataset = TensorDataset(torch.tensor(images_lst), torch.tensor(labels_lst))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        advs_images_lst = []
        advs_labels_lst = []
        print('Poisoining images')
        for imgs, targets in tqdm(dataloader):
            imgs = imgs.cuda()
            targets = targets.cuda()
            adv_images = self.fb(imgs, targets)
            advs_images_lst.append(adv_images)
            advs_labels_lst.append(targets)
        self.advs_images_lst = torch.concat(advs_images_lst, dim=0)
        self.advs_labels_lst = torch.concat(advs_labels_lst, dim=0)

    def __getitem__(self, index):
        # img, target = self.data[index], self.targets[index]
        # path, target = self.samples[index]
        # img = self.loader(path)

        img = self.advs_images_lst[index]
        target = self.advs_labels_lst[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        labels = {}
        # img = Image.fromarray(img)
        target = torch.tensor(target).to('cuda')
        labels['label_orig'] = target

        if self.target_transform is not None:
            target = self.target_transform(target)

        # pois_target = self.model(img)
        # labels['label_pois'] = pois_target
        # print(pois_target, target)
        labels['label_pois'] = target

        # if self.transform is not None:
        #     img = self.transform(img)
        return img, labels


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 poisoned_rate,
                 model):

        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=True)
        total_num = len(benign_dataset)
        poisoned_num = int(total_num * poisoned_rate)
        assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
        tmp_list = list(range(total_num))
        random.shuffle(tmp_list)
        self.poisoned_set = frozenset(tmp_list[:poisoned_num])

        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([ToTensor()])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        # self.data = torch.stack([self.poisoned_transform(img) for img in self.data])
        # self.targets = torch.tensor(self.targets)

        # Add trigger to targets
        bounds = (0, 1)
        # model = torchvision.models.resnet18(pretrained=True)
        fmodel = foolbox.PyTorchModel(model, bounds=bounds, device='cuda')
        self.fb = FoolBoxLib(fmodel, str(type(benign_dataset)))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        labels = {}
        img = Image.fromarray(img)
        target = torch.tensor(target).to('cuda')
        labels['label_orig'] = target

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            img = img.to('cuda')
            # img = self.fb(img, target)
            img = img.squeeze(0)
            # target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        labels['label_pois'] = target
        # if self.transform is not None:
        #     img = self.transform(img)

        return img, labels


def CreatePoisonedDataset(benign_dataset, poisoned_rate, model, attack_type):
    class_name = type(benign_dataset)
    if class_name == Origdataset:
        return PoisonedDataset(benign_dataset, poisoned_rate, model, attack_type)
    elif class_name == CIFAR10:
        return PoisonedCIFAR10(benign_dataset, poisoned_rate, model, attack_type)

    else:
        raise NotImplementedError


class FoolBox(Base):
    """Construct poisoned datasets with BadNets method.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        y_target (int): N-to-1 attack target label.
        poisoned_rate (float): Ratio of poisoned samples.
        pattern (None | torch.Tensor): Trigger pattern, shape (C, H, W) or (H, W).
        weight (None | torch.Tensor): Trigger pattern weight, shape (C, H, W) or (H, W).
        poisoned_transform_train_index (int): The position index that poisoned transform will be inserted in train dataset. Default: 0.
        poisoned_transform_test_index (int): The position index that poisoned transform will be inserted in test dataset. Default: 0.
        poisoned_target_transform_index (int): The position that poisoned target transform will be inserted. Default: 0.
        schedule (dict): Training or testing schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self,
                 train_dataset,
                 test_dataset,
                 attack_type,
                 model,
                 loss,
                 y_target,
                 poisoned_rate,
                 pattern=None,
                 weight=None,
                 poisoned_transform_train_index=0,
                 poisoned_transform_test_index=0,
                 poisoned_target_transform_index=0,
                 schedule=None,
                 seed=0,
                 deterministic=False):
        # assert pattern is None or (isinstance(pattern, torch.Tensor) and ((0 < pattern) & (pattern < 1)).sum() == 0), 'pattern should be None or 0-1 torch.Tensor.'

        self.model = model
        self.attack_label = y_target

        super(FoolBox, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.poisoned_train_dataset = CreatePoisonedDataset(train_dataset, poisoned_rate, model, attack_type)
        self.poisoned_test_dataset = CreatePoisonedDataset(test_dataset, 1.0, model, attack_type)
