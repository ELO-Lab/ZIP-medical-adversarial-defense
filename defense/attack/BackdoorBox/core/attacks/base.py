import os
import os.path as osp
import time
from copy import deepcopy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10, CelebA
from attack.originalimagenet import Origdataset
from preprocess.guided_diffusion.purify import  Purify, SplitDataset, nonSplitDataset, SplitCLeanDataset
import torchvision.utils as tvu

from ..utils import Log
from settings import base_args, base_config
args, config = base_args, base_config


support_list = (
    DatasetFolder,
    CIFAR10,
    Origdataset,
    SplitDataset,
    SplitCLeanDataset,
    nonSplitDataset
)


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def get_class_accuracy(output, label):
    """
    Calculates the accuracy for each class.

    Args:
        output (list): List of model outputs.
        label (list): List of true labels.

    Returns:
        class_accuracy (dict): Dictionary of accuracies for each class.
    """
    num_classes = len(torch.unique(label))
    if num_classes==1:
        class_accuracy = {}
        true_positives = 0
        total = 0
        for i in range(len(label)):
            if label[i] == label[0]:
                total += 1
                if torch.argmax(output[i]) == label[i]:
                    true_positives += 1
        class_accuracy[label[0].item()] = true_positives / total if total > 0 else 0
    else:
        class_accuracy = {}
        for class_label in range(num_classes):
            true_positives = 0
            total = 0
            for i in range(len(label)):
                if label[i] == class_label:
                    total += 1
                    if torch.argmax(output[i]) == label[i]:
                        true_positives += 1
            class_accuracy[class_label] = true_positives / total if total > 0 else 0
    return class_accuracy




class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(self, train_dataset, test_dataset, model, loss, schedule=None, seed=0, deterministic=False):
        assert isinstance(train_dataset, support_list), 'train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list.'
        self.train_dataset = train_dataset

        assert isinstance(test_dataset, support_list), 'test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list.'
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule['schedule']:
            self.current_schedule['lr'] *= self.current_schedule['gamma']
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.current_schedule['lr']

    def train(self, schedule=None):
        
        if schedule is None and self.global_schedule is None:
            raise AttributeError("Training schedule is None, please check your schedule setting.")
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)
            
        work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'],'Res_' + str(self.current_schedule['layers'])+ '_' +str(self.current_schedule['img_size']))
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, 'log.txt'))

        if self.current_schedule['pretrain'] is True:
            
            last_time = time.time()
            
            print("Using Pre-trained model:", work_dir)
            device = args.gpu
                
            
            print(f"=================================================Loading Model ==================================================")
            # ckpt_model_filename = "ckpt_epoch_250" + ".pth"
            # ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
            
            if args.dataset == "Alzheimer":
                ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification2/Alzheimer/model_76.pth'
            elif args.dataset == "ChestCancer":
                ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification2/Chest Cancer/model_97.pth'
            elif args.dataset == "KidneyCancer":
                ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification2/Kidney Cancer/model_99.pth'
            elif args.dataset == "Monkeypox":
                ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Monkeypox/model_84.pth'
            
            # if args.dataset == "Alzheimer":
            #     ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Alzheimer/model_42.pth'
            # elif args.dataset == "ChestCancer":
            #     ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Chest Cancer/model_97.pth'
            # elif args.dataset == "KidneyCancer":
            #     ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Kidney Cancer/model_85.pth'
            # elif args.dataset == "Monkeypox":
            #     ckpt_model_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Monkeypox/model_84.pth'
            

            self.model.load_state_dict(torch.load(ckpt_model_path))
            # self.model.load_state_dict(torch.load('/home/chaunm/Projects/medical-cls/saved_models/classification/Alzheimer/model_42.pth'))
            self.model = self.model.to(device)
            self.model = nn.DataParallel(self.model.cuda(), device_ids=args.gpulist)
            self.model.eval()
            print(f"=================================================    Loaded    ==================================================")

            predict_digits, labels = self._cleantest(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
            total_num = labels.size(0)
            print(predict_digits)
            print(labels)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            class_accuracy  = get_class_accuracy(predict_digits, labels)
            msg = "==========Test result on benign test dataset(CA)==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
            print(msg)

            # test result on poisoned test dataset
            # if self.current_schedule['benign_training'] is False:
            predict_digits, labels = self._asrtest(self.poisoned_test_dataset, device, 1, self.current_schedule['num_workers'], label_type='label_pois') 
            #batch_size must set to 1, we do not includ images with original labels that are the same with  the posioned label, if this step is too slow, you can adjust in the self._asrtest
            total_num = labels.size(0)
            print(predict_digits)
            print(labels)
            print('ASR_lables', labels)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            class_accuracy  = get_class_accuracy(predict_digits, labels)
            msg = "==========Test result on poisoned test dataset(ASR)(Not including images with original label same as attack label)==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
            print(msg)
            
            '''
            predict_digits, labels = self._asrtest2(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], label_type='label_pois') 
            # we  includ images with original labels that are the same with the posioned label.
            total_num = labels.size(0)
            print('ASR_lables', labels)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            class_accuracy  = get_class_accuracy(predict_digits, labels)
            msg = "==========Test result on poisoned test dataset(ASR)(Including images with original label same as attack label)==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
            print(msg)
            '''
            
            predict_digits, labels = self._patest(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], label_type='label_orig')
            total_num = labels.size(0)
            print(predict_digits)
            print(labels)
            print('PA_lables', labels)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            class_accuracy  = get_class_accuracy(predict_digits, labels)
            msg = "==========Test result on poisoned test dataset(PA)==========\n" + \
                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
            print(msg)    
            print(f"=================================================End==================================================")
        else:
             # Use GPU
            print(self.current_schedule)
            if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
                if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
                    os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

                assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
                assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
                print(f"the gpu number is {torch.cuda.device_count()}")
                print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

                if self.current_schedule['GPU_num'] == 1:
                    device = torch.device("cuda:0")
                else:
                    gpus = list(range(self.current_schedule['GPU_num']))
                    device = torch.device("cuda:0")
                    self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])

            # Use CPU
            else:
                device = torch.device("cpu")

            if self.current_schedule['benign_training'] is True:
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.current_schedule['batch_size'],
                    shuffle=True,
                    num_workers=self.current_schedule['num_workers'],
                    drop_last=False,
                    pin_memory=True,
                    worker_init_fn=self._seed_worker
                )
            elif self.current_schedule['benign_training'] is False:
                print(f"The model is current using {self.poisoned_train_dataset} for training")
                train_loader = DataLoader(
                    self.poisoned_train_dataset,
                    batch_size=self.current_schedule['batch_size'],
                    shuffle=True,
                    num_workers=self.current_schedule['num_workers'],
                    drop_last=False,
                    pin_memory=True,
                    worker_init_fn=self._seed_worker
                )
            else:
                raise AttributeError("self.current_schedule['benign_training'] should be True or False.")

            self.model = self.model.to(device)
            self.model.train()

            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.current_schedule['lr'], momentum=self.current_schedule['momentum'], weight_decay=self.current_schedule['weight_decay'])

            work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'],'Res_' + str(self.current_schedule['layers'])+ '_' +str(self.current_schedule['img_size']))
            os.makedirs(work_dir, exist_ok=True)
            log = Log(osp.join(work_dir, 'log.txt'))

            iteration = 0
            last_time = time.time()

            if self.current_schedule['benign_training'] is True:
                msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
            else:
                msg = f"Total train samples: {len(self.poisoned_train_dataset)}\nTotal test samples: {len(self.poisoned_test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.poisoned_train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
            log(msg)

            for i in range(self.current_schedule['epochs']):
                self.adjust_learning_rate(optimizer, i)
                for batch_id, batch in enumerate(train_loader):
                    batch_img = batch[0]
                    batch_label = batch[1]['label_pois']
                    batch_img = batch_img.to(device)
                    batch_label = batch_label.to(device)
                    optimizer.zero_grad()
                    predict_digits = self.model(batch_img)
                    loss = self.loss(predict_digits, batch_label)
                    loss.backward()
                    optimizer.step()

                    iteration += 1

                    if iteration % self.current_schedule['log_iteration_interval'] == 0:
                        msg = time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                        last_time = time.time()
                        log(msg)

                if (i + 1) % self.current_schedule['test_epoch_interval'] == 0:
                    # test result on benign test dataset
                    predict_digits, labels = self._cleantest(self.test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'])
                    total_num = labels.size(0)
                    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
                    top1_correct = int(round(prec1.item() / 100.0 * total_num))
                    top5_correct = int(round(prec5.item() / 100.0 * total_num))
                    class_accuracy  = get_class_accuracy(predict_digits, labels)
                    msg = "==========Test result on benign test dataset(CA)==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
                    log(msg)

                    predict_digits, labels = self._asrtest(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], label_type='label_pois')
                    total_num = labels.size(0)
                    print('ASR_lables', labels)
                    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
                    top1_correct = int(round(prec1.item() / 100.0 * total_num))
                    top5_correct = int(round(prec5.item() / 100.0 * total_num))
                    class_accuracy  = get_class_accuracy(predict_digits, labels)
                    msg = "==========Test result on poisoned test dataset(ASR)==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
                    log(msg)
                    
                    predict_digits, labels = self._patest(self.poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], label_type='label_orig')
                    total_num = labels.size(0)
                    print('PA_lables', labels)
                    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
                    top1_correct = int(round(prec1.item() / 100.0 * total_num))
                    top5_correct = int(round(prec5.item() / 100.0 * total_num))
                    class_accuracy  = get_class_accuracy(predict_digits, labels)
                    msg = "==========Test result on poisoned test dataset(PA)==========\n" + \
                        time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
                        f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
                    log(msg)

                    self.model = self.model.to(device)
                    self.model.train()

                if (i + 1) % self.current_schedule['save_epoch_interval'] == 0:
                    self.model.eval()
                    self.model = self.model.cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(i+1) + ".pth"
                    ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                    torch.save(self.model.state_dict(), ckpt_model_path)
                    self.model = self.model.to(device)
                    self.model.train()

    def _cleantest(self, dataset, device, batch_size=16, num_workers=8, model=None):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
        
    def _asrtest(self, dataset, device, batch_size=1, num_workers=8, model=None, label_type='label_pois'):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                #print(batch_label)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                # if the speed is too slow, you can choose cite the following five lines, and uncite the next 2 lines
                # if batch_label['label_orig'] == batch_label['label_pois']: ##delete those instances with labels that is the same to the poisoned label
                
                # if torch.equal(batch_label['label_orig'], batch_label['label_pois']):
                #     pass
                # else:
                predict_digits.append(batch_img)
                labels.append(batch_label[label_type])
                
                #predict_digits.append(batch_img)
                #labels.append(batch_label[label_type])

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
        
        
    def _asrtest2(self, dataset, device, batch_size=1, num_workers=8, model=None, label_type='label_pois'):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                #print(batch_label)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()

                predict_digits.append(batch_img)
                labels.append(batch_label[label_type])

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels
        
    def _patest(self, dataset, device, batch_size=16, num_workers=8, model=None, label_type='label_orig'):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                #print(batch_label)
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label[label_type])

            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    # def test(self, schedule=None, model=None, test_dataset=None, poisoned_test_dataset=None):
    #     if schedule is None and self.global_schedule is None:
    #         raise AttributeError("Test schedule is None, please check your schedule setting.")
    #     elif schedule is not None and self.global_schedule is None:
    #         self.current_schedule = deepcopy(schedule)
    #     elif schedule is None and self.global_schedule is not None:
    #         self.current_schedule = deepcopy(self.global_schedule)
    #     elif schedule is not None and self.global_schedule is not None:
    #         self.current_schedule = deepcopy(schedule)

    #     if model is None:
    #         model = self.model

    #     if 'test_model' in self.current_schedule:
    #         model.load_state_dict(torch.load(self.current_schedule['test_model']), strict=False)

    #     if test_dataset is None and poisoned_test_dataset is None:
    #         test_dataset = self.test_dataset
    #         poisoned_test_dataset = self.poisoned_test_dataset

    #     # Use GPU
    #     if 'device' in self.current_schedule and self.current_schedule['device'] == 'GPU':
    #         if 'CUDA_VISIBLE_DEVICES' in self.current_schedule:
    #             os.environ['CUDA_VISIBLE_DEVICES'] = self.current_schedule['CUDA_VISIBLE_DEVICES']

    #         assert torch.cuda.device_count() > 0, 'This machine has no cuda devices!'
    #         assert self.current_schedule['GPU_num'] >0, 'GPU_num should be a positive integer'
            
    #         print(f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train.")

    #         if self.current_schedule['GPU_num'] == 1:
    #             device = torch.device("cuda:0")
    #         else:
    #             gpus = list(range(self.current_schedule['GPU_num']))
    #             model = nn.DataParallel(model.cuda(), device_ids=gpus, output_device=gpus[0])
    #             # TODO: DDP training
    #             pass
    #     # Use CPU
    #     else:
    #         device = torch.device("cpu")

    #     work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
    #     os.makedirs(work_dir, exist_ok=True)
    #     log = Log(osp.join(work_dir, 'log.txt'))

    #     if test_dataset is not None:
    #         last_time = time.time()
    #         # test result on benign test dataset
    #         predict_digits, labels = self._test(test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
    #         total_num = labels.size(0)
    #         prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
    #         top1_correct = int(round(prec1.item() / 100.0 * total_num))
    #         top5_correct = int(round(prec5.item() / 100.0 * total_num))
    #         class_accuracy  = get_class_accuracy(predict_digits, labels)
    #         msg = "==========Test result on benign test dataset==========\n" + \
    #               time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
    #               f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
    #         log(msg)

    #     if poisoned_test_dataset is not None:
    #         last_time = time.time()
    #         # test result on poisoned test dataset
    #         predict_digits, labels = self._test(poisoned_test_dataset, device, self.current_schedule['batch_size'], self.current_schedule['num_workers'], model)
    #         total_num = labels.size(0)
    #         prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 2))
    #         top1_correct = int(round(prec1.item() / 100.0 * total_num))
    #         top5_correct = int(round(prec5.item() / 100.0 * total_num))
    #         class_accuracy  = get_class_accuracy(predict_digits, labels)
    #         msg = f"==========Test result on Poisoned test dataset==========\n" + \
    #               time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime()) + \
    #               f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n class_acc:{class_accuracy}"
    #         log(msg)
