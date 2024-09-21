from model import CNN, CNNModel
from data import Medical_Classification
import os
import torch
import datetime
import wandb
import math
import time
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import argparse


seed = 42
torch.manual_seed(seed)

def arg_parse():
    parser = argparse.ArgumentParser(description="Training and Evaluating classification models.")

    parser.add_argument("--train", help="Training mode", action='store_true')
    parser.add_argument("--root", type=str, help="Data root", required=True)
    # parser.add_argument("--data_cls", type=str, help="Data class", required=True)
    parser.add_argument("--batch", type=int, help="Batch size", default=64)
    parser.add_argument("--img_size", type=int, help="Image size", default=64)
    parser.add_argument("--epoch", type=int, help="Training epoch", default=100)
    parser.add_argument("--saved_path", type=str, help="Save path for chekpoint models", default='./saved_models/classification/')
    parser.add_argument("--pretrained", type=str, help="Path for pretrained models", \
                        default='/home/chaunm/Projects/medical-cls/saved_models/classification2/Chest Cancer/model_97.pth')
    
    
    args = parser.parse_args()
    return args

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

def train_one_epoch(training_loader, model, optimizer, scheduler, loss_fn):
    running_loss = 0.
    last_loss = 0.

    for i, data in enumerate(training_loader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.long().cuda()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    scheduler.step()
    return last_loss

def train(args):
    # data_root = '/home/chaunm/Projects/dataset/medical-scan-classification'
    # data_names = ['Chest Cancer', 'Kidney Cancer', 'Monkeypox']
    # data_root = "/home/chaunm/Projects/ZIP/pur/Mode3/KidneyCancer/Foolbox/Demo/4.0/50/val_copy"
    data_root = args.root
    data_names = ['Kidney Cancer']
    IMG_SIZE = args.img_size
    EPOCHS = args.epoch
    BATCH_SIZE = args.batch
    INIT_LR = .003
    # saved_path = './saved_models/classification/'
    saved_path = args.saved_path

    wandb.login()
    for data_name in data_names:
        run = wandb.init(
            project="medical-" + data_name,
            config={
                "img_size": IMG_SIZE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": INIT_LR,
            },
        )

        data = Medical_Classification(data_root, IMG_SIZE)
        num_classes = len(data.get_classes())
        model = CNNModel(num_classes).cuda()
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR, weight_decay=1e-5)

        def schedule(epoch):
            if epoch < 5:
                return ((epoch)+1)*1/5
            if epoch < 15:
                return 1
            else:
                return math.exp(-0.1)
            
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        labels = data.labels

        train_idx, test_idx = train_test_split(
            range(len(labels)), test_size=0.2, stratify=data.labels, random_state=42
        )

        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.25, stratify=[labels[i] for i in train_idx], random_state=42
        )

        # Create the corresponding subsets
        train_set = Subset(data, train_idx)
        val_set = Subset(data, val_idx)
        test_set = Subset(data, test_idx)

        training_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

        best_acc = 0.
        best_epoch = 0

        for epoch in range(EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            model.train()
            avg_loss = train_one_epoch(training_loader, model, optimizer, scheduler, loss_fn)


            running_vloss = 0.0
            correct = 0
            total = 0
            topk = [1]

            model.eval()

            predict_digits_arr = []
            labels_arr = []

            with torch.no_grad():
                for i, data in enumerate(validation_loader):
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.long().cuda()

                    outputs = model(inputs)
                    vloss = loss_fn(outputs, labels)
                    running_vloss += vloss.item()

                    predict_digits_arr.append(outputs)
                    labels_arr.append(labels)

            avg_vloss = running_vloss / len(validation_loader)
            predict_digits_arr = torch.cat(predict_digits_arr, dim=0)
            labels_arr = torch.cat(labels_arr, dim=0)
            total_num = labels_arr.size(0)
            val_accuracy = 0
            precs = accuracy(predict_digits_arr, labels_arr, topk=topk)
            for top, prec in zip(topk, precs):
                top_correct = int(round(prec.item() / 100.0 * total_num))
                val_accuracy = top_correct/total_num
            print('LOSS train {} LOSS valid {}, ACC valid {}'.format(avg_loss, avg_vloss, val_accuracy))
            wandb.log({"train_loss": avg_loss, "valid_loss": avg_vloss, "valid_acc": val_accuracy, "_epoch": epoch})
            
            if val_accuracy >= best_acc:
                best_acc = val_accuracy
                best_epoch = epoch
                saved_data_path = os.path.join(saved_path, data_name)
                if not os.path.exists(saved_data_path):
                    os.makedirs(saved_data_path)
                model_path = os.path.join(saved_data_path, 'model_{}.pth'.format(epoch))
                torch.save(model.state_dict(), model_path)
        
        wandb.finish()
    # ========================================================================================================================================
    model.eval()

    predict_digits_arr = []
    labels_arr = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.long().cuda()
            outputs = model(inputs)
            
            # preds = outputs > 0.5
            predict_digits_arr.append(outputs)
            labels_arr.append(labels)

    predict_digits_arr = torch.cat(predict_digits_arr, dim=0)
    labels_arr = torch.cat(labels_arr, dim=0)
    total_num = labels_arr.size(0)
    precs = accuracy(predict_digits_arr, labels_arr, topk=topk)

    print("==========Test result on benign test dataset(CA)==========")
    for top, prec in zip(topk, precs):
        top_correct = int(round(prec.item() / 100.0 * total_num))
        print(f'Top {top} correct: {top_correct}/{total_num}, Top {top} accuracy: {top_correct/total_num}')


def eval(args):
    # data_root = '/home/chaunm/Projects/dataset/medical-scan-classification'
    data_root = args.root
    data_name = 'Chest Cancer'
    IMG_SIZE = args.img_size
    data = Medical_Classification(os.path.join(data_root, data_name), IMG_SIZE)
    BATCH_SIZE = args.batch
    topk = [1]
    pretrained_path = args.pretrained
    
    test_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    print('Classes:', data.get_classes())
    num_classes = len(data.get_classes())
    model = CNNModel(num_classes).cuda()
    model.load_state_dict(torch.load(pretrained_path))

    model.eval()

    predict_digits_arr = []
    labels_arr = []
    last_time = time.time()

        
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.long().cuda()

            outputs = model(inputs)
            
            # preds = outputs > 0.5
            predict_digits_arr.append(outputs)
            labels_arr.append(labels)

    predict_digits_arr = torch.cat(predict_digits_arr, dim=0)
    labels_arr = torch.cat(labels_arr, dim=0)
    # labels_arr = labels_arr.argmax(dim=1)
    total_num = labels_arr.size(0)
    # print(predict_digits_arr.shape, labels_arr.shape)
    print(predict_digits_arr)
    precs = accuracy(predict_digits_arr, labels_arr, topk=topk)

    print("==========Test result on benign test dataset(CA)==========")
    for top, prec in zip(topk, precs):
        top_correct = int(round(prec.item() / 100.0 * total_num))
        print(f'Top {top} correct: {top_correct}/{total_num}, Top {top} accuracy: {top_correct/total_num}')

if __name__ == '__main__':
    args = arg_parse()
    if args.train:
        print('====================Training mode====================')
        train(args)
    else:
        print('====================Evaluating mode====================')
        eval(args)