import foolbox as fb
from model import CNN, CNNModel
from data import Medical_Classification
import torch, os
from tqdm import tqdm
from torchvision.utils import save_image

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


IMG_SIZE = 64
BATCH_SIZE = 4
topk = [1]
# data_root = '/home/chaunm/Projects/dataset/medical-scan-classification'
data_root = '/home/chaunm/Projects/ZIP/datasets/'
data_name = 'Alzheimer'
# data_name = 'ChestCancer'
# data_name = 'KidneyCancer'

pretrained_path = '/home/chaunm/Projects/medical-cls/saved_models/classification/Alzheimer/model_42.pth'
# pretrained_path = "/home/chaunm/Projects/medical-cls/saved_models/classification/Chest Cancer/model_97.pth"
# pretrained_path = "/home/chaunm/Projects/medical-cls/saved_models/classification/Kidney Cancer/model_85.pth"
data = Medical_Classification(os.path.join(data_root, data_name, 'val'), IMG_SIZE)

labels = data.labels

print('Classes:', data.get_classes())
num_classes = len(data.get_classes())
model = CNNModel(num_classes).cuda()
model.load_state_dict(torch.load(pretrained_path))
# fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
model.eval()

# Example custom collate function
def adversarial_collate_fn(batch):
    # Unpack the batch
    images, labels = zip(*batch)
    images = torch.stack(images).cuda()
    labels = torch.tensor(labels).cuda()

    # Define your model (must be a PyTorch model compatible with Foolbox)
    # Assume `model` is already defined and loaded with your trained weights
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Define the attack
    attack = fb.attacks.LinfFastGradientAttack()

    # Convert images to NumPy arrays (required by Foolbox)
    # images_np = images.numpy()
    # labels_np = labels.numpy()

    # Generate adversarial examples
    _, adv_images, success = attack(fmodel, images, labels, epsilons=0.3)

    # Convert adversarial images back to PyTorch tensors
    # adv_images_tensor = torch.from_numpy(adv_images).float()

    return adv_images, labels

test_loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=adversarial_collate_fn)

predict_digits_arr = []
labels_arr = []
# images = torch.stack(data.imgs).cuda()
# labels = torch.tensor(data.labels).cuda()


for i, data in tqdm(enumerate(test_loader)):
    inputs, labels = data
    inputs = inputs.cuda()
    labels = labels.long().cuda()
    # _, advs, success = attack(fmodel, inputs, labels, epsilons=0.03)
    # print(advs[0].shape, len(advs), inputs.shape)
    # save_image(inputs, f'./imgs/{i}.png')
    
    with torch.no_grad():
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