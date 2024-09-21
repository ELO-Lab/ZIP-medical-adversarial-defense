import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import torchvision
import torch.nn.functional as F


class Medical_Classification(Dataset):
    def __init__(self, data_path, img_size):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
        ])

        self.data_path = data_path
        self.imgs = []
        self.labels = []
        self.classes_set = set()
        self.build_dataset()

    def build_dataset(self):
        for path, subdirs, files in os.walk(self.data_path):
            for name in files:
                img_path = os.path.join(path, name)
                temp = Image.open(img_path).convert('RGB')
                img = temp.copy()
                label = path.split('/')[-1]
                self.classes_set.add(label)
                self.imgs.append(img)
                self.labels.append(label)
                temp.close()
        
        sorted_classes = sorted(self.classes_set)
        self.class_to_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
        self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}

        indices = [self.class_to_index[cls] for cls in self.labels]
        self.labels = torch.tensor(indices)
        # self.imgs = torch.tensor(self.imgs)


        # sorted_classes = sorted(self.classes_set)
        # self.class_to_index = {cls: idx for idx, cls in enumerate(sorted_classes)}
        # self.index_to_class = {idx: cls for cls, idx in self.class_to_index.items()}

        # indices = [self.class_to_index[cls] for cls in self.labels]
        # indices = torch.tensor(indices)
        # num_classes = len(self.class_to_index)
        # self.labels = F.one_hot(indices, num_classes=num_classes)

    def __len__(self):
        return len(self.imgs)
    
    def get_classes(self):
        return list(self.classes_set)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        imgs = self.imgs[idx]
        labels = self.labels[idx]
        imgs = self.transform(imgs)
        return imgs, labels


class Medical_Segmentation(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            mask_dir (str): Path to the directory containing masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_list = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert mask to long tensor
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        return image, mask

if __name__=='__main__':
    data_root = '/home/chaunm/Projects/dataset/medical-scan-classification'
    data_name = 'Chest Cancer'
    IMG_SIZE = 64
    data = Medical_Classification(os.path.join(data_root, data_name), IMG_SIZE)
    num_classes = len(data.get_classes())
    print(num_classes)
#     root_dir = './COVID-19_Radiography_Dataset/COVID'
#     image_dir = os.path.join(root_dir, 'images')
#     mask_dir = os.path.join(root_dir, 'masks')

#     # Define transformations
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor()
#     ])

#     # Create dataset
#     dataset = Medical_Segmentation(image_dir=image_dir, mask_dir=mask_dir, transform=transform)

#     # Create data loader
#     # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#     print(len(dataset))