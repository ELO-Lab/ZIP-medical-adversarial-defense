import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_V2_L_Weights
import torchvision


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.build_model()
        
    def build_model(self):
        base_model = torchvision.models.efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*(list(base_model.children())[:-1]))
        # Output torch.Size([1, 1280, 1, 1])
        for param in self.model.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.swish = nn.SiLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(1280, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)
        self.head = nn.Linear(128, self.num_classes)
    
    def forward(self, X):
        X = self.model(X)
        X = self.flatten(X)
        X = self.bn1(self.relu(self.fc1(X)))
        X = self.bn2(self.swish(self.fc2(X)))
        X = self.bn3(self.relu(self.fc3(X)))
        X = self.dropout(X)
        X = self.head(X)
        return X
    
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)  # After 3 max-pooling layers, 64x64 image becomes 8x8
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
if __name__=='__main__':
    x = torch.rand(8, 3, 64, 64)
    model = CNN(64, 10)
    print(model(x).shape)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(0.5)

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(0.5)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(0.5)

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(0.5)

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256)
        )

        self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        )

        self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        )

        self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, padding=0)
        self.conv9 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.conv10 = nn.Conv2d(32, self.num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.drop1(self.pool1(c1))

        c2 = self.conv2(p1)
        p2 = self.drop2(self.pool2(c2))

        c3 = self.conv3(p2)
        p3 = self.drop3(self.pool3(c3))

        c4 = self.conv4(p3)
        p4 = self.drop4(self.pool4(c4))

        c5 = self.conv5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.conv6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.conv7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.conv8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.conv9(u9)

        outputs = torch.sigmoid(self.conv10(c9))

        return outputs