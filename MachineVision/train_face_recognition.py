import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import random

from tqdm import tqdm


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 512)  # 提取特征的输出维度

    def forward_one(self, x):
        return self.resnet(x)

    def forward(self, input1, input2):
        out1 = self.forward_one(input1)
        out2 = self.forward_one(input2)
        return torch.abs(out1 - out2)  # 输出两个图像的绝对差异，用于计算相似度

# 数据加载器，读取配对数据
class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.pairs_file = os.path.join(root_dir, 'pairs.csv')  # 假设pair文件名为pairs.csv
        self.image_dir = os.path.join(root_dir, 'lfw-deepfunneled')  # 图像所在的文件夹
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()[1:]  # Skip the header line
            for line in lines:
                parts = line.strip().split(',')
                person1, person2 = parts[0], parts[1]
                try:
                    same = int(parts[2])  # Try to convert the third column to an integer
                except ValueError:
                    print(f"Error converting value for {person1} and {person2}: {parts[2]}")
                    continue  # Skip this pair if conversion fails
                pairs.append((person1, person2, same))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person1, person2, same = self.pairs[idx]
        img1_path = os.path.join(self.image_dir, person1, f"{person1}_0001.jpg")  # 假设每个文件夹中都以'0001.jpg'作为文件名
        img2_path = os.path.join(self.image_dir, person2, f"{person2}_0001.jpg")

        img1 = Image.open(img1_path)
        img2 = Image.open(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, same  # 返回图像对和标签

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = LFWDataset(root_dir='E:/LearningProjects/MachineLearning/MachineVision/lfw', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 定义模型、损失函数和优化器
model = SiameseNetwork()
criterion = nn.BCEWithLogitsLoss()  # 用于二分类
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for img1, img2, same in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        optimizer.zero_grad()
        outputs = model(img1, img2)
        loss = criterion(outputs, same.float())  # 计算损失

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")

# 保存训练好的模型
torch.save(model.state_dict(), 'siamese_model.pth')
